"""service

Rôle
	Services "non-UI" utilisés par le laboratoire 1.

	Ce module encapsule principalement la gestion du fichier de configuration
	`parametres.txt` :
		- lecture/écriture/suppression de lignes
		- validation du nouveau format (blocs entre crochets)

Utilisation
	- Appelé depuis l'interface (UI) et depuis le lanceur.
	- Ne doit pas faire planter l'application sur une erreur d'I/O.

Notes
	Les fonctions sont volontairement simples et sans dépendances externes.
"""

###################################
###  fonction complementaire 1  ###
###################################

from __future__ import annotations

from pathlib import Path
from typing import Sequence

try:
	from lab1 import layout  # import via package si disponible
except Exception:
	import layout  # fallback local


def read_parametres_text(parametres_path: str | Path | None = None) -> str:
	"""Lit `parametres.txt` et retourne tout le contenu en un bloc.

	- Par défaut, lit `lab1/parametres.txt`.
	- Retourne une chaîne vide si le fichier n'existe pas.
	"""
	default_path = Path(__file__).with_name("parametres.txt")
	path = Path(parametres_path) if parametres_path is not None else default_path
	try:
		return path.read_text(encoding="utf-8")
	except FileNotFoundError:
		return ""
	except Exception:
		# Ne pas faire planter le lanceur sur une erreur d'I/O.
		return ""


def add_parametres_line(
	formatted_line: str,
	parametres_path: str | Path | None = None,
) -> tuple[bool, str]:
	"""Ajoute une ligne formatée dans parametres.txt si elle n'existe pas déjà.

	Retourne (ok, message):
	- ok=True  : la ligne a été ajoutée
	- ok=False : doublon ou erreur
	"""
	line = (formatted_line or "").strip()
	if not line:
		return False, "Ligne vide"
	try:
		validate_parametres_line_new_format(line)
	except Exception as exc:
		return False, f"Format invalide: {exc}"

	default_path = Path(__file__).with_name("parametres.txt")
	path = Path(parametres_path) if parametres_path is not None else default_path

	try:
		existing = path.read_text(encoding="utf-8") if path.exists() else ""
		existing_lines = [l.strip() for l in existing.splitlines() if l.strip()]
		if any(l == line for l in existing_lines):
			return False, "Refus d'écriture: doublon déjà présent dans parametres.txt"

		# Si le fichier est vide, on écrit l'entête d'abord.
		out_lines: list[str] = []
		if not existing_lines:
			out_lines.append(layout.PARAMETRES_HEADER)
		else:
			out_lines.extend(existing.splitlines())
			# S'assure que la dernière ligne finit par une newline à l'écriture.

		out_lines.append(line)
		path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
		return True, ""
	except Exception as exc:
		return False, f"Erreur d'écriture: {exc}"


def delete_parametres_line(
	raw_line: str,
	parametres_path: str | Path | None = None,
) -> tuple[bool, str]:
	"""Supprime une ligne exacte de parametres.txt.

	- Ne supprime pas l'en-tête.
	- La comparaison se fait sur la ligne strip() (exact match).
	"""
	needle = (raw_line or "").strip()
	if not needle:
		return False, "Ligne vide"

	default_path = Path(__file__).with_name("parametres.txt")
	path = Path(parametres_path) if parametres_path is not None else default_path
	if not path.exists():
		return False, "parametres.txt introuvable"

	try:
		content = path.read_text(encoding="utf-8")
		lines = content.splitlines()
		kept: list[str] = []
		removed = 0
		for i, line in enumerate(lines):
			strip_line = (line or "").strip()
			# Conserve l'en-tête même si la chaîne correspond.
			if i == 0 and strip_line.startswith("[Focntion act.]"):
				kept.append(line)
				continue
			if strip_line == needle:
				removed += 1
				continue
			kept.append(line)

		if removed == 0:
			return False, "Ligne non trouvée dans parametres.txt"
		path.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")
		return True, ""
	except Exception as exc:
		return False, f"Erreur de suppression: {exc}"


def _extract_bracket_groups(text: str) -> list[str]:
	"""Extrait les groupes `[...]` de premier niveau d'une ligne.

	Retourne les contenus sans les crochets externes.
	"""
	groups: list[str] = []
	buf: list[str] = []
	depth = 0
	in_group = False
	for ch in text:
		if ch == "[":
			if depth == 0:
				in_group = True
				buf = []
			else:
				buf.append(ch)
			depth += 1
		elif ch == "]":
			if depth == 0:
				continue
			depth -= 1
			if depth == 0 and in_group:
				groups.append("".join(buf).strip())
				in_group = False
			else:
				buf.append(ch)
		else:
			if in_group:
				buf.append(ch)
	return groups


def _parse_int_list(text: str) -> list[int] | None:
	"""Parse une liste d'entiers depuis un texte (ex: "2, 3" / "[2,3]")."""
	raw = (text or "").strip()
	if not raw:
		return []
	clean = raw.replace("[", "").replace("]", "").strip()
	if not clean:
		return []
	if "," in clean:
		parts = [p.strip() for p in clean.split(",") if p.strip()]
	else:
		parts = [p.strip() for p in clean.split() if p.strip()]
	try:
		return [int(float(p.replace(",", "."))) for p in parts]
	except Exception:
		return None


def validate_parametres_line_new_format(line: str) -> None:
	"""Valide strictement une ligne parametres.txt selon le nouveau format.

	Règle imposée:
	- `nb_neurones` contient uniquement les couches cachées (exactement nb_couches valeurs)
	- `nb_sorties` est la taille de la couche de sortie (séparée)
	"""
	s = (line or "").strip()
	groups = _extract_bracket_groups(s)
	if len(groups) < 3:
		raise ValueError("ligne illisible (groupes [] insuffisants)")
	inner = _extract_bracket_groups(groups[1])
	if len(inner) < 5:
		raise ValueError("bloc [[...]] invalide (attendu 5 champs)")

	try:
		nb_entrees = int(inner[0].strip())
		nb_couches = int(inner[1].strip())
		nb_sorties = int(inner[3].strip())
	except Exception:
		raise ValueError("nb_entrees/nb_couches/nb_sorties doivent être des entiers")

	if nb_entrees <= 0:
		raise ValueError("nb_entrees doit être ≥ 1")
	if nb_couches < 0:
		raise ValueError("nb_couches doit être ≥ 0")
	if nb_sorties <= 0:
		raise ValueError("nb_sorties doit être ≥ 1")

	vals = _parse_int_list(inner[2])
	if vals is None:
		raise ValueError("nb_neurones illisible")
	if len(vals) != nb_couches:
		raise ValueError("nb_neurones doit contenir exactement nb_couches valeurs")
	if any(int(v) <= 0 for v in vals):
		raise ValueError("nb_neurones: toutes les valeurs doivent être ≥ 1")


###################################
###  fonction complementaire 2  ###
###################################


def fonction_max(valeurs: Sequence[float]) -> list[int]:
	"""Convertit une liste analogique en liste binaire (0/1) via la règle MAX.

	Règle:
	- l'élément de plus grande valeur devient 1
	- tous les autres deviennent 0

	Note:
	- en cas d'égalité, le premier maximum reçoit le 1 (comportement argmax).
	"""
	if not valeurs:
		return []

	# Argmax en une passe (tie-break: premier maximum).
	max_index = max(range(len(valeurs)), key=lambda i: valeurs[i])
	return [1 if i == max_index else 0 for i in range(len(valeurs))]
