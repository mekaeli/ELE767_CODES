
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
import random


# ==================== convert_label =========================
def convert_label(label: int, nb_sorties: int) -> list[int]:
	"""Convertit un label en vecteur binaire (one-hot) de taille `nb_sorties`.

	Cas spécial (nb_sorties=10, chiffres):
	- ordre des classes = 1,2,3,4,5,6,7,8,9,0
	  donc 0 est encodé sur la dernière position.

	Sinon (générique):
	- accepte des labels 0..nb_sorties-1 (0-based)
	- accepte aussi 1..nb_sorties (1-based)
	- si le label est hors plage, applique un mapping modulo (lab % nb_sorties)
	  pour forcer l'encodage dans la taille demandée.
	"""
	try:
		lab = int(label)
		n_out = int(nb_sorties)
	except Exception as exc:
		raise ValueError(
			f"arguments invalides: label={label!r}, nb_sorties={nb_sorties!r}"
		) from exc

	if n_out <= 0:
		raise ValueError("nb_sorties doit être >= 1")

	# Cas digits: nb_sorties=10, classes {1..9,0}
	if n_out == 10:
		if not (0 <= lab <= 9):
			raise ValueError(
				"label invalide pour nb_sorties=10 (attendu un chiffre 0..9)"
			)
		vec = [0] * 10
		idx = 9 if lab == 0 else (lab - 1)
		vec[idx] = 1
		return vec

	# Cas générique: 0-based prioritaire
	if 0 <= lab < n_out:
		idx = lab
	elif 1 <= lab <= n_out:
		idx = lab - 1
	else:
		# Exigence: ne pas lever d'erreur, mais replier via modulo.
		# Exemple: label=8, nb_sorties=5 -> idx=8%5=3.
		idx = int(lab) % int(n_out)

	vec = [0] * n_out
	vec[idx] = 1
	return vec


# ==================== _parse_dataset_line =========================
def _parse_dataset_line(
	line: str,
	*,
	n_in: int,
	nb_sorties: int,
	convert_label: Callable[[int, int], list[int]] = convert_label,
) -> tuple[list[int], list[float]]:
	"""Parse une ligne de dataset `label: x1 x2 ...` -> (Dn_s, Xn)."""
	use_n_in = int(n_in)
	use_nb_sorties = int(nb_sorties)
	if use_n_in <= 0:
		raise ValueError("n_in doit être >= 1")
	if use_nb_sorties <= 0:
		raise ValueError("nb_sorties doit être >= 1")

	s = (line or "").strip()
	if not s:
		raise ValueError("Ligne invalide (vide)")
	if ":" not in s:
		raise ValueError(f"Ligne invalide (pas de ':'): {s[:80]!r}")

	left, _sep, right = s.partition(":")
	label_str = left.strip()
	values_str = right.strip()

	Dn_s = convert_label(int(label_str), use_nb_sorties)
	tokens = values_str.split()
	if not tokens:
		raise ValueError(f"Ligne invalide (aucune valeur X): {s[:80]!r}")
	try:
		X_all = [float(t.replace(",", ".")) for t in tokens]
	except Exception as exc:
		raise ValueError(f"Valeur X invalide sur la ligne: {s[:80]!r}") from exc

	if len(X_all) < use_n_in:
		raise ValueError(f"Taille Xn insuffisante: {len(X_all)} (attendu >= {use_n_in})")
	Xn = X_all[:use_n_in]
	return Dn_s, Xn


# ==================== resolve_validation_file =========================
def resolve_validation_file(
	n_in: int,
	*,
	root_dir: Path,
	fichier: Path | None = None,
) -> Path:
	"""Retourne le chemin du fichier de validation.

	- Par défaut: `root_dir / f"{n_in}_data_vc.txt"`
	- Fallback: `root_dir / "data_vc.txt"`
	- Si `fichier` est fourni, il est utilisé tel quel.
	"""
	if fichier is not None:
		file_path = Path(fichier)
		if not file_path.exists():
			raise FileNotFoundError(f"Fichier introuvable: {file_path}")
		return file_path

	use_n_in = int(n_in)
	file_path = root_dir / f"{use_n_in}_data_vc.txt"
	if file_path.exists():
		return file_path

	fallback = root_dir / "data_vc.txt"
	if fallback.exists():
		return fallback

	raise FileNotFoundError(f"Fichier de validation introuvable: {file_path}")


@lru_cache(maxsize=32)
# ==================== _read_non_empty_lines_cached =========================
def _read_non_empty_lines_cached(file_path_str: str) -> tuple[str, ...]:
	"""Lit et met en cache les lignes non vides d'un fichier.

	Le cache est basé sur le chemin (string). Si le fichier change sur disque,
	il faut vider le cache avec `_read_non_empty_lines_cached.cache_clear()`.
	"""
	file_path = Path(file_path_str)
	lines: list[str] = []
	with open(file_path, "r", encoding="utf-8") as f:
		for raw in f:
			line = raw.strip()
			if line:
				lines.append(line)
	return tuple(lines)


# ==================== count_non_empty_lines =========================
def count_non_empty_lines(file_path: Path) -> int:
	"""Compte le nombre de lignes non vides (strip) dans un fichier."""
	return len(_read_non_empty_lines_cached(str(Path(file_path))))


# ==================== get_validation_sample_seq =========================
def get_validation_sample_seq(
	file_path: Path,
	*,
	n_ligne: int,
	n_in: int,
	nb_sorties: int,
	convert_label: Callable[[int, int], list[int]] = convert_label,
) -> tuple[list[list[int]], list[list[float]]]:
	"""Retourne l'exemple à la ligne `n_ligne` sous forme de listes (D_list, X_list).

	Spécification
	- Reçoit `n_ligne`: numéro de la ligne à lire (1-based)
	- Valide `n_ligne`
	- Retourne uniquement (D_list, X_list)

	Notes
	- Les lignes vides sont ignorées (donc `n_ligne` compte parmi les lignes non vides).
	- (D_list, X_list) contiennent chacun un seul élément.
	"""
	try:
		idx_1 = int(n_ligne)
	except Exception as exc:
		raise ValueError(f"n_ligne invalide: {n_ligne!r}") from exc

	if idx_1 < 1:
		raise ValueError("n_ligne doit être >= 1")

	lines = _read_non_empty_lines_cached(str(Path(file_path)))
	if idx_1 > len(lines):
		raise ValueError(
			f"n_ligne hors plage: {idx_1} (attendu 1..{len(lines)})"
		)

	line = lines[idx_1 - 1]
	Dn_s, Xn = _parse_dataset_line(
		line,
		n_in=n_in,
		nb_sorties=nb_sorties,
		convert_label=convert_label,
	)
	return [Dn_s], [Xn]


# ==================== _load_all_samples_cached =========================
def _load_all_samples_cached(
	file_path: Path,
	*,
	n_in: int,
	nb_sorties: int,
	convert_label: Callable[[int, int], list[int]] = convert_label,
) -> tuple[list[list[int]], list[list[float]]]:
	"""Charge toutes les lignes non vides via le cache et retourne (D_list, X_list)."""
	D_list: list[list[int]] = []
	X_list: list[list[float]] = []
	lines = _read_non_empty_lines_cached(str(Path(file_path)))
	for line in lines:
		Dn_s, Xn = _parse_dataset_line(
			line,
			n_in=n_in,
			nb_sorties=nb_sorties,
			convert_label=convert_label,
		)
		D_list.append(Dn_s)
		X_list.append(Xn)
	return D_list, X_list


# ==================== get_validation_samples_random =========================
def get_validation_samples_random(
	file_path: Path,
	*,
	n_in: int,
	nb_sorties: int,
	convert_label: Callable[[int, int], list[int]] = convert_label,
	seed: int | None = None,

) -> tuple[list[list[int]], list[list[float]]]:
	"""Lit les données de validation en ordre aléatoire et retourne (D_shuffled, X_shuffled).

	Propriété importante
	- Contenu identique à la lecture séquentielle: les couples (Dn_s, Xn) sont
	  strictement les mêmes, seul l'ordre change.
	- Si `seed` est fourni, l'ordre est déterministe/reproductible.
	"""
	D_list, X_list = _load_all_samples_cached(
		file_path,
		n_in=n_in,
		nb_sorties=nb_sorties,
		convert_label=convert_label,
	)
	indices = list(range(len(D_list)))
	rng = random.Random(seed)
	rng.shuffle(indices)
	D_shuffled = [D_list[i] for i in indices]
	X_shuffled = [X_list[i] for i in indices]
	return D_shuffled, X_shuffled
