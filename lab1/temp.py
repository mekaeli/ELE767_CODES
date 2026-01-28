"""temp

Copie fonctionnelle de `neurone/c2p59.py` (sans modifier ce fichier), avec une
fonction utilitaire qui exécute le même scénario et append une ligne dans
`lab1/parametres.txt`.

Format écrit (une struct_reso par ligne, append):
	[nom_fonction d'activation] [n_in n_c-1 N_b[-2] N_b[-1] eta] [Wn_c] [Bn_c] [60%]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Permet d'exécuter ce fichier directement (ajoute la racine du projet à sys.path)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))

from neurone.reso2neurone import console_utils


PRECISION = 6


def _env_int(name: str, default: int) -> int:
	try:
		return int(os.environ.get(name, str(default)).strip())
	except Exception:
		return default


def _env_float(name: str, default: float) -> float:
	try:
		return float(os.environ.get(name, str(default)).strip().replace(",", "."))
	except Exception:
		return default


def _activation_name(n_fct: int) -> str:
	return {1: "sigmoïde", 2: "tan", 3: "tanh", 4: "gelu"}.get(int(n_fct), f"n_fct={n_fct}")


def _extract_WB_flat(struct_reso: Dict[str, object]) -> Tuple[List[List[float]], List[List[float]]]:
	"""Extrait Wn_c et Bn_c de struct_reso en valeurs numériques (sans les noms)."""
	kw = next((k for k in struct_reso.keys() if isinstance(k, str) and k.startswith("W")), None)
	kb = next((k for k in struct_reso.keys() if isinstance(k, str) and k.startswith("B")), None)
	if kw is None or kb is None:
		raise ValueError("struct_reso invalide: clés W*/B* introuvables")

	W_layers = struct_reso[kw]
	B_layers = struct_reso[kb]
	if not isinstance(W_layers, list) or not isinstance(B_layers, list):
		raise ValueError("struct_reso invalide: W*/B* doivent être des listes")

	Wn_c = [[float(val) for (_name, val) in layer] for layer in W_layers]
	Bn_c = [[float(val) for (_name, val) in layer] for layer in B_layers]
	return Wn_c, Bn_c


def _fmt_float(x: float, *, decimals: int = PRECISION) -> str:
	"""Formate un float sans notation scientifique.

	- Utilise un format fixe (decimals) puis supprime les zéros inutiles.
	- 0.000000... devient 0
	"""
	try:
		xf = float(x)
	except Exception:
		return str(x)

	if abs(xf) < 10 ** (-(decimals + 2)):
		xf = 0.0

	s = f"{xf:.{int(decimals)}f}"
	if "." in s:
		s = s.rstrip("0").rstrip(".")
	if s == "-0":
		s = "0"
	return s


def _fmt_list_of_floats(values: List[float]) -> str:
	return "[" + ", ".join(_fmt_float(v) for v in values) + "]"


def _fmt_layers(layers: List[List[float]]) -> str:
	return "[" + ", ".join(_fmt_list_of_floats(layer) for layer in layers) + "]"


def _format_parametres_line(
	*,
	activation_name: str,
	n_in: int,
	n_c: int,
	N_b: List[int],
	eta: float,
	Wn_c: List[List[float]],
	Bn_c: List[List[float]],
	pourcentage: str = "60%",
) -> str:
	"""Construit une ligne au format demandé pour parametres.txt."""
	n_c_moins_1 = int(n_c) - 1
	# On écrit les deux dernières tailles de couches: N_b[-2] et N_b[-1]
	if len(N_b) >= 2:
		nb_prev = int(N_b[-2])
		nb_last = int(N_b[-1])
	elif len(N_b) == 1:
		nb_prev = int(N_b[-1])
		nb_last = int(N_b[-1])
	else:
		nb_prev = 0
		nb_last = 0
	eta_s = _fmt_float(float(eta))
	W_s = _fmt_layers(Wn_c)
	B_s = _fmt_layers(Bn_c)
	return (
		f"[{activation_name}] "
		f"[{int(n_in)} {n_c_moins_1} {nb_prev} {nb_last} {eta_s}] "
		f"{W_s} "
		f"{B_s} "
		f"[{pourcentage}]"
	)


# ==================== test_reso =========================
def test_reso(*, n_fct: int | None = None, eta: float | None = None, return_struct: bool = False):
	"""Construit un petit réseau, le met dans struct_reso, puis lance 1 itération.

	- Par défaut, n_fct et eta viennent des variables d'environnement N_FCT / ETA.
	- Si return_struct=True, retourne (struct_reso, meta).
	"""
	from neurone.reso2neurone import backpp, mon_reso

	# ==================== Configuration du réseau (rarement modifiée) ====================
	n_in = 2
	N_b = [2, 1]
	n_c = 2
	biais = [1, 5]
	poids = [-0.1, 0.1]
	X_domain = [0, 1]
	D_domain = [0, 1]

	# ==================== À MODIFIER MANUELLEMENT (par l'utilisateur) ====================
	# 2) Entrées
	x1 = 1.0
	x2 = 0.0

	# 3) Poids + biais (couche 1 : 2 -> 2)
	w11_1, w12_1 = 3.0, 4.0
	w21_1, w22_1 = 6.0, 5.0
	W1_aplatie = [w11_1, w12_1, w21_1, w22_1]

	b1_1, b2_1 = 1.0, 0.0
	b1 = [b1_1, b2_1]

	# 4) Poids + biais (couche 2 : 2 -> 1)
	w11_2, w21_2 = 2.0, 4.0
	W2_aplatie = [w11_2, w21_2]

	b1_2 = -3.92
	b2 = [b1_2]

	# 5) Sortie désirée
	d1 = 1.0

	# 6) Assemblage final (format set_reso)
	Xn = [x1, x2]
	Wn_c = [W1_aplatie, W2_aplatie]
	Bn_c = [b1, b2]
	Dn_s = [d1]

	# 6) Hyperparamètres
	if n_fct is None:
		n_fct = _env_int("N_FCT", 1)
	if eta is None:
		eta = _env_float("ETA", 0.1)

	# ==================== Exécution (NE PAS MODIFIER) ====================
	reso = mon_reso(n_in=n_in, n_c=n_c, N_b=N_b, biais=biais, poids=poids, X=X_domain, D=D_domain)
	reso.cree_reso()
	struct_reso = reso.get_reso()

	try:
		reso.set_reso(struct_reso, Xn, Wn_c, Bn_c, Dn_s)
	except ValueError as exc:
		print(f"ERREUR (set_reso): {exc}")
		raise

	reso.Affiche_reso(struct_reso, precision=PRECISION, n_fct=int(n_fct), eta=float(eta))

	# Résolution identique à c2p59_mat.py (avec affichage géré dans backpp)
	solver = backpp(struct_reso, n_fct=int(n_fct), eta=float(eta))
	solver.resolution_affiche(update_struct=True, show_params=False, precision=PRECISION)

	if return_struct:
		Wn_c_export, Bn_c_export = _extract_WB_flat(struct_reso)
		meta = {
			"n_in": int(n_in),
			"n_c": int(n_c),
			"N_b": [int(x) for x in N_b],
			"eta": float(eta),
			"n_fct": int(n_fct),
			"activation_name": _activation_name(int(n_fct)),
			"Wn_c": Wn_c_export,
			"Bn_c": Bn_c_export,
		}
		return struct_reso, meta
	return None


def execute_and_append_parametres(
	*,
	parametres_path: str | os.PathLike | None = None,
	n_fct: int | None = None,
	eta: float | None = None,
	pourcentage: str = "60%",
) -> str:
	"""Exécute le scénario type c2p59 et append une ligne dans parametres.txt.

	Retourne la ligne écrite.
	"""
	struct_reso, meta = test_reso(n_fct=n_fct, eta=eta, return_struct=True)
	_ = struct_reso

	line = _format_parametres_line(
		activation_name=str(meta["activation_name"]),
		n_in=int(meta["n_in"]),
		n_c=int(meta["n_c"]),
		N_b=list(meta["N_b"]),
		eta=float(meta["eta"]),
		Wn_c=list(meta["Wn_c"]),
		Bn_c=list(meta["Bn_c"]),
		pourcentage=pourcentage,
	)

	if parametres_path is None:
		parametres_path = _PROJECT_ROOT / "lab1" / "parametres.txt"

	path = Path(parametres_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as f:
		f.write(line + "\n")
	return line


def executer_logique_et_ecrire_parametres(
	*,
	parametres_path: str | os.PathLike | None = None,
	n_fct: int | None = None,
	eta: float | None = None,
	pourcentage: str = "60%",
) -> str:
	"""Fonction demandée: exécute la logique de temp.py puis append dans parametres.txt.

	- Une struct_reso par ligne
	- N'efface pas les données existantes (append)
	- Format:
	  [nom_fonction d'activation] [n_in n_c-1 N_b[-2] N_b[-1] eta] [Wn_c] [Bn_c] [60%]

	Retourne la ligne écrite.
	"""
	return execute_and_append_parametres(
		parametres_path=parametres_path,
		n_fct=n_fct,
		eta=eta,
		pourcentage=pourcentage,
	)


if __name__ == "__main__":
	# Nettoie la console (démo minimale).
	console_utils.clear_console()

	# Par défaut, on exporte TOUJOURS (append) dans lab1/parametres.txt.
	# Pour désactiver l'export: NO_EXPORT=1/true/yes
	no_export = os.environ.get("NO_EXPORT", "").strip().lower() in {"1", "true", "yes", "y"}
	if no_export:
		test_reso()
	else:
		line = executer_logique_et_ecrire_parametres(
			n_fct=_env_int("N_FCT", 1),
			eta=_env_float("ETA", 0.1),
			pourcentage=os.environ.get("POURCENTAGE", "60%"),
		)
		print(f"\nLigne ajoutée à parametres.txt:\n{line}")
