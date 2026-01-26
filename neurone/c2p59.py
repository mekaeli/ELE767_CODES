"""c2p59

Fichier de test pour la classe `mon_reso`.
"""

from __future__ import annotations

import os
import sys

# Permet d'exécuter ce fichier directement (ajoute la racine du projet à sys.path)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
	sys.path.insert(0, _PROJECT_ROOT)

from neurone.reso2neurone import console_utils


PRECISION = 6


# ==================== _flatten_row_major =========================
def _flatten_row_major(M: list[list[float]]) -> list[float]:
	"""Aplatis une matrice en une liste (ordre ligne).

	Exemple:
		[[1,2],[3,4]] -> [1,2,3,4]
	"""
	return [float(v) for row in M for v in row]


# ==================== _pack_params =========================
def _pack_params(
	*,
	X: list[float],
	W_list: list[list[list[float]]],
	b_list: list[list[float]],
	d_list: list[float],
) -> tuple[list[float], list[list[float]], list[list[float]], list[float]]:
	"""Convertit X/W/b/d vers le format attendu par mon_reso.set_reso.

	Format attendu:
		- Xn: list[float]
		- Wn_c: list[list[float]] où chaque couche est une liste aplatie row-major
		- Bn_c: list[list[float]] où chaque couche est un vecteur de biais
		- Dn_s: list[float]

	Validation minimale des dimensions pour rester "dynamique" si le réseau change.
	"""
	if not W_list:
		raise ValueError("W_list ne peut pas être vide.")
	if len(W_list) != len(b_list):
		raise ValueError(
			f"Incohérence: len(W_list)={len(W_list)} != len(b_list)={len(b_list)}"
		)

	# Vérifie que la première couche est compatible avec X
	n_in = len(X)
	if len(W_list[0]) != n_in:
		raise ValueError(
			f"Dimension invalide: W1 a {len(W_list[0])} lignes, attendu {n_in} (len(X))."
		)

	# Vérifie que chaque b correspond au nombre de neurones de la couche
	for c, (Wc, bc) in enumerate(zip(W_list, b_list), start=1):
		n_dst = len(Wc[0]) if Wc else 0
		if any(len(row) != n_dst for row in Wc):
			raise ValueError(f"W{c}: matrice non rectangulaire.")
		if len(bc) != n_dst:
			raise ValueError(f"b{c}: taille invalide. Reçu {len(bc)}, attendu {n_dst}.")
		# Vérifie le chaînage des couches
		if c >= 2:
			prev_dst = len(W_list[c - 2][0]) if W_list[c - 2] else 0
			if len(Wc) != prev_dst:
				raise ValueError(
					f"Dimension invalide: W{c} a {len(Wc)} lignes, attendu {prev_dst} (sorties couche précédente)."
				)

	Xn = [float(v) for v in X]
	Wn_c = [_flatten_row_major(Wc) for Wc in W_list]
	Bn_c = [[float(v) for v in bc] for bc in b_list]
	Dn_s = [float(v) for v in d_list]
	return Xn, Wn_c, Bn_c, Dn_s


# ==================== test_reso =========================
def test_reso() -> None:
	"""Construit un petit réseau, le met dans struct_reso, puis lance 1 itération."""
	from neurone.reso2neurone import backpp, mon_reso

	# ==================== Paramètres (style c2p59_mat.py) ====================
	# entrees
	x1 = 1
	x2 = 0
	X = [x1, x2]

	# couche 1
	w11_1, w12_1 = 3, 4
	w21_1, w22_1 = 6, 5
	b1_1 = 1
	b2_1 = 0
	W1 = [[w11_1, w12_1], [w21_1, w22_1]]
	b1 = [b1_1, b2_1]

	# couche 2
	w11_2, w21_2 = 2, 4
	b1_2 = -3.92
	W2 = [[w11_2], [w21_2]]
	b2 = [b1_2]

	# sorties desirées
	d1 = 1
	d_list = [d1]

	# param
	n_fct = 1
	eta = 0.1

	# Variables de configuration du réseau (dynamiques)
	W_list = [W1, W2]
	b_list = [b1, b2]
	n_in = len(X)
	N_b = [len(bc) for bc in b_list]
	n_c = len(N_b)
	# Intervalle de génération aléatoire des biais et des poids
	biais = [1, 5]
	poids = [-0.1, 0.1]
	# Domaines possibles pour initialiser X (entrées) et D (sorties)
	X_domain = [0, 1]
	D_domain = [0, 1]

	reso = mon_reso(n_in=n_in, n_c=n_c, N_b=N_b, biais=biais, poids=poids, X=X_domain, D=D_domain)
	reso.cree_reso()
	struct_reso = reso.get_reso()

	# Mise à jour de la structure (packaging dynamique)
	Xn, Wn_c, Bn_c, Dn_s = _pack_params(X=[x1, x2], W_list=W_list, b_list=b_list, d_list=d_list)
	print("\n--- Packaging (utilisé pour set_reso) ---")
	print(f"Xn   = {Xn}")
	print(f"Wn_c = {Wn_c}")
	print(f"Bn_c = {Bn_c}")
	print(f"Dn_s = {Dn_s}")

	try:
		reso.set_reso(struct_reso, Xn, Wn_c, Bn_c, Dn_s)
	except ValueError as exc:
		print(f"ERREUR (set_reso): {exc}")
		raise

	reso.Affiche_reso(struct_reso, precision=PRECISION, n_fct=n_fct, eta=eta)

	# Résolution identique à c2p59_mat.py (avec affichage géré dans backpp)
	solver = backpp(struct_reso, n_fct=n_fct, eta=eta)
	solver.resolution_affiche(update_struct=True, show_params=False, precision=PRECISION)


if __name__ == "__main__":
	# Nettoie la console (démo minimale).
	console_utils.clear_console()
	test_reso()
