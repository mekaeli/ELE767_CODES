"""c2p69

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


# ==================== test_reso =========================
def test_reso() -> None:
	"""Construit un petit réseau, le met dans struct_reso, puis lance 1 itération."""
	from neurone.reso2neurone import backpp, mon_reso

	# ==================== Configuration du réseau (rarement modifiée) ====================
	# Topologie: 2 entrées -> couche 1 (2 neurones) -> couche 2 (3 neurones)
	n_in = 2
	N_b = [2, 3]
	n_c = 2
	# Intervalle de génération aléatoire des biais et des poids
	biais = [1, 5]
	poids = [-0.1, 0.1]
	# Domaines possibles pour initialiser X (entrées) et D (sorties)
	X_domain = [0, 1]
	D_domain = [0, 1]

	# ==================== À MODIFIER MANUELLEMENT (par l'utilisateur) ====================
	# Objectif: fournir directement les 4 objets attendus par reso.set_reso():
	#    - Xn   : [x1, x2, ...]
	#    - Wn_c : [W1_aplatie, W2_aplatie, ...]  (ordre row-major)
	#    - Bn_c : [b1, b2, ...]
	#    - Dn_s : [d1, d2, ...]
	#
	# Exemple (couche 1 = 2x2):
	#   W1_aplatie = [w11_1, w12_1, w21_1, w22_1]
	# Exemple (couche 2 = 2x3):
	#   W2_aplatie = [w11_2, w12_2, w13_2, w21_2, w22_2, w23_2]

	# 1) Entrées
	x1, x2 = 2.0, 4.0
	Xn = [x1, x2]

	# 2) Poids + biais (couche 1 : 2 -> 2)
	w11_1, w12_1 = 1.0, 3.0
	w21_1, w22_1 = 2.0, 4.0
	W1_aplatie = [w11_1, w12_1, w21_1, w22_1]

	b1_1, b2_1 = 2.0, 1.0
	b1 = [b1_1, b2_1]

	# 3) Poids + biais (couche 2 : 2 -> 3)
	w11_2, w12_2, w13_2 = 0.3, -0.2, 0.2
	w21_2, w22_2, w23_2 = 1.3, -0.4, -1.2
	W2_aplatie = [w11_2, w12_2, w13_2, w21_2, w22_2, w23_2]

	b1_2, b2_2, b3_2 = -0.1, 0.1, -0.2
	b2 = [b1_2, b2_2, b3_2]

	# 4) Sorties désirées
	d1, d2, d3 = 1.0, 2.0, 1.0

	# 5) Assemblage final (format set_reso)
	Xn = [x1, x2]
	Wn_c = [W1_aplatie, W2_aplatie]
	Bn_c = [b1, b2]
	Dn_s = [d1, d2, d3]

	# 6) Hyperparamètres
	n_fct = 1
	eta = 0.1

	# ==================== Exécution (NE PAS MODIFIER) ====================

	reso = mon_reso(n_in=n_in, n_c=n_c, N_b=N_b, biais=biais, poids=poids, X=X_domain, D=D_domain)
	reso.cree_reso()
	struct_reso = reso.get_reso()

	try:
		reso.set_reso(struct_reso, Xn, Wn_c, Bn_c, Dn_s)
	except ValueError as exc:
		print(f"ERREUR (set_reso): {exc}")
		raise

	reso.Affiche_reso(struct_reso, precision=PRECISION, n_fct=n_fct, eta=eta)

	# Résolution identique à c2p69_mat.py (avec affichage géré dans backpp)
	solver = backpp(struct_reso, n_fct=n_fct, eta=eta)
	solver.resolution_affiche(update_struct=True, show_params=False, precision=PRECISION)


if __name__ == "__main__":
	console_utils.clear_console()
	test_reso()
