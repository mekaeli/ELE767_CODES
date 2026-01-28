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

# ==================== test_reso =========================
def test_reso() -> None:
	"""Construit un petit réseau, le met dans struct_reso, puis lance 1 itération."""
	from neurone.reso2neurone import backpp, mon_reso

	# ==================== Configuration du réseau (rarement modifiée) ====================
	# Ces valeurs décrivent la topologie utilisée pour créer le squelette du réseau.
	# Si tu changes le nombre d'entrées ou la taille des couches, il faut mettre à
	# jour n_in / N_b / n_c pour correspondre.
	n_in = 2
	N_b = [2, 1]
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
	#    Exemple (couche 1 = 2x2):
	#      W1_aplatie = [w11_1, w12_1, w21_1, w22_1]
	#    Exemple (couche 2 = 2x1):
	#      W2_aplatie = [w11_2, w21_2]

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
	n_fct = _env_int("N_FCT", 1)
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

	reso.Affiche_reso(struct_reso, precision=PRECISION, n_fct=n_fct, eta=eta)

	# Résolution identique à c2p59_mat.py (avec affichage géré dans backpp)
	solver = backpp(struct_reso, n_fct=n_fct, eta=eta)
	solver.resolution_affiche(update_struct=True, show_params=False, precision=PRECISION)


if __name__ == "__main__":
	# Nettoie la console (démo minimale).
	console_utils.clear_console()
	test_reso()
