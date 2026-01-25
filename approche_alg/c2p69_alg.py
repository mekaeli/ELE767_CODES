"""Exercice c2p69 — Réseau 2→2→3.

Ce script montre toutes les étapes du calcul:
- choix de la fonction d'activation avec n_fct
- affichage pédagogique (Entrées / Sorties)
- propagation avant + rétropropagation (deltas, corrections, mises à jour)

Note:
	Les valeurs (poids/biais/cibles) proviennent du schéma fourni.
"""

# Configuration du chemin pour importer les modules du dossier parent
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import des fonctions nécessaires
from generic_func import clear_console
from neurone.backpp_neurone_alg import Neurone


_IND = "\t"


# ==================== choisir_fonction_activation =========================
def choisir_fonction_activation(n_fct: int):
	"""Choisit la fonction d'activation selon n_fct.

	Règle (ordre officiel):
		- n_fct = 1 : sigmoïde
		- n_fct = 2 : tan
		- n_fct = 3 : tanh
		- n_fct = 4 : GELU

	Retour:
		(nom_affiche, fonction_activation)

	Note:
		La fonction retournée prend une entrée i et retourne (Fi, Fp):
		- Fi : valeur activée
		- Fp : dérivée de la fonction au point i
	"""

	if n_fct == 1:
		from neurone.fct_dactivation import sigmoide_et_derivative

		return "sigmoïde", sigmoide_et_derivative

	if n_fct == 2:
		from neurone.fct_dactivation import tan_et_derivative

		return "tan", tan_et_derivative

	if n_fct == 3:
		from neurone.fct_dactivation import tanh_et_derivative

		return "tanh", tanh_et_derivative

	if n_fct == 4:
		from neurone.fct_dactivation import gelu_et_derivative

		return "gelu", gelu_et_derivative

	raise ValueError("n_fct doit être 1 (sigmoïde), 2 (tan), 3 (tanh) ou 4 (gelu)")


# ==================== _print_entrees =========================
def _print_entrees(lines):
	"""Affiche un bloc 'Entrées' avec une indentation constante.

	But:
		Rendre l'affichage plus lisible et identique partout.
	"""

	print(f"{_IND}Entrées")
	for line in lines:
		print(f"{_IND}  {line}")
	print()


# ==================== _print_sorties =========================
def _print_sorties(lines):
	"""Affiche un bloc 'Sorties' avec une indentation constante.

	But:
		Rendre l'affichage plus lisible et identique partout.
	"""

	print(f"{_IND}Sorties")
	for line in lines:
		print(f"{_IND}  {line}")
	print()


# ==================== resolution_reseau_exemple =========================
def resolution_reseau_exemple(n_fct: int):
	"""Résout le réseau du schéma (2 entrées → 2 cachés → 3 sorties).

	Ce que la fonction fait:
		- Initialise les poids et les biais comme sur le schéma
		- Calcule la propagation avant (i, Fi, Fp)
		- Calcule la rétropropagation (deltas, corrections, mises à jour)

	Paramètre:
		n_fct (int): choix de la fonction d'activation (voir choisir_fonction_activation).
	"""

	nom_fct, fct_activation = choisir_fonction_activation(n_fct)

	# 1) Paramètres (du schéma)
	# entrees (à ajuster si ton énoncé donne d'autres valeurs)
	x1, x2 = 2, 4
	X = [x1, x2]

	# couche 1 (poids des entrées vers les 2 neurones cachés)
	# w11_1 : x1 -> h1, w21_1 : x2 -> h1
	# w12_1 : x1 -> h2, w22_1 : x2 -> h2
	w11_1, w12_1 = 1, 3
	w21_1, w22_1 = 2, 4

	# biais couche cachée (non indiqué sur le schéma → 0)
	b_h1 = 2
	b_h2 = 1

	# couche 2 (poids des 2 cachés vers les 3 sorties)
	# h1 -> (o1,o2,o3)
	w11_2, w12_2, w13_2 = 0.3, -0.2, 0.2
	# h2 -> (o1,o2,o3)
	w21_2, w22_2, w23_2 = 1.3, -0.4, -1.2

	# biais des sorties (valeurs affichées dans les neurones du schéma)
	b_o1, b_o2, b_o3 = -0.1, 0.1, -0.2

	# cibles
	d1, d2, d3 = 1, 2, 1

	# taux d'apprentissage
	eta = 0.1

	# Outil pour calculer i = somme(x*w) + b
	neurone = Neurone()

	# -------------------- Calculs (sans affichage) --------------------
	# Activations — couche cachée
	i_h1 = neurone.calcul_activation_i([(x1, w11_1), (x2, w21_1)], b_h1)
	i_h2 = neurone.calcul_activation_i([(x1, w12_1), (x2, w22_1)], b_h2)

	# Fonction d'activation + dérivée — couche cachée
	Fi_h1, Fp_h1 = fct_activation(i_h1)
	Fi_h2, Fp_h2 = fct_activation(i_h2)

	# Activations — couche sortie
	i_o1 = neurone.calcul_activation_i([(Fi_h1, w11_2), (Fi_h2, w21_2)], b_o1)
	i_o2 = neurone.calcul_activation_i([(Fi_h1, w12_2), (Fi_h2, w22_2)], b_o2)
	i_o3 = neurone.calcul_activation_i([(Fi_h1, w13_2), (Fi_h2, w23_2)], b_o3)

	# Fonction d'activation + dérivée — couche sortie
	Fi_o1, Fp_o1 = fct_activation(i_o1)
	Fi_o2, Fp_o2 = fct_activation(i_o2)
	Fi_o3, Fp_o3 = fct_activation(i_o3)

	# Signaux d'erreur (deltas) — couche sortie
	delta_o1 = Neurone.Delta(d1, Fi_o1, Fp_o1)
	delta_o2 = Neurone.Delta(d2, Fi_o2, Fp_o2)
	delta_o3 = Neurone.Delta(d3, Fi_o3, Fp_o3)

	# Signaux d'erreur (deltas) — couche cachée
	delta_h1 = Neurone.Delta_cache([(delta_o1, w11_2), (delta_o2, w12_2), (delta_o3, w13_2)], Fp_h1)
	delta_h2 = Neurone.Delta_cache([(delta_o1, w21_2), (delta_o2, w22_2), (delta_o3, w23_2)], Fp_h2)

	# Facteurs de correction — couche 2
	corr_w11_2 = Neurone.Correcteur(eta, Fi_h1, delta_o1)
	corr_w21_2 = Neurone.Correcteur(eta, Fi_h2, delta_o1)
	corr_b_o1 = Neurone.Correcteur(eta, 1, delta_o1)

	corr_w12_2 = Neurone.Correcteur(eta, Fi_h1, delta_o2)
	corr_w22_2 = Neurone.Correcteur(eta, Fi_h2, delta_o2)
	corr_b_o2 = Neurone.Correcteur(eta, 1, delta_o2)

	corr_w13_2 = Neurone.Correcteur(eta, Fi_h1, delta_o3)
	corr_w23_2 = Neurone.Correcteur(eta, Fi_h2, delta_o3)
	corr_b_o3 = Neurone.Correcteur(eta, 1, delta_o3)

	# Facteurs de correction — couche 1
	corr_w11_1 = Neurone.Correcteur(eta, x1, delta_h1)
	corr_w21_1 = Neurone.Correcteur(eta, x2, delta_h1)
	corr_b_h1 = Neurone.Correcteur(eta, 1, delta_h1)

	corr_w12_1 = Neurone.Correcteur(eta, x1, delta_h2)
	corr_w22_1 = Neurone.Correcteur(eta, x2, delta_h2)
	corr_b_h2 = Neurone.Correcteur(eta, 1, delta_h2)

	# Mises à jour — couche 2
	new_w11_2 = Neurone.maj(w11_2, corr_w11_2)
	new_w21_2 = Neurone.maj(w21_2, corr_w21_2)
	new_b_o1 = Neurone.maj(b_o1, corr_b_o1)

	new_w12_2 = Neurone.maj(w12_2, corr_w12_2)
	new_w22_2 = Neurone.maj(w22_2, corr_w22_2)
	new_b_o2 = Neurone.maj(b_o2, corr_b_o2)

	new_w13_2 = Neurone.maj(w13_2, corr_w13_2)
	new_w23_2 = Neurone.maj(w23_2, corr_w23_2)
	new_b_o3 = Neurone.maj(b_o3, corr_b_o3)

	# Mises à jour — couche 1
	new_w11_1 = Neurone.maj(w11_1, corr_w11_1)
	new_w21_1 = Neurone.maj(w21_1, corr_w21_1)
	new_b_h1 = Neurone.maj(b_h1, corr_b_h1)

	new_w12_1 = Neurone.maj(w12_1, corr_w12_1)
	new_w22_1 = Neurone.maj(w22_1, corr_w22_1)
	new_b_h2 = Neurone.maj(b_h2, corr_b_h2)

	# -------------------- Affichage par phases --------------------
	print("=== ACTIVATIONS ===")
	print("h1")
	_print_entrees([f"x1 = {x1}, w11_1 = {w11_1}", f"x2 = {x2}, w21_1 = {w21_1}", f"b_h1 = {b_h1}"])
	_print_sorties([f"i_h1 = {i_h1}"])

	print("h2")
	_print_entrees([f"x1 = {x1}, w12_1 = {w12_1}", f"x2 = {x2}, w22_1 = {w22_1}", f"b_h2 = {b_h2}"])
	_print_sorties([f"i_h2 = {i_h2}"])

	print("o1")
	_print_entrees([f"Fi_h1 = {Fi_h1}, w11_2 = {w11_2}", f"Fi_h2 = {Fi_h2}, w21_2 = {w21_2}", f"b_o1 = {b_o1}"])
	_print_sorties([f"i_o1 = {i_o1}"])

	print("o2")
	_print_entrees([f"Fi_h1 = {Fi_h1}, w12_2 = {w12_2}", f"Fi_h2 = {Fi_h2}, w22_2 = {w22_2}", f"b_o2 = {b_o2}"])
	_print_sorties([f"i_o2 = {i_o2}"])

	print("o3")
	_print_entrees([f"Fi_h1 = {Fi_h1}, w13_2 = {w13_2}", f"Fi_h2 = {Fi_h2}, w23_2 = {w23_2}", f"b_o3 = {b_o3}"])
	_print_sorties([f"i_o3 = {i_o3}"])

	print(f"=== FONCTIONS D'ACTIVATION + DERIVEES ({nom_fct}) ===")
	print("h1")
	_print_entrees([f"i_h1 = {i_h1}"])
	_print_sorties([f"Fi_h1 = {Fi_h1}", f"Fp_h1 = {Fp_h1}"])
	print("h2")
	_print_entrees([f"i_h2 = {i_h2}"])
	_print_sorties([f"Fi_h2 = {Fi_h2}", f"Fp_h2 = {Fp_h2}"])
	print("o1")
	_print_entrees([f"i_o1 = {i_o1}"])
	_print_sorties([f"Fi_o1 = {Fi_o1}", f"Fp_o1 = {Fp_o1}"])
	print("o2")
	_print_entrees([f"i_o2 = {i_o2}"])
	_print_sorties([f"Fi_o2 = {Fi_o2}", f"Fp_o2 = {Fp_o2}"])
	print("o3")
	_print_entrees([f"i_o3 = {i_o3}"])
	_print_sorties([f"Fi_o3 = {Fi_o3}", f"Fp_o3 = {Fp_o3}"])

	print("=== SIGNAUX D'ERREUR ===")
	print("sortie")
	_print_entrees([f"d1 = {d1}", f"Fi_o1 = {Fi_o1}", f"Fp_o1 = {Fp_o1}"])
	_print_sorties([f"delta_o1 = {delta_o1}"])
	_print_entrees([f"d2 = {d2}", f"Fi_o2 = {Fi_o2}", f"Fp_o2 = {Fp_o2}"])
	_print_sorties([f"delta_o2 = {delta_o2}"])
	_print_entrees([f"d3 = {d3}", f"Fi_o3 = {Fi_o3}", f"Fp_o3 = {Fp_o3}"])
	_print_sorties([f"delta_o3 = {delta_o3}"])

	print("cachée")
	_print_entrees([f"delta_o1 = {delta_o1}, w11_2 = {w11_2}", f"delta_o2 = {delta_o2}, w12_2 = {w12_2}", f"delta_o3 = {delta_o3}, w13_2 = {w13_2}", f"Fp_h1 = {Fp_h1}"])
	_print_sorties([f"delta_h1 = {delta_h1}"])
	_print_entrees([f"delta_o1 = {delta_o1}, w21_2 = {w21_2}", f"delta_o2 = {delta_o2}, w22_2 = {w22_2}", f"delta_o3 = {delta_o3}, w23_2 = {w23_2}", f"Fp_h2 = {Fp_h2}"])
	_print_sorties([f"delta_h2 = {delta_h2}"])

	print("=== FACTEURS DE CORRECTION ===")
	print("couche 2 (cachée -> sorties)")
	_print_entrees([f"eta = {eta}", f"x = Fi_h1 = {Fi_h1}, delta = {delta_o1}", f"x = Fi_h2 = {Fi_h2}, delta = {delta_o1}", f"x = 1 (biais), delta = {delta_o1}"])
	_print_sorties([f"delta_w11_2 = {corr_w11_2}", f"delta_w21_2 = {corr_w21_2}", f"delta_b_o1  = {corr_b_o1}"])
	_print_entrees([f"eta = {eta}", f"x = Fi_h1 = {Fi_h1}, delta = {delta_o2}", f"x = Fi_h2 = {Fi_h2}, delta = {delta_o2}", f"x = 1 (biais), delta = {delta_o2}"])
	_print_sorties([f"delta_w12_2 = {corr_w12_2}", f"delta_w22_2 = {corr_w22_2}", f"delta_b_o2  = {corr_b_o2}"])
	_print_entrees([f"eta = {eta}", f"x = Fi_h1 = {Fi_h1}, delta = {delta_o3}", f"x = Fi_h2 = {Fi_h2}, delta = {delta_o3}", f"x = 1 (biais), delta = {delta_o3}"])
	_print_sorties([f"delta_w13_2 = {corr_w13_2}", f"delta_w23_2 = {corr_w23_2}", f"delta_b_o3  = {corr_b_o3}"])

	print("couche 1 (entrées -> cachée)")
	_print_entrees([f"eta = {eta}", f"x1 = {x1}, delta = {delta_h1}", f"x2 = {x2}, delta = {delta_h1}", f"x = 1 (biais), delta = {delta_h1}"])
	_print_sorties([f"delta_w11_1 = {corr_w11_1}", f"delta_w21_1 = {corr_w21_1}", f"delta_b_h1  = {corr_b_h1}"])
	_print_entrees([f"eta = {eta}", f"x1 = {x1}, delta = {delta_h2}", f"x2 = {x2}, delta = {delta_h2}", f"x = 1 (biais), delta = {delta_h2}"])
	_print_sorties([f"delta_w12_1 = {corr_w12_1}", f"delta_w22_1 = {corr_w22_1}", f"delta_b_h2  = {corr_b_h2}"])

	print("=== MISES A JOUR ===")
	print("couche 2")
	_print_entrees([f"w11_2 = {w11_2}, delta_w11_2 = {corr_w11_2}", f"w21_2 = {w21_2}, delta_w21_2 = {corr_w21_2}", f"b_o1  = {b_o1}, delta_b_o1  = {corr_b_o1}"])
	_print_sorties([f"w11_2 -> {new_w11_2}", f"w21_2 -> {new_w21_2}", f"b_o1  -> {new_b_o1}"])
	_print_entrees([f"w12_2 = {w12_2}, delta_w12_2 = {corr_w12_2}", f"w22_2 = {w22_2}, delta_w22_2 = {corr_w22_2}", f"b_o2  = {b_o2}, delta_b_o2  = {corr_b_o2}"])
	_print_sorties([f"w12_2 -> {new_w12_2}", f"w22_2 -> {new_w22_2}", f"b_o2  -> {new_b_o2}"])
	_print_entrees([f"w13_2 = {w13_2}, delta_w13_2 = {corr_w13_2}", f"w23_2 = {w23_2}, delta_w23_2 = {corr_w23_2}", f"b_o3  = {b_o3}, delta_b_o3  = {corr_b_o3}"])
	_print_sorties([f"w13_2 -> {new_w13_2}", f"w23_2 -> {new_w23_2}", f"b_o3  -> {new_b_o3}"])

	print("couche 1")
	_print_entrees([f"w11_1 = {w11_1}, delta_w11_1 = {corr_w11_1}", f"w21_1 = {w21_1}, delta_w21_1 = {corr_w21_1}", f"b_h1  = {b_h1}, delta_b_h1  = {corr_b_h1}"])
	_print_sorties([f"w11_1 -> {new_w11_1}", f"w21_1 -> {new_w21_1}", f"b_h1  -> {new_b_h1}"])
	_print_entrees([f"w12_1 = {w12_1}, delta_w12_1 = {corr_w12_1}", f"w22_1 = {w22_1}, delta_w22_1 = {corr_w22_1}", f"b_h2  = {b_h2}, delta_b_h2  = {corr_b_h2}"])
	_print_sorties([f"w12_1 -> {new_w12_1}", f"w22_1 -> {new_w22_1}", f"b_h2  -> {new_b_h2}"])

	# Résumé court des mises à jour (pratique pour comparer)
	print("\n=== RESUME (poids mis à jour) ===")
	_print_sorties(
		[
			f"Couche 2: (w11_2,w21_2,b_o1) -> ({new_w11_2},{new_w21_2},{new_b_o1})",
			f"Couche 2: (w12_2,w22_2,b_o2) -> ({new_w12_2},{new_w22_2},{new_b_o2})",
			f"Couche 2: (w13_2,w23_2,b_o3) -> ({new_w13_2},{new_w23_2},{new_b_o3})",
			f"Couche 1: (w11_1,w21_1,b_h1) -> ({new_w11_1},{new_w21_1},{new_b_h1})",
			f"Couche 1: (w12_1,w22_1,b_h2) -> ({new_w12_1},{new_w22_1},{new_b_h2})",
		]
	)


# ==================== main =========================
def main():
	"""Point d'entrée du script.

	Tu peux changer n_fct pour sélectionner la fonction d'activation.
	Au démarrage, le choix courant est affiché.
	"""

	# Choix de la fonction d'activation
	# n_fct = 1 : sigmoïde
	# n_fct = 2 : tan
	# n_fct = 3 : tanh
	# n_fct = 4 : gelu
	n_fct = 1

	clear_console()

	nom_fct, _ = choisir_fonction_activation(n_fct)
	print(f"Choix courant : n_fct = {n_fct} ({nom_fct})")
	print()

	resolution_reseau_exemple(n_fct)


if __name__ == "__main__":
	main()

