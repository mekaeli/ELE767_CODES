"""Exercice c2p69 — Réseau 2→2→3 (approche matricielle).

Objectif:
	- Propagation avant + rétropropagation en utilisant NeuroneMat
	- Affichage regroupé par phases (activations, Fi/Fp, deltas, corrections, mises à jour)
	- Matrices affichées correctement (lignes/colonnes)
"""

# Configuration du chemin pour importer les modules du dossier parent
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import des fonctions nécessaires
from generic_func import clear_console
from approch_mat.format_mat import matrix_lines, shape_str, vector_lines
from neurone.backpp_neurone_mat import NeuroneMat


_IND = "\t"


def _nom_fct(n_fct: int) -> str:
	return {1: "sigmoïde", 2: "tan", 3: "tanh", 4: "gelu"}.get(n_fct, f"n_fct={n_fct}")


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

	nom_fct = _nom_fct(n_fct)

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

	# ==================== Paramètres matriciels ====================
	W1 = [[w11_1, w12_1], [w21_1, w22_1]]
	b1 = [b_h1, b_h2]

	W2 = [
		[w11_2, w12_2, w13_2],
		[w21_2, w22_2, w23_2],
	]
	b2 = [b_o1, b_o2, b_o3]

	d_list = [d1, d2, d3]
	w_list = [W1, W2]
	b_list = [b1, b2]

	nn = NeuroneMat()

	# ==================== CALCULS (matriciel uniquement) ====================
	y, cache = nn.forward(X, w_list, b_list, n_fct)
	z1, z2 = cache.z_list
	a1, a2 = cache.a_list
	fp1, fp2 = cache.fp_list

	delta2 = nn.Delta(d_list, y, fp2)
	delta1 = nn.Delta_cache(delta2, W2, fp1)

	dW2, db2 = nn.Correcteur(eta, a1, delta2)
	dW1, db1 = nn.Correcteur(eta, X, delta1)

	new_W1, new_b1 = nn.maj(W1, dW1, b1, db1)
	new_W2, new_b2 = nn.maj(W2, dW2, b2, db2)

	results = {
		"X": X,
		"W1": W1,
		"b1": b1,
		"W2": W2,
		"b2": b2,
		"d": d_list,
		"eta": eta,
		# forward
		"z1": z1,
		"a1": a1,
		"fp1": fp1,
		"z2": z2,
		"a2": a2,
		"fp2": fp2,
		# deltas
		"delta2": delta2,
		"delta1": delta1,
		# correcteurs
		"dW2": dW2,
		"db2": db2,
		"dW1": dW1,
		"db1": db1,
		# mise à jour
		"new_W1": new_W1,
		"new_b1": new_b1,
		"new_W2": new_W2,
		"new_b2": new_b2,
	}

	# ==================== AFFICHAGE (phases) ====================
	print("=== ACTIVATIONS ===")
	_print_entrees([f"X {shape_str(X)} (colonne)"] + vector_lines(X, as_column=True))

	_print_entrees([f"W1 {shape_str(W1)}", f"b1 {shape_str(b1)} (colonne)"] + matrix_lines(W1) + ["b1="] + vector_lines(b1, as_column=True))
	_print_sorties([f"z1 = X@W1 + b1 {shape_str(z1)} (colonne)"] + vector_lines(z1, as_column=True))

	_print_entrees([f"W2 {shape_str(W2)}", f"b2 {shape_str(b2)} (colonne)"] + matrix_lines(W2) + ["b2="] + vector_lines(b2, as_column=True))
	_print_sorties([f"z2 = a1@W2 + b2 {shape_str(z2)} (colonne)"] + vector_lines(z2, as_column=True))

	print(f"=== FONCTIONS D'ACTIVATION + DERIVEES ({nom_fct}) ===")
	_print_sorties([f"a1 = Fi(z1) {shape_str(a1)} (colonne)"] + vector_lines(a1, as_column=True))
	_print_sorties([f"fp1 = Fp(z1) {shape_str(fp1)} (colonne)"] + vector_lines(fp1, as_column=True))
	_print_sorties([f"a2 = Fi(z2) {shape_str(a2)} (colonne)"] + vector_lines(a2, as_column=True))
	_print_sorties([f"fp2 = Fp(z2) {shape_str(fp2)} (colonne)"] + vector_lines(fp2, as_column=True))

	print("=== SIGNAUX D'ERREUR ===")
	_print_entrees([f"d {shape_str(d_list)} (colonne)"] + vector_lines(d_list, as_column=True))
	_print_sorties([f"delta2 (sortie) {shape_str(delta2)} (colonne)"] + vector_lines(delta2, as_column=True))
	_print_sorties([f"delta1 (cachée) {shape_str(delta1)} (colonne)"] + vector_lines(delta1, as_column=True))

	print("=== FACTEURS DE CORRECTION ===")
	_print_entrees([f"eta = {eta}"])
	_print_sorties([f"dW2 {shape_str(dW2)}"] + matrix_lines(dW2))
	_print_sorties([f"db2 {shape_str(db2)} (colonne)"] + vector_lines(db2, as_column=True))
	_print_sorties([f"dW1 {shape_str(dW1)}"] + matrix_lines(dW1))
	_print_sorties([f"db1 {shape_str(db1)} (colonne)"] + vector_lines(db1, as_column=True))

	print("=== MISES A JOUR ===")
	_print_sorties([f"W2 <- W2 + dW2"] + matrix_lines(new_W2, precision=12))
	_print_sorties([f"b2 <- b2 + db2 (colonne)"] + vector_lines(new_b2, as_column=True, precision=12))
	_print_sorties([f"W1 <- W1 + dW1"] + matrix_lines(new_W1, precision=12))
	_print_sorties([f"b1 <- b1 + db1 (colonne)"] + vector_lines(new_b1, as_column=True, precision=12))

	print("\n=== RESUME (poids mis à jour) ===")
	_print_sorties(
		[
			f"Couche 2: W2 {shape_str(new_W2)}, b2 {shape_str(new_b2)}",
			"W2=",
			*matrix_lines(new_W2, precision=12),
			"b2=",
			*vector_lines(new_b2, as_column=True, precision=12),
			f"Couche 1: W1 {shape_str(new_W1)}, b1 {shape_str(new_b1)}",
			"W1=",
			*matrix_lines(new_W1, precision=12),
			"b1=",
			*vector_lines(new_b1, as_column=True, precision=12),
		]
	)

	return results


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

