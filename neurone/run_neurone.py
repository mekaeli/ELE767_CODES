# import librairy standard
import os
import math
import sys

# Permettre l'import du module generic_func depuis le dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import generic_func
# Import de la classe Neurone du module backpp_neurone
from backpp_neurone import Neurone

# =============================================================================================
# Nettoyer la console (Windows et Unix)
generic_func.clear_console()

# =============================================================================================
# Exemple d'utilisation de la classe Neurone
if __name__ == "__main__":
    # Création d'une instance de la classe Neurone
    neurone = Neurone()
    # Exemple de liste de paires (xk, wki)
    inlist = [(1, 3), (0, 6)]
    bi = 1  # biais du neurone
    try:
        activation = neurone.calcul_activation_i(inlist, bi)
        print(f"Activation i = {activation}")
    except ValueError as e:
        print(f"Erreur : {e}")

    # Exemple d'appel de la fonction d'activation/derivée selon n_fct
    for n_fct in [1, 2, 3, 4]:
        try:
            result = neurone.activation_et_derivative(activation, n_fct)
            if n_fct == 1:
                nom = "sigmoïde"
            elif n_fct == 2:
                nom = "tanh"
            elif n_fct == 3:
                nom = "gelu"
            elif n_fct == 4:
                nom = "tan"
            print(f"{nom} : Fi = {result[0]}, Fp = {result[1]}")
        except ValueError as e:
            print(f"Erreur activation_et_derivative : {e}")
