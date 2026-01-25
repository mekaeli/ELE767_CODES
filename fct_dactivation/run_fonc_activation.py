# import librairy standard
import os
import math
import sys

# Permettre l'import du module generic_func depuis le dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import generic_func
# Import de chaque fonction d'activation
from sigmoide import sigmoide_et_derivative as sigmoide
from tanh import tanh_et_derivative as tanh
from gelu import gelu_et_derivative as gelu
from tan import tan_et_derivative as tan

# =============================================================================================
# Nettoyer la console (Windows et Unix)
generic_func.clear_console()

# =============================================================================================
# Exemple d'utilisation de chaque fonction, séparément
if __name__ == "__main__":
    activation_i = 1.9720  # Exemple de valeur d'activation

    # Sigmoïde
    result_sig = sigmoide(activation_i)
    print(f"Sigmoïde : Fi = {result_sig[0]}, Fp = {result_sig[1]}")

    # Tanh
    result_tanh = tanh(activation_i)
    print(f"Tanh : Fi = {result_tanh[0]}, Fp = {result_tanh[1]}")

    # GELU
    result_gelu = gelu(activation_i)
    print(f"GELU : Fi = {result_gelu[0]}, Fp = {result_gelu[1]}")

    # TAN
    result_tan = tan(activation_i)
    print(f"TAN : Fi = {result_tan[0]}, Fp = {result_tan[1]}")
