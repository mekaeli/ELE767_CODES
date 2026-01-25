# import librairy standard
import os
import math
import sys

# Permettre l'import du module generic_func depuis le dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import generic_func

# =============================================================================================
# Nettoyer la console (Windows et Unix)
generic_func.clear_console()

# =============================================================================================
# Import de la fonction Delta du module BackPP
from BackPP import Delta, Delta_cache, Correcteur, maj

# Exemple d'utilisation de la fonction Delta
if __name__ == "__main__":
    d = 1.0    # valeur désirée
    Fi = 0.8778   # sortie réelle de la neurone
    Fp = 0.1072  # dérivée de la fonction d'activation
    delta = Delta(d, Fi, Fp)
    print(f"Signal d'erreur delta = {delta}")

    # Exemple d'utilisation de la fonction Delta_cache
    delta_w_list = [(delta, 2)]  # liste de paires (delta, w)
    Fp_cache = 0.017676  # dérivée de la fonction d'activation de la neurone cachée
    delta_cache = Delta_cache(delta_w_list, Fp_cache)
    print(f"Signal d'erreur delta_cache = {delta_cache}")

    # Exemple d'utilisation de la fonction Correcteur
    eta = 0.1   # taux d'apprentissage
    x = 1.0     # entrée de la neurone
    delta = 0.000463105  # signal d'erreur
    correction = Correcteur(eta, x, delta)
    print(f"Facteur de correction = {correction}")

    # Exemple d'utilisation de la fonction maj
    w = 3        # ancien poids
    delta_w = correction  # facteur de correction
    nouveau_w = maj(w, delta_w)
    print(f"Nouveau poids = {nouveau_w}")
