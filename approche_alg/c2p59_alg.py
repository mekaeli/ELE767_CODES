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
    """Résout l'exemple du réseau (2 entrées → 2 cachés → 1 sortie).

    Ce que la fonction fait:
        - Initialise les poids et les biais comme sur le schéma
        - Calcule la propagation avant (i, Fi, Fp)
        - Calcule la rétropropagation (delta, corrections, mises à jour)

    Paramètre:
        n_fct (int): choix de la fonction d'activation (voir choisir_fonction_activation).
    """

    nom_fct, fct_activation = choisir_fonction_activation(n_fct)

    # 1. Initialisation des paramètres du schéma
    # entrees
    x1, x2 = 1, 0
    X = [x1, x2]

    # couche 1
    w11_1, w12_1 = 3, 4
    w21_1, w22_1 = 6, 5

    # couche 2
    w11_2, w21_2 = 2, 4
    d = 1
    eta = 0.1

    # Biais des neurones
    b_h1 = 1  # biais neurone caché 1
    b_h2 = 0  # biais neurone caché 2
    b_s = -3.92  # biais neurone sortie

    # Création d'une instance de la classe Neurone
    neurone = Neurone()

    # -------------------- Calculs (sans affichage) --------------------
    # Activations
    i_h1 = neurone.calcul_activation_i([(x1, w11_1), (x2, w21_1)], b_h1)
    i_h2 = neurone.calcul_activation_i([(x1, w12_1), (x2, w22_1)], b_h2)

    # Fonction d'activation + dérivée
    Fi_h1, Fp_h1 = fct_activation(i_h1)
    Fi_h2, Fp_h2 = fct_activation(i_h2)

    # Activation sortie
    i_s = neurone.calcul_activation_i([(Fi_h1, w11_2), (Fi_h2, w21_2)], b_s)
    Fi_s, Fp_s = fct_activation(i_s)

    # Signaux d'erreur (deltas)
    delta_s = Neurone.Delta(d, Fi_s, Fp_s)
    delta_h1 = Neurone.Delta_cache([(delta_s, w11_2)], Fp_h1)
    delta_h2 = Neurone.Delta_cache([(delta_s, w21_2)], Fp_h2)

    # Facteurs de correction (deltas_w / deltas_b)
    corr_w11_2 = Neurone.Correcteur(eta, Fi_h1, delta_s)
    corr_w21_2 = Neurone.Correcteur(eta, Fi_h2, delta_s)
    corr_b_s = Neurone.Correcteur(eta, 1, delta_s)

    corr_w11_1 = Neurone.Correcteur(eta, x1, delta_h1)
    corr_w21_1 = Neurone.Correcteur(eta, x2, delta_h1)
    corr_b_h1 = Neurone.Correcteur(eta, 1, delta_h1)

    corr_w12_1 = Neurone.Correcteur(eta, x1, delta_h2)
    corr_w22_1 = Neurone.Correcteur(eta, x2, delta_h2)
    corr_b_h2 = Neurone.Correcteur(eta, 1, delta_h2)

    # Mises à jour
    new_w11_2 = Neurone.maj(w11_2, corr_w11_2)
    new_w21_2 = Neurone.maj(w21_2, corr_w21_2)
    new_b_s = Neurone.maj(b_s, corr_b_s)

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

    print("s")
    _print_entrees([f"Fi_h1 = {Fi_h1}, w11_2 = {w11_2}", f"Fi_h2 = {Fi_h2}, w21_2 = {w21_2}", f"b_s = {b_s}"])
    _print_sorties([f"i_s = {i_s}"])

    print(f"=== FONCTIONS D'ACTIVATION + DERIVEES ({nom_fct}) ===")
    print("h1")
    _print_entrees([f"i_h1 = {i_h1}"])
    _print_sorties([f"Fi_h1 = {Fi_h1}", f"Fp_h1 = {Fp_h1}"])

    print("h2")
    _print_entrees([f"i_h2 = {i_h2}"])
    _print_sorties([f"Fi_h2 = {Fi_h2}", f"Fp_h2 = {Fp_h2}"])

    print("s")
    _print_entrees([f"i_s = {i_s}"])
    _print_sorties([f"Fi_s = {Fi_s}", f"Fp_s = {Fp_s}"])

    print("=== SIGNAUX D'ERREUR ===")
    print("sortie")
    _print_entrees([f"d = {d}", f"Fi_s = {Fi_s}", f"Fp_s = {Fp_s}"])
    _print_sorties([f"delta_s = {delta_s}"])

    print("cachée")
    _print_entrees([f"delta_s = {delta_s}", f"w11_2 = {w11_2}", f"Fp_h1 = {Fp_h1}"])
    _print_sorties([f"delta_h1 = {delta_h1}"])
    _print_entrees([f"delta_s = {delta_s}", f"w21_2 = {w21_2}", f"Fp_h2 = {Fp_h2}"])
    _print_sorties([f"delta_h2 = {delta_h2}"])

    print("=== FACTEURS DE CORRECTION ===")
    print("couche 2 (cachée -> sortie)")
    _print_entrees([f"eta = {eta}", f"x = Fi_h1 = {Fi_h1}, delta = {delta_s}", f"x = Fi_h2 = {Fi_h2}, delta = {delta_s}", f"x = 1 (biais), delta = {delta_s}"])
    _print_sorties([f"delta_w11_2 = {corr_w11_2}", f"delta_w21_2 = {corr_w21_2}", f"delta_b_s   = {corr_b_s}"])

    print("couche 1 (entrées -> cachée)")
    _print_entrees([f"eta = {eta}", f"x1 = {x1}, delta = {delta_h1}", f"x2 = {x2}, delta = {delta_h1}", f"x = 1 (biais), delta = {delta_h1}"])
    _print_sorties([f"delta_w11_1 = {corr_w11_1}", f"delta_w21_1 = {corr_w21_1}", f"delta_b_h1  = {corr_b_h1}"])
    _print_entrees([f"eta = {eta}", f"x1 = {x1}, delta = {delta_h2}", f"x2 = {x2}, delta = {delta_h2}", f"x = 1 (biais), delta = {delta_h2}"])
    _print_sorties([f"delta_w12_1 = {corr_w12_1}", f"delta_w22_1 = {corr_w22_1}", f"delta_b_h2  = {corr_b_h2}"])

    print("=== MISES A JOUR ===")
    print("couche 2")
    _print_entrees([f"w11_2 = {w11_2}, delta_w11_2 = {corr_w11_2}", f"w21_2 = {w21_2}, delta_w21_2 = {corr_w21_2}", f"b_s   = {b_s}, delta_b_s   = {corr_b_s}"])
    _print_sorties([f"w11_2 -> {new_w11_2}", f"w21_2 -> {new_w21_2}", f"b_s   -> {new_b_s}"])

    print("couche 1")
    _print_entrees([f"w11_1 = {w11_1}, delta_w11_1 = {corr_w11_1}", f"w21_1 = {w21_1}, delta_w21_1 = {corr_w21_1}", f"b_h1  = {b_h1}, delta_b_h1  = {corr_b_h1}"])
    _print_sorties([f"w11_1 -> {new_w11_1}", f"w21_1 -> {new_w21_1}", f"b_h1  -> {new_b_h1}"])
    _print_entrees([f"w12_1 = {w12_1}, delta_w12_1 = {corr_w12_1}", f"w22_1 = {w22_1}, delta_w22_1 = {corr_w22_1}", f"b_h2  = {b_h2}, delta_b_h2  = {corr_b_h2}"])
    _print_sorties([f"w12_1 -> {new_w12_1}", f"w22_1 -> {new_w22_1}", f"b_h2  -> {new_b_h2}"])

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
