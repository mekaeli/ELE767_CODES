# import librairy standard
import os
import math
import sys

# Permettre l'import des modules depuis le dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import des fonctions d'activation (module regroupé)
from neurone.fct_dactivation import (
    sigmoide_et_derivative,
    tanh_et_derivative,
    gelu_et_derivative,
    tan_et_derivative,
)


# ==================== Neurone =========================
class Neurone:
    """Neurone + utilitaires Backpropagation (BackPP).

    Cette classe:
    - calcule l'activation i = somme(x*w) + b
    - applique une fonction d'activation (Fi) et sa dérivée (Fp)
    - fournit les fonctions Backpropagation classiques: Delta, Delta_cache, Correcteur, maj
    """

    # ==================== calcul_activation_i =========================
    def calcul_activation_i(self, inlist, bi):
        """Calcule l'activation du neurone.

        Chaque élément de inlist doit être une paire (xk, wki):
            - xk : valeur d'entrée
            - wki : poids associé à cette entrée

        Args:
            inlist (list of tuple): liste de paires (xk, wki)
            bi (float): biais du neurone

        Returns:
            float: valeur de l'activation i

        Raises:
            ValueError: si la liste n'est pas bien formée
        """

        if not all(isinstance(pair, (list, tuple)) and len(pair) == 2 for pair in inlist):
            raise ValueError("Chaque élément de inlist doit être une paire (xk, wki).")

        xk = [pair[0] for pair in inlist]
        wki = [pair[1] for pair in inlist]

        if len(xk) != len(wki):
            raise ValueError("Les listes xk et wki doivent avoir la même longueur.")

        activation = sum(x * w for x, w in zip(xk, wki)) + bi
        return activation

    # ==================== activation_et_derivative =========================
    def activation_et_derivative(self, activation_i, n_fct):
        """Retourne [Fi, Fp] selon le choix n_fct.

        But:
            Convertir i (activation brute) en Fi (sortie) et Fp (dérivée).

        n_fct:
            1 : sigmoïde
            2 : tan
            3 : tanh
            4 : gelu
        """

        if n_fct == 1:
            return sigmoide_et_derivative(activation_i)
        if n_fct == 2:
            return tan_et_derivative(activation_i)
        if n_fct == 3:
            return tanh_et_derivative(activation_i)
        if n_fct == 4:
            return gelu_et_derivative(activation_i)

        raise ValueError("n_fct doit être 1 (sigmoïde), 2 (tan), 3 (tanh) ou 4 (gelu)")

    # =============================
    # Backpropagation (BackPP)
    # =============================

    # ==================== maj =========================
    @staticmethod
    def maj(w, delta_w):
        """Met à jour un poids ou un biais.

        Règle:
            nouveau = ancien + delta
        """

        return w + delta_w

    # ==================== Correcteur =========================
    @staticmethod
    def Correcteur(eta, x, delta):
        """Calcule la correction (delta_w) d'un poids ou d'un biais.

        Formule:
            delta_w = eta * x * delta
        """

        return eta * x * delta

    # ==================== Delta_cache =========================
    @staticmethod
    def Delta_cache(delta_w_list, Fp):
        """Calcule le delta d'un neurone caché.

        Entrées:
            delta_w_list: liste de paires (delta_suivant, poids_vers_suivant)
            Fp: dérivée de l'activation du neurone caché

        Sortie:
            delta_cache
        """

        somme = sum(delta * w for delta, w in delta_w_list)
        return somme * Fp

    # ==================== Delta =========================
    @staticmethod
    def Delta(d, Fi, Fp):
        """Calcule le delta d'un neurone de sortie.

        Formule:
            delta = (d - Fi) * Fp
        """

        return (d - Fi) * Fp
