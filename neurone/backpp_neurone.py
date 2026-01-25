# import librairy standard
import os
import math
import sys

# Permettre l'import des modules depuis le dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import des fonctions d'activation
from fct_dactivation.sigmoide import sigmoide_et_derivative
from fct_dactivation.tanh import tanh_et_derivative
from fct_dactivation.gelu import gelu_et_derivative
from fct_dactivation.tan import tan_et_derivative


class Neurone:
    """Neurone + utilitaires Backpropagation (BackPP).

    Cette classe:
    - calcule l'activation i = somme(x*w) + b
    - applique une fonction d'activation (Fi) et sa dérivée (Fp)
    - fournit les fonctions Backpropagation classiques: Delta, Delta_cache, Correcteur, maj
    """

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

    def activation_et_derivative(self, activation_i, n_fct):
        """Retourne [Fi, Fp] selon le choix n_fct.

        n_fct:
            1 : sigmoïde
            2 : tanh
            3 : gelu
            4 : tan
        """

        if n_fct == 1:
            return sigmoide_et_derivative(activation_i)
        if n_fct == 2:
            return tanh_et_derivative(activation_i)
        if n_fct == 3:
            return gelu_et_derivative(activation_i)
        if n_fct == 4:
            return tan_et_derivative(activation_i)

        raise ValueError("n_fct doit être 1 (sigmoïde), 2 (tanh), 3 (gelu) ou 4 (tan)")

    # =============================
    # Backpropagation (BackPP)
    # =============================

    @staticmethod
    def maj(w, delta_w):
        """Mise à jour d'un poids (Backpropagation classique)."""

        return w + delta_w

    @staticmethod
    def Correcteur(eta, x, delta):
        """Facteur de correction d'un poids (Backpropagation classique)."""

        return eta * x * delta

    @staticmethod
    def Delta_cache(delta_w_list, Fp):
        """Signal d'erreur (delta) d'un neurone caché (Backpropagation classique)."""

        somme = sum(delta * w for delta, w in delta_w_list)
        return somme * Fp

    @staticmethod
    def Delta(d, Fi, Fp):
        """Signal d'erreur (delta) d'un neurone de sortie (Backpropagation classique)."""

        return (d - Fi) * Fp
