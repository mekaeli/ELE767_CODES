"""Backpropagation (approche matricielle).

Ce module est une version "matricielle" de [neurone/backpp_neurone_alg.py](../neurone/backpp_neurone_alg.py):
- on travaille avec des vecteurs (listes) et des matrices (listes de listes)
- on garde les mêmes fonctionnalités: calcul de i, application activation, deltas,
  corrections et mise à jour des poids/biais

Données attendues:
    X: liste d'entrées (vecteur)
    w_list: liste de couches; chaque élément est une matrice W (liste de listes)
            W est de taille (n_in, n_out)
    b_list: liste de couches; chaque élément est un vecteur b (liste)
            b est de taille (n_out)
    d_list: liste des sorties désirées (vecteur cible), taille (n_out)

Choix n_fct (ordre officiel):
    1 : sigmoïde
    2 : tan
    3 : tanh
    4 : gelu
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from neurone.fct_dactivation import (
    gelu_et_derivative,
    sigmoide_et_derivative,
    tan_et_derivative,
    tanh_et_derivative,
)


Vector = List[float]
Matrix = List[List[float]]


# ==================== _validate_shapes =========================
def _validate_shapes(X: Sequence[float], w_list: Sequence[Matrix], b_list: Sequence[Vector], d_list: Sequence[float]) -> None:
    """Valide les dimensions de X, w_list, b_list et d_list.

    But:
        Détecter tôt les erreurs de dimensions (très fréquentes en matriciel).

    Règles:
        - len(w_list) == len(b_list)
        - pour chaque couche: W est (n_in, n_out) et b est (n_out)
        - la sortie finale doit avoir la même taille que d_list
    """

    if len(w_list) == 0:
        raise ValueError("w_list ne doit pas être vide.")

    if len(w_list) != len(b_list):
        raise ValueError("w_list et b_list doivent avoir la même longueur (une entrée par couche).")

    current_size = len(X)
    if current_size == 0:
        raise ValueError("X ne doit pas être vide.")

    for layer_index, (W, b) in enumerate(zip(w_list, b_list)):
        if len(W) == 0:
            raise ValueError(f"Couche {layer_index}: W ne doit pas être vide.")

        # Vérifie que W a bien current_size lignes
        if len(W) != current_size:
            raise ValueError(
                f"Couche {layer_index}: W doit avoir {current_size} lignes (n_in), mais en a {len(W)}."
            )

        # Vérifie que toutes les lignes de W ont la même longueur (n_out)
        n_out = len(W[0])
        if n_out == 0:
            raise ValueError(f"Couche {layer_index}: W ne doit pas contenir de lignes vides.")

        for row in W:
            if len(row) != n_out:
                raise ValueError(f"Couche {layer_index}: toutes les lignes de W doivent avoir la même taille.")

        if len(b) != n_out:
            raise ValueError(
                f"Couche {layer_index}: b doit avoir {n_out} éléments (n_out), mais en a {len(b)}."
            )

        current_size = n_out

    if len(d_list) != current_size:
        raise ValueError(
            f"d_list doit avoir la même taille que la sortie finale ({current_size}), mais a {len(d_list)}."
        )


# ==================== _matvec =========================
def _matvec(W: Matrix, x: Sequence[float]) -> Vector:
    """Calcule x @ W (produit vecteur-matrice).

    Convention:
        - x est un vecteur taille (n_in)
        - W est une matrice taille (n_in, n_out)
        - résultat est un vecteur taille (n_out)
    """

    if len(W) != len(x):
        raise ValueError("Dimensions incompatibles: len(W) doit être égal à len(x).")

    n_out = len(W[0])
    out = [0.0] * n_out
    for i, xi in enumerate(x):
        row = W[i]
        for j in range(n_out):
            out[j] += xi * row[j]
    return out


# ==================== _vec_add =========================
def _vec_add(a: Sequence[float], b: Sequence[float]) -> Vector:
    """Additionne deux vecteurs a + b."""

    if len(a) != len(b):
        raise ValueError("Addition impossible: vecteurs de tailles différentes.")
    return [ai + bi for ai, bi in zip(a, b)]


# ==================== _vec_mul =========================
def _vec_mul(a: Sequence[float], b: Sequence[float]) -> Vector:
    """Produit élément-par-élément: a * b."""

    if len(a) != len(b):
        raise ValueError("Produit impossible: vecteurs de tailles différentes.")
    return [ai * bi for ai, bi in zip(a, b)]


# ==================== _outer =========================
def _outer(x: Sequence[float], y: Sequence[float]) -> Matrix:
    """Produit externe: outer(x, y).

    Sortie:
        matrice M de taille (len(x), len(y)) avec M[i][j] = x[i] * y[j]
    """

    return [[xi * yj for yj in y] for xi in x]


# ==================== _scale_matrix =========================
def _scale_matrix(W: Matrix, alpha: float) -> Matrix:
    """Multiplie une matrice par un scalaire."""

    return [[alpha * wij for wij in row] for row in W]


# ==================== _scale_vector =========================
def _scale_vector(v: Sequence[float], alpha: float) -> Vector:
    """Multiplie un vecteur par un scalaire."""

    return [alpha * vi for vi in v]


# ==================== _matrix_add =========================
def _matrix_add(A: Matrix, B: Matrix) -> Matrix:
    """Additionne deux matrices A + B."""

    if len(A) != len(B):
        raise ValueError("Addition impossible: matrices de tailles différentes (nb lignes).")

    out: Matrix = []
    for row_a, row_b in zip(A, B):
        if len(row_a) != len(row_b):
            raise ValueError("Addition impossible: matrices de tailles différentes (nb colonnes).")
        out.append([a + b for a, b in zip(row_a, row_b)])
    return out


# ==================== _transpose_matvec =========================
def _transpose_matvec(W: Matrix, delta_next: Sequence[float]) -> Vector:
    """Calcule l'erreur rétro-propagée vers la couche précédente.

    Ici, W est la matrice des poids (n_in, n_out) allant de la couche courante vers
    la couche suivante.

    On calcule:
        err_prev[k] = sum_j delta_next[j] * W[k][j]

    Sortie:
        err_prev (taille n_in)
    """

    n_in = len(W)
    n_out = len(W[0])
    if len(delta_next) != n_out:
        raise ValueError("delta_next doit avoir la taille n_out (nb neurones de la couche suivante).")

    err_prev = [0.0] * n_in
    for k in range(n_in):
        s = 0.0
        row = W[k]
        for j in range(n_out):
            s += delta_next[j] * row[j]
        err_prev[k] = s
    return err_prev


# ==================== _activation_scalar =========================
def _activation_scalar(i: float, n_fct: int) -> Tuple[float, float]:
    """Applique la fonction d'activation choisie sur un scalaire i.

    Retour:
        (Fi, Fp)
    """

    if n_fct == 1:
        Fi, Fp = sigmoide_et_derivative(i)
        return float(Fi), float(Fp)
    if n_fct == 2:
        Fi, Fp = tan_et_derivative(i)
        return float(Fi), float(Fp)
    if n_fct == 3:
        Fi, Fp = tanh_et_derivative(i)
        return float(Fi), float(Fp)
    if n_fct == 4:
        Fi, Fp = gelu_et_derivative(i)
        return float(Fi), float(Fp)

    raise ValueError("n_fct doit être 1 (sigmoïde), 2 (tan), 3 (tanh) ou 4 (gelu)")


@dataclass
class ForwardCache:
    """Cache de la propagation avant (utile pour la rétropropagation)."""

    z_list: List[Vector]  # activations brutes i (une liste par couche)
    a_list: List[Vector]  # sorties Fi (une liste par couche)
    fp_list: List[Vector]  # dérivées Fp (une liste par couche)


# ==================== NeuroneMat =========================
class NeuroneMat:
    """Neurone + Backpropagation en approche matricielle.

    Idée:
        Au lieu de calculer un neurone à la fois, on calcule toute une couche
        avec des opérations vecteur/matrice.

    Convention des dimensions:
        - X est un vecteur (n_in)
        - W est une matrice (n_in, n_out)
        - b est un vecteur (n_out)
    """

    # ==================== forward_layer =========================
    def forward_layer(self, x: Sequence[float], W: Matrix, b: Sequence[float], n_fct: int) -> Tuple[Vector, Vector, Vector]:
        """Propagation avant pour une couche.

        Entrées:
            x: vecteur d'entrée de la couche
            W: matrice des poids (n_in, n_out)
            b: biais (n_out)
            n_fct: choix de fonction d'activation

        Sorties:
            (z, a, fp)
            - z: i = x @ W + b
            - a: Fi appliquée sur z
            - fp: Fp (dérivée) appliquée sur z
        """

        z = _vec_add(_matvec(W, x), b)
        a: Vector = []
        fp: Vector = []
        for zi in z:
            Fi, Fp = _activation_scalar(zi, n_fct)
            a.append(Fi)
            fp.append(Fp)
        return z, a, fp

    # ==================== forward =========================
    def forward(self, X: Sequence[float], w_list: Sequence[Matrix], b_list: Sequence[Vector], n_fct: int) -> Tuple[Vector, ForwardCache]:
        """Propagation avant sur toutes les couches.

        Entrées:
            X: vecteur d'entrée
            w_list: liste de matrices W (une par couche)
            b_list: liste de vecteurs b (un par couche)
            n_fct: choix de fonction d'activation

        Sorties:
            (y, cache)
            - y: sortie finale (Fi de la dernière couche)
            - cache: contient z/a/fp de chaque couche
        """

        if len(w_list) != len(b_list):
            raise ValueError("w_list et b_list doivent avoir la même longueur.")

        z_list: List[Vector] = []
        a_list: List[Vector] = []
        fp_list: List[Vector] = []

        x = list(X)
        for W, b in zip(w_list, b_list):
            z, a, fp = self.forward_layer(x, W, b, n_fct)
            z_list.append(z)
            a_list.append(a)
            fp_list.append(fp)
            x = a

        return a_list[-1], ForwardCache(z_list=z_list, a_list=a_list, fp_list=fp_list)

    # ==================== Delta =========================
    @staticmethod
    def Delta(d_list: Sequence[float], Fi_list: Sequence[float], Fp_list: Sequence[float]) -> Vector:
        """Delta (signal d'erreur) pour la couche de sortie.

        Formule (élément par élément):
            delta = (d - Fi) * Fp
        """

        if len(d_list) != len(Fi_list) or len(Fi_list) != len(Fp_list):
            raise ValueError("d_list, Fi_list et Fp_list doivent avoir la même taille.")
        return [(d - Fi) * Fp for d, Fi, Fp in zip(d_list, Fi_list, Fp_list)]

    # ==================== Delta_cache =========================
    @staticmethod
    def Delta_cache(delta_next: Sequence[float], W_next: Matrix, Fp_current: Sequence[float]) -> Vector:
        """Delta (signal d'erreur) pour une couche cachée.

        Entrées:
            delta_next: delta de la couche suivante (taille n_out)
            W_next: poids entre couche courante et suivante (n_in, n_out)
            Fp_current: dérivée de la couche courante (taille n_in)

        Sortie:
            delta_current (taille n_in)
        """

        err_prev = _transpose_matvec(W_next, delta_next)
        return _vec_mul(err_prev, Fp_current)

    # ==================== Correcteur =========================
    @staticmethod
    def Correcteur(eta: float, a_prev: Sequence[float], delta_current: Sequence[float]) -> Tuple[Matrix, Vector]:
        """Calcule les corrections (delta_W, delta_b) pour une couche.

        Formules:
            delta_W = eta * outer(a_prev, delta_current)
            delta_b = eta * delta_current
        """

        delta_W = _scale_matrix(_outer(a_prev, delta_current), eta)
        delta_b = _scale_vector(delta_current, eta)
        return delta_W, delta_b

    # ==================== maj =========================
    @staticmethod
    def maj(W: Matrix, delta_W: Matrix, b: Sequence[float], delta_b: Sequence[float]) -> Tuple[Matrix, Vector]:
        """Met à jour (W, b) en ajoutant (delta_W, delta_b)."""

        new_W = _matrix_add(W, delta_W)
        new_b = _vec_add(b, delta_b)
        return new_W, new_b

    # ==================== backprop_update =========================
    def backprop_update(
        self,
        X: Sequence[float],
        w_list: Sequence[Matrix],
        b_list: Sequence[Vector],
        d_list: Sequence[float],
        eta: float,
        n_fct: int,
    ) -> Tuple[List[Matrix], List[Vector], Vector]:
        """Fait une itération complète: forward + backprop + mise à jour.

        Entrées:
            X: vecteur d'entrée
            w_list: liste des matrices de poids (une par couche)
            b_list: liste des vecteurs de biais (un par couche)
            d_list: cible (vecteur de sortie désirée)
            eta: taux d'apprentissage
            n_fct: choix de fonction d'activation

        Sorties:
            (new_w_list, new_b_list, y)
            - new_w_list/new_b_list: paramètres mis à jour
            - y: sortie du réseau avant mise à jour (utile pour affichage)
        """

        _validate_shapes(X, w_list, b_list, d_list)

        y, cache = self.forward(X, w_list, b_list, n_fct)

        # 1) delta de sortie
        delta_list: List[Vector] = [None] * len(w_list)  # type: ignore[assignment]
        delta_out = self.Delta(d_list, y, cache.fp_list[-1])
        delta_list[-1] = delta_out

        # 2) deltas des couches cachées (de la fin vers le début)
        for layer_index in range(len(w_list) - 2, -1, -1):
            W_next = w_list[layer_index + 1]
            Fp_current = cache.fp_list[layer_index]
            delta_list[layer_index] = self.Delta_cache(delta_list[layer_index + 1], W_next, Fp_current)

        # 3) corrections + mise à jour (de l'entrée vers la sortie)
        new_w_list: List[Matrix] = []
        new_b_list: List[Vector] = []

        a_prev = list(X)
        for layer_index, (W, b) in enumerate(zip(w_list, b_list)):
            delta_current = delta_list[layer_index]
            delta_W, delta_b = self.Correcteur(eta, a_prev, delta_current)
            new_W, new_b = self.maj(W, delta_W, b, delta_b)
            new_w_list.append(new_W)
            new_b_list.append(new_b)
            a_prev = cache.a_list[layer_index]

        return new_w_list, new_b_list, y
