# import librairy standard
import os
import math

# Fonction pour calculer tanh et sa dérivée

def tanh_et_derivative(activation_i):
    """
    Reçoit l'activation i d'un neurone.
    Calcule et retourne une liste contenant :
        - Fi : la valeur de la fonction d'activation tanh
        - Fp : la dérivée de la fonction d'activation tanh
    Args:
        activation_i (float): valeur d'activation du neurone
    Returns:
        list: [Fi, Fp] où Fi est tanh et Fp sa dérivée
    """
    import math
    Fi = math.tanh(activation_i)
    Fp = 1 - Fi ** 2
    return [Fi, Fp]
