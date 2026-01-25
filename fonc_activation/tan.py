# import librairy standard
import os
import math


# Fonction pour calculer tan et sa dérivée
def tan_et_derivative(activation_i):

    """Reçoit l'activation i d'un neurone.
    Calcule et retourne une liste contenant :
        - Fi : la valeur de la fonction d'activation tan
        - Fp : la dérivée de la fonction d'activation tan
    Args:
        activation_i (float): valeur d'activation du neurone
    Returns:
        list: [Fi, Fp] où Fi est tan et Fp sa dérivée
    """

    Fi = math.tan(activation_i)
    # F'(i) = 1 / cos(i)^2
    cos_i = math.cos(activation_i)
    Fp = 1 / (cos_i**2)
    return [Fi, Fp]
