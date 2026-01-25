# import librairy standard
import os
import math

# Fonction pour calculer gelu et sa dérivée
def gelu_et_derivative(activation_i):
    """
    Reçoit l'activation i d'un neurone.
    Calcule et retourne une liste contenant :
        - Fi : la valeur de la fonction d'activation GELU (approximation)
        - Fp : la dérivée de la fonction d'activation GELU (approximation)
    Args:
        activation_i (float): valeur d'activation du neurone
    Returns:
        list: [Fi, Fp] où Fi est GELU et Fp sa dérivée
    """
    import math
    # Approximation de la GELU
    Fi = 0.5 * activation_i * (1 + math.tanh(math.sqrt(2 / math.pi) * (activation_i + 0.044715 * activation_i ** 3)))
    # Dérivée de la GELU (approximation)
    tanh_arg = math.sqrt(2 / math.pi) * (activation_i + 0.044715 * activation_i ** 3)
    tanh_val = math.tanh(tanh_arg)
    sech2 = 1 - tanh_val ** 2
    Fp = 0.5 * (1 + tanh_val) + 0.5 * activation_i * sech2 * (
        math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * activation_i ** 2)
    )
    return [Fi, Fp]





