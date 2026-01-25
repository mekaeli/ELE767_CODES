# import librairy standard
import os
import math

# =============================================================================================
# Fonction pour calculer la sigmoïde et sa dérivée
def sigmoide_et_derivative(activation_i):
	"""
	Reçoit l'activation i d'un neurone.
	Calcule et retourne une liste contenant :
		- Fi : la valeur de la fonction d'activation sigmoïde
		- Fp : la dérivée de la fonction d'activation sigmoïde
	Args:
		activation_i (float): valeur d'activation du neurone
	Returns:
		list: [Fi, Fp] où Fi est la sigmoïde et Fp sa dérivée
	"""
	Fi = 1 / (1 + math.exp(-activation_i))
	Fp = Fi * (1 - Fi)
	return [Fi, Fp]
