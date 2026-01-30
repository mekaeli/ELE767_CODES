"""Fonctions d'activation (Fi) et leurs dérivées (Fp).

Ce module regroupe les 4 fonctions d'activation utilisées dans le cours:
- sigmoïde
- tanh
- gelu (approx.)
- tan

Chaque fonction:
- reçoit une activation i (float)
- retourne [Fi, Fp] (liste de 2 floats)
"""

import math


# ==================== sigmoide_et_derivative =========================
def sigmoide_et_derivative(activation_i):
	"""Calcule la sigmoïde et sa dérivée.

	Entrée:
		activation_i: valeur i (avant activation).

	Sortie:
		[Fi, Fp]
		- Fi = 1 / (1 + exp(-i))
		- Fp = Fi * (1 - Fi)
	"""

	Fi = 1 / (1 + math.exp(-activation_i))
	Fp = Fi * (1 - Fi)
	return [Fi, Fp]


# ==================== tanh_et_derivative =========================
def tanh_et_derivative(activation_i):
	"""Calcule tanh et sa dérivée.

	Entrée:
		activation_i: valeur i (avant activation).

	Sortie:
		[Fi, Fp]
		- Fi = tanh(i)
		- Fp = 1 - Fi^2
	"""

	Fi = math.tanh(activation_i)
	Fp = 1 - Fi**2
	return [Fi, Fp]


# ==================== gelu_et_derivative =========================
def gelu_et_derivative(activation_i):
	"""Calcule GELU (approximation) et sa dérivée (approximation).

	Remarque:
		On utilise une approximation courante de GELU avec tanh.

	Entrée:
		activation_i: la valeur i (avant activation).

	Sortie:
		[Fi, Fp]
		- Fi = 0.5 * i * (1 + tanh( sqrt(2/pi) * (i + 0.044715 * i^3) ))
		- Fp = dérivée approchée associée
	"""

	i = activation_i
	a = math.sqrt(2 / math.pi)

	tanh_arg = a * (i + 0.044715 * i**3)
	tanh_val = math.tanh(tanh_arg)

	Fi = 0.5 * i * (1 + tanh_val)

	sech2 = 1 - tanh_val**2
	Fp = 0.5 * (1 + tanh_val) + 0.5 * i * sech2 * (a * (1 + 3 * 0.044715 * i**2))

	return [Fi, Fp]


# ==================== tan_et_derivative =========================
def tan_et_derivative(activation_i):
	"""Calcule tan et sa dérivée.

	Attention:
		tan(i) explose quand cos(i) est proche de 0.

	Entrée:
		activation_i: la valeur i (avant activation).

	Sortie:
		[Fi, Fp]
		- Fi = tan(i)
		- Fp = 1 / cos(i)^2
	"""

	Fi = math.tan(activation_i)
	cos_i = math.cos(activation_i)
	Fp = 1 / (cos_i**2)
	return [Fi, Fp]

