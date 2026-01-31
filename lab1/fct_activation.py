r"""fct_activation

Rôle
	Fonctions d'activation (Fi) et dérivées (Fp) utilisées par le laboratoire.

Conventions
	- Les fonctions prennent en entrée une valeur scalaire `i` (pré-activation).
	- Elles retournent une paire `[Fi, Fp]` (liste de deux floats) afin d'être
	  directement déballable: `Fi, Fp = ...`.

Fonctions disponibles
	- `sigmoide_et_derivative(i)` : $\sigma(i)$ et $\sigma(i)\,(1-\sigma(i))$
	- `tanh_et_derivative(i)` : $\tanh(i)$ et $1-\tanh(i)^2$
	- `gelu_et_derivative(i)` : GELU (approx.) et dérivée (approx.)
	- `tan_et_derivative(i)` : $\tan(i)$ et $1/\cos(i)^2$

Notes numériques
	Certaines activations (ex: tan) peuvent produire de très grandes valeurs
	lorsque le cosinus est proche de 0. Les fonctions ici sont volontairement
	"brutes" (pas de clipping) pour coller au cours.
"""

import math


def _sigmoid_stable(x: float) -> float:
	"""Sigmoïde numériquement stable.

	Évite les overflows de exp() quand |x| est grand.
	"""
	xf = float(x)
	if xf >= 0.0:
		# exp(-x) est dans (0, 1] ici (pas d'overflow)
		z = math.exp(-xf)
		return 1.0 / (1.0 + z)
	# x < 0: exp(x) est dans (0, 1] (pas d'overflow)
	z = math.exp(xf)
	return z / (1.0 + z)


# ==================== sigmoide_et_derivative =========================
def sigmoide_et_derivative(activation_i):
	"""Calcule la sigmoïde et sa dérivée.

	Entrée:
		activation_i: valeur i (avant activation).

	Sortie:
		[Fi, Fp] (liste de deux floats)
		- Fi = 1 / (1 + exp(-i))
		- Fp = Fi * (1 - Fi)
	"""

	Fi = _sigmoid_stable(float(activation_i))
	Fp = Fi * (1.0 - Fi)
	return [Fi, Fp]


# ==================== tanh_et_derivative =========================
def tanh_et_derivative(activation_i):
	"""Calcule tanh et sa dérivée.

	Entrée:
		activation_i: valeur i (avant activation).

	Sortie:
		[Fi, Fp] (liste de deux floats)
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
		[Fi, Fp] (liste de deux floats)
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
		[Fi, Fp] (liste de deux floats)
		- Fi = tan(i)
		- Fp = 1 / cos(i)^2
	"""

	Fi = math.tan(activation_i)
	cos_i = math.cos(activation_i)
	Fp = 1 / (cos_i**2)
	return [Fi, Fp]

