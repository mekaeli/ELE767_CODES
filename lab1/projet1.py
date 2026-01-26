"""Génération + apprentissage (approche matricielle) d'un réseau dynamique.

Objectif:
	Fournir des fonctions simples pour:
	  - générer automatiquement un réseau (X, W, b, d)
	  - appliquer une itération de backpropagation en utilisant NeuroneMat

Le but est de pouvoir tester "n'importe quel système" en changeant:
	nx, nc, nsc1, nsc2, ncs, n_fct, bornes des poids (w) et bornes des biais (b).
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# Permettre l'import des modules depuis le dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurone.reso2neurone import NeuroneMat  # noqa: E402


Vector = List[float]
Matrix = List[List[float]]


@dataclass
class ReseauDynamique:
	"""Conteneur clair pour les données d'un réseau généré."""

	nx: int
	nc: int
	nsc1: int
	nsc2: int
	ncs: int
	n_fct: int
	eta: float

	vecX: Vector
	vecD: Vector
	vecW: List[Matrix]
	vecB: List[Vector]

	# Noms explicites (utile pour debug/affichage)
	named_X: Dict[str, float]
	named_D: Dict[str, float]
	named_W: Dict[str, float]
	named_B: Dict[str, float]


# ==================== _rand_binary =========================
def _rand_binary(rng: random.Random) -> float:
	"""Retourne 0.0 ou 1.0 aléatoirement."""

	return float(rng.randint(0, 1))


# ==================== _rand_uniform =========================
def _rand_uniform(rng: random.Random, vmin: float, vmax: float) -> float:
	"""Retourne une valeur uniforme dans [vmin, vmax]."""

	return float(rng.uniform(vmin, vmax))


# ==================== _build_layer_sizes =========================
def _build_layer_sizes(nx: int, nc: int, nsc1: int, nsc2: int, ncs: int) -> List[int]:
	"""Construit la taille de chaque couche.

	Convention:
		- Couche 0 = entrées: nx
		- Couche 1 = cachée 1: nsc1
		- Couche 2 = cachée 2: nsc2
		- Couche (nc-1) = sortie: ncs
		- Les autres couches (si nc > 4) ont ncs neurones.
	"""

	if nx <= 0:
		raise ValueError("nx doit être > 0")
	if nc < 2:
		raise ValueError("nc doit être >= 2 (au moins une couche de sortie)")
	if nsc1 <= 0 or nsc2 <= 0 or ncs <= 0:
		raise ValueError("nsc1, nsc2, ncs doivent être > 0")

	sizes = [nx]
	for layer_index in range(1, nc):
		if layer_index == 1:
			sizes.append(nsc1)
		elif layer_index == 2:
			sizes.append(nsc2)
		elif layer_index == nc - 1:
			sizes.append(ncs)
		else:
			sizes.append(ncs)
	return sizes


# ==================== generer_reseau_dynamique =========================
def generer_reseau_dynamique(
	*,
	nx: int = 5,
	nc: int = 4,
	nsc1: int = 15,
	nsc2: int = 15,
	ncs: int = 10,
	n_fct: int = 1,
	b: Sequence[float] = (1.0, 5.0),
	w: Sequence[float] = (-0.1, 0.1),
	eta: float = 0.1,
	seed: int | None = None,
) -> ReseauDynamique:
	"""Génère un réseau complet (X, W, B, D) avec des valeurs aléatoires.

	Paramètres:
		nx: nombre d'entrées.
		nc: nombre de couches (incluant la couche de sortie).
		nsc1: nombre de neurones de la couche 1.
		nsc2: nombre de neurones de la couche 2.
		ncs: nombre de neurones de sortie.
		n_fct: numéro de fonction d'activation (1..4).
		b: (min,max) pour les biais.
		w: (min,max) pour les poids.
		eta: taux d'apprentissage.
		seed: graine pour reproduire les mêmes tirages.

	Sortie:
		Un objet ReseauDynamique contenant vecX, vecW, vecB, vecD + des noms explicites.
	"""

	if len(b) != 2 or len(w) != 2:
		raise ValueError("b et w doivent être des couples (min,max)")
	bmin, bmax = float(b[0]), float(b[1])
	wmin, wmax = float(w[0]), float(w[1])
	if bmin > bmax:
		raise ValueError("b: min doit être <= max")
	if wmin > wmax:
		raise ValueError("w: min doit être <= max")

	rng = random.Random(seed)
	layer_sizes = _build_layer_sizes(nx, nc, nsc1, nsc2, ncs)

	# -------------------- Génération de vecX et vecD --------------------
	vecX: Vector = [_rand_binary(rng) for _ in range(nx)]
	vecD: Vector = [_rand_binary(rng) for _ in range(ncs)]

	named_X = {f"x1_{i+1}": vecX[i] for i in range(nx)}
	named_D = {f"d{nc-1}_{j+1}": vecD[j] for j in range(ncs)}

	# -------------------- Génération de vecW et vecB --------------------
	vecW: List[Matrix] = []
	vecB: List[Vector] = []
	named_W: Dict[str, float] = {}
	named_B: Dict[str, float] = {}

	# Pour chaque couche L (1..nc-1), on crée W_L et b_L
	for layer_index in range(1, nc):
		n_in = layer_sizes[layer_index - 1]
		n_out = layer_sizes[layer_index]

		W_layer: Matrix = []
		for i in range(n_in):
			row: List[float] = []
			for j in range(n_out):
				val = _rand_uniform(rng, wmin, wmax)
				row.append(val)
				named_W[f"w{layer_index}_{i+1}_{j+1}"] = val
			W_layer.append(row)

		b_layer: Vector = []
		for j in range(n_out):
			val = _rand_uniform(rng, bmin, bmax)
			b_layer.append(val)
			named_B[f"b{layer_index}_{j+1}"] = val

		vecW.append(W_layer)
		vecB.append(b_layer)

	return ReseauDynamique(
		nx=nx,
		nc=nc,
		nsc1=nsc1,
		nsc2=nsc2,
		ncs=ncs,
		n_fct=n_fct,
		eta=eta,
		vecX=vecX,
		vecD=vecD,
		vecW=vecW,
		vecB=vecB,
		named_X=named_X,
		named_D=named_D,
		named_W=named_W,
		named_B=named_B,
	)


# ==================== apprendre_une_iteration =========================
def apprendre_une_iteration(reseau: ReseauDynamique) -> Tuple[List[Matrix], List[Vector], Vector]:
	"""Applique une itération d'apprentissage (forward + backprop + mise à jour).

	Cette fonction utilise uniquement l'approche matricielle via NeuroneMat.

	Sortie:
		(new_vecW, new_vecB, y)
		- new_vecW/new_vecB: poids et biais mis à jour
		- y: sortie du réseau avant mise à jour
	"""

	nn = NeuroneMat()
	new_vecW, new_vecB, y = nn.backprop_update(
		reseau.vecX,
		reseau.vecW,
		reseau.vecB,
		reseau.vecD,
		reseau.eta,
		reseau.n_fct,
	)
	return new_vecW, new_vecB, y


# ==================== demo =========================
def demo() -> None:
	"""Petite démo: génère un réseau et fait une itération d'apprentissage."""

	reseau = generer_reseau_dynamique(seed=123)
	new_W, new_B, y = apprendre_une_iteration(reseau)

	print("=== DEMO learn.py ===")
	print(f"nx={reseau.nx}, nc={reseau.nc}, ncs={reseau.ncs}, n_fct={reseau.n_fct}")
	print(f"vecX (len={len(reseau.vecX)}): {reseau.vecX}")
	print(f"vecD (len={len(reseau.vecD)}): {reseau.vecD}")
	print(f"y (len={len(y)}): {y}")
	print(f"nb couches de poids: {len(new_W)} ; nb couches de biais: {len(new_B)}")


if __name__ == "__main__":
	demo()

