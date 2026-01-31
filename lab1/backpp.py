"""backpp

Rôle
	Ce module regroupe la logique *numérique* d'un petit MLP (propagation avant,
	rétropropagation, mise à jour des poids/biais) *et* la logique d'*affichage*
	détaillé utilisée dans le laboratoire.

Contrainte de projet
	Le module est volontairement *orienté classe* (pas de fonctions/variables
	globales), afin de rester compatible avec l'architecture du projet.

Structure `struct_reso`
	La structure de réseau est un dictionnaire produit par `reseau.mon_reso`.
	Les clés dépendent des dimensions et ont la forme :
		- `X{n_in}` : entrées, liste de paires (nom, valeur), ex. ('x1', 1.0)
		- `W{n_layers}` : poids, liste de couches, chaque couche = liste (nom, valeur)
		- `B{n_layers}` : biais, liste de couches, chaque couche = liste (nom, valeur)
		- `D{n_out}` : sorties désirées, liste (nom, valeur), ex. ('d1', 1)

	Convention de noms (couche k en 1..n_layers)
		- poids : w{i}{j}_{k} = poids de l'entrée i vers le neurone j (couche k)
		- biais  : b{j}_{k} = biais du neurone j (couche k)

Classes
	- `console_utils` : utilitaires console (effacer l'écran)
	- `ForwardCache` : cache des valeurs intermédiaires (z, a, fp) pour backprop
	- `NeuroneMat` : cœur numérique (forward, delta, correcteurs, maj)
	- `backpp` : façade de haut niveau autour de `struct_reso` + affichage

Affichage (séquences)
	`backpp.resolution_affiche()` peut afficher plusieurs sections.
	Par défaut, si `show_resume=True`, `struct_reso` est aussi affiché.
	L'ordre par défaut de `struct_reso` dépend de `show_params`:
		- option 1 (test unitaire) : `show_params=False` -> `struct_reso` avant Résumé
		- autres options : `show_params=True` -> Paramètres -> Résumé -> `struct_reso`

	Le style d'impression de `struct_reso` est identique à l'affichage des
	paramètres (réutilise `reseau.mon_reso.Affiche_reso`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import unicodedata
from typing import Dict, List, Sequence, Tuple

# Imports robustes : permettent d'exécuter depuis la racine du workspace
# (import lab1.*) ou directement depuis le répertoire lab1/.
try:
	from lab1.reseau import mon_reso  # type: ignore
except Exception:
	from reseau import mon_reso  # type: ignore

try:
	from lab1.fct_activation import (  # type: ignore
		gelu_et_derivative,
		sigmoide_et_derivative,
		tan_et_derivative,
		tanh_et_derivative,
	)
except Exception:
	from fct_activation import (  # type: ignore
		gelu_et_derivative,
		sigmoide_et_derivative,
		tan_et_derivative,
		tanh_et_derivative,
	)


class console_utils:
	"""Utilitaires console (évite les dépendances externes)."""

	# ==================== clear_console =========================
	@staticmethod
	def clear_console() -> None:
		"""Nettoie la console.

		- Windows : commande `cls`
		- Linux/Mac : commande `clear`
		"""
		os.system("cls" if os.name == "nt" else "clear")
		print("Console nettoyée.")


@dataclass
class ForwardCache:
	"""Cache de la propagation avant.

	Attributs
		z_list : liste des vecteurs i (pré-activations) par couche
		a_list : liste des vecteurs a (activations) par couche
		fp_list : liste des vecteurs F' (dérivées de l'activation) par couche

	Ces listes sont indexées par couche (0..L-1).
	"""

	z_list: List[List[float]]
	a_list: List[List[float]]
	fp_list: List[List[float]]


@dataclass
class BackpropRunResult:
	"""Résultat d'une itération (calcul) pour l'affichage pédagogique.

	Cette structure sépare le *calcul* (pur) du *print*.
	"""

	X: List[float]
	w_list: List[List[List[float]]]
	b_list: List[List[float]]
	d_list: List[float]
	eta: float
	n_fct: int
	nom_fct: str
	y: List[float]
	z_list: List[List[float]]
	a_list: List[List[float]]
	fp_list: List[List[float]]
	delta_list: List[List[float]]
	err_list: List[List[float] | None]
	dW_list: List[List[List[float]]]
	db_list: List[List[float]]
	new_w_list: List[List[List[float]]]
	new_b_list: List[List[float]]
	prec_up: int


class NeuroneMat:
	"""Cœur numérique (approche matricielle) : forward + rétroprop.

	Cette classe manipule des matrices/vecteurs Python (listes) et implémente :
		- forward : calcule y et retourne un cache (z, a, fp)
		- delta (sortie) et delta_cache (caché) : signaux d'erreur
		- correcteurs (dW/db) et maj (W/b)

	Elle est indépendante de `struct_reso` : elle travaille uniquement avec des
	vecteurs et des matrices déjà ordonnés.
	"""

	# ==================== helpers (privés) ====================
	# ==================== _validate_shapes =========================
	@staticmethod
	def _validate_shapes(
		X: Sequence[float],
		w_list: Sequence[List[List[float]]],
		b_list: Sequence[List[float]],
		d_list: Sequence[float],
	) -> None:
		"""Vérifie la cohérence des dimensions (X, W, b, d).

		Paramètres
			X : vecteur d'entrée (taille n_in)
			w_list : liste de matrices W_k de taille (n_in_k, n_out_k)
			b_list : liste de vecteurs b_k de taille (n_out_k)
			d_list : sortie désirée (taille n_out_final)
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

			if len(W) != current_size:
				raise ValueError(
					f"Couche {layer_index}: W doit avoir {current_size} lignes (n_in), mais en a {len(W)}."
				)

			n_out = len(W[0])
			if n_out == 0:
				raise ValueError(f"Couche {layer_index}: W ne doit pas contenir de lignes vides.")

			for row in W:
				if len(row) != n_out:
					raise ValueError(
						f"Couche {layer_index}: toutes les lignes de W doivent avoir la même taille."
					)

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
	@staticmethod
	def _matvec(W: List[List[float]], x: Sequence[float]) -> List[float]:
		"""Calcule x @ W (x vecteur ligne).

		W est stockée par lignes (une ligne par entrée).
		Si W est de taille (n_in, n_out) et x de taille (n_in), la sortie est
		de taille (n_out).
		"""
		if len(W) != len(x):
			raise ValueError("Dimensions incompatibles: len(W) doit être égal à len(x).")

		n_out = len(W[0])
		out = [0.0] * n_out
		for i, xi in enumerate(x):
			row = W[i]
			for j in range(n_out):
				out[j] += float(xi) * float(row[j])
		return out

	# ==================== _vec_add =========================
	@staticmethod
	def _vec_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
		"""Additionne deux vecteurs de même taille."""
		if len(a) != len(b):
			raise ValueError("Addition impossible: vecteurs de tailles différentes.")
		return [float(ai) + float(bi) for ai, bi in zip(a, b)]

	# ==================== _vec_mul =========================
	@staticmethod
	def _vec_mul(a: Sequence[float], b: Sequence[float]) -> List[float]:
		"""Multiplie deux vecteurs élément par élément (même taille)."""
		if len(a) != len(b):
			raise ValueError("Produit impossible: vecteurs de tailles différentes.")
		return [float(ai) * float(bi) for ai, bi in zip(a, b)]

	# ==================== _outer =========================
	@staticmethod
	def _outer(x: Sequence[float], y: Sequence[float]) -> List[List[float]]:
		"""Calcule le produit extérieur: outer(x, y)."""
		return [[float(xi) * float(yj) for yj in y] for xi in x]

	# ==================== _scale_matrix =========================
	@staticmethod
	def _scale_matrix(W: List[List[float]], alpha: float) -> List[List[float]]:
		"""Multiplie une matrice par un scalaire."""
		return [[float(alpha) * float(wij) for wij in row] for row in W]

	# ==================== _scale_vector =========================
	@staticmethod
	def _scale_vector(v: Sequence[float], alpha: float) -> List[float]:
		"""Multiplie un vecteur par un scalaire."""
		return [float(alpha) * float(vi) for vi in v]

	# ==================== _matrix_add =========================
	@staticmethod
	def _matrix_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
		"""Additionne deux matrices de mêmes dimensions."""
		if len(A) != len(B):
			raise ValueError("Addition impossible: matrices de tailles différentes (nb lignes).")

		out: List[List[float]] = []
		for row_a, row_b in zip(A, B):
			if len(row_a) != len(row_b):
				raise ValueError("Addition impossible: matrices de tailles différentes (nb colonnes).")
			out.append([float(a) + float(b) for a, b in zip(row_a, row_b)])
		return out

	# ==================== _transpose_matvec =========================
	@staticmethod
	def _transpose_matvec(W: List[List[float]], delta_next: Sequence[float]) -> List[float]:
		"""Remonte un signal d'erreur depuis la couche suivante.

		Ici, W est la matrice de la couche suivante (taille n_in x n_out).
		Avec delta_next de taille n_out, on calcule err_prev de taille n_in :
			err_prev[k] = Σ_j W[k][j] * delta_next[j]
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
				s += float(delta_next[j]) * float(row[j])
			err_prev[k] = s
		return err_prev

	# ==================== _activation_scalar =========================
	@staticmethod
	def _activation_scalar(i: float, n_fct: int) -> Tuple[float, float]:
		"""Retourne (Fi, Fp) pour un scalaire i.

		Fi : activation
		Fp : dérivée de l'activation évaluée au même i
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

	# ==================== API ====================
	# ==================== forward_layer =========================
	def forward_layer(
		self,
		x: Sequence[float],
		W: List[List[float]],
		b: Sequence[float],
		n_fct: int,
	) -> Tuple[List[float], List[float], List[float]]:
		"""Propagation avant pour une couche.

		Calcule :
			z = x @ W + b
			a = Fi(z)
			fp = F'(z)

		Retour
			(z, a, fp) avec des vecteurs de taille n_out.
		"""
		z = self._vec_add(self._matvec(W, x), b)
		a: List[float] = []
		fp: List[float] = []
		for zi in z:
			Fi, Fp = self._activation_scalar(float(zi), int(n_fct))
			a.append(Fi)
			fp.append(Fp)
		return z, a, fp

	# ==================== forward =========================
	def forward(
		self,
		X: Sequence[float],
		w_list: Sequence[List[List[float]]],
		b_list: Sequence[List[float]],
		n_fct: int,
	) -> Tuple[List[float], ForwardCache]:
		"""Propagation avant complète.

		Retourne
			y : vecteur de sortie final
			cache : z_list/a_list/fp_list par couche (pour la rétropropagation)
		"""
		if len(w_list) != len(b_list):
			raise ValueError("w_list et b_list doivent avoir la même longueur.")

		z_list: List[List[float]] = []
		a_list: List[List[float]] = []
		fp_list: List[List[float]] = []

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
	def Delta(d_list: Sequence[float], Fi_list: Sequence[float], Fp_list: Sequence[float]) -> List[float]:
		"""Delta de sortie.

		Formule : delta_L = (d - y) ⊙ fp_L
		- d : sortie désirée
		- y : sortie courante (activations de la dernière couche)
		- fp : dérivée de l'activation de la dernière couche
		"""
		if len(d_list) != len(Fi_list) or len(Fi_list) != len(Fp_list):
			raise ValueError("d_list, Fi_list et Fp_list doivent avoir la même taille.")
		return [(float(d) - float(Fi)) * float(Fp) for d, Fi, Fp in zip(d_list, Fi_list, Fp_list)]

	# ==================== Delta_cache =========================
	@staticmethod
	def Delta_cache(delta_next: Sequence[float], W_next: List[List[float]], Fp_current: Sequence[float]) -> List[float]:
		"""Delta d'une couche cachée.

		Formule :
			err_prev = W_next @ delta_next
			delta_current = err_prev ⊙ fp_current
		"""
		err_prev = NeuroneMat._transpose_matvec(W_next, delta_next)
		return NeuroneMat._vec_mul(err_prev, Fp_current)

	# ==================== Correcteur =========================
	@staticmethod
	def Correcteur(
		eta: float,
		a_prev: Sequence[float],
		delta_current: Sequence[float],
	) -> Tuple[List[List[float]], List[float]]:
		"""Calcule les corrections (dW, db) pour une couche.

		Formules :
			dW = eta * outer(a_prev, delta)
			db = eta * delta
		"""
		delta_W = NeuroneMat._scale_matrix(NeuroneMat._outer(a_prev, delta_current), float(eta))
		delta_b = NeuroneMat._scale_vector(delta_current, float(eta))
		return delta_W, delta_b

	# ==================== maj =========================
	@staticmethod
	def maj(
		W: List[List[float]],
		delta_W: List[List[float]],
		b: Sequence[float],
		delta_b: Sequence[float],
	) -> Tuple[List[List[float]], List[float]]:
		"""Applique la mise à jour des paramètres.

		Convention de ce laboratoire :
			W <- W + dW
			b <- b + db
		"""
		new_W = NeuroneMat._matrix_add(W, delta_W)
		new_b = NeuroneMat._vec_add(b, delta_b)
		return new_W, new_b

	# ==================== backprop_update =========================
	def backprop_update(
		self,
		X: Sequence[float],
		w_list: Sequence[List[List[float]]],
		b_list: Sequence[List[float]],
		d_list: Sequence[float],
		eta: float,
		n_fct: int,
	) -> Tuple[List[List[List[float]]], List[List[float]], List[float]]:
		"""Une itération complète (forward + rétroprop + mise à jour).

		Retourne
			(new_w_list, new_b_list, y)
		avec y la sortie avant mise à jour (issue du forward).
		"""
		self._validate_shapes(X, w_list, b_list, d_list)
		y, cache = self.forward(X, w_list, b_list, n_fct)

		delta_list: List[List[float]] = [None] * len(w_list)  # type: ignore[assignment]
		delta_out = self.Delta(d_list, y, cache.fp_list[-1])
		delta_list[-1] = delta_out

		for layer_index in range(len(w_list) - 2, -1, -1):
			W_next = w_list[layer_index + 1]
			Fp_current = cache.fp_list[layer_index]
			delta_list[layer_index] = self.Delta_cache(delta_list[layer_index + 1], W_next, Fp_current)

		new_w_list: List[List[List[float]]] = []
		new_b_list: List[List[float]] = []

		a_prev = list(X)
		for layer_index, (W, b) in enumerate(zip(w_list, b_list)):
			delta_current = delta_list[layer_index]
			delta_W, delta_b = self.Correcteur(float(eta), a_prev, delta_current)
			new_W, new_b = self.maj(W, delta_W, b, delta_b)
			new_w_list.append(new_W)
			new_b_list.append(new_b)
			a_prev = cache.a_list[layer_index]

		return new_w_list, new_b_list, y
# ==================== Classe backpp ====================


class backpp:
	"""Façade de haut niveau autour de `struct_reso`.

	Cette classe :
		- lit `struct_reso` (paires nom/valeur) et reconstruit X, W, b, d
		- appelle le cœur numérique `NeuroneMat`
		- réécrit les nouveaux poids/biais dans `struct_reso` (si demandé)
		- fournit un affichage détaillé conforme aux exigences du laboratoire

	Important
		`struct_reso` est modifiée *en place* lors des mises à jour.
	"""

	# ==================== __init__ =========================
	def __init__(self, struct_reso: Dict[str, object], n_fct: int = 1, eta: float = 0.1):
		"""Initialise le solveur.

		Paramètres
			struct_reso : dictionnaire (voir docstring du module)
			n_fct : identifiant de la fonction d'activation (1..4)
			eta : taux d'apprentissage
		"""
		self.struct_reso = struct_reso
		self.n_fct = int(n_fct)
		self.eta = float(eta)
		self._core = NeuroneMat()

	# ==================== Console policy =========================
	@dataclass(frozen=True)
	class ConsolePolicy:
		"""Politique d'affichage console.

		Objectif: centraliser la politique d'affichage (niveau minimal/détaillé)
		au lieu d'avoir de nombreux flags éparpillés.
		"""

		show_params: bool
		params_style: str
		show_equations: bool
		show_resume: bool
		only_sections: bool

	@staticmethod
	def _normalize_console_level(value: str) -> str:
		"""Normalise un niveau de log console (lower + sans accents)."""
		s = str(value or "").strip().lower()
		s = unicodedata.normalize("NFKD", s)
		s = "".join(ch for ch in s if not unicodedata.combining(ch))
		return s

	@classmethod
	def _console_policy(cls, level: str) -> "backpp.ConsolePolicy":
		lvl = cls._normalize_console_level(level)
		# Synonymes acceptés
		if lvl in {"min", "minimal", "mini", "m"}:
			# Exigence labo (options != 1): uniquement Paramètres / Résumé / struct_reso.
			# Le contenu du Résumé peut rester détaillé; c'est une section permise.
			return cls.ConsolePolicy(
				show_params=True,
				params_style="assign",
				show_equations=True,
				show_resume=True,
				only_sections=True,
			)
		if lvl in {"detail", "detaille", "detaile", "d", "detailed"}:
			# Mode détaillé: forward/deltas/correcteurs/maj + résumé.
			# Les paramètres sont typiquement déjà affichés via reseau.Affiche_reso en option 1.
			return cls.ConsolePolicy(
				show_params=False,
				params_style="matrix",
				show_equations=True,
				show_resume=True,
				only_sections=False,
			)
		raise ValueError(f"log_console invalide: '{level}' (attendu: minimal/détaillé)")

	# ==================== _validate_shapes_detailed =========================
	@staticmethod
	def _validate_shapes_detailed(
		X: Sequence[float],
		w_list: Sequence[List[List[float]]],
		b_list: Sequence[List[float]],
		d_list: Sequence[float],
	) -> None:
		"""Validation stricte des dimensions avec messages "attendu vs reçu"."""
		if len(w_list) == 0:
			raise ValueError("Dimensions invalides: w_list ne doit pas être vide.")

		if len(w_list) != len(b_list):
			raise ValueError(
				f"Dimensions invalides: nb couches W attendu={len(w_list)} vs nb couches b reçu={len(b_list)}."
			)

		n_in_expected = len(X)
		if n_in_expected == 0:
			raise ValueError("Dimensions invalides: X ne doit pas être vide (attendu n_in>0).")

		current_size = n_in_expected
		for layer_index, (W, b) in enumerate(zip(w_list, b_list), start=1):
			if len(W) == 0:
				raise ValueError(f"Dimensions invalides: W{layer_index} vide (attendu {current_size} lignes).")

			n_in_received = len(W)
			if n_in_received != current_size:
				raise ValueError(
					f"Dimensions invalides: W{layer_index} nb lignes attendu={current_size} vs reçu={n_in_received}."
				)

			# Colonnes
			n_out_expected = len(W[0])
			if n_out_expected == 0:
				raise ValueError(f"Dimensions invalides: W{layer_index} a 0 colonne (attendu n_out>0).")
			for row_idx, row in enumerate(W, start=1):
				if len(row) != n_out_expected:
					raise ValueError(
						f"Dimensions invalides: W{layer_index} ligne {row_idx} nb colonnes attendu={n_out_expected} vs reçu={len(row)}."
					)

			if len(b) != n_out_expected:
				raise ValueError(
					f"Dimensions invalides: b{layer_index} taille attendue={n_out_expected} vs reçue={len(b)}."
				)

			current_size = n_out_expected

		if len(d_list) != current_size:
			raise ValueError(
				f"Dimensions invalides: d taille attendue={current_size} vs reçue={len(d_list)}."
			)

	# ==================== _run_from_struct (calcul) =========================
	def _run_from_struct(
		self,
		*,
		eta: float | None,
		n_fct: int | None,
		update_struct: bool,
		precision: int,
	) -> BackpropRunResult:
		"""Effectue le calcul complet (forward + rétroprop + maj) sans affichage."""
		X, w_list, b_list, d_list = self._struct_to_matrices()
		use_eta = self.eta if eta is None else float(eta)
		use_n_fct = self.n_fct if n_fct is None else int(n_fct)
		nom_fct = self._nom_fct(use_n_fct)

		# Validation stricte des dimensions (messages attendu vs reçu)
		self._validate_shapes_detailed(X, w_list, b_list, d_list)

		y, cache = self._core.forward(X, w_list, b_list, use_n_fct)
		z_list = cache.z_list
		a_list = cache.a_list
		fp_list = cache.fp_list

		delta_list: List[List[float]] = [None] * len(w_list)  # type: ignore[assignment]
		err_list: List[List[float] | None] = [None] * len(w_list)
		delta_list[-1] = self._core.Delta(d_list, y, fp_list[-1])
		for layer_index in range(len(w_list) - 2, -1, -1):
			delta_list[layer_index] = self._core.Delta_cache(
				delta_list[layer_index + 1], w_list[layer_index + 1], fp_list[layer_index]
			)

		# Calcule les err (utilisés par le résumé) indépendamment de l'affichage.
		for layer_index in range(len(w_list) - 1, 0, -1):
			W_next = w_list[layer_index]
			delta_next = delta_list[layer_index]
			err_prev: List[float] = []
			for k in range(len(W_next)):
				s = 0.0
				row = W_next[k]
				for j in range(len(row)):
					s += float(row[j]) * float(delta_next[j])
				err_prev.append(s)
			err_list[layer_index - 1] = err_prev

		dW_list: List[List[List[float]]] = []
		db_list: List[List[float]] = []
		new_w_list: List[List[List[float]]] = []
		new_b_list: List[List[float]] = []

		a_prev = list(X)
		for layer_index, (W, b) in enumerate(zip(w_list, b_list)):
			delta_current = delta_list[layer_index]
			dW, db = self._core.Correcteur(use_eta, a_prev, delta_current)
			new_W, new_b = self._core.maj(W, dW, b, db)
			dW_list.append(dW)
			db_list.append(db)
			new_w_list.append(new_W)
			new_b_list.append(new_b)
			a_prev = a_list[layer_index]

		# Mise à jour de la structure avant tout affichage dépendant de struct_reso.
		if update_struct:
			self._matrices_to_struct(new_w_list, new_b_list)

		prec_up = max(int(precision), 12)
		return BackpropRunResult(
			X=list(X),
			w_list=[list(map(list, W)) for W in w_list],
			b_list=[list(b) for b in b_list],
			d_list=list(d_list),
			eta=use_eta,
			n_fct=use_n_fct,
			nom_fct=nom_fct,
			y=list(y),
			z_list=[list(z) for z in z_list],
			a_list=[list(a) for a in a_list],
			fp_list=[list(fp) for fp in fp_list],
			delta_list=[list(d) for d in delta_list],
			err_list=[(list(e) if e is not None else None) for e in err_list],
			dW_list=[list(map(list, dW)) for dW in dW_list],
			db_list=[list(db) for db in db_list],
			new_w_list=[list(map(list, W)) for W in new_w_list],
			new_b_list=[list(b) for b in new_b_list],
			prec_up=prec_up,
		)

	# ==================== _get_by_prefix =========================
	@staticmethod
	def _get_by_prefix(struct_reso: Dict[str, object], prefix: str):
		"""Retourne (clé, valeur) du premier champ dont la clé commence par `prefix`.

		Exemples
			prefix='X' -> ('X2', [...])
			prefix='W' -> ('W2', [...])
		"""
		for k, v in struct_reso.items():
			if isinstance(k, str) and k.startswith(prefix):
				return k, v
		return None, None

	# ==================== _pairs_to_vector =========================
	@staticmethod
	def _pairs_to_vector(pairs, prefix: str) -> List[float]:
		"""Convertit une liste (nom, valeur) en vecteur ordonné.

		Le nom doit commencer par `prefix` suivi d'un indice entier, ex. 'x1', 'x2'.
		La sortie est ordonnée par indice croissant.
		"""
		if not isinstance(pairs, list):
			raise ValueError(f"{prefix}: attendu une liste de (nom,valeur).")

		values_by_index: Dict[int, float] = {}
		for item in pairs:
			if not (isinstance(item, tuple) and len(item) == 2):
				raise ValueError(f"{prefix}: élément invalide. Attendu (nom,valeur).")
			name, value = item
			if not isinstance(name, str) or not name.startswith(prefix):
				raise ValueError(f"{prefix}: nom invalide '{name}'.")
			index_str = name[len(prefix) :]
			if not index_str.isdigit():
				raise ValueError(f"{prefix}: nom invalide '{name}' (indice manquant).")
			idx = int(index_str)
			values_by_index[idx] = float(value)

		if len(values_by_index) == 0:
			raise ValueError(f"{prefix}: liste vide.")

		out: List[float] = []
		for idx in range(1, max(values_by_index.keys()) + 1):
			if idx not in values_by_index:
				raise ValueError(f"{prefix}: indice manquant {prefix}{idx}.")
			out.append(values_by_index[idx])
		return out

	# ==================== _struct_to_matrices =========================
	def _struct_to_matrices(self):
		"""Reconstruit les objets numériques depuis `struct_reso`.

		Retour
			X : vecteur d'entrée [x1..x_n]
			w_list : liste de matrices W_k (list[list[float]])
			b_list : liste de vecteurs b_k
			d : vecteur des sorties désirées [d1..d_m]

		Cette fonction valide aussi la présence des poids/biais attendus (noms).
		"""
		kx, Xn = self._get_by_prefix(self.struct_reso, "X")
		kw, Wn_c = self._get_by_prefix(self.struct_reso, "W")
		kb, Bn_c = self._get_by_prefix(self.struct_reso, "B")
		kd, Dn_s = self._get_by_prefix(self.struct_reso, "D")

		if kx is None or kw is None or kb is None or kd is None:
			raise ValueError("struct_reso invalide: attendu X*, W*, B*, D*.")

		if not isinstance(Wn_c, list) or not isinstance(Bn_c, list):
			raise ValueError("struct_reso invalide: W* et B* doivent être des listes de couches.")

		X = self._pairs_to_vector(Xn, "x")
		d = self._pairs_to_vector(Dn_s, "d")

		w_list: List[List[List[float]]] = []
		b_list: List[List[float]] = []

		if len(Wn_c) != len(Bn_c):
			raise ValueError(
				f"struct_reso invalide: nombre de couches W ({len(Wn_c)}) != B ({len(Bn_c)})."
			)

		for layer_index, (Wc_pairs, Bc_pairs) in enumerate(zip(Wn_c, Bn_c), start=1):
			if not isinstance(Wc_pairs, list) or not isinstance(Bc_pairs, list):
				raise ValueError(f"Couche {layer_index}: W/B doivent être des listes de (nom,valeur).")

			n_out = len(Bc_pairs)
			if n_out == 0:
				raise ValueError(f"Couche {layer_index}: B ne doit pas être vide.")
			if len(Wc_pairs) % n_out != 0:
				raise ValueError(
					f"Couche {layer_index}: nb poids ({len(Wc_pairs)}) non divisible par nb biais ({n_out})."
				)
			n_in = len(Wc_pairs) // n_out
			if n_in == 0:
				raise ValueError(f"Couche {layer_index}: dimensions invalides (n_in=0).")

			w_by_name = {name: float(val) for (name, val) in Wc_pairs}
			b_by_name = {name: float(val) for (name, val) in Bc_pairs}

			W: List[List[float]] = []
			for i in range(1, n_in + 1):
				row: List[float] = []
				for j in range(1, n_out + 1):
					key = f"w{i}{j}_{layer_index}"
					if key not in w_by_name:
						raise ValueError(f"Couche {layer_index}: poids manquant '{key}'.")
					row.append(w_by_name[key])
				W.append(row)

			b: List[float] = []
			for j in range(1, n_out + 1):
				key = f"b{j}_{layer_index}"
				if key not in b_by_name:
					raise ValueError(f"Couche {layer_index}: biais manquant '{key}'.")
				b.append(b_by_name[key])

			w_list.append(W)
			b_list.append(b)

		return X, w_list, b_list, d

	# ==================== _matrices_to_struct =========================
	def _matrices_to_struct(self, new_w_list: List[List[List[float]]], new_b_list: List[List[float]]) -> None:
		"""Réécrit W/B dans `struct_reso`.

		Les paires (nom, valeur) sont reconstruites selon la convention de noms
		(wij_k, bj_k) en conservant la clé globale existante ('W*', 'B*').
		"""
		kw, _ = self._get_by_prefix(self.struct_reso, "W")
		kb, _ = self._get_by_prefix(self.struct_reso, "B")
		if kw is None or kb is None:
			raise ValueError("struct_reso invalide: clés W*/B* introuvables.")
		if len(new_w_list) != len(new_b_list):
			raise ValueError("new_w_list et new_b_list doivent avoir la même longueur.")

		W_layers = []
		B_layers = []
		for layer_index, (W, b) in enumerate(zip(new_w_list, new_b_list), start=1):
			if len(W) == 0 or len(W[0]) == 0:
				raise ValueError(f"Couche {layer_index}: matrice W invalide.")
			n_in = len(W)
			n_out = len(W[0])
			if len(b) != n_out:
				raise ValueError(f"Couche {layer_index}: b a {len(b)} éléments, attendu {n_out}.")

			Wc_pairs: List[Tuple[str, float]] = []
			for i in range(1, n_in + 1):
				if len(W[i - 1]) != n_out:
					raise ValueError(f"Couche {layer_index}: lignes de W de tailles différentes.")
				for j in range(1, n_out + 1):
					Wc_pairs.append((f"w{i}{j}_{layer_index}", float(W[i - 1][j - 1])))
			W_layers.append(Wc_pairs)

			Bc_pairs: List[Tuple[str, float]] = []
			for j in range(1, n_out + 1):
				Bc_pairs.append((f"b{j}_{layer_index}", float(b[j - 1])))
			B_layers.append(Bc_pairs)

		self.struct_reso[kw] = W_layers
		self.struct_reso[kb] = B_layers

	# ==================== forward =========================
	def forward(self, n_fct: int | None = None):
		"""Résout le réseau (propagation avant) à partir de `struct_reso`.

		Retourne y (vecteur de sortie final).
		"""
		X, w_list, b_list, _ = self._struct_to_matrices()
		use_n_fct = self.n_fct if n_fct is None else int(n_fct)
		y, _cache = self._core.forward(X, w_list, b_list, use_n_fct)
		return y

	def solve(self, n_fct: int | None = None):
		"""Alias de `forward()` (résolution par propagation avant)."""
		return self.forward(n_fct=n_fct)

	def forward_layer(self, x, W, b, n_fct: int | None = None):
		"""Propagation avant d'une couche (wrapper).

		Utilisé surtout pour des tests ou pour expliquer le calcul couche par couche.
		"""
		use_n_fct = self.n_fct if n_fct is None else int(n_fct)
		return self._core.forward_layer(x, W, b, use_n_fct)

	@staticmethod
	def Delta(d_list, Fi_list, Fp_list):
		"""Wrapper de NeuroneMat.Delta (couche de sortie)."""
		return NeuroneMat.Delta(d_list, Fi_list, Fp_list)

	@staticmethod
	def Delta_cache(delta_next, W_next, Fp_current):
		"""Wrapper de NeuroneMat.Delta_cache (couche cachée)."""
		return NeuroneMat.Delta_cache(delta_next, W_next, Fp_current)

	@staticmethod
	def Correcteur(eta, a_prev, delta_current):
		"""Wrapper de NeuroneMat.Correcteur (corrections delta_W/delta_b)."""
		return NeuroneMat.Correcteur(eta, a_prev, delta_current)

	@staticmethod
	def maj(W, delta_W, b, delta_b):
		"""Wrapper de NeuroneMat.maj (mise à jour W/b)."""
		return NeuroneMat.maj(W, delta_W, b, delta_b)

	def backprop_update(self, eta: float | None = None, n_fct: int | None = None):
		"""Une itération complète (forward + backprop + maj) sur `struct_reso`.

		Effet de bord
			Met à jour les poids/biais dans `struct_reso`.
		"""
		X, w_list, b_list, d_list = self._struct_to_matrices()
		use_eta = self.eta if eta is None else float(eta)
		use_n_fct = self.n_fct if n_fct is None else int(n_fct)
		new_w_list, new_b_list, y = self._core.backprop_update(X, w_list, b_list, d_list, use_eta, use_n_fct)
		self._matrices_to_struct(new_w_list, new_b_list)
		return y

	@staticmethod
	def _nom_fct(n_fct: int) -> str:
		"""Retourne le nom lisible de la fonction d'activation (1..4)."""
		return {1: "sigmoïde", 2: "tan", 3: "tanh", 4: "gelu"}.get(n_fct, f"n_fct={n_fct}")

	@staticmethod
	def _fmt(x: float, precision: int = 6) -> str:
		"""Format numérique lisible (sans zéros inutiles)."""
		try:
			xf = float(x)
		except Exception:
			return str(x)
		if abs(xf) < 1e-15:
			xf = 0.0
		if abs(xf - round(xf)) < 1e-12:
			return str(int(round(xf)))
		return format(xf, f".{int(precision)}g")

	@classmethod
	def _fmt_vec(cls, v: List[float], precision: int = 6) -> str:
		"""Formate un vecteur sous la forme: [v1, v2, ...]."""
		return "[" + ", ".join(cls._fmt(x, precision) for x in v) + "]"

	@classmethod
	def _fmt_mat_no_header(
		cls,
		M: List[List[float]],
		precision: int = 6,
		row_labels: List[str] | None = None,
	) -> str:
		"""Formatte une matrice sans l'en-tête de colonnes (c1, c2, ...)."""
		if not M:
			return "[]"
		n_rows = len(M)
		n_cols = len(M[0]) if M else 0

		cell = [[cls._fmt(M[r][c], precision) for c in range(n_cols)] for r in range(n_rows)]
		row_labels_local = row_labels if row_labels is not None else [f"r{r + 1}" for r in range(n_rows)]

		widths = [0 for _ in range(n_cols)]
		for c in range(n_cols):
			widths[c] = max((len(cell[r][c]) for r in range(n_rows)), default=0)

		row_label_w = max((len(lbl) for lbl in row_labels_local), default=0)
		lines: List[str] = []
		for r in range(n_rows):
			label = row_labels_local[r].ljust(row_label_w)
			vals = "  ".join(cell[r][c].rjust(widths[c]) for c in range(n_cols))
			lines.append(f"{label}: {vals}")
		return "[\n" + "\n".join("  " + ln for ln in lines) + "\n]"

	@staticmethod
	def _print_step(title: str) -> None:
		"""Affiche un séparateur de section du style: --- Titre ---"""
		print(f"\n--- {title} ---")

	def resolution_affiche(
		self,
		eta: float | None = None,
		n_fct: int | None = None,
		update_struct: bool = True,
		log_console: str | None = None,
		show_params: bool = True,
		params_style: str = "matrix",
		show_equations: bool = True,
		show_resume: bool = False,
		show_struct_reso: bool | None = None,
		struct_reso_after_resume: bool | None = None,
		only_sections: bool = False,
		precision: int = 6,
	) -> Dict[str, object]:
		"""Affiche les étapes (forward/backprop/maj) à partir de `struct_reso`.

		Cette méthode sert d'outil pédagogique : elle détaille les calculs
		intermédiaires (z, a, fp), les deltas (sortie + couches cachées), puis les
		correcteurs et les mises à jour.

		Paramètres principaux
			log_console : niveau d'affichage centralisé (optionnel)
				- 'minimal'  : n'affiche que Paramètres / Résumé / struct_reso
				- 'détaillé' : affiche aussi Forward/Deltas/Correcteurs/Mises à jour
				Si fourni, ce paramètre pilote/écrase les flags show_* ci-dessous.
			show_params : affiche les paramètres (X, W, b, d) au début
			params_style : 'assign' (affectations) ou 'matrix' (blocs matriciels)
			show_equations : affiche les équations détaillées (sinon mode compact)
			show_resume : affiche un résumé final (sections [1]..[5])
			update_struct : si True, écrit les nouveaux W/b dans `self.struct_reso`

		Affichage de `struct_reso`
			- `show_struct_reso` :
				- None (défaut) : suit `show_resume` (si résumé, on montre aussi struct_reso)
				- True/False : force l'affichage
			- `struct_reso_after_resume` :
				- None (défaut) : dépend de `show_params`
					- show_params=False (option 1 / test unitaire) => struct_reso avant Résumé
					- show_params=True (autres options) => Paramètres -> Résumé -> struct_reso
				- True/False : force l'ordre

		Le style d'impression de `struct_reso` est identique à l'affichage des
		paramètres, car il réutilise `reseau.mon_reso.Affiche_reso`.

		Mode "sections seulement"
			Si `only_sections=True`, on n'affiche que les trois sections exigées :
				1) Paramètres
				2) Résumé
				3) struct_reso
			Le calcul complet est tout de même effectué (pour alimenter le résumé),
			mais les sections intermédiaires (Forward, Sortie, Deltas, Correcteurs,
			Mises à jour) ne sont pas imprimées.
		"""

		# Politique centralisée (optionnelle): si fournie, elle pilote les flags.
		if log_console is not None:
			policy = self._console_policy(log_console)
			show_params = policy.show_params
			params_style = policy.params_style
			show_equations = policy.show_equations
			show_resume = policy.show_resume
			only_sections = policy.only_sections

		res = self._run_from_struct(eta=eta, n_fct=n_fct, update_struct=update_struct, precision=precision)
		X = res.X
		w_list = res.w_list
		b_list = res.b_list
		d_list = res.d_list
		use_eta = res.eta
		use_n_fct = res.n_fct
		nom_fct = res.nom_fct
		y = res.y
		z_list = res.z_list
		a_list = res.a_list
		fp_list = res.fp_list
		delta_list = res.delta_list
		err_list = res.err_list
		dW_list = res.dW_list
		db_list = res.db_list
		new_w_list = res.new_w_list
		new_b_list = res.new_b_list

		def _fmt_mul_terms(values: List[float], weights: List[float], precision_local: int) -> str:
			terms = [f"{self._fmt(values[i], precision_local)}*{self._fmt(weights[i], precision_local)}" for i in range(len(values))]
			return " + ".join(terms)

		def _inp_name(layer_index: int, i: int) -> str:
			if layer_index == 1:
				return f"x{i}"
			return f"a{i}_{layer_index - 1}"

		if show_params:
			self._print_step(f"Paramètres (n_fct={use_n_fct} / {nom_fct}, eta={use_eta})")
			if str(params_style).strip().lower() == "assign":
				# Style affectations: x1=..., puis blocs "# couche k".
				for i, xi in enumerate(X, start=1):
					print(f"x{i} = {self._fmt(xi, precision)}")
				for layer_index, (W, b) in enumerate(zip(w_list, b_list), start=1):
					n_in = len(W)
					n_out = len(W[0]) if W else 0
					print(f"\n# couche {layer_index}")
					# Poids
					if n_out == 1:
						w_names = [f"w{i}1_{layer_index}" for i in range(1, n_in + 1)]
						w_vals = [W[i - 1][0] for i in range(1, n_in + 1)]
						print(f"\t{', '.join(w_names)} = {', '.join(self._fmt(v, precision) for v in w_vals)}")
					else:
						for i in range(1, n_in + 1):
							w_names = [f"w{i}{j}_{layer_index}" for j in range(1, n_out + 1)]
							w_vals = [W[i - 1][j - 1] for j in range(1, n_out + 1)]
							print(f"\t{', '.join(w_names)} = {', '.join(self._fmt(v, precision) for v in w_vals)}")
					# Biais
					for j in range(1, n_out + 1):
						print(f"\tb{j}_{layer_index} = {self._fmt(b[j - 1], precision)}")
				for i, di in enumerate(d_list, start=1):
					print(f"d{i} = {self._fmt(di, precision)}")
			else:
				# Style matriciel existant.
				print(f"X  = {self._fmt_vec(X, precision)}")
				for layer_index, (W, b) in enumerate(zip(w_list, b_list), start=1):
					n_in = len(W)
					n_out = len(W[0]) if W else 0
					w_row_labels = [
						", ".join(f"w{i}{j}_{layer_index}" for j in range(1, n_out + 1))
						for i in range(1, n_in + 1)
					]
					print(
						f"W{layer_index} (w_ij_{layer_index}) = {self._fmt_mat_no_header(W, precision, row_labels=w_row_labels)}"
					)
					print(f"b{layer_index} (b_j_{layer_index}) = {self._fmt_vec(b, precision)}")
				print(f"d  = {self._fmt_vec(d_list, precision)}")

		if not only_sections:
			for layer_index, (W, b, z, a, fp) in enumerate(zip(w_list, b_list, z_list, a_list, fp_list), start=1):
				inp_name = "X" if layer_index == 1 else f"a{layer_index - 1}"
				self._print_step(f"Forward couche {layer_index}")
				if show_equations:
					print(f"i{layer_index} = {inp_name}@W{layer_index} + b{layer_index}")
				else:
					print(
						f"i{layer_index} = {inp_name}@W{layer_index} + b{layer_index} = {self._fmt_vec(z, precision)}"
					)
				if show_equations:
					# Détail scalaire (même ordre que la sortie attendue): i -> a -> fp.
					a_prev_vec = X if layer_index == 1 else a_list[layer_index - 2]
					for j in range(1, len(a) + 1):
						w_in = [W[i - 1][j - 1] for i in range(1, len(a_prev_vec) + 1)]
						expr_left = " + ".join(
							f"{_inp_name(layer_index, i)}*w{i}{j}_{layer_index}" for i in range(1, len(a_prev_vec) + 1)
						)
						expr_vals = _fmt_mul_terms(list(a_prev_vec), w_in, precision)
						print(
							f"  i{j}_{layer_index} = {expr_left} + b{j}_{layer_index} = {expr_vals} + {self._fmt(b[j - 1], precision)} = {self._fmt(z[j - 1], precision)}"
						)
				if show_equations:
					print(f"a{layer_index} = Fi(i{layer_index})")
				else:
					print(f"a{layer_index} = Fi(i{layer_index}) = {self._fmt_vec(a, precision)}")
				if show_equations:
					for j in range(1, len(a) + 1):
						print(f"  a{j}_{layer_index} = Fi(i{j}_{layer_index}) = {self._fmt(a[j - 1], precision)}")
				if show_equations:
					print(f"fp{layer_index} = Fp(i{layer_index})")
				else:
					print(f"fp{layer_index} = Fp(i{layer_index}) = {self._fmt_vec(fp, precision)}")
				if show_equations:
					for j in range(1, len(fp) + 1):
						print(f"  fp{j}_{layer_index} = Fp(i{j}_{layer_index}) = {self._fmt(fp[j - 1], precision)}")

			self._print_step("Sortie")
			L = len(a_list)
			if len(y) == 1:
				print(f"y: a1_{L} = [{self._fmt(y[0], precision)}]")
			else:
				print("y:")
				for j in range(len(y)):
					print(f"  a{j + 1}_{L} = [{self._fmt(y[j], precision)}]")

			self._print_step("Deltas")
			L = len(delta_list)
			if show_equations:
				print(f"delta{L} = (d - y) ⊙ fp{L}")
			else:
				print(f"delta{L} = (d - y) ⊙ fp{L} = {self._fmt_vec(delta_list[-1], precision)}")
			if show_equations:
				for j in range(1, len(delta_list[-1]) + 1):
					lhs = f"delta{j}_{L}"
					diff = float(d_list[j - 1]) - float(y[j - 1])
					print(
						f"  {lhs} = (d{j} - y{j})*fp{j}_{L} = ({self._fmt(d_list[j - 1], precision)} - {self._fmt(y[j - 1], precision)})*{self._fmt(fp_list[-1][j - 1], precision)} = {self._fmt(diff, precision)}*{self._fmt(fp_list[-1][j - 1], precision)} = {self._fmt(delta_list[-1][j - 1], precision)}"
					)
			for layer_index in range(L - 1, 0, -1):
				W_next = w_list[layer_index]
				delta_next = delta_list[layer_index]
				err_prev = err_list[layer_index - 1] or []
				if show_equations:
					print(f"err{layer_index} = W{layer_index + 1} @ delta{layer_index + 1}")
				else:
					print(
						f"err{layer_index} = W{layer_index + 1} @ delta{layer_index + 1} = {self._fmt_vec(err_prev, precision)}"
					)
				if show_equations:
					# err_k_layer = Σ_j w_{k,j}_{next} * delta_j_{nextLayer}
					n_in = len(W_next)
					n_out = len(W_next[0]) if W_next else 0
					for k in range(1, n_in + 1):
						terms_left = " + ".join(
							f"w{k}{j}_{layer_index + 1}*delta{j}_{layer_index + 1}" for j in range(1, n_out + 1)
						)
						terms_vals = " + ".join(
							f"{self._fmt(W_next[k - 1][j - 1], precision)}*{self._fmt(delta_next[j - 1], precision)}"
							for j in range(1, n_out + 1)
						)
						print(
							f"  err{k}_{layer_index} = {terms_left} = {terms_vals} = {self._fmt(err_prev[k - 1], precision)}"
						)
				if show_equations:
					print(f"delta{layer_index} = err{layer_index} ⊙ fp{layer_index}")
				else:
					print(
						f"delta{layer_index} = err{layer_index} ⊙ fp{layer_index} = {self._fmt_vec(delta_list[layer_index - 1], precision)}"
					)
				if show_equations:
					for k in range(1, len(delta_list[layer_index - 1]) + 1):
						print(
							f"  delta{k}_{layer_index} = err{k}_{layer_index}*fp{k}_{layer_index} = {self._fmt(err_prev[k - 1], precision)}*{self._fmt(fp_list[layer_index - 1][k - 1], precision)} = {self._fmt(delta_list[layer_index - 1][k - 1], precision)}"
						)

			self._print_step("Correcteurs")
			print(f"eta = {self._fmt(use_eta, precision)}")
			for layer_index, (dW, db) in enumerate(zip(dW_list, db_list), start=1):
				n_in = len(dW)
				n_out = len(dW[0]) if dW else 0
				w_row_labels = [
					", ".join(f"w{i}{j}_{layer_index}" for j in range(1, n_out + 1))
					for i in range(1, n_in + 1)
				]
				outer_left = "X" if layer_index == 1 else f"a{layer_index - 1}"
				if show_equations:
					print(f"dW{layer_index} = eta * outer({outer_left}, delta{layer_index})")
					print(f"db{layer_index} = eta * delta{layer_index}")
				else:
					print(
						f"dW{layer_index} = eta * outer({outer_left}, delta{layer_index}) = {self._fmt_mat_no_header(dW, precision, row_labels=w_row_labels)}"
					)
					print(f"db{layer_index} = eta * delta{layer_index} = {self._fmt_vec(db, precision)}")
				if show_equations:
					# Détail des correcteurs élémentaires
					a_prev_vec = X if layer_index == 1 else a_list[layer_index - 2]
					delta_vec = delta_list[layer_index - 1]
					n_in_local = len(a_prev_vec)
					n_out_local = len(delta_vec)
					for i in range(1, n_in_local + 1):
						for j in range(1, n_out_local + 1):
							print(
								f"  dW{layer_index}_{i},{j} = eta*{_inp_name(layer_index, i)}*delta{j}_{layer_index} = {self._fmt(use_eta, precision)}*{self._fmt(a_prev_vec[i - 1], precision)}*{self._fmt(delta_vec[j - 1], precision)} = {self._fmt(dW[i - 1][j - 1], precision)}"
							)
					for j in range(1, n_out_local + 1):
						print(
							f"  db{j}_{layer_index} = eta*delta{j}_{layer_index} = {self._fmt(use_eta, precision)}*{self._fmt(delta_vec[j - 1], precision)} = {self._fmt(db[j - 1], precision)}"
						)

			self._print_step("Mises à jour")
			prec_up = max(int(precision), 12)
			for layer_index, (new_W, new_b) in enumerate(zip(new_w_list, new_b_list), start=1):
				n_in = len(new_W)
				n_out = len(new_W[0]) if new_W else 0
				w_row_labels = [
					", ".join(f"w{i}{j}_{layer_index}" for j in range(1, n_out + 1))
					for i in range(1, n_in + 1)
				]
				if show_equations:
					print(f"W{layer_index}_new = W{layer_index} + dW{layer_index}")
					print(f"b{layer_index}_new = b{layer_index} + db{layer_index}")
				else:
					print(
						f"W{layer_index}_new = W{layer_index} + dW{layer_index} = {self._fmt_mat_no_header(new_W, prec_up, row_labels=w_row_labels)}"
					)
					print(f"b{layer_index}_new = b{layer_index} + db{layer_index} = {self._fmt_vec(new_b, prec_up)}")
				if show_equations:
					old_W = w_list[layer_index - 1]
					old_b = b_list[layer_index - 1]
					layer_dW = dW_list[layer_index - 1]
					layer_db = db_list[layer_index - 1]
					n_in_local = len(new_W)
					n_out_local = len(new_W[0]) if new_W else 0
					for i in range(1, n_in_local + 1):
						for j in range(1, n_out_local + 1):
							print(
								f"  W{layer_index}_new[{i},{j}] (w{i}{j}_{layer_index}) = {self._fmt(old_W[i - 1][j - 1], prec_up)} + {self._fmt(layer_dW[i - 1][j - 1], prec_up)} = {self._fmt(new_W[i - 1][j - 1], prec_up)}"
							)
					for j in range(1, len(new_b) + 1):
						print(
							f"  b{layer_index}_new[{j}] (b{j}_{layer_index}) = {self._fmt(old_b[j - 1], prec_up)} + {self._fmt(layer_db[j - 1], prec_up)} = {self._fmt(new_b[j - 1], prec_up)}"
						)

		prec_up = res.prec_up

		# Par défaut:
		# - si show_resume=True, on affiche aussi struct_reso.
		# - l'option 1 (test unitaire) appelle show_params=False, donc struct_reso avant Résumé.
		# - les autres options (typiquement show_params=True) veulent paramètres -> résumé -> struct_reso.
		want_struct_reso = bool(show_resume) if show_struct_reso is None else bool(show_struct_reso)
		after_resume = bool(show_params) if struct_reso_after_resume is None else bool(struct_reso_after_resume)

		def _print_struct_reso() -> None:
			self._print_step("struct_reso")
			# Même style d'affichage qu'au début (Affiche_reso).
			n_in_local = len(X)
			n_layers_local = len(w_list)
			n_s_local = 1 if n_layers_local > 0 else 0
			n_c_local = max(n_layers_local - n_s_local, 0)
			N_b_local = [len(b) for b in b_list]
			_tmp_reso = mon_reso(
				n_in=n_in_local,
				n_c=n_c_local,
				n_s=n_s_local,
				N_b=N_b_local,
				biais=[0.0, 0.0],
				poids=[0.0, 0.0],
				X=[0.0, 1.0],
				D=[0, 1],
			)
			_tmp_reso.Affiche_reso(self.struct_reso, precision=int(precision), n_fct=use_n_fct, eta=use_eta)

		if show_resume:
			if want_struct_reso and not after_resume:
				_print_struct_reso()
			self._print_step("Résumé")
			if show_equations:
				print("\n[1] Activations")
				for layer_index, z in enumerate(z_list, start=1):
					for j in range(1, len(z) + 1):
						print(f"i{j}_{layer_index} = {self._fmt(z[j - 1], precision)}")
					a = a_list[layer_index - 1]
					for j in range(1, len(a) + 1):
						print(f"a{j}_{layer_index} = {self._fmt(a[j - 1], precision)}")

				print("\n[2] Fonctions d'activation + dérivées")
				print(f"f = {nom_fct}")
				for layer_index, (a, fp) in enumerate(zip(a_list, fp_list), start=1):
					for j in range(1, len(a) + 1):
						print(f"Fi(i{j}_{layer_index}) = a{j}_{layer_index} = {self._fmt(a[j - 1], precision)}")
						print(f"Fp(i{j}_{layer_index}) = fp{j}_{layer_index} = {self._fmt(fp[j - 1], precision)}")

				print("\n[3] Signaux d'erreur")
				for layer_index in range(1, len(delta_list) + 1):
					err = err_list[layer_index - 1]
					if err is not None:
						for j in range(1, len(err) + 1):
							print(f"err{j}_{layer_index} = {self._fmt(err[j - 1], precision)}")
					delta_vec = delta_list[layer_index - 1]
					for j in range(1, len(delta_vec) + 1):
						print(f"delta{j}_{layer_index} = {self._fmt(delta_vec[j - 1], precision)}")

				print("\n[4] Facteurs de correction")
				print(f"eta = {self._fmt(use_eta, precision)}")
				for layer_index, (dW, db) in enumerate(zip(dW_list, db_list), start=1):
					n_in_local = len(dW)
					n_out_local = len(dW[0]) if dW else 0
					for i in range(1, n_in_local + 1):
						for j in range(1, n_out_local + 1):
							print(f"dW{layer_index}_{i},{j} = {self._fmt(dW[i - 1][j - 1], precision)}")
					for j in range(1, len(db) + 1):
						print(f"db{j}_{layer_index} = {self._fmt(db[j - 1], precision)}")

				print("\n[5] Mises à jour")
				for layer_index, (new_W, new_b) in enumerate(zip(new_w_list, new_b_list), start=1):
					n_in_local = len(new_W)
					n_out_local = len(new_W[0]) if new_W else 0
					for i in range(1, n_in_local + 1):
						for j in range(1, n_out_local + 1):
							print(f"W{layer_index}_new[{i},{j}] (w{i}{j}_{layer_index}) = {self._fmt(new_W[i - 1][j - 1], prec_up)}")
					for j in range(1, len(new_b) + 1):
						print(f"b{layer_index}_new[{j}] (b{j}_{layer_index}) = {self._fmt(new_b[j - 1], prec_up)}")
			else:
				# Fallback compact: vecteurs/matrices.
				print("\n[1] Activations")
				for layer_index, z in enumerate(z_list, start=1):
					print(f"i{layer_index} = {self._fmt_vec(z, precision)}")
					print(f"a{layer_index} = {self._fmt_vec(a_list[layer_index - 1], precision)}")
				print("\n[2] Fonctions d'activation + dérivées")
				print(f"f = {nom_fct}")
				for layer_index in range(1, len(z_list) + 1):
					print(f"Fi(i{layer_index}) = a{layer_index} = {self._fmt_vec(a_list[layer_index - 1], precision)}")
					print(f"Fp(i{layer_index}) = fp{layer_index} = {self._fmt_vec(fp_list[layer_index - 1], precision)}")
				print("\n[3] Signaux d'erreur")
				for layer_index in range(1, len(err_list) + 1):
					err = err_list[layer_index - 1]
					if err is not None:
						print(f"err{layer_index} = {self._fmt_vec(err, precision)}")
					print(f"delta{layer_index} = {self._fmt_vec(delta_list[layer_index - 1], precision)}")
				print("\n[4] Facteurs de correction")
				print(f"eta = {self._fmt(use_eta, precision)}")
				for layer_index, (dW, db) in enumerate(zip(dW_list, db_list), start=1):
					n_in = len(dW)
					n_out = len(dW[0]) if dW else 0
					w_row_labels = [
						", ".join(f"w{i}{j}_{layer_index}" for j in range(1, n_out + 1))
						for i in range(1, n_in + 1)
					]
					print(f"dW{layer_index} = {self._fmt_mat_no_header(dW, precision, row_labels=w_row_labels)}")
					print(f"db{layer_index} = {self._fmt_vec(db, precision)}")
				print("\n[5] Mises à jour")
				for layer_index, (new_W, new_b) in enumerate(zip(new_w_list, new_b_list), start=1):
					n_in = len(new_W)
					n_out = len(new_W[0]) if new_W else 0
					w_row_labels = [
						", ".join(f"w{i}{j}_{layer_index}" for j in range(1, n_out + 1))
						for i in range(1, n_in + 1)
					]
					print(f"W{layer_index}_new = {self._fmt_mat_no_header(new_W, prec_up, row_labels=w_row_labels)}")
					print(f"b{layer_index}_new = {self._fmt_vec(new_b, prec_up)}")
			if want_struct_reso and after_resume:
				_print_struct_reso()
		elif want_struct_reso:
			_print_struct_reso()

		return {
			"X": X,
			"w_list": w_list,
			"b_list": b_list,
			"d": d_list,
			"eta": use_eta,
			"n_fct": use_n_fct,
			"y": y,
			"z_list": z_list,
			"a_list": a_list,
			"fp_list": fp_list,
			"delta_list": delta_list,
			"dW_list": dW_list,
			"db_list": db_list,
			"new_w_list": new_w_list,
			"new_b_list": new_b_list,
		}


