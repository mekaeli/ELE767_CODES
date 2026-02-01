"""reso2neurone

Module pour générer, afficher et exploiter une structure de réseau de neurones.

Contrainte: ce module est *orienté classe*. Il expose uniquement des classes
(pas de variables globales ni de fonctions module-level).

La méthode `mon_reso.cree_reso()` retourne un dictionnaire `struct_reso` avec des
clés basées sur les valeurs:
	- `X{n_in}` pour les entrées
	- `W{n_c}` pour les poids (liste de couches)
	- `B{n_c}` pour les biais (liste de couches)
	- `D{n_s}` pour les sorties (n_s = nb neurones couche sortie)

Chaque variable élémentaire est un couple (nom, valeur).

Classes:
	- mon_reso : crée/affiche/met à jour `struct_reso`
	- backpp : résout (propagation avant) et peut mettre à jour les poids/biais
								(une itération) à partir de `struct_reso` en s'appuyant sur
								l'approche matricielle.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from neurone.fct_dactivation import (
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
	"""Cache de la propagation avant (utile pour la rétropropagation)."""

	z_list: List[List[float]]
	a_list: List[List[float]]
	fp_list: List[List[float]]


class NeuroneMat:
	"""Neurone + Backpropagation en approche matricielle.

	Note:
		Cette classe était auparavant dans neurone/neurone_mat.py.
		Elle est intégrée ici pour garder le projet minimal (uniquement des classes)
		et permettre la suppression du fichier externe.
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
		"""Vérifie la cohérence des dimensions (X, W, b, d) avant le calcul."""
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
		"""Calcule le produit matriciel simplifié: x @ W (x vecteur ligne)."""
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
		"""Calcule err_prev = W @ delta_next (utile pour remonter l'erreur)."""
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
		"""Retourne (Fi, Fp) pour un scalaire i selon n_fct (1..4)."""
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
		"""Fait la propagation avant d'une couche: z, a, fp."""
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
		"""Fait la propagation avant complète et retourne (y, cache)."""
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
		"""Calcule le delta de la couche de sortie: (d - y) ⊙ fp."""
		if len(d_list) != len(Fi_list) or len(Fi_list) != len(Fp_list):
			raise ValueError("d_list, Fi_list et Fp_list doivent avoir la même taille.")
		return [(float(d) - float(Fi)) * float(Fp) for d, Fi, Fp in zip(d_list, Fi_list, Fp_list)]

	# ==================== Delta_cache =========================
	@staticmethod
	def Delta_cache(delta_next: Sequence[float], W_next: List[List[float]], Fp_current: Sequence[float]) -> List[float]:
		"""Calcule le delta d'une couche cachée à partir de la couche suivante."""
		err_prev = NeuroneMat._transpose_matvec(W_next, delta_next)
		return NeuroneMat._vec_mul(err_prev, Fp_current)

	# ==================== Correcteur =========================
	@staticmethod
	def Correcteur(
		eta: float,
		a_prev: Sequence[float],
		delta_current: Sequence[float],
	) -> Tuple[List[List[float]], List[float]]:
		"""Calcule les corrections delta_W et delta_b pour une couche."""
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
		"""Applique la mise à jour: W <- W + delta_W, b <- b + delta_b."""
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
		"""Fait une itération complète (forward + backprop + mise à jour)."""
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


# ==================== Classe mon_reso ====================


class mon_reso:
	"""Réseau (structure) configurable.

	La classe s'initialise avec:
		n_in, n_c, N_b, biais, poids, X, D

	Les fonctions cree_reso, Affiche_reso, set_reso appartiennent à la classe.
	"""

	# ==================== __init__ =========================
	def __init__(self, n_in: int, n_c: int, N_b: List[int], biais, poids, X, D):
		"""Initialise la configuration du réseau (dimensions + bornes)."""
		self.n_in = int(n_in)
		self.n_c = int(n_c)
		self.N_b = list(N_b)
		self.biais = list(biais)
		self.poids = list(poids)
		self.X = list(X)
		self.D = list(D)
		self.struct_reso: Dict[str, object] | None = None

	# ==================== get_reso =========================
	def get_reso(self) -> Dict[str, object]:
		"""Expose la structure interne `struct_reso`.

		`struct_reso` est créée par `cree_reso()` et peut ensuite être modifiée par
		`set_reso()`.
		"""
		if self.struct_reso is None:
			raise ValueError(
				"Aucune structure disponible. Appelle d'abord cree_reso() pour initialiser struct_reso."
			)
		return self.struct_reso

	# ==================== cree_reso =========================
	def cree_reso(self) -> Dict[str, object]:
		"""Génère toutes les variables d'un réseau de neurones (struct_reso)."""

		n_in_local = self.n_in
		n_c_local = self.n_c
		N_b_local = self.N_b

		if n_in_local <= 0:
			raise ValueError("n_in doit être > 0")
		if n_c_local <= 0:
			raise ValueError("n_c doit être > 0")
		if len(N_b_local) != n_c_local:
			raise ValueError("N_b doit contenir exactement n_c valeurs")
		if any(nb <= 0 for nb in N_b_local):
			raise ValueError("Toutes les valeurs de N_b doivent être > 0")
		if len(self.X) == 0:
			raise ValueError("La liste X ne doit pas être vide")
		if len(self.D) == 0:
			raise ValueError("La liste D ne doit pas être vide")
		if len(self.poids) != 2:
			raise ValueError("poids doit être [min, max]")
		if len(self.biais) != 2:
			raise ValueError("biais doit être [min, max]")

		p_min, p_max = float(self.poids[0]), float(self.poids[1])
		b_min, b_max = float(self.biais[0]), float(self.biais[1])

		Xn: List[Tuple[str, int]] = [(f"x{i}", random.choice(self.X)) for i in range(1, n_in_local + 1)]

		Wn_c_local: List[List[Tuple[str, float]]] = []
		Bn_c_local: List[List[Tuple[str, float]]] = []

		for c in range(1, n_c_local + 1):
			n_src = n_in_local if c == 1 else N_b_local[c - 2]
			n_dst = N_b_local[c - 1]

			Wc: List[Tuple[str, float]] = []
			for i in range(1, n_src + 1):
				for j in range(1, n_dst + 1):
					name = f"w{i}{j}_{c}"
					value = random.uniform(p_min, p_max)
					Wc.append((name, value))
			Wn_c_local.append(Wc)

			Bc: List[Tuple[str, float]] = []
			for j in range(1, n_dst + 1):
				name = f"b{j}_{c}"
				value = random.uniform(b_min, b_max)
				Bc.append((name, value))
			Bn_c_local.append(Bc)

		n_s = N_b_local[-1]
		Dn_s_local: List[Tuple[str, int]] = [(f"d{i}", random.choice(self.D)) for i in range(1, n_s + 1)]

		key_x = f"X{n_in_local}"
		key_w = f"W{n_c_local}"
		key_b = f"B{n_c_local}"
		key_d = f"D{n_s}"
		self.struct_reso = {key_x: Xn, key_w: Wn_c_local, key_b: Bn_c_local, key_d: Dn_s_local}
		return self.struct_reso

	# ==================== Affiche_reso =========================
	def Affiche_reso(
		self,
		struct_reso: Dict[str, object],
		precision: int = 6,
		n_fct: int | None = None,
		eta: float | None = None,
	) -> None:
		"""Affichage des paramètres (même style que les blocs matriciels).

		Affiche:
			X, W1..Wc, b1..bc, d

		Si `n_fct` et `eta` sont fournis, ajoute un en-tête "--- Paramètres (...) ---".
		"""

		# ==================== _get_by_prefix =========================
		def _get_by_prefix(prefix: str):
			"""Retourne (clé, valeur) du premier champ dont la clé commence par prefix."""
			for k, v in struct_reso.items():
				if isinstance(k, str) and k.startswith(prefix):
					return k, v
			return None, []

		# ==================== _fmt =========================
		def _fmt(x: float, p: int = 6) -> str:
			"""Formate un nombre pour l'affichage (entiers propres, petits zéros, précision)."""
			try:
				xf = float(x)
			except Exception:
				return str(x)
			if abs(xf) < 1e-15:
				xf = 0.0
			if abs(xf - round(xf)) < 1e-12:
				return str(int(round(xf)))
			return format(xf, f".{int(p)}g")

		# ==================== _fmt_vec =========================
		def _fmt_vec(v: List[float], p: int = 6) -> str:
			"""Formate un vecteur sous la forme: [v1, v2, ...]."""
			return "[" + ", ".join(_fmt(x, p) for x in v) + "]"

		# ==================== _print_assign =========================
		def _print_assign(names: List[str], values: List[float], indent: str = "\t") -> None:
			"""Affiche des affectations du style: w11_1, w12_1 = 3, 4"""
			if not names:
				return
			left = ", ".join(names)
			right = ", ".join(_fmt(v, precision) for v in values)
			print(f"{indent}{left} = {right}")

		kx, Xn_local = _get_by_prefix("X")
		kw, Wn_c_local = _get_by_prefix("W")
		kb, Bn_c_local = _get_by_prefix("B")
		kd, Dn_s_local = _get_by_prefix("D")

		if n_fct is not None and eta is not None:
			nom_fct = {1: "sigmoïde", 2: "tan", 3: "tanh", 4: "gelu"}.get(int(n_fct), f"n_fct={n_fct}")
			print(f"\n--- Paramètres (n_fct={int(n_fct)} / {nom_fct}, eta={_fmt(float(eta), precision)}) ---")
		else:
			print("\n--- Paramètres ---")

		# Entrées (style affectations)
		x_names: List[str] = []
		x_vals: List[float] = []
		for i in range(1, int(self.n_in) + 1):
			name = f"x{i}"
			val = None
			for n, v in Xn_local:
				if n == name:
					val = v
					break
			if val is None:
				val = Xn_local[i - 1][1] if i - 1 < len(Xn_local) else 0
			x_names.append(name)
			x_vals.append(float(val))
		# Affichage 1 valeur par ligne (x1 = 1, x2 = 0, ...)
		for name, val in zip(x_names, x_vals):
			_print_assign([name], [val], indent="")

		# Couches (style affectations)
		for c in range(1, int(self.n_c) + 1):
			n_src = int(self.n_in) if c == 1 else int(self.N_b[c - 2])
			n_dst = int(self.N_b[c - 1])

			Wc_pairs = Wn_c_local[c - 1] if c - 1 < len(Wn_c_local) else []
			Bc_pairs = Bn_c_local[c - 1] if c - 1 < len(Bn_c_local) else []
			W_map = {name: float(val) for name, val in Wc_pairs}
			B_map = {name: float(val) for name, val in Bc_pairs}

			W: List[List[float]] = []
			for i in range(1, n_src + 1):
				row: List[float] = []
				for j in range(1, n_dst + 1):
					row.append(float(W_map.get(f"w{i}{j}_{c}", 0.0)))
				W.append(row)
			b: List[float] = [float(B_map.get(f"b{j}_{c}", 0.0)) for j in range(1, n_dst + 1)]

			print(f"\n# couche {c}")
			# Poids: même logique d'affichage que dans l'exemple demandé.
			# - si 1 seul neurone (n_dst == 1) : afficher tous les poids entrants sur une seule ligne
			# - sinon : afficher une ligne par entrée (i) avec tous les poids sortants
			if n_dst == 1:
				w_names = [f"w{i}1_{c}" for i in range(1, n_src + 1)]
				w_vals = [float(W_map.get(w_name, 0.0)) for w_name in w_names]
				_print_assign(w_names, w_vals)
			else:
				for i in range(1, n_src + 1):
					w_names = [f"w{i}{j}_{c}" for j in range(1, n_dst + 1)]
					w_vals = [float(W_map.get(w_name, 0.0)) for w_name in w_names]
					_print_assign(w_names, w_vals)

			# Biais: une ligne par biais (b1_c = ..., b2_c = ...)
			b_names = [f"b{j}_{c}" for j in range(1, n_dst + 1)]
			b_vals = [float(B_map.get(b_name, 0.0)) for b_name in b_names]
			for b_name, b_val in zip(b_names, b_vals):
				_print_assign([b_name], [b_val])

		# Sorties attendues (style affectations)
		d_names: List[str] = []
		d_vals: List[float] = []
		for i in range(1, int(self.N_b[-1]) + 1):
			name = f"d{i}"
			val = None
			for n, v in Dn_s_local:
				if n == name:
					val = v
					break
			if val is None:
				val = Dn_s_local[i - 1][1] if i - 1 < len(Dn_s_local) else 0
			d_names.append(name)
			d_vals.append(float(val))
		# Affichage 1 valeur par ligne (d1 = ..., d2 = ..., ...)
		for name, val in zip(d_names, d_vals):
			_print_assign([name], [val], indent="")

	# ==================== set_reso =========================
	def set_reso(self, struct_reso: Dict[str, object], Xn, Wn_c, Bn_c, Dn_s) -> Dict[str, object]:
		"""Écrit les valeurs reçues dans struct_reso (avec vérification de dimensions).

		Accepte:
			- Xn/Dn_s: soit liste de valeurs, soit liste de (nom,valeur)
			- Wn_c/Bn_c: soit liste de listes de valeurs, soit liste de listes de (nom,valeur)
		"""

		# ==================== _get_by_prefix =========================
		def _get_by_prefix(prefix: str):
			"""Retourne (clé, valeur) du premier champ dont la clé commence par prefix."""
			for k, v in struct_reso.items():
				if isinstance(k, str) and k.startswith(prefix):
					return k, v
			return None, None

		kx, old_Xn = _get_by_prefix("X")
		kw, old_Wn_c = _get_by_prefix("W")
		kb, old_Bn_c = _get_by_prefix("B")
		kd, old_Dn_s = _get_by_prefix("D")

		if kx is None or kw is None or kb is None or kd is None:
			raise ValueError(
				"struct_reso invalide: clés manquantes. Attendu des clés commençant par X, W, B, D."
			)

		# Validation de base de la structure existante
		if not isinstance(old_Xn, list):
			raise ValueError(f"struct_reso invalide: {kx} doit être une liste.")
		if not isinstance(old_Dn_s, list):
			raise ValueError(f"struct_reso invalide: {kd} doit être une liste.")
		if not isinstance(old_Wn_c, list) or any(not isinstance(layer, list) for layer in old_Wn_c):
			raise ValueError(f"struct_reso invalide: {kw} doit être une liste de listes (couches).")
		if not isinstance(old_Bn_c, list) or any(not isinstance(layer, list) for layer in old_Bn_c):
			raise ValueError(f"struct_reso invalide: {kb} doit être une liste de listes (couches).")

		# Helpers de conversion (valeurs -> paires en gardant les noms)
		# ==================== _is_pair =========================
		def _is_pair(x) -> bool:
			"""Vrai si x est un couple (nom, valeur)."""
			return isinstance(x, tuple) and len(x) == 2

		# ==================== _is_pair_list =========================
		def _is_pair_list(lst) -> bool:
			"""Vrai si lst est une liste de couples (nom, valeur) (ou vide)."""
			return isinstance(lst, list) and (len(lst) == 0 or _is_pair(lst[0]))

		# ==================== _ensure_list =========================
		def _ensure_list(obj, label: str) -> None:
			"""Valide que obj est une liste, sinon lève une erreur lisible."""
			if not isinstance(obj, list):
				raise ValueError(f"{label}: type invalide. Attendu une liste, reçu {type(obj).__name__}.")

		# ==================== _ensure_pair_list =========================
		def _ensure_pair_list(lst: list, label: str, expected_len: int | None = None) -> None:
			"""Valide que lst est une liste de (nom,valeur) et (optionnel) la longueur attendue."""
			for idx, item in enumerate(lst, start=1):
				if not _is_pair(item):
					raise ValueError(
						f"{label}: élément #{idx} invalide. Attendu un couple (nom,valeur)."
					)
			if expected_len is not None and len(lst) != expected_len:
				raise ValueError(
					f"{label}: dimension invalide. Attendu {expected_len} élément(s), reçu {len(lst)}."
				)

		# ==================== _to_pairs_1d =========================
		def _to_pairs_1d(old_pairs, new_data, label: str):
			"""Convertit un vecteur (1D) en liste de paires en conservant les noms d'origine."""
			_ensure_list(old_pairs, f"struct_reso.{label}")
			_ensure_list(new_data, label)

			if _is_pair_list(new_data):
				_ensure_pair_list(new_data, label, expected_len=len(old_pairs))
				return new_data

			if len(new_data) != len(old_pairs):
				raise ValueError(
					f"{label}: dimension invalide. Attendu {len(old_pairs)} valeur(s), reçu {len(new_data)}."
				)
			return [(name, new_data[i]) for i, (name, _) in enumerate(old_pairs)]

		# ==================== _to_pairs_2d =========================
		def _to_pairs_2d(old_layers, new_layers, label: str):
			"""Convertit des couches (2D) en paires en conservant les noms d'origine."""
			_ensure_list(old_layers, f"struct_reso.{label}")
			_ensure_list(new_layers, label)
			if len(new_layers) != len(old_layers):
				raise ValueError(
					f"{label}: nombre de couches invalide. Attendu {len(old_layers)}, reçu {len(new_layers)}."
				)

			out = []
			for layer_index in range(len(old_layers)):
				old_layer = old_layers[layer_index]
				new_layer = new_layers[layer_index]
				layer_label = f"{label}[{layer_index + 1}]"

				if not isinstance(old_layer, list):
					raise ValueError(
						f"struct_reso.{label}[{layer_index + 1}] invalide: attendu une liste, reçu {type(old_layer).__name__}."
					)
				if not isinstance(new_layer, list):
					raise ValueError(
						f"{layer_label}: type invalide. Attendu une liste, reçu {type(new_layer).__name__}."
					)

				if _is_pair_list(new_layer):
					_ensure_pair_list(new_layer, layer_label, expected_len=len(old_layer))
					out.append(new_layer)
				else:
					if len(new_layer) != len(old_layer):
						raise ValueError(
							f"{layer_label}: dimension invalide. Attendu {len(old_layer)} valeur(s), reçu {len(new_layer)}."
						)
					out.append([(name, new_layer[i]) for i, (name, _) in enumerate(old_layer)])
			return out

		# Vérification + conversion
		new_Xn = _to_pairs_1d(old_Xn, Xn, "Xn")
		new_Dn_s = _to_pairs_1d(old_Dn_s, Dn_s, "Dn_s")
		new_Wn_c = _to_pairs_2d(old_Wn_c, Wn_c, "Wn_c")
		new_Bn_c = _to_pairs_2d(old_Bn_c, Bn_c, "Bn_c")

		# Mise à jour
		struct_reso[kx] = new_Xn
		struct_reso[kw] = new_Wn_c
		struct_reso[kb] = new_Bn_c
		struct_reso[kd] = new_Dn_s
		self.struct_reso = struct_reso
		return struct_reso


# ==================== Classe backpp ====================


class backpp:
	"""Résolution/Backprop (approche matricielle) à partir de `struct_reso`.

	Objectif:
		Pouvoir utiliser les fonctions de [neurone/neurone_mat.py] avec la
		structure `struct_reso` (X*, W*, B*, D*).

	Notes:
		- `forward()` calcule la sortie y à partir de X, W, b.
		- `backprop_update()` fait une itération complète et met à jour W/B dans
		  `struct_reso`.
	"""

	# ==================== __init__ =========================
	def __init__(self, struct_reso: Dict[str, object], n_fct: int = 1, eta: float = 0.1):
		"""Initialise le solveur (struct_reso + paramètres n_fct/eta)."""
		self.struct_reso = struct_reso
		self.n_fct = int(n_fct)
		self.eta = float(eta)
		self._core = NeuroneMat()

	# ==================== _get_by_prefix =========================
	@staticmethod
	def _get_by_prefix(struct_reso: Dict[str, object], prefix: str):
		"""Retourne (clé, valeur) du premier champ dont la clé commence par prefix."""
		for k, v in struct_reso.items():
			if isinstance(k, str) and k.startswith(prefix):
				return k, v
		return None, None

	# ==================== _pairs_to_vector =========================
	@staticmethod
	def _pairs_to_vector(pairs, prefix: str) -> List[float]:
		"""Convertit une liste de (nom,valeur) en vecteur ordonné par indice."""
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
		"""Extrait (X, w_list, b_list, d) depuis struct_reso."""
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
		"""Écrit new_w_list/new_b_list dans struct_reso (en conservant les noms)."""
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
				raise ValueError(
					f"Couche {layer_index}: b a {len(b)} éléments, attendu {n_out}."
				)

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
		"""Résout le réseau (propagation avant) à partir de struct_reso.

		Retour:
			y (liste)
		"""
		X, w_list, b_list, _ = self._struct_to_matrices()
		use_n_fct = self.n_fct if n_fct is None else int(n_fct)
		y, _cache = self._core.forward(X, w_list, b_list, use_n_fct)
		return y

	# ==================== solve =========================
	def solve(self, n_fct: int | None = None):
		"""Alias de `forward()` (résolution par propagation avant)."""
		return self.forward(n_fct=n_fct)

	# ==================== forward_layer =========================
	def forward_layer(self, x, W, b, n_fct: int | None = None):
		"""Propagation avant d'une couche (wrapper de NeuroneMat.forward_layer)."""
		use_n_fct = self.n_fct if n_fct is None else int(n_fct)
		return self._core.forward_layer(x, W, b, use_n_fct)

	# ==================== Delta =========================
	@staticmethod
	def Delta(d_list, Fi_list, Fp_list):
		"""Wrapper de NeuroneMat.Delta (couche de sortie)."""
		return NeuroneMat.Delta(d_list, Fi_list, Fp_list)

	# ==================== Delta_cache =========================
	@staticmethod
	def Delta_cache(delta_next, W_next, Fp_current):
		"""Wrapper de NeuroneMat.Delta_cache (couche cachée)."""
		return NeuroneMat.Delta_cache(delta_next, W_next, Fp_current)

	# ==================== Correcteur =========================
	@staticmethod
	def Correcteur(eta, a_prev, delta_current):
		"""Wrapper de NeuroneMat.Correcteur (corrections delta_W/delta_b)."""
		return NeuroneMat.Correcteur(eta, a_prev, delta_current)

	# ==================== maj =========================
	@staticmethod
	def maj(W, delta_W, b, delta_b):
		"""Wrapper de NeuroneMat.maj (mise à jour W/b)."""
		return NeuroneMat.maj(W, delta_W, b, delta_b)

	# ==================== backprop_update =========================
	def backprop_update(self, eta: float | None = None, n_fct: int | None = None):
		"""Fait une itération complète (forward + backprop + maj) et met à jour struct_reso."""
		X, w_list, b_list, d_list = self._struct_to_matrices()
		use_eta = self.eta if eta is None else float(eta)
		use_n_fct = self.n_fct if n_fct is None else int(n_fct)
		new_w_list, new_b_list, y = self._core.backprop_update(X, w_list, b_list, d_list, use_eta, use_n_fct)
		self._matrices_to_struct(new_w_list, new_b_list)
		return y

	# ==================== _nom_fct =========================
	@staticmethod
	def _nom_fct(n_fct: int) -> str:
		"""Retourne le nom lisible de la fonction d'activation (1..4)."""
		return {1: "sigmoïde", 2: "tan", 3: "tanh", 4: "gelu"}.get(n_fct, f"n_fct={n_fct}")

	# ==================== _fmt =========================
	@staticmethod
	def _fmt(x: float, precision: int = 6) -> str:
		"""Format numérique lisible (sans zéros inutiles).

		`precision` est un nombre de chiffres significatifs (style `g`).
		"""
		try:
			xf = float(x)
		except Exception:
			return str(x)

		if abs(xf) < 1e-15:
			xf = 0.0
		if abs(xf - round(xf)) < 1e-12:
			return str(int(round(xf)))
		return format(xf, f".{int(precision)}g")

	# ==================== _fmt_vec =========================
	@classmethod
	def _fmt_vec(cls, v: List[float], precision: int = 6) -> str:
		"""Formate un vecteur sous la forme: [v1, v2, ...]."""
		return "[" + ", ".join(cls._fmt(x, precision) for x in v) + "]"

	# ==================== _fmt_mat =========================
	@classmethod
	def _fmt_mat(
		cls,
		M: List[List[float]],
		precision: int = 6,
		row_labels: List[str] | None = None,
		col_labels: List[str] | None = None,
	) -> str:
		"""Formate une matrice en tableau aligné (avec en-tête optionnel)."""
		if not M:
			return "[]"
		n_rows = len(M)
		n_cols = len(M[0]) if M else 0

		cell = [[cls._fmt(M[r][c], precision) for c in range(n_cols)] for r in range(n_rows)]
		col_labels_local = col_labels if col_labels is not None else [f"c{c + 1}" for c in range(n_cols)]
		row_labels_local = row_labels if row_labels is not None else [f"r{r + 1}" for r in range(n_rows)]

		widths = [len(col_labels_local[c]) for c in range(n_cols)]
		for c in range(n_cols):
			for r in range(n_rows):
				widths[c] = max(widths[c], len(cell[r][c]))

		row_label_w = max(len(lbl) for lbl in row_labels_local)
		header = " " * (row_label_w + 2) + "  ".join(
			col_labels_local[c].rjust(widths[c]) for c in range(n_cols)
		)
		lines = [header]
		for r in range(n_rows):
			line = row_labels_local[r].rjust(row_label_w) + ": " + "  ".join(
				cell[r][c].rjust(widths[c]) for c in range(n_cols)
			)
			lines.append(line)
		return "[\n" + "\n".join("  " + ln for ln in lines) + "\n]"

	# ==================== _fmt_mat_no_header =========================
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

	# ==================== _print_step =========================
	@staticmethod
	def _print_step(title: str) -> None:
		"""Affiche un séparateur de section du style: --- Titre ---"""
		print(f"\n--- {title} ---")

	# ==================== resolution_affiche =========================
	def resolution_affiche(
		self,
		eta: float | None = None,
		n_fct: int | None = None,
		update_struct: bool = True,
		show_params: bool = True,
		precision: int = 6,
	) -> Dict[str, object]:
		"""Résolution identique aux scripts matriciels (ex: c2p59_mat.py), mais depuis struct_reso.

		Phases affichées:
			- ACTIVATIONS
			- FONCTIONS D'ACTIVATION + DERIVEES
			- SIGNAUX D'ERREUR
			- FACTEURS DE CORRECTION
			- MISES A JOUR

		Si update_struct=True, les poids/biais de struct_reso sont mis à jour.
		"""

		X, w_list, b_list, d_list = self._struct_to_matrices()
		use_eta = self.eta if eta is None else float(eta)
		use_n_fct = self.n_fct if n_fct is None else int(n_fct)
		nom_fct = self._nom_fct(use_n_fct)

		# ==================== CALCULS (matriciel) ====================
		y, cache = self._core.forward(X, w_list, b_list, use_n_fct)

		z_list = cache.z_list
		a_list = cache.a_list
		fp_list = cache.fp_list

		# deltas (de la sortie vers l'entrée)
		delta_list: List[List[float]] = [None] * len(w_list)  # type: ignore[assignment]
		err_list: List[List[float] | None] = [None] * len(w_list)
		delta_list[-1] = self._core.Delta(d_list, y, fp_list[-1])
		for layer_index in range(len(w_list) - 2, -1, -1):
			delta_list[layer_index] = self._core.Delta_cache(
				delta_list[layer_index + 1], w_list[layer_index + 1], fp_list[layer_index]
			)

		# correcteurs + maj (de l'entrée vers la sortie)
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

		# ==================== AFFICHAGE (calculs numériques) ====================
		if show_params:
			self._print_step(f"Paramètres (n_fct={use_n_fct} / {nom_fct}, eta={use_eta})")
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

		# Étapes forward
		for layer_index, (W, b, z, a, fp) in enumerate(zip(w_list, b_list, z_list, a_list, fp_list), start=1):
			inp_name = "X" if layer_index == 1 else f"a{layer_index - 1}"
			inp_vec = X if layer_index == 1 else a_list[layer_index - 2]
			self._print_step(f"Forward couche {layer_index}")

			# Détail terme-à-terme (utile quand les dimensions sont petites)
			n_in = len(W)
			n_out = len(W[0]) if W else 0
			show_terms = n_in <= 3 and n_out <= 3 and len(inp_vec) <= 3 and len(b) <= 3
			if show_terms:
				print(f"i{layer_index} = {inp_name}@W{layer_index} + b{layer_index}")
			else:
				print(
					f"i{layer_index} = {inp_name}@W{layer_index} + b{layer_index} = {self._fmt_vec(z, precision)}"
				)
			if show_terms:
				for j in range(n_out):
					# forme symbolique + forme numérique
					sym_terms = []
					num_terms = []
					for i in range(n_in):
						var_i = f"x{i + 1}" if layer_index == 1 else f"a{i + 1}_{layer_index - 1}"
						w_name = f"w{i + 1}{j + 1}_{layer_index}"
						sym_terms.append(f"{var_i}*{w_name}")
						num_terms.append(f"{self._fmt(inp_vec[i], precision)}*{self._fmt(W[i][j], precision)}")
					b_name = f"b{j + 1}_{layer_index}"
					print(
						f"  i{j + 1}_{layer_index} = "
						+ " + ".join(sym_terms)
						+ f" + {b_name} = "
						+ " + ".join(num_terms)
						+ f" + {self._fmt(b[j], precision)} = {self._fmt(z[j], precision)}"
					)

			if len(a) <= 3:
				print(f"a{layer_index} = Fi(i{layer_index})")
				for j in range(len(a)):
					print(
						f"  a{j + 1}_{layer_index} = Fi(i{j + 1}_{layer_index}) = {self._fmt(a[j], precision)}"
					)
			else:
				print(f"a{layer_index} = Fi(i{layer_index}) = {self._fmt_vec(a, precision)}")
			if len(fp) <= 3:
				print(f"fp{layer_index} = Fp(i{layer_index})")
				for j in range(len(fp)):
					print(
						f"  fp{j + 1}_{layer_index} = Fp(i{j + 1}_{layer_index}) = {self._fmt(fp[j], precision)}"
					)
			else:
				print(f"fp{layer_index} = Fp(i{layer_index}) = {self._fmt_vec(fp, precision)}")

		self._print_step("Sortie")
		L = len(a_list)
		# Convention d'affichage: a<rang>_<couche>
		Y_mat = [[float(v)] for v in y]
		Y_row_labels = [f"a{j + 1}_{L}" for j in range(len(y))]
		print(f"Y (matrice) = {self._fmt_mat_no_header(Y_mat, precision, row_labels=Y_row_labels)}")
		print(f"y (liste) = {self._fmt_vec(y, precision)}")
		if len(y) == 1:
			print(f"y: a1_{L} = [{self._fmt(y[0], precision)}]")
		else:
			print("y:")
			for j in range(len(y)):
				print(f"  a{j + 1}_{L} = [{self._fmt(y[j], precision)}]")

		# Étapes backward (deltas)
		self._print_step("Deltas")
		L = len(delta_list)
		d_minus_y = [float(d_list[i]) - float(y[i]) for i in range(len(y))]
		if len(delta_list[-1]) <= 3:
			print(f"delta{L} = (d - y) ⊙ fp{L}")
			for j in range(len(delta_list[-1])):
				print(
					f"  delta{j + 1}_{L} = (d{j + 1} - y{j + 1})*fp{j + 1}_{L} = "
					f"({self._fmt(d_list[j], precision)} - {self._fmt(y[j], precision)})*{self._fmt(fp_list[-1][j], precision)} = "
					f"{self._fmt(d_minus_y[j], precision)}*{self._fmt(fp_list[-1][j], precision)} = {self._fmt(delta_list[-1][j], precision)}"
				)
		else:
			print(
				f"delta{L} = (d - y) ⊙ fp{L} = {self._fmt_vec(delta_list[-1], precision)}"
			)
		for layer_index in range(L - 1, 0, -1):
			W_next = w_list[layer_index]  # W_{layer_index+1} (0-based)
			delta_next = delta_list[layer_index]
			# err = W_next @ delta_next (car W est (n_in,n_out) et delta_next est (n_out))
			err_prev: List[float] = []
			for k in range(len(W_next)):
				s = 0.0
				row = W_next[k]
				for j in range(len(row)):
					s += float(row[j]) * float(delta_next[j])
				err_prev.append(s)

			# Stocke err pour le résumé (err{layer_index})
			err_list[layer_index - 1] = err_prev

			# Détail terme-à-terme (utile quand les dimensions sont petites)
			show_err_terms = len(W_next) <= 3 and (len(W_next[0]) if W_next else 0) <= 3 and len(delta_next) <= 3
			if show_err_terms:
				print(f"err{layer_index} = W{layer_index + 1} @ delta{layer_index + 1}")
			else:
				print(
					f"err{layer_index} = W{layer_index + 1} @ delta{layer_index + 1} = {self._fmt_vec(err_prev, precision)}"
				)

			if show_err_terms:
				for k in range(len(W_next)):
					row = W_next[k]
					sym_terms = []
					num_terms = []
					for j in range(len(row)):
						w_name = f"w{k + 1}{j + 1}_{layer_index + 1}"
						d_name = f"delta{j + 1}_{layer_index + 1}"
						sym_terms.append(f"{w_name}*{d_name}")
						num_terms.append(f"{self._fmt(row[j], precision)}*{self._fmt(delta_next[j], precision)}")
					print(
						f"  err{k + 1}_{layer_index} = "
						+ " + ".join(sym_terms)
						+ " = "
						+ " + ".join(num_terms)
						+ f" = {self._fmt(err_prev[k], precision)}"
					)

			if len(delta_list[layer_index - 1]) <= 3:
				print(f"delta{layer_index} = err{layer_index} ⊙ fp{layer_index}")
				for j in range(len(delta_list[layer_index - 1])):
					print(
						f"  delta{j + 1}_{layer_index} = err{j + 1}_{layer_index}*fp{j + 1}_{layer_index} = "
						f"{self._fmt(err_prev[j], precision)}*{self._fmt(fp_list[layer_index - 1][j], precision)} = {self._fmt(delta_list[layer_index - 1][j], precision)}"
					)
			else:
				print(
					f"delta{layer_index} = err{layer_index} ⊙ fp{layer_index} = {self._fmt_vec(delta_list[layer_index - 1], precision)}"
				)

		# Correcteurs
		self._print_step("Correcteurs")
		print(f"eta = {self._fmt(use_eta, precision)}")
		for layer_index, (dW, db) in enumerate(zip(dW_list, db_list), start=1):
			prev_name = "X" if layer_index == 1 else f"a{layer_index - 1}"
			prev_vec = X if layer_index == 1 else a_list[layer_index - 2]
			delta_current = delta_list[layer_index - 1]
			n_in = len(dW)
			n_out = len(dW[0]) if dW else 0
			w_row_labels = [
				", ".join(f"w{i}{j}_{layer_index}" for j in range(1, n_out + 1))
				for i in range(1, n_in + 1)
			]
			print(
				f"dW{layer_index} = eta * outer({prev_name}, delta{layer_index}) = {self._fmt_mat_no_header(dW, precision, row_labels=w_row_labels)}"
			)
			# Détail terme-à-terme
			if n_in <= 3 and n_out <= 3 and len(prev_vec) <= 3 and len(delta_current) <= 3:
				for i in range(n_in):
					for j in range(n_out):
						var_i = f"x{i + 1}" if layer_index == 1 else f"a{i + 1}_{layer_index - 1}"
						delta_name = f"delta{j + 1}_{layer_index}"
						print(
							f"  dW{layer_index}_{i + 1},{j + 1} = eta*{var_i}*{delta_name} = "
							f"{self._fmt(use_eta, precision)}*{self._fmt(prev_vec[i], precision)}*{self._fmt(delta_current[j], precision)} = {self._fmt(dW[i][j], precision)}"
						)
			print(f"db{layer_index} = eta * delta{layer_index} = {self._fmt_vec(db, precision)}")
			if len(db) <= 3 and len(delta_current) <= 3:
				for j in range(len(db)):
					print(
						f"  db{j + 1}_{layer_index} = eta*delta{j + 1}_{layer_index} = "
						f"{self._fmt(use_eta, precision)}*{self._fmt(delta_current[j], precision)} = {self._fmt(db[j], precision)}"
					)

		# Mises à jour
		self._print_step("Mises à jour")
		prec_up = max(int(precision), 12)
		for layer_index, (W, b, new_W, new_b) in enumerate(zip(w_list, b_list, new_w_list, new_b_list), start=1):
			n_in = len(new_W)
			n_out = len(new_W[0]) if new_W else 0
			w_row_labels = [
				", ".join(f"w{i}{j}_{layer_index}" for j in range(1, n_out + 1))
				for i in range(1, n_in + 1)
			]
			print(
				f"W{layer_index}_new = W{layer_index} + dW{layer_index} = {self._fmt_mat_no_header(new_W, prec_up, row_labels=w_row_labels)}"
			)
			# Détail terme-à-terme
			n_in = len(W)
			n_out = len(W[0]) if W else 0
			if n_in <= 3 and n_out <= 3:
				for i in range(n_in):
					for j in range(n_out):
						w_name = f"w{i + 1}{j + 1}_{layer_index}"
						print(
							f"  W{layer_index}_new[{i + 1},{j + 1}] ({w_name}) = {self._fmt(W[i][j], prec_up)} + {self._fmt(dW_list[layer_index - 1][i][j], prec_up)} = {self._fmt(new_W[i][j], prec_up)}"
						)
			print(f"b{layer_index}_new = b{layer_index} + db{layer_index} = {self._fmt_vec(new_b, prec_up)}")
			if len(b) <= 3:
				for j in range(len(b)):
					b_name = f"b{j + 1}_{layer_index}"
					print(
						f"  b{layer_index}_new[{j + 1}] ({b_name}) = {self._fmt(b[j], prec_up)} + {self._fmt(db_list[layer_index - 1][j], prec_up)} = {self._fmt(new_b[j], prec_up)}"
					)

		if update_struct:
			self._matrices_to_struct(new_w_list, new_b_list)

		# ==================== RÉSUMÉ ====================
		self._print_step("Résumé")
		# 1) Activations
		print("\n[1] Activations")
		for layer_index, (z, a) in enumerate(zip(z_list, a_list), start=1):
			print(f"i{layer_index} = {self._fmt_vec(z, precision)}")
			print(f"a{layer_index} = {self._fmt_vec(a, precision)}")

		# 2) Fonctions d'activation + dérivées
		print("\n[2] Fonctions d'activation + dérivées")
		print(f"f = {nom_fct}")
		for layer_index, (z, a, fp) in enumerate(zip(z_list, a_list, fp_list), start=1):
			print(f"Fi(i{layer_index}) = a{layer_index} = {self._fmt_vec(a, precision)}")
			print(f"Fp(i{layer_index}) = fp{layer_index} = {self._fmt_vec(fp, precision)}")

		# 3) Signaux d'erreur
		print("\n[3] Signaux d'erreur")
		for layer_index in range(1, len(w_list)):
			err_prev = err_list[layer_index - 1]
			if err_prev is not None:
				print(f"err{layer_index} = {self._fmt_vec(err_prev, precision)}")
		for layer_index, delta in enumerate(delta_list, start=1):
			print(f"delta{layer_index} = {self._fmt_vec(delta, precision)}")

		# 4) Facteurs de correction
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

		# 5) Mises à jour
		print("\n[5] Mises à jour")
		prec_up = max(int(precision), 12)
		for layer_index, (new_W, new_b) in enumerate(zip(new_w_list, new_b_list), start=1):
			n_in = len(new_W)
			n_out = len(new_W[0]) if new_W else 0
			w_row_labels = [
				", ".join(f"w{i}{j}_{layer_index}" for j in range(1, n_out + 1))
				for i in range(1, n_in + 1)
			]
			print(
				f"W{layer_index}_new = {self._fmt_mat_no_header(new_W, prec_up, row_labels=w_row_labels)}"
			)
			print(f"b{layer_index}_new = {self._fmt_vec(new_b, prec_up)}")

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


# Alias rétrocompatible (ancien nom)
backpp_mat = backpp

