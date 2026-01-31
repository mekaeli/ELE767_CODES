"""reseau

Rôle
	Décrit et construit la *structure* d'un réseau de neurones (MLP) sous la
	forme d'un dictionnaire `struct_reso`.

Ce module est volontairement autonome
	Il ne dépend d'aucun autre module du projet. Il sert de brique de base pour :
		- initialiser les paramètres (X, W, b, d)
		- afficher/éditer ces paramètres dans le style attendu par le laboratoire

Structure `struct_reso`
	`struct_reso` est un dictionnaire contenant des paires (nom, valeur). Les clés
	portent les dimensions et ont la forme :
		- X{n_in} : liste des entrées (nom, valeur) ex. ('x1', 1)
		- W{n_layers} : liste de couches de poids, une couche = liste (nom, valeur)
		- B{n_layers} : liste de couches de biais, une couche = liste (nom, valeur)
		- D{n_out} : liste des sorties désirées (nom, valeur) ex. ('d1', 0)

Dimensions
	- n_c : nombre de couches cachées
	- n_s : nombre de couches de sortie (typiquement 1)
	- n_layers = n_c + n_s
	- n_out = N_b[-1]

Point d'entrée principal
	La classe `mon_reso` expose :
		- `cree_reso()` : génère un `struct_reso`
		- `Affiche_reso()` : imprime les paramètres (style affectations)
		- `set_reso()` : injecte X/W/B/D fournis dans une structure existante
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple


class mon_reso:
	"""Réseau (structure) configurable.

	Cette classe ne fait *pas* d'apprentissage :
		- elle génère et manipule la structure `struct_reso` (noms + valeurs)
		- elle fournit l'affichage au format attendu par le laboratoire

	Les calculs (forward/backprop) sont réalisés dans `backpp.py`.
	"""

	# ==================== __init__ =========================
	def __init__(
		self,
		n_in: int,
		n_c: int,
		n_s: int,
		N_b: List[int],
		biais,
		poids,
		X,
		D,
		seed: int | None = None,
	):
		"""Initialise la configuration du réseau (dimensions + bornes)."""
		self.n_in = int(n_in)
		self.n_c = int(n_c)
		self.n_s = int(n_s)
		self.N_b = list(N_b)
		self.biais = list(biais)
		self.poids = list(poids)
		self.X = list(X)
		self.D = list(D)
		# RNG local (répétable avec seed, et n'impacte pas le hasard global).
		self._rng = random.Random(seed)
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
		n_s_local = self.n_s
		n_layers = n_c_local + n_s_local
		N_b_local = self.N_b

		if n_in_local <= 0:
			raise ValueError("n_in doit être > 0")
		if n_c_local < 0:
			raise ValueError("n_c doit être >= 0 (couches cachées)")
		if n_s_local <= 0:
			raise ValueError("n_s doit être > 0 (couches de sortie)")
		if n_layers <= 0:
			raise ValueError("Le réseau doit contenir au moins une couche.")
		if len(N_b_local) != n_layers:
			raise ValueError("N_b doit contenir exactement n_c + n_s valeurs")
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

		Xn: List[Tuple[str, int]] = [(f"x{i}", self._rng.choice(self.X)) for i in range(1, n_in_local + 1)]

		Wn_c_local: List[List[Tuple[str, float]]] = []
		Bn_c_local: List[List[Tuple[str, float]]] = []

		for c in range(1, n_layers + 1):
			n_src = n_in_local if c == 1 else N_b_local[c - 2]
			n_dst = N_b_local[c - 1]

			Wc: List[Tuple[str, float]] = []
			for i in range(1, n_src + 1):
				for j in range(1, n_dst + 1):
					name = f"w{i}{j}_{c}"
					value = self._rng.uniform(p_min, p_max)
					Wc.append((name, value))
			Wn_c_local.append(Wc)

			Bc: List[Tuple[str, float]] = []
			for j in range(1, n_dst + 1):
				name = f"b{j}_{c}"
				value = self._rng.uniform(b_min, b_max)
				Bc.append((name, value))
			Bn_c_local.append(Bc)

		n_out = int(N_b_local[-1])
		Dn_s_local: List[Tuple[str, int]] = [(f"d{i}", self._rng.choice(self.D)) for i in range(1, n_out + 1)]

		key_x = f"X{n_in_local}"
		key_w = f"W{n_layers}"
		key_b = f"B{n_layers}"
		key_d = f"D{n_out}"
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
		n_layers = int(self.n_c) + int(self.n_s)
		for c in range(1, n_layers + 1):
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
			# Poids:
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

