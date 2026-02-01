"""lanceur

Rôle
	Point d'entrée "application" du laboratoire 1.

	Ce module orchestre :
		- la lecture/écriture des configurations (via `service.py`)
		- le lancement de l'interface graphique (via `interface.py`)
		- l'exécution d'un "payload" provenant de l'UI (validation + exécution)
		- le mode *Test unitaire* (option 1) qui affiche le calcul détaillé
		  (via `backpp.backpp.resolution_affiche`)

Notes
	- La logique d'apprentissage/backprop est dans `backpp.py`.
	- La structure `struct_reso` est produite par `reseau.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import ast
import copy
import os
import sys
import unicodedata

# Le module est conçu pour être utilisé comme package.
# Exécution recommandée: `python -m lab1.lanceur`.

# ==================== _env_int =========================
def _env_int(name: str, default: int) -> int:
	"""Lit un entier depuis les variables d'environnement.

	- Si la variable n'existe pas ou est invalide, retourne `default`.
	"""
	try:
		return int(os.environ.get(name, str(default)).strip())
	except Exception:
		return default


# ==================== _env_float =========================
def _env_float(name: str, default: float) -> float:
	"""Lit un float depuis les variables d'environnement.

	- Accepte la virgule comme séparateur décimal.
	- Si la variable n'existe pas ou est invalide, retourne `default`.
	"""
	try:
		return float(os.environ.get(name, str(default)).strip().replace(",", "."))
	except Exception:
		return default


# ==================== _env_bool =========================
def _env_bool(name: str, default: bool) -> bool:
	"""Lit un booléen depuis l'environnement.

	Accepte (case-insensitive): 1/0, true/false, yes/no, y/n, on/off.
	"""
	try:
		raw = os.environ.get(name)
		if raw is None:
			return bool(default)
		s = str(raw).strip().lower()
		if s in {"1", "true", "yes", "y", "on"}:
			return True
		if s in {"0", "false", "no", "n", "off"}:
			return False
		return bool(default)
	except Exception:
		return bool(default)


@dataclass(frozen=True)
class varEntryManuel:
	"""Exemple de données manuelles au format attendu par `reso.set_reso()`.

	Format attendu:
	- Xn   : entrées
	- Wn_c : poids aplatis par couche (row-major)
	- Bn_c : biais par couche
	- Dn_s : sortie(s) désirée(s)
	- n_fct / eta : hyperparamètres
	"""
	# Garder les annotations de type: elles aident la lecture et l'auto-complétion.

	# 1) Entrées (X)
	x1: float = 2.0
	x2: float = 4.0

	# 2) Poids + biais (couche 1)
	w11_1: float = 1.0
	w12_1: float = 3.0
	w21_1: float = 2.0
	w22_1: float = 4.0
	b1_1: float = 2.0
	b2_1: float = 1.0

	# 3) Poids + biais (couche 2)
	w11_2: float = 0.3
	w12_2: float = -0.2
	w13_2: float = 0.2
	w21_2: float = 1.3
	w22_2: float = -0.4
	w23_2: float = -1.2

	b1_2: float = -0.1
	b2_2: float = 0.1
	b3_2: float = -1.2

	# 4) Sorties désirées (D)
	d1: float = 1.0
	d2: float = 2.0
	d3: float = 1.0

	# 5) Hyperparamètres
	n_fct: int = 1
	eta: float = 0.1
	test_unitaire: bool = True

	@staticmethod
	def from_env() -> "varEntryManuel":
		return varEntryManuel(
			n_fct=_env_int("N_FCT", 1),
			eta=_env_float("ETA", 0.1),
			test_unitaire=_env_bool("TEST_UNITAIRE", True),
		)

	@property
	def Xn(self) -> list[float]:
		return [self.x1, self.x2]

	@property
	def Wn_c(self) -> list[list[float]]:
		W1_aplatie = [self.w11_1, self.w12_1, self.w21_1, self.w22_1]
		W2_aplatie = [self.w11_2, self.w12_2, self.w13_2, self.w21_2, self.w22_2, self.w23_2]
		return [W1_aplatie, W2_aplatie]

	@property
	def Bn_c(self) -> list[list[float]]:
		b1 = [self.b1_1, self.b2_1]
		b2 = [self.b1_2, self.b2_2, self.b3_2]
		return [b1, b2]

	@property
	def Dn_s(self) -> list[float]:
		return [self.d1, self.d2, self.d3]


# ==================== _format_score_percent =========================
def _format_score_percent(raw: object) -> str:
	"""Normalise un score pour affichage/stockage en pourcentage (ex: "60% ")."""
	s = str(raw).strip() if raw is not None else ""
	if not s or s == "-":
		return "-"
	if s.endswith("%"):
		return s.replace(" ", "")
	try:
		val = float(s.replace(",", "."))
	except Exception:
		return s
	if abs(val - round(val)) < 1e-9:
		return f"{int(round(val))}%"
	formatted = f"{val:.2f}".rstrip("0").rstrip(".")
	return f"{formatted}%"


# ==================== _parse_score_ref =========================
def _parse_score_ref(value: object) -> float:
	"""Parse un score de référence en % (0..100).

	Accepte: 60, 60.0, "60", "60%", " 60 % ".
	"""
	if value is None:
		return 0.0
	s = str(value).strip()
	if not s or s == "-":
		return 0.0
	if s.endswith("%"):
		s = s[:-1].strip()
	try:
		return float(s.replace(",", "."))
	except Exception:
		return 0.0


# ==================== _ask_continue_apprentissage =========================
def _ask_continue_apprentissage(message: str) -> bool:
	"""Demande à l'utilisateur s'il veut continuer l'apprentissage (GUI).

	Fallback: retourne False si la boîte de dialogue n'est pas disponible.
	"""
	try:
		import tkinter.messagebox as messagebox

		return bool(messagebox.askyesno("Apprentissage", message))
	except Exception:
		return False


# ==================== _extract_bracket_groups =========================
def _extract_bracket_groups(text: str) -> list[str]:
	"""Extrait les groupes `[...]` de premier niveau d'une ligne.

	Retourne les contenus sans les crochets externes.
	"""
	groups: list[str] = []
	buf: list[str] = []
	depth = 0
	in_group = False
	for ch in str(text or ""):
		if ch == "[":
			if depth == 0:
				in_group = True
				buf = []
			else:
				buf.append(ch)
			depth += 1
		elif ch == "]":
			if depth == 0:
				continue
			depth -= 1
			if depth == 0 and in_group:
				groups.append("".join(buf).strip())
				in_group = False
			else:
				buf.append(ch)
		else:
			if in_group:
				buf.append(ch)
	return groups


# ==================== _parse_parametres_config_line =========================
def _parse_parametres_config_line(raw_line: str) -> tuple[str, int, int, list[int], int, float, list[list[float]], list[list[float]], str]:
	"""Parse une ligne complète de `parametres.txt` (nouveau format) et retourne:
	(activation, nb_entrees, nb_couches, hidden_list, nb_sorties, eta, Wn_c, Bn_c, score).

	Note: `hidden_list` = couches cachées uniquement (taille == nb_couches).
	"""
	s = (raw_line or "").strip()
	if not s:
		raise ValueError("ligne vide")
	# En-tête
	lower = s.lower()
	if "nb d'entr" in lower and "eta" in lower and "poid" in lower:
		raise ValueError("ligne d'en-tête")

	groups = _extract_bracket_groups(s)
	if len(groups) < 5:
		raise ValueError("ligne illisible (groupes [] insuffisants)")
	activation = groups[0].strip() or "-"
	inner = _extract_bracket_groups(groups[1])
	if len(inner) < 5:
		raise ValueError("bloc [[...]] invalide (attendu 5 champs)")
	try:
		nb_entrees = int(inner[0].strip())
		nb_couches = int(inner[1].strip())
		nb_sorties = int(inner[3].strip())
		eta = float(str(inner[4]).strip().replace(",", "."))
	except Exception as exc:
		raise ValueError(f"métadonnées invalides: {exc}") from exc

	# nb_neurones (cachées)
	hidden_raw = (inner[2] or "").strip().replace("[", "").replace("]", "")
	if not hidden_raw:
		hidden_list = []
	else:
		parts = [p.strip() for p in (hidden_raw.split(",") if "," in hidden_raw else hidden_raw.split()) if p.strip()]
		hidden_list = [int(float(p.replace(",", "."))) for p in parts]
	if len(hidden_list) != nb_couches:
		raise ValueError("nb_neurones doit contenir exactement nb_couches valeurs")

	# Wn_c et Bn_c sont aussi des groupes [] top-level dans la ligne.
	# _extract_bracket_groups enlève les crochets externes, donc on les reconstitue.
	try:
		Wn_c = ast.literal_eval("[" + groups[2] + "]")
		Bn_c = ast.literal_eval("[" + groups[3] + "]")
	except Exception as exc:
		raise ValueError(f"Wn_c/Bn_c illisibles: {exc}") from exc
	if not isinstance(Wn_c, list) or not all(isinstance(layer, list) for layer in Wn_c):
		raise ValueError("Wn_c invalide")
	if not isinstance(Bn_c, list) or not all(isinstance(layer, list) for layer in Bn_c):
		raise ValueError("Bn_c invalide")

	Wn_c_f = [[float(v) for v in layer] for layer in Wn_c]
	Bn_c_f = [[float(v) for v in layer] for layer in Bn_c]
	return activation, nb_entrees, nb_couches, hidden_list, nb_sorties, float(eta), Wn_c_f, Bn_c_f, groups[-1].strip()


# ==================== _check_WnBn_dimensions =========================
def _check_WnBn_dimensions(*, nb_entrees: int, layer_sizes: list[int], Wn_c: list[list[float]], Bn_c: list[list[float]]) -> None:
	"""Valide dimensions Wn_c/Bn_c pour une topologie (nb_entrees -> layer_sizes...)."""
	if len(Wn_c) != len(layer_sizes) or len(Bn_c) != len(layer_sizes):
		raise ValueError("Wn_c/Bn_c incompatibles avec la topologie")
	prev = int(nb_entrees)
	for idx, layer_size in enumerate(layer_sizes):
		expected_w = prev * int(layer_size)
		if len(Wn_c[idx]) != expected_w:
			raise ValueError(f"Dimension invalide pour Wn_c[{idx}] (attendu {expected_w})")
		if len(Bn_c[idx]) != int(layer_size):
			raise ValueError(f"Dimension invalide pour Bn_c[{idx}] (attendu {layer_size})")
		prev = int(layer_size)


# ==================== test_general_option2 =========================
def test_general_option2(
	struct_reso: dict[str, object],
	Wn_c: list[list[float]],
	Bn_c: list[list[float]],
	n_in: int,
	*,
	eta: float = 0.1,
	n_fct: int = 1,
	root_dir: Path | None = None,
) -> tuple[float, int, int, int]:
	"""Option 2: Test général sur `n_in_data_test.txt`.

	- Boucle sur toutes les lignes du fichier de test.
	- Compare Dn_s vs Dn_snew (one-hot max).
	- Affiche OK/NOK/taille/Score et retourne le %.
	"""
	from .reseau import mon_reso
	from .backpp import backpp
	from . import loader, service

	use_n_in_selector = int(n_in)
	if use_n_in_selector not in {40, 50, 60}:
		raise ValueError("Aucun ficher correspondante n_in (attendu 40/50/60)")

	# Déduit n_in (dimension réelle) et nb_sorties depuis struct_reso.
	key_x = next((k for k in struct_reso.keys() if isinstance(k, str) and k.startswith("X")), None)
	key_b = next((k for k in struct_reso.keys() if isinstance(k, str) and k.startswith("B")), None)
	key_d = next((k for k in struct_reso.keys() if isinstance(k, str) and k.startswith("D")), None)
	if key_x is None or key_b is None or key_d is None:
		raise ValueError("struct_reso invalide: clés X/B/D manquantes")
	try:
		use_n_in_features = int(str(key_x)[1:])
	except Exception as exc:
		raise ValueError(f"struct_reso invalide: clé X inattendue: {key_x!r}") from exc

	D_pairs = struct_reso.get(key_d)
	if not isinstance(D_pairs, list):
		raise ValueError("struct_reso invalide: D doit être une liste")
	use_nb_sorties = len(D_pairs)
	if use_nb_sorties <= 0:
		raise ValueError("struct_reso invalide: nb_sorties <= 0")

	B_layers = struct_reso.get(key_b)
	if not isinstance(B_layers, list) or any(not isinstance(layer, list) for layer in B_layers):
		raise ValueError("struct_reso invalide: B doit être une liste de listes")
	N_b = [len(layer) for layer in B_layers]
	if not N_b or any(n <= 0 for n in N_b):
		raise ValueError("struct_reso invalide: dimensions B invalides")

	# Sanity dimensions W/B selon topologie
	_check_WnBn_dimensions(nb_entrees=use_n_in_features, layer_sizes=N_b, Wn_c=Wn_c, Bn_c=Bn_c)

	use_root_dir = root if root_dir is None else Path(root_dir)
	file_path = use_root_dir / f"{use_n_in_selector}_data_test.txt"
	if not file_path.exists():
		raise FileNotFoundError(f"Fichier introuvable: {file_path}")

	taille = loader.count_non_empty_lines(file_path)
	if taille <= 0:
		pct = 0.0
		print(f"OK=0 | NOK=0 | taille=0 | Score={pct:.2f}%")
		return pct, 0, 0, 0

	# Instancie mon_reso uniquement pour set_reso
	n_layers = len(N_b)
	use_n_s = 1
	use_n_c = max(0, n_layers - use_n_s)
	r = mon_reso(
		n_in=use_n_in_features,
		n_c=use_n_c,
		n_s=use_n_s,
		N_b=N_b,
		biais=[0.0, 0.0],
		poids=[0.0, 0.0],
		X=[0, 1],
		D=[0, 1],
		seed=None,
	)

	bp = backpp(copy.deepcopy(struct_reso), n_fct=int(n_fct), eta=float(eta))
	n_ok = 0
	n_nok = 0

	for n_ligne in range(1, taille + 1):
		D_list, X_list = loader.get_validation_sample_seq(
			file_path,
			n_ligne=n_ligne,
			n_in=use_n_in_features,
			nb_sorties=use_nb_sorties,
		)
		Dn_s_local = D_list[0]
		Xn_local = X_list[0]
		bp.struct_reso = r.set_reso(bp.struct_reso, Xn_local, Wn_c, Bn_c, Dn_s_local)
		y = bp.forward(n_fct=int(n_fct))
		Dn_snew = service.fonction_max(y)
		if Dn_snew == Dn_s_local:
			n_ok += 1
		else:
			n_nok += 1

	total = n_ok + n_nok
	pct = 0.0 if total == 0 else (100.0 * n_ok / total)
	print(f"OK={n_ok} | NOK={n_nok} | taille={taille} | Score={pct:.2f}%")
	return pct, n_ok, n_nok, int(taille)


# ==================== _extract_struct_meta =========================
def _extract_struct_meta(struct_reso: dict[str, object]) -> tuple[int, int, list[int]]:
	"""Retourne (n_in_features, nb_sorties, N_b) depuis struct_reso."""
	key_x = next((k for k in struct_reso.keys() if isinstance(k, str) and k.startswith("X")), None)
	key_b = next((k for k in struct_reso.keys() if isinstance(k, str) and k.startswith("B")), None)
	key_d = next((k for k in struct_reso.keys() if isinstance(k, str) and k.startswith("D")), None)
	if key_x is None or key_b is None or key_d is None:
		raise ValueError("struct_reso invalide: clés X/B/D manquantes")
	try:
		n_in_features = int(str(key_x)[1:])
	except Exception as exc:
		raise ValueError(f"struct_reso invalide: clé X inattendue: {key_x!r}") from exc

	D_pairs = struct_reso.get(key_d)
	if not isinstance(D_pairs, list):
		raise ValueError("struct_reso invalide: D doit être une liste")
	nb_sorties = len(D_pairs)
	if nb_sorties <= 0:
		raise ValueError("struct_reso invalide: nb_sorties <= 0")

	B_layers = struct_reso.get(key_b)
	if not isinstance(B_layers, list) or any(not isinstance(layer, list) for layer in B_layers):
		raise ValueError("struct_reso invalide: B doit être une liste de listes")
	N_b = [len(layer) for layer in B_layers]
	if not N_b or any(n <= 0 for n in N_b):
		raise ValueError("struct_reso invalide: dimensions B invalides")
	return n_in_features, nb_sorties, N_b


# ==================== _extract_Wn_c_Bn_c =========================
def _extract_Wn_c_Bn_c(struct_reso: dict[str, object]) -> tuple[list[list[float]], list[list[float]]]:
	"""Extrait Wn_c/Bn_c (valeurs) depuis struct_reso."""
	key_w = next((k for k in struct_reso.keys() if isinstance(k, str) and k.startswith("W")), None)
	key_b = next((k for k in struct_reso.keys() if isinstance(k, str) and k.startswith("B")), None)
	if key_w is None or key_b is None:
		raise ValueError("struct_reso invalide: clés W/B manquantes")

	W_layers = struct_reso.get(key_w)
	B_layers = struct_reso.get(key_b)
	if not isinstance(W_layers, list) or not isinstance(B_layers, list):
		raise ValueError("struct_reso invalide: W/B doivent être des listes")

	Wn_c: list[list[float]] = []
	Bn_c: list[list[float]] = []
	for layer in W_layers:
		if not isinstance(layer, list):
			raise ValueError("struct_reso invalide: W doit être une liste de listes")
		Wn_c.append([float(v) for _name, v in layer])  # type: ignore[misc]
	for layer in B_layers:
		if not isinstance(layer, list):
			raise ValueError("struct_reso invalide: B doit être une liste de listes")
		Bn_c.append([float(v) for _name, v in layer])  # type: ignore[misc]
	return Wn_c, Bn_c


# ==================== _format_parametres_line_for_current_run =========================
def _format_parametres_line_for_current_run(
	*,
	activation: str,
	nb_entrees: int,
	nb_couches: int,
	hidden_list: list[int],
	nb_sorties: int,
	eta: float,
	Wn_c: list[list[float]],
	Bn_c: list[list[float]],
	score_attendu: object,
) -> str:
	"""Construit une ligne parametres.txt (nouveau format) pour la sauvegarde."""
	hidden_str = ", ".join(str(int(v)) for v in hidden_list)
	score = _format_score_percent(score_attendu)
	return (
		f"[{activation}] [[{nb_entrees}] [{nb_couches}] [{hidden_str}] [{nb_sorties}] [{eta}]] "
		f"{Wn_c} {Bn_c} [{score}]"
	)


# ==================== run_apprentissage_option3 =========================
def run_apprentissage_option3(
	*,
	payload: dict,
	r,
	struct_reso: dict[str, object],
	eta: float,
	root_dir: Path | None = None,
) -> tuple[bool, str, dict[str, object]]:
	"""Option 3: apprentissage + erreurs + validation + ajout config si score atteint."""
	use_root_dir = root if root_dir is None else Path(root_dir)
	values = payload.get("values") or {}
	if not isinstance(values, dict):
		return False, "Payload invalide (values)", struct_reso

	try:
		n_in_selector = int(values.get("nb_entrees"))
		k_epoques = int(values.get("k_epoques"))
		score_ref = _parse_score_ref(values.get("score_attendu"))
		max_iterations_raw = values.get("max_iterations", os.environ.get("APPRENTISSAGE_MAX_ITERS"))
		max_iterations: int | None
		if max_iterations_raw is None or str(max_iterations_raw).strip() == "":
			max_iterations = None
		else:
			max_iterations = int(str(max_iterations_raw).strip())
			if max_iterations <= 0:
				max_iterations = None
		nb_couches = int(values.get("nb_couches"))
		nb_sorties_ui = int(values.get("nb_sorties"))
		hidden_raw = values.get("nb_neurones")
		if not isinstance(hidden_raw, list):
			return False, "nb_neurones invalide", struct_reso
		hidden_list = [int(v) for v in hidden_raw]
		activation = str(payload.get("activation") or "-").strip() or "-"
	except Exception as exc:
		return False, f"Paramètres invalides (Apprentissage): {exc}", struct_reso

	if n_in_selector not in {40, 50, 60}:
		return False, "Aucun ficher correspondante n_in (attendu 40/50/60)", struct_reso
	if k_epoques <= 0:
		return False, "k_epoques doit être >= 1", struct_reso

	# Imports internes (package)
	from . import loader, service
	from .backpp import backpp

	train_path = use_root_dir / f"{n_in_selector}_data_train.txt"
	if not train_path.exists():
		return False, f"Fichier introuvable: {train_path}", struct_reso

	# Dimensions depuis struct_reso
	try:
		n_in_features, nb_sorties_struct, _N_b = _extract_struct_meta(struct_reso)
	except Exception as exc:
		return False, f"struct_reso invalide: {exc}", struct_reso
	if nb_sorties_ui != nb_sorties_struct:
		return False, "nb_sorties UI incompatible avec struct_reso", struct_reso

	taille = loader.count_non_empty_lines(train_path)
	if taille <= 0:
		return False, f"Fichier d'entraînement vide: {train_path.name}", struct_reso

	# Exigence UI: i (iterations) doit refléter la taille du fichier.
	STATE.i = int(taille)

	# IMPORTANT: `erreur.txt` est écrit dans le dossier du module (lab1/erreur.txt)
	# pour éviter la confusion avec `lab1/data/`.
	erreur_path = Path(__file__).resolve().parent / "erreur.txt"
	# Repart à zéro à chaque exécution
	erreur_path.write_text("", encoding="utf-8")

	iteration = 0

	while True:
		stop_early = False
		with open(erreur_path, "a", encoding="utf-8") as f:
			for epoch_idx in range(1, k_epoques + 1):
				epoch_delta_sum: list[float] | None = None
				epoch_count = 0
				# Shuffle une fois par époque
				D_shuffled, X_shuffled = loader.get_validation_samples_random(
					train_path,
					n_in=n_in_features,
					nb_sorties=nb_sorties_struct,
					seed=None,
				)
				for Dn_s, Xn in zip(D_shuffled, X_shuffled):
					iteration += 1
					if max_iterations is not None and iteration > max_iterations:
						stop_early = True
						break
					# Met à jour X/D dans struct_reso (garde W/B actuels)
					Wn_c_current, Bn_c_current = _extract_Wn_c_Bn_c(struct_reso)
					struct_reso = r.set_reso(struct_reso, Xn, Wn_c_current, Bn_c_current, Dn_s)

					bp = backpp(struct_reso, n_fct=STATE.n_fct, eta=float(eta))
					# Delta de sortie attendu: (D - y) ⊙ f'(z)
					X_vec, w_list, b_list, d_list = bp._struct_to_matrices()  # type: ignore[attr-defined]
					y_forward, cache = bp._core.forward(X_vec, w_list, b_list, int(STATE.n_fct))
					delta_out = bp._core.Delta(d_list, y_forward, cache.fp_list[-1])

					y = bp.backprop_update(eta=float(eta), n_fct=int(STATE.n_fct))
					struct_reso = bp.struct_reso

					# Accumule les deltas de sortie pour l'époque
					if epoch_delta_sum is None:
						epoch_delta_sum = [0.0 for _ in delta_out]
					for i, v in enumerate(delta_out):
						epoch_delta_sum[i] += float(v)
					epoch_count += 1

				# Fin d'époque: écrit UNE seule ligne (moyenne des deltas) dans erreur.txt
				if epoch_count > 0 and epoch_delta_sum is not None:
					epoch_delta_mean = [v / float(epoch_count) for v in epoch_delta_sum]
					f.write(f"{epoch_idx} {epoch_delta_mean}\n")
				else:
					# Cas rare: aucune itération traitée dans l'époque (ex: max_iterations=0)
					f.write(f"{epoch_idx} []\n")

				if stop_early:
					break
			if stop_early:
				print(f"Apprentissage limité à {max_iterations} itérations (debug/test)")
				# on sort du bloc d'époques et passe directement à la validation

		# Validation finale
		Wn_c_final, Bn_c_final = _extract_Wn_c_Bn_c(struct_reso)
		pct = test_validation(
			copy.deepcopy(struct_reso),
			Wn_c_final,
			Bn_c_final,
			n_in_selector,
			eta=float(eta),
			n_fct=int(STATE.n_fct),
			root_dir=use_root_dir,
		)
		# Met à jour le score obtenu (utilisé par l'interface).
		STATE.score_ob = f"{pct:.2f}%"
		print(f"Retour validation: {pct:.2f}%")

		if pct >= score_ref:
			formatted_line = _format_parametres_line_for_current_run(
				activation=activation,
				nb_entrees=n_in_selector,
				nb_couches=nb_couches,
				hidden_list=hidden_list,
				nb_sorties=nb_sorties_ui,
				eta=float(eta),
				Wn_c=Wn_c_final,
				Bn_c=Bn_c_final,
				# Exigence: la colonne "score" doit refléter le score obtenu.
				score_attendu=pct,
			)
			ok, msg = service.add_parametres_line(formatted_line)
			if ok:
				return True, f"Apprentissage terminé: score={pct:.2f}% >= ref={score_ref:.2f}% (config ajoutée)", struct_reso
			return False, msg or "Refus d'écriture (doublon)", struct_reso

		# Score trop bas: demander de continuer
		cont = _ask_continue_apprentissage(
			f"Score obtenu {pct:.2f}% < score attendu {score_ref:.2f}%. Continuer l'apprentissage (encore {k_epoques} époques) ?"
		)
		if not cont:
			return False, f"Stop: score={pct:.2f}% < ref={score_ref:.2f}%", struct_reso


# Racine des fichiers d'entrée (datasets). Chemin local au package.
root = Path(__file__).resolve().parent / "data"


# Dictionnaire: nom d'activation -> identifiant numérique (n_fct).
ACTIVATION_TO_N_FCT: dict[str, int] = {
	"sigmoide": 1,
	"tan": 2,
	"tanh": 3,
	"gelu": 4,
}

# ==================== _normalize_activation =========================
def _normalize_activation(value: object) -> str:
	"""Normalise un libellé d'activation (lower + sans accents)."""
	s = str(value or "").strip().lower()
	# Décompose les accents puis les enlève.
	s = unicodedata.normalize("NFKD", s)
	s = "".join(ch for ch in s if not unicodedata.combining(ch))
	return s

# ==================== clear_console =========================
def clear_console() -> None:
	"""Nettoie la console.

	- Windows : commande `cls`
	- Linux/Mac : commande `clear`
	"""
	os.system("cls" if os.name == "nt" else "clear")
	print("Console nettoyée.")

@dataclass
class LanceurState:
	"""État centralisé du lanceur.

	Objectif: regrouper les paramètres dispersés (génération, apprentissage,
	UI) tout en gardant la compatibilité avec les variables globales.
	"""

	# Paramètres de base / génération
	n_in: int = 2
	n_c: int = 1
	n_s: int = 1
	N_b: list[int] = field(default_factory=lambda: [2, 1])
	biais: list[float] = field(default_factory=lambda: [1.0, 5.0])
	poids: list[float] = field(default_factory=lambda: [-0.1, 0.1])
	X: list[int] = field(default_factory=lambda: [0, 1])
	D: list[int] = field(default_factory=lambda: [0, 1])

	# Apprentissage / exécution
	n_fct: int = 1
	i: int = 1
	N: int = 1
	j: int = 1
	k: int = 1
	test_unitaire: bool = True

	# UI / suivi
	score_ob: str = "0%"
	score_ref: str = "25%"

	# Config réseau (courante)
	Wn_c: list[list[float]] = field(default_factory=list)
	Bn_c: list[list[float]] = field(default_factory=list)

	# Exemple d'initialisation (optionnel, utile pour debug/démos)
	manual_entry: varEntryManuel | None = None

	def startup_defaults(self) -> dict:
		return {
			"score_ob": self.score_ob,
			"biais": self.biais,
			"poids": self.poids,
			"i": self.i,
			"k": self.k,
			"Wn_c": self.Wn_c,
			"Bn_c": self.Bn_c,
			# Mode UI au démarrage (un seul choix parmi 3)
			"mode": ("Test unitaire" if self.test_unitaire else "Test général"),
		}


# État global du lanceur (utilisé par l'UI et par les tests).
STATE = LanceurState()


# ==================== _init_manual_entry_defaults =========================
def _init_manual_entry_defaults() -> None:
	"""Initialise les valeurs d'exemple (et l'état) depuis l'environnement."""
	ex = varEntryManuel.from_env()
	STATE.manual_entry = ex
	STATE.Wn_c = ex.Wn_c
	STATE.Bn_c = ex.Bn_c
	STATE.n_fct = ex.n_fct
	STATE.test_unitaire = bool(ex.test_unitaire)


_init_manual_entry_defaults()


# ==================== execute_payload =========================
def execute_payload(payload: dict) -> tuple[bool, str]:
	"""Chemin d'exécution "Exécuter" (testable hors-GUI).

	Valide côté lanceur, met à jour l'état, puis:
	- crée/initialise `struct_reso` via `reseau.py`
	- exécute l'option demandée (test unitaire ou mode)
	"""
	from .reseau import mon_reso
	from .backpp import backpp

	try:
		values = payload.get("values") or {}
		if not isinstance(values, dict):
			return False, "Payload invalide (values)"

		mode = str(payload.get("mode") or "").strip()
		# Un seul choix parmi 4: Test unitaire OU un des 3 modes
		test_unitaire = payload.get("test_unitaire", values.get("test_unitaire"))
		STATE.test_unitaire = bool(test_unitaire) or (mode == "Test unitaire")

		seed_raw = payload.get("seed", values.get("seed"))
		seed: int | None
		if seed_raw is None or str(seed_raw).strip() == "":
			seed = None
		else:
			seed = int(seed_raw)

		STATE.n_in = int(values.get("nb_entrees"))
		STATE.n_c = int(values.get("nb_couches"))
		# nb_sorties = nb neurones de sortie (i.e. taille de la dernière couche)
		nb_sorties = int(values.get("nb_sorties"))

		N_b_raw = payload.get("N_b")
		if not isinstance(N_b_raw, list) or not N_b_raw:
			return False, "N_b invalide"
		STATE.N_b = [int(v) for v in N_b_raw]
		# n_s = nb de couches de sortie (théoriquement toujours 1)
		STATE.n_s = 1
		# Cohérence minimale
		if nb_sorties <= 0 or STATE.N_b[-1] != nb_sorties:
			return False, "nb_sorties invalide (doit correspondre à N_b[-1])"

		STATE.biais = [float(values.get("biais_min")), float(values.get("biais_max"))]
		STATE.poids = [float(values.get("poids_min")), float(values.get("poids_max"))]
		STATE.i = int(values.get("iterations"))
		STATE.k = int(values.get("k_epoques"))

		# Activation: map -> n_fct
		activation = _normalize_activation(payload.get("activation"))
		STATE.n_fct = ACTIVATION_TO_N_FCT.get(activation, 1)

		STATE.score_ob = str(values.get("score_obtenu", STATE.score_ob))
		eta = float(values.get("eta", values.get("taux_apprentissage", 0.1)))

	except Exception as exc:
		return False, f"Erreur paramètres: {exc}"

	# 1) Créer & initialiser struct_reso via reseau.py
	try:
		r = mon_reso(
			n_in=STATE.n_in,
			n_c=STATE.n_c,
			n_s=STATE.n_s,
			N_b=STATE.N_b,
			biais=STATE.biais,
			poids=STATE.poids,
			X=STATE.X,
			D=STATE.D,
			seed=seed,
		)
		struct_reso = r.cree_reso()
	except Exception as exc:
		return False, f"Impossible de créer/initialiser struct_reso: {exc}"

	# 2) Option 1: test unitaire
	ex = STATE.manual_entry
	if STATE.test_unitaire:
		if ex is None:
			return False, "test_unitaire demandé mais aucun varEntryManuel n'est chargé"
		try:
			clear_console()
			# Vérifie la cohérence minimale entre l'exemple manuel et la topologie.
			n_layers = int(STATE.n_c) + int(STATE.n_s)
			if len(ex.Xn) != int(STATE.n_in):
				return False, "test_unitaire invalide: taille Xn incompatible avec nb_entrees"
			if len(ex.Wn_c) != n_layers or len(ex.Bn_c) != n_layers:
				return False, "test_unitaire invalide: Wn_c/Bn_c incompatibles avec nb_couches/nb_sorties"
			if len(ex.Dn_s) != int(STATE.N_b[-1]):
				return False, "test_unitaire invalide: taille Dn_s incompatible avec nb_sorties"

			# Modifie struct_reso via reseau.py (écrit X/W/B/D dans la structure).
			struct_reso = r.set_reso(
				struct_reso,
				ex.Xn,
				ex.Wn_c,
				ex.Bn_c,
				ex.Dn_s,
			)

			# Affiche les paramètres (style affectations) via reseau.
			r.Affiche_reso(struct_reso, precision=6, n_fct=STATE.n_fct, eta=eta)

			# Résout + affiche via backpp (utilise fct_activation côté backpp).
			bp = backpp(struct_reso, n_fct=STATE.n_fct, eta=eta)
			bp.resolution_affiche(
				eta=eta,
				n_fct=STATE.n_fct,
				update_struct=True,
				log_console="détaillé",
				precision=6,
			)
		except Exception as exc:
			return False, f"test_unitaire: échec résolution/affichage: {exc}"

		mode = str(payload.get("mode") or "").strip()
		return True, f"OK (mode={mode}, test_unitaire=True)"

	# 3) Options 2-3: modes
	if mode == "Apprentissage":
		ok, msg, _struct_after = run_apprentissage_option3(
			payload=payload,
			r=r,
			struct_reso=struct_reso,
			eta=eta,
			root_dir=root,
		)
		return ok, msg

	if mode == "Test général":
		# Exigence: Wn_c/Bn_c doivent provenir de la config sélectionnée dans parametres.txt.
		selected_raw_line = str(payload.get("selected_raw_line") or "").strip()
		if not selected_raw_line:
			return False, "Aucune configuration sélectionnée dans le tableau"
		from . import service
		text = service.read_parametres_text()
		# Cherche une correspondance exacte (strip)
		needle = selected_raw_line.strip()
		match_line = None
		for line in (text or "").splitlines():
			if (line or "").strip() == needle:
				match_line = (line or "").strip()
				break
		if match_line is None:
			return False, "Aucune config correspondante (introuvable dans parametres.txt)"
		try:
			_act, cfg_n_in, _cfg_n_c, _cfg_hidden, cfg_nb_sorties, _cfg_eta, Wn_c_cfg, Bn_c_cfg, _score = _parse_parametres_config_line(match_line)
		except Exception as exc:
			return False, f"Config invalide dans parametres.txt: {exc}"

		# Cohérences minimales (évite de lancer un test avec une mauvaise config)
		try:
			_n_in_features, nb_sorties_struct, _N_b = _extract_struct_meta(struct_reso)
		except Exception as exc:
			return False, f"struct_reso invalide: {exc}"
		if int(cfg_n_in) != int(STATE.n_in):
			return False, "Aucune config correspondante (nb_entrees ne correspond pas)"
		if int(cfg_nb_sorties) != int(nb_sorties_struct):
			return False, "Aucune config correspondante (nb_sorties ne correspond pas)"

		try:
			pct, n_ok, n_nok, taille = test_general_option2(
				struct_reso=struct_reso,
				Wn_c=Wn_c_cfg,
				Bn_c=Bn_c_cfg,
				n_in=cfg_n_in,
				eta=float(eta),
				n_fct=int(STATE.n_fct),
				root_dir=root,
			)
		except Exception as exc:
			return False, str(exc)

		STATE.score_ob = f"{pct:.2f}%"
		return True, f"OK={n_ok} | NOK={n_nok} | taille={taille} | Score={pct:.2f}%"

	if mode == "Généralisation":
		return False, f"Mode '{mode}' à venir"
	return False, f"Mode invalide: '{mode}'"


# ==================== test_validation =========================
def test_validation(
	struct_reso: dict[str, object],
	Wn_c: list[list[float]] | list[list[tuple[str, float]]],
	Bn_c: list[list[float]] | list[list[tuple[str, float]]],
	n_in: int,
	*,
	eta: float = 0.1,
	n_fct: int = 1,
	root_dir: Path = root,
) -> float:
	"""Valide un réseau sur un fichier `n_in_data_vc.txt`.

	Comportement demandé
	- Reçoit `struct_reso` + `Wn_c` + `Bn_c` + `n_in`.
	- Ouvre `root_dir / f"{n_in}_data_vc.txt"`.
	- Calcule `taille = loader.count_non_empty_lines(file)`.
	- Pour chaque ligne (1..taille):
		- récupère (D_list, X_list) via loader (un seul élément)
		- met à jour `struct_reso` avec Wn_c/Bn_c, D_list, X_list
		- exécute le forward via backpp
		- compare Dn_s (désirée) vs Dn_snew (one-hot max)
	- Affiche et retourne le pourcentage de réussite.

	Important
	- `Wn_c` et `Bn_c` ne doivent pas être modifiés par cette fonction.
	"""
	from .reseau import mon_reso
	from .backpp import backpp
	from . import loader
	from . import service

	use_n_in_selector = int(n_in)
	use_eta = float(eta)
	use_n_fct = int(n_fct)

	if use_n_in_selector <= 0:
		raise ValueError("n_in doit être >= 1")

	# Ne pas muter les paramètres reçus.
	Wn_c_fixed = copy.deepcopy(Wn_c)
	Bn_c_fixed = copy.deepcopy(Bn_c)
	struct_reso_local = copy.deepcopy(struct_reso)

	# Déduit n_in (dimension réelle) et nb_sorties depuis struct_reso.
	key_x = next((k for k in struct_reso_local.keys() if isinstance(k, str) and k.startswith("X")), None)
	key_b = next((k for k in struct_reso_local.keys() if isinstance(k, str) and k.startswith("B")), None)
	key_d = next((k for k in struct_reso_local.keys() if isinstance(k, str) and k.startswith("D")), None)
	if key_x is None or key_b is None or key_d is None:
		raise ValueError("struct_reso invalide: clés X/B/D manquantes")
	try:
		use_n_in_features = int(str(key_x)[1:])
	except Exception as exc:
		raise ValueError(f"struct_reso invalide: clé X inattendue: {key_x!r}") from exc

	D_pairs = struct_reso_local.get(key_d)
	if not isinstance(D_pairs, list):
		raise ValueError("struct_reso invalide: D doit être une liste")
	use_nb_sorties = len(D_pairs)
	if use_nb_sorties <= 0:
		raise ValueError("struct_reso invalide: nb_sorties <= 0")

	B_layers = struct_reso_local.get(key_b)
	if not isinstance(B_layers, list) or any(not isinstance(layer, list) for layer in B_layers):
		raise ValueError("struct_reso invalide: B doit être une liste de listes")
	N_b = [len(layer) for layer in B_layers]
	if not N_b or any(n <= 0 for n in N_b):
		raise ValueError("struct_reso invalide: dimensions B invalides")

	# Fichier à ouvrir: n_in_data_vc.txt (sans fallback)
	file_path = Path(root_dir) / f"{use_n_in_selector}_data_vc.txt"
	if not file_path.exists():
		raise FileNotFoundError(f"Fichier introuvable: {file_path}")

	taille = loader.count_non_empty_lines(file_path)
	if taille <= 0:
		pct = 0.0
		print(f"OK=0 | NOK=0 | taille=0 | Score={pct:.2f}%")
		return pct

	# Pour utiliser set_reso, on instancie un mon_reso cohérent avec struct_reso.
	n_layers = len(N_b)
	use_n_s = 1
	use_n_c = max(0, n_layers - use_n_s)
	r = mon_reso(
		n_in=use_n_in_features,
		n_c=use_n_c,
		n_s=use_n_s,
		N_b=N_b,
		biais=[0.0, 0.0],
		poids=[0.0, 0.0],
		X=[0, 1],
		D=[0, 1],
		seed=None,
	)

	bp = backpp(struct_reso_local, n_fct=use_n_fct, eta=use_eta)

	n_ok = 0
	n_nok = 0

	for n_ligne in range(1, taille + 1):
		D_list, X_list = loader.get_validation_sample_seq(
			file_path,
			n_ligne=n_ligne,
			n_in=use_n_in_features,
			nb_sorties=use_nb_sorties,
		)
		Dn_s_local = D_list[0]
		Xn_local = X_list[0]

		# Met à jour struct_reso avec W/B reçus + X/D de la ligne.
		struct_reso_local = r.set_reso(
			struct_reso_local,
			Xn_local,
			Wn_c_fixed,
			Bn_c_fixed,
			Dn_s_local,
		)
		bp.struct_reso = struct_reso_local

		# Résolution (forward)
		y = bp.forward(n_fct=use_n_fct)
		Dn_snew = service.fonction_max(y)

		if Dn_snew == Dn_s_local:
			n_ok += 1
		else:
			n_nok += 1

	# Sanity: Wn_c/Bn_c ne doivent pas avoir changé.
	if Wn_c_fixed != Wn_c or Bn_c_fixed != Bn_c:
		raise RuntimeError("Wn_c/Bn_c ont été modifiés pendant la validation (inattendu)")

	total = n_ok + n_nok
	pct = 0.0 if total == 0 else (100.0 * n_ok / total)
	print(f"OK={n_ok} | NOK={n_nok} | taille={taille} | Score={pct:.2f}%")
	return pct


# ==================== main =========================
def main() -> None:
	"""Démarre l'application (interface + actions).

	- Charge `parametres.txt`.
	- Instancie l'interface.
	- Branche les callbacks (ajout/suppression/exécution).
	"""
	clear_console()
	from . import interface, service
	import tkinter.messagebox as messagebox

	parametres_text = service.read_parametres_text()

	def on_set_config(payload: dict) -> tuple[bool, str]:
		# Met à jour les variables "courantes" côté lanceur.
		if isinstance(payload.get("Wn_c"), list):
			STATE.Wn_c = payload.get("Wn_c")  # type: ignore[assignment]
		if isinstance(payload.get("Bn_c"), list):
			STATE.Bn_c = payload.get("Bn_c")  # type: ignore[assignment]

		formatted_line = str(payload.get("formatted_line", "")).strip()
		ok, msg = service.add_parametres_line(formatted_line)
		if ok:
			# Rafraîchit le tableau dans l'interface après écriture.
			app.load_parametres_text(service.read_parametres_text())
			try:
				app.after(80, app._select_last_non_empty_table_row)
			except Exception:
				pass
		return ok, msg

	def on_delete_config(payload: dict) -> tuple[bool, str]:
		raw_line = str(payload.get("raw_line", "")).strip()
		if not raw_line:
			return False, "Ligne invalide"
		confirm = messagebox.askyesno(
			"Suppression",
			"Confirmer la suppression de la ligne sélectionnée dans parametres.txt ?",
		)
		if not confirm:
			return False, "Suppression annulée"
		ok, msg = service.delete_parametres_line(raw_line)
		if ok:
			app.load_parametres_text(service.read_parametres_text())
			try:
				app.after(80, app._select_last_non_empty_table_row)
			except Exception:
				pass
			messagebox.showinfo("Suppression", "Ligne supprimée")
		return ok, msg

	def on_execute(payload: dict) -> tuple[bool, str]:
		ok, msg = execute_payload(payload)
		# Exigence: après Apprentissage, si une config est ajoutée, rafraîchir le tableau.
		if str(payload.get("mode") or "").strip() == "Apprentissage":
			# 1) Met à jour le champ "Score obtenu"
			try:
				app._set_entry_text("score_obtenu", app._format_score_percent(STATE.score_ob))
			except Exception:
				pass
			# 1b) Met à jour i (iterations) avec la taille du fichier
			try:
				app._set_entry_text("iterations", str(int(STATE.i)))
			except Exception:
				pass
			# 2) Si une config est ajoutée, rafraîchir le tableau
			if ok:
				try:
					app.load_parametres_text(service.read_parametres_text())
					app.after(80, app._select_last_non_empty_table_row)
				except Exception:
					pass
		# Exigence option 2: met à jour l'interface avec le % de réussite.
		if str(payload.get("mode") or "").strip() == "Test général":
			try:
				app._set_entry_text("score_obtenu", app._format_score_percent(STATE.score_ob))
			except Exception:
				pass
		return ok, msg

	app = interface.HMIApp(
		parametres_text=parametres_text,
		startup_defaults=STATE.startup_defaults(),
		on_set_config=on_set_config,
		on_delete_config=on_delete_config,
		on_execute=on_execute,
	)
	app.mainloop()


if __name__ == "__main__":
	# Ce module utilise des imports relatifs ("from . import ...").
	# En exécution directe (python lab1/lanceur.py), Python ne définit pas le package.
	# On configure le contexte package pour permettre les imports relatifs.
	if __package__ in (None, ""):
		# Ajoute le dossier parent dans le chemin d'import pour que `lab1` soit importable.
		# (N'affecte pas le fonctionnement en mode module: `python -m lab1.lanceur`.)
		parent_dir = str(Path(__file__).resolve().parent.parent)
		if parent_dir not in sys.path:
			sys.path.insert(0, parent_dir)
		__package__ = "lab1"
	main()
