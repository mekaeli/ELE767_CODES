from __future__ import annotations

from dataclasses import dataclass, field
import os
import unicodedata


ACTIVATION_TO_N_FCT: dict[str, int] = {
	"sigmoide": 1,
	"tan": 2,
	"tanh": 3,
	"gelu": 4,
}


def _normalize_activation(value: object) -> str:
	"""Normalise un libellé d'activation (lower + sans accents)."""
	s = str(value or "").strip().lower()
	# Décompose les accents puis les enlève.
	s = unicodedata.normalize("NFKD", s)
	s = "".join(ch for ch in s if not unicodedata.combining(ch))
	return s


def _env_int(name: str, default: int) -> int:
	try:
		return int(os.environ.get(name, str(default)).strip())
	except Exception:
		return default


def _env_float(name: str, default: float) -> float:
	try:
		return float(os.environ.get(name, str(default)).strip().replace(",", "."))
	except Exception:
		return default


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
	
	# 1) Entrées
	x1: float = 1.0
	x2: float = 0.0

	# 2) Poids + biais (couche 1 : 2 -> 2)
	w11_1: float = 3.0
	w12_1: float = 4.0
	w21_1: float = 6.0
	w22_1: float = 5.0
	b1_1: float = 1.0
	b2_1: float = 0.0

	# 3) Poids + biais (couche 2 : 2 -> 1)
	w11_2: float = 2.0
	w21_2: float = 4.0
	b1_2: float = -3.92

	# 4) Sortie désirée
	d1: float = 1.0

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
		W2_aplatie = [self.w11_2, self.w21_2]
		return [W1_aplatie, W2_aplatie]

	@property
	def Bn_c(self) -> list[list[float]]:
		b1 = [self.b1_1, self.b2_1]
		b2 = [self.b1_2]
		return [b1, b2]

	@property
	def Dn_s(self) -> list[float]:
		return [self.d1]


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
	i: int = 4
	k: int = 40
	test_unitaire: bool = True

	# UI / suivi
	score_ob: str = "0%"
	n_ok: int = 0
	n_nok: int = 0
	N_fich: int = 0

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
			# Mode UI au démarrage (un seul choix parmi 4)
			"mode": ("Test unitaire" if self.test_unitaire else "Généralisation"),
		}


STATE = LanceurState()


def _init_manual_entry_defaults() -> None:
	"""Initialise les valeurs d'exemple (et l'état) depuis l'environnement."""
	ex = varEntryManuel.from_env()
	STATE.manual_entry = ex
	STATE.Wn_c = ex.Wn_c
	STATE.Bn_c = ex.Bn_c
	STATE.n_fct = ex.n_fct
	STATE.test_unitaire = bool(ex.test_unitaire)


_init_manual_entry_defaults()


def execute_payload(payload: dict) -> tuple[bool, str]:
	"""Chemin d'exécution "Exécuter" (testable hors-GUI).

	Valide côté lanceur, met à jour l'état, puis:
	- crée/initialise `struct_reso` via `reseau.py`
	- exécute l'option demandée (test unitaire ou mode)
	"""
	# Import robuste: fonctionne depuis la racine ou depuis lab1/
	try:
		from lab1.reseau import mon_reso
	except Exception:
		from reseau import mon_reso  # type: ignore

	# Import robuste: backprop (affichage console)
	try:
		from lab1.backpp import backpp
	except Exception:
		from backpp import backpp  # type: ignore

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
				show_params=False,
				show_equations=True,
				show_resume=True,
				precision=6,
			)
		except Exception as exc:
			return False, f"test_unitaire: échec résolution/affichage: {exc}"

		mode = str(payload.get("mode") or "").strip()
		return True, f"OK (mode={mode}, test_unitaire=True)"

	# 3) Options 2-4: modes (à venir)
	if mode in {"Généralisation", "Validation", "Apprentissage"}:
		return False, f"Mode '{mode}' à venir (test_unitaire désactivé)"
	return False, f"Mode invalide: '{mode}'"


def main() -> None:
	clear_console()
	# Imports robustes: fonctionne depuis la racine ou depuis lab1/
	try:
		from lab1 import interface, service
	except Exception:
		import interface  # type: ignore
		import service  # type: ignore
		import tkinter.messagebox as messagebox  # type: ignore
	else:
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
		return execute_payload(payload)

	app = interface.HMIApp(
		parametres_text=parametres_text,
		startup_defaults=STATE.startup_defaults(),
		on_set_config=on_set_config,
		on_delete_config=on_delete_config,
		on_execute=on_execute,
	)
	app.mainloop()


if __name__ == "__main__":
	main()
