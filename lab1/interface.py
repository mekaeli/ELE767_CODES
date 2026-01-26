
"""Interface HMI (Projet 1) avec CustomTkinter.

But:
	Créer une interface graphique proche de l'image fournie:
	  - sélection de mode (Généralisation / Validation / Apprentissage)
	  - champs de configuration (entrées, couches, marges, etc.)
	  - boutons Get config / Set config
	  - flèches (haut/bas)
	  - bouton Exécuter

Notes:
	- Dépendance: customtkinter
		pip install customtkinter
	- Ici on construit surtout l'UI. Les actions Get/Set/Exécuter sont fournies
	  sous forme de fonctions simples (faciles à brancher au reste du projet).
"""

from __future__ import annotations

import json
import importlib
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional


# ==================== Import CustomTkinter =========================
try:
	import customtkinter as ctk
except ModuleNotFoundError as exc:  # pragma: no cover
	raise ModuleNotFoundError(
		"customtkinter n'est pas installé. Lance: pip install customtkinter"
	) from exc


# ==================== Constantes UI =========================
_APP_TITLE = "Projet 1 — HMI"
_APP_SIZE = "860x560"

# Paramètres demandés
appearance_mode: str = "dark"
color_theme: str = "blue"
root_bg_color: str = "#0f172a"

_BG = root_bg_color
_TEXT = "#cbd5e1"  # blanc moins vif
_BORDER = "#334155"  # bordures non blanches (slate)
_BLUE = "#3B8ED0"
_GREEN = "#2E9E2E"

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config,json")


# ==================== FieldSpec =========================
@dataclass(frozen=True)
class FieldSpec:
	"""Description d'un champ (label + plage de valeurs)."""

	key: str
	title: str
	hint: str
	default_value: str


# ==================== _safe_int =========================
def _safe_int(value: str, default: int) -> int:
	"""Convertit en int; si impossible, retourne default."""

	try:
		return int(value.strip())
	except Exception:
		return default


# ==================== _safe_float =========================
def _safe_float(value: str, default: float) -> float:
	"""Convertit en float; si impossible, retourne default."""

	try:
		return float(value.strip().replace(",", "."))
	except Exception:
		return default


# ==================== _try_float =========================
def _try_float(value: str) -> Optional[float]:
	"""Retourne un float ou None si vide/invalide."""

	v = value.strip()
	if not v:
		return None
	try:
		return float(v.replace(",", "."))
	except Exception:
		return None


# ==================== _mse =========================
def _mse(d, y) -> float:
	"""Mean Squared Error entre deux vecteurs."""

	if not d or not y:
		return 0.0
	n = min(len(d), len(y))
	if n == 0:
		return 0.0
	return sum((float(d[i]) - float(y[i])) ** 2 for i in range(n)) / float(n)


# ==================== _resolve_learning_api =========================
def _resolve_learning_api():
	"""Retourne (apprendre_une_iteration, generer_reseau_dynamique).

	Essaye d'abord `projet1`, puis `learn`, et s'assure que le dossier courant
	de ce fichier est dans `sys.path` pour éviter les problèmes de chemin.
	"""

	this_dir = os.path.dirname(__file__)
	if this_dir and this_dir not in sys.path:
		sys.path.insert(0, this_dir)

	last_err: Optional[Exception] = None
	for module_name in ("projet1", "learn"):
		try:
			module = importlib.import_module(module_name)
			apprendre = getattr(module, "apprendre_une_iteration")
			generer = getattr(module, "generer_reseau_dynamique")
			return apprendre, generer
		except (ModuleNotFoundError, ImportError, AttributeError) as exc:
			last_err = exc
			continue

	raise ImportError(
		"Impossible de trouver l'API d'apprentissage (apprendre_une_iteration / "
		"generer_reseau_dynamique) dans projet1.py ou learn.py."
	) from last_err


# ==================== ConfigField =========================
class ConfigField(ctk.CTkFrame):
	"""Petit widget: rond (case) + entrée + texte de plage."""

	# ==================== __init__ =========================
	def __init__(self, master, spec: FieldSpec):
		super().__init__(master, fg_color=_BG)
		self.spec = spec

		self.label = ctk.CTkLabel(self, text=spec.title, text_color=_TEXT)
		self.label.grid(row=0, column=0, sticky="w")

		# Petit rond (ressemble au cercle de l'image)
		self.enabled_var = ctk.BooleanVar(value=True)
		self.enabled = ctk.CTkCheckBox(
			self,
			text="",
			variable=self.enabled_var,
			width=24,
			checkbox_width=22,
			checkbox_height=22,
			corner_radius=11,
			border_width=2,
			fg_color=_BLUE,
			border_color=_BORDER,
		)
		self.enabled.grid(row=1, column=0, pady=(6, 6))

		self.entry = ctk.CTkEntry(
			self,
			width=70,
			height=34,
			border_width=2,
			border_color=_BORDER,
			fg_color=_BG,
			text_color=_TEXT,
		)
		self.entry.grid(row=2, column=0)
		self.entry.insert(0, spec.default_value)

		self.hint = ctk.CTkLabel(self, text=spec.hint, text_color=_TEXT)
		self.hint.grid(row=3, column=0, pady=(6, 0))

	# ==================== get_value =========================
	def get_value(self) -> str:
		"""Retourne le texte de l'entrée (même si invalide)."""

		return self.entry.get().strip()

	# ==================== set_value =========================
	def set_value(self, value: str) -> None:
		"""Écrit une valeur dans l'entrée."""

		self.entry.delete(0, "end")
		self.entry.insert(0, str(value))


# ==================== Projet1HMI =========================
class Projet1HMI(ctk.CTk):
	"""Fenêtre principale."""

	# ==================== __init__ =========================
	def __init__(self):
		super().__init__()

		ctk.set_appearance_mode(appearance_mode)
		ctk.set_default_color_theme(color_theme)

		self.title(_APP_TITLE)
		self.geometry(_APP_SIZE)
		self.minsize(860, 560)
		self.resizable(False, False)
		self.configure(fg_color=_BG)

		# Cadre avec une bordure noire (comme l'image)
		self.root_frame = ctk.CTkFrame(self, fg_color=_BG, border_width=2, border_color=_BORDER, corner_radius=0)
		self.root_frame.pack(fill="both", expand=True, padx=12, pady=12)

		self._build_top_mode()
		self._build_config_grid()
		self._build_bottom_bar()

	# ==================== _build_top_mode =========================
	def _build_top_mode(self) -> None:
		"""Construit la zone du haut: radios + texte "Mode sélectionné"."""

		top = ctk.CTkFrame(self.root_frame, fg_color=_BG)
		top.pack(fill="x", padx=18, pady=(18, 6))

		self.mode_var = ctk.StringVar(value="Généralisation")

		radio_frame = ctk.CTkFrame(top, fg_color=_BG)
		radio_frame.pack(anchor="w")

		for label in ("Généralisation", "Validation", "Apprentissage"):
			rb = ctk.CTkRadioButton(
				radio_frame,
				text=label,
				value=label,
				variable=self.mode_var,
				text_color=_TEXT,
				fg_color=_BLUE,
				border_color=_BORDER,
				command=self._on_mode_change,
			)
			rb.pack(side="left", padx=(0, 30))

		self.mode_label = ctk.CTkLabel(
			top,
			text=f"Mode sélectionné : {self.mode_var.get()}",
			text_color=_TEXT,
			font=ctk.CTkFont(size=22, weight="normal"),
		)
		self.mode_label.pack(anchor="w", pady=(14, 6))

		self.source_label = ctk.CTkLabel(top, text="Fichier source", text_color=_TEXT)
		self.source_label.pack(anchor="center", pady=(2, 0))

	# ==================== _on_mode_change =========================
	def _on_mode_change(self) -> None:
		"""Met à jour le texte après changement de mode."""

		self.mode_label.configure(text=f"Mode sélectionné : {self.mode_var.get()}")

	# ==================== _build_config_grid =========================
	def _build_config_grid(self) -> None:
		"""Construit la grille principale des champs + boutons Get/Set + flèches."""

		grid = ctk.CTkFrame(self.root_frame, fg_color=_BG)
		grid.pack(fill="both", expand=True, padx=18, pady=(10, 6))

		# Row 1 (comme l'image)
		specs_row1 = [
			FieldSpec("nx", "Entrées", "4 ≤ x ≤ 128", "5"),
			FieldSpec("nsc1", "Sous couche 1", "4 ≤ x ≤ 128", "15"),
			FieldSpec("nsc2", "Sous couche 2", "4 ≤ x ≤ 128", "15"),
			FieldSpec("ncs", "couche de sortie", "4 ≤ x ≤ 128", "10"),
			FieldSpec("others", "Autres couches", "", ""),
		]

		# Row 2 (comme l'image)
		specs_row2 = [
			FieldSpec("nc", "Nb de couches", "3 ≤ x ≤ 10", "4"),
			FieldSpec("b_margin", "Marge biais", "1 ≤ x ≤ 5", ""),
			FieldSpec("w_margin", "Marge poids", "-0,1 ≤ x ≤ 0,1", ""),
			FieldSpec("todo1", "a venir", "4 ≤ x ≤ 128", ""),
			FieldSpec("todo2", "a venir", "", ""),
		]

		self.fields: Dict[str, ConfigField] = {}

		# ==================== add_row =========================
		def add_row(row_index: int, specs):
			row_frame = ctk.CTkFrame(grid, fg_color=_BG)
			row_frame.grid(row=row_index, column=0, sticky="w", pady=(10, 16))

			for col, spec in enumerate(specs):
				f = ConfigField(row_frame, spec)
				f.grid(row=0, column=col, padx=(0, 26))
				self.fields[spec.key] = f

		add_row(0, specs_row1)
		add_row(1, specs_row2)

		# Colonne droite: boutons Get/Set
		btn_frame = ctk.CTkFrame(grid, fg_color=_BG)
		btn_frame.grid(row=0, column=1, rowspan=2, sticky="ne", padx=(10, 0), pady=(48, 0))

		self.btn_get = ctk.CTkButton(
			btn_frame,
			text="Get config",
			width=140,
			height=40,
			fg_color=_BLUE,
			text_color="white",
			corner_radius=10,
			command=self.get_config,
		)
		self.btn_get.pack(pady=(0, 10))

		self.btn_set = ctk.CTkButton(
			btn_frame,
			text="Set config",
			width=140,
			height=40,
			fg_color=_BLUE,
			text_color="white",
			corner_radius=10,
			command=self.set_config,
		)
		self.btn_set.pack()

		# Flèches (haut/bas) à droite comme l'image
		arrows = ctk.CTkFrame(grid, fg_color=_BG)
		arrows.grid(row=1, column=1, sticky="se", padx=(10, 0), pady=(0, 20))

		self.btn_up = ctk.CTkButton(
			arrows,
			text="▲",
			width=44,
			height=34,
			fg_color="transparent",
			hover=False,
			text_color=_TEXT,
			border_width=0,
			command=lambda: None,
		)
		self.btn_up.pack(pady=(0, 0))

		self.btn_down = ctk.CTkButton(
			arrows,
			text="▼",
			width=44,
			height=34,
			fg_color="transparent",
			hover=False,
			text_color=_TEXT,
			border_width=0,
			command=lambda: None,
		)
		self.btn_down.pack(pady=(0, 0))

	# ==================== _build_bottom_bar =========================
	def _build_bottom_bar(self) -> None:
		"""Construit la barre du bas: option menu + scores + bouton Exécuter."""

		bottom = ctk.CTkFrame(self.root_frame, fg_color=_BG)
		bottom.pack(fill="x", padx=18, pady=(0, 18))

		# Fonction d'activation
		act_frame = ctk.CTkFrame(bottom, fg_color=_BG)
		act_frame.pack(side="left", padx=(0, 16))
		ctk.CTkLabel(act_frame, text="Fonction act.", text_color=_TEXT).pack(anchor="w")
		self.act_var = ctk.StringVar(value="sigmoïde")
		self.act_menu = ctk.CTkOptionMenu(
			act_frame,
			values=["sigmoïde", "tan", "tanh", "gelu"],
			variable=self.act_var,
			fg_color=_BG,
			button_color=_BORDER,
			button_hover_color=_BORDER,
			text_color=_TEXT,
			dropdown_fg_color=_BG,
			dropdown_text_color=_TEXT,
			dropdown_hover_color="#1f2937",
			corner_radius=10,
			width=180,
		)
		self.act_menu.pack()

		# Score attendu
		score_frame = ctk.CTkFrame(bottom, fg_color=_BG)
		score_frame.pack(side="left", padx=(0, 16))
		ctk.CTkLabel(score_frame, text="Score attendu", text_color=_TEXT).pack(anchor="w")
		self.score_expected = ctk.CTkEntry(
			score_frame,
			width=220,
			height=36,
			border_width=2,
			border_color=_BORDER,
			fg_color=_BG,
			text_color=_TEXT,
		)
		self.score_expected.pack()

		# Score obtenu
		score2_frame = ctk.CTkFrame(bottom, fg_color=_BG)
		score2_frame.pack(side="left", padx=(0, 16))
		ctk.CTkLabel(score2_frame, text="Score obtenu", text_color=_TEXT).pack(anchor="w")
		self.score_obtained = ctk.CTkEntry(
			score2_frame,
			width=220,
			height=36,
			border_width=2,
			border_color=_BORDER,
			fg_color=_BG,
			text_color=_TEXT,
		)
		self.score_obtained.pack()

		# Bouton Exécuter
		self.btn_run = ctk.CTkButton(
			bottom,
			text="Exécuter",
			width=160,
			height=40,
			fg_color=_GREEN,
			hover_color="#258525",
			text_color="white",
			corner_radius=10,
			command=self.execute,
		)
		self.btn_run.pack(side="right")

	# ==================== _activation_to_n_fct =========================
	def _activation_to_n_fct(self) -> int:
		"""Convertit le nom de la fonction d'activation en n_fct (1..4)."""

		mapping = {"sigmoïde": 1, "tan": 2, "tanh": 3, "gelu": 4}
		return mapping.get(self.act_var.get(), 1)

	# ==================== get_config =========================
	def get_config(self) -> None:
		"""Charge une configuration depuis lab1/config.json et remplit les champs."""

		if not os.path.exists(_CONFIG_PATH):
			return

		try:
			with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
				data = json.load(f)
		except Exception:
			return

		for key, field in self.fields.items():
			if key in data:
				field.set_value(str(data[key]))

		if "mode" in data:
			self.mode_var.set(str(data["mode"]))
			self._on_mode_change()
		if "activation" in data:
			self.act_var.set(str(data["activation"]))

	# ==================== set_config =========================
	def set_config(self) -> None:
		"""Sauvegarde les valeurs de l'UI dans lab1/config.json."""

		data: Dict[str, object] = {
			"mode": self.mode_var.get(),
			"activation": self.act_var.get(),
		}

		for key, field in self.fields.items():
			data[key] = field.get_value()

		try:
			with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
				json.dump(data, f, indent=2, ensure_ascii=False)
		except Exception:
			return

	# ==================== execute =========================
	def execute(self) -> None:
		"""Action du bouton Exécuter.

		Version simple:
			- lit quelques champs
			- calcule un n_fct
			- (à brancher) lance le code du projet
		"""

		# Import local pour éviter tout souci de chemin au démarrage.
		# Le projet peut avoir le code dans learn.py ou dans projet1.py.
		apprendre_une_iteration, generer_reseau_dynamique = _resolve_learning_api()

		nx = _safe_int(self.fields["nx"].get_value(), 5)
		nc = _safe_int(self.fields["nc"].get_value(), 4)
		nsc1 = _safe_int(self.fields["nsc1"].get_value(), 15)
		nsc2 = _safe_int(self.fields["nsc2"].get_value(), 15)
		ncs = _safe_int(self.fields["ncs"].get_value(), 10)
		n_fct = self._activation_to_n_fct()

		# Marges (optionnelles)
		b_margin = _try_float(self.fields["b_margin"].get_value())
		w_margin = _try_float(self.fields["w_margin"].get_value())

		# Interprétation simple des marges (cohérente avec les bornes par défaut de learn.py)
		b_range = (1.0, 5.0) if b_margin is None else (1.0, float(max(1.0, b_margin)))
		w_range = (-0.1, 0.1)
		if w_margin is not None:
			m = abs(float(w_margin))
			w_range = (-m, m)

		reseau = generer_reseau_dynamique(
			nx=nx,
			nc=nc,
			nsc1=nsc1,
			nsc2=nsc2,
			ncs=ncs,
			n_fct=n_fct,
			b=b_range,
			w=w_range,
			eta=0.1,
			seed=123,
		)

		_, _, y = apprendre_une_iteration(reseau)
		score = _mse(reseau.vecD, y)

		# Affichage du score obtenu
		self.score_obtained.delete(0, "end")
		self.score_obtained.insert(0, f"MSE(y,D)={score:.6g}")

		# Si l'utilisateur a écrit un score attendu, on montre aussi l'écart
		expected = _try_float(self.score_expected.get())
		if expected is not None:
			diff = float(score) - float(expected)
			self.score_obtained.insert("end", f" ; Δ={diff:+.6g}")


# ==================== main =========================
def main() -> None:
	"""Point d'entrée: lance l'interface."""

	app = Projet1HMI()
	app.mainloop()


if __name__ == "__main__":
	main()

