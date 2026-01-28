from __future__ import annotations

"""Interface HMI du laboratoire 1 (CustomTkinter).

- Affiche une interface sombre (thème) pour configurer un petit réseau de neurones.
- Valide les champs (plages min/max) avant exécution.
- Peut sauvegarder/charger la configuration dans `lab1/config,json`.
- Lance `neurone/c2p59.py` avec des variables d'environnement (N_FCT, ETA).
"""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path
import tkinter.messagebox as messagebox
from tkinter import filedialog

import customtkinter as ctk


# ========================
# Couleurs (inspirées de l'image: fond sombre + accent violet)
# ========================
COLOR_BG = "#141418"            # fond principal (gris très sombre)
COLOR_PANEL = "#1C1C22"         # panneaux
COLOR_ENTRY = "#16161B"         # champs / dropdown
COLOR_BORDER = "#2D2D36"        # contours

# Couleur de texte unifiée (à appliquer partout sauf boutons et radios)
# NOTE: Mets ici le hex exact si tu veux une nuance différente.
COLOR_TEXT_UNIFIED = "#DADAE6"

# Déclinaisons (conservées pour compatibilité dans le code)
COLOR_TEXT_TITLE = COLOR_TEXT_UNIFIED
COLOR_TEXT = COLOR_TEXT_UNIFIED
COLOR_TEXT_MUTED = COLOR_TEXT_UNIFIED
COLOR_TEXT_HINT = COLOR_TEXT_UNIFIED

# Exceptions demandées: boutons et radios gardent leurs couleurs.
COLOR_TEXT_RADIO = "#DADAE6"  # ancien COLOR_TEXT (avant unification)

# Accent (violet)
COLOR_ACCENT = "#7B6CFF"        # violet bouton
COLOR_ACCENT_HOVER = "#6B5BF3"  # violet plus dense
COLOR_ON_ACCENT = "#0C0C10"     # texte sur accent (presque noir)

CONTENT_WIDTH = 700
ENTRY_W = 90
SCORE_W = 120
ACT_W = 170
CELL_W = 100
CELL_H = 112
BTN_W = 120
BTN_H = 30


# ==================== _enable_windows_dpi_awareness =========================
def _enable_windows_dpi_awareness() -> None:
	"""Active le DPI-aware sous Windows.

	But: éviter que Windows "étire" l'application en bitmap (texte flou).
	Aucun impact sur les tailles définies dans le code.
	"""
	if sys.platform != "win32":
		return
	try:
		# 2 = PROCESS_PER_MONITOR_DPI_AWARE
		ctypes.windll.shcore.SetProcessDpiAwareness(2)
	except Exception:
		try:
			ctypes.windll.user32.SetProcessDPIAware()
		except Exception:
			pass


FONT_FAMILY = "Segoe UI"


# ========================
# Typographie (harmonisée)
# ========================
FONT_TITLE: ctk.CTkFont | None = None
FONT_SECTION: ctk.CTkFont | None = None
FONT_LABEL: ctk.CTkFont | None = None
FONT_HINT: ctk.CTkFont | None = None
FONT_ENTRY: ctk.CTkFont | None = None
FONT_BUTTON: ctk.CTkFont | None = None
FONT_RADIO: ctk.CTkFont | None = None


def _init_fonts() -> None:
	"""Initialise les polices après création de la racine Tk.

	Important: `ctk.CTkFont(...)` échoue si appelé avant la création de
	la fenêtre (pas de default root). On initialise donc à la demande.
	"""
	global FONT_TITLE, FONT_SECTION, FONT_LABEL, FONT_HINT, FONT_ENTRY, FONT_BUTTON, FONT_RADIO
	if FONT_TITLE is not None:
		return

	FONT_TITLE = ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold")
	FONT_SECTION = ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold")
	FONT_LABEL = ctk.CTkFont(family=FONT_FAMILY, size=12)
	FONT_HINT = ctk.CTkFont(family=FONT_FAMILY, size=11)
	FONT_ENTRY = ctk.CTkFont(family=FONT_FAMILY, size=12)
	FONT_BUTTON = ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold")
	FONT_RADIO = ctk.CTkFont(family=FONT_FAMILY, size=12)


ctk.set_appearance_mode("Dark")
_enable_windows_dpi_awareness()


CONFIG_PATH = Path(__file__).with_name("config,json")


class HMIApp(ctk.CTk):
	"""Fenêtre principale de l'interface."""

	# ==================== __init__ =========================
	def __init__(self) -> None:
		"""Crée la fenêtre et construit tous les widgets."""
		super().__init__()
		_init_fonts()
		self.title("HMI - Réseaux de neurones")
		self.geometry("760x760")
		self.minsize(760, 760)
		self.resizable(False, False)
		self.configure(fg_color=COLOR_BG)

		self._build_ui()

	# ==================== _build_ui =========================
	def _build_ui(self) -> None:
		"""Construit toute l'interface (layout + widgets)."""
		root_frame = ctk.CTkFrame(
			self,
			fg_color=COLOR_PANEL,
			border_color=COLOR_BORDER,
			border_width=2,
			corner_radius=18,
		)
		root_frame.pack(fill="both", expand=True, padx=18, pady=18)

		root_frame.grid_columnconfigure(0, weight=1)
		# Répartir l'espace vertical: le tableau (bas) prend le reste.
		root_frame.grid_rowconfigure(0, weight=0)
		root_frame.grid_rowconfigure(1, weight=0)
		root_frame.grid_rowconfigure(2, weight=1)

		# ---------- En-tête ----------
		header_frame = ctk.CTkFrame(root_frame, fg_color="transparent")
		header_frame.grid(row=0, column=0, pady=(14, 4), sticky="ew")

		self.mode_var = ctk.StringVar(value="Généralisation")
		radio_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
		radio_frame.pack(pady=(2, 8))

		self._add_radio(radio_frame, "Généralisation").pack(side="left", padx=16)
		self._add_radio(radio_frame, "Validation").pack(side="left", padx=16)
		self._add_radio(radio_frame, "Apprentissage").pack(side="left", padx=16)

		self.mode_label = ctk.CTkLabel(
			header_frame,
			text="Mode sélectionné : Généralisation",
			text_color=COLOR_TEXT_TITLE,
			font=FONT_TITLE,
		)
		self.mode_label.pack(pady=(0, 10))

		file_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
		file_frame.pack(pady=(0, 4))

		ctk.CTkLabel(
			file_frame,
			text="Fichier source",
			text_color=COLOR_TEXT_MUTED,
			font=FONT_LABEL,
		).grid(row=0, column=0, columnspan=2, pady=(0, 6))

		self.file_entry = ctk.CTkEntry(
			file_frame,
			width=360,
			height=28,
			fg_color=COLOR_ENTRY,
			border_color=COLOR_BORDER,
			text_color=COLOR_TEXT,
			font=FONT_ENTRY,
		)
		self.file_entry.grid(row=1, column=0, padx=(0, 8))

		ctk.CTkButton(
			file_frame,
			text="...",
			width=34,
			height=28,
			fg_color=COLOR_ACCENT,
			hover_color=COLOR_ACCENT_HOVER,
			text_color=COLOR_ON_ACCENT,
			font=FONT_BUTTON,
			command=self._choose_file,
		).grid(row=1, column=1)

		# ---------- Zone paramètres ----------
		param_container = ctk.CTkFrame(
			root_frame,
			fg_color="transparent",
			border_color=COLOR_BORDER,
			border_width=2,
			corner_radius=16,
		)
		param_container.configure(width=CONTENT_WIDTH)
		param_container.grid(row=1, column=0, padx=16, pady=(10, 10), sticky="ew")

		param_container.grid_columnconfigure(0, weight=1, minsize=CONTENT_WIDTH - (BTN_W + 16))
		param_container.grid_columnconfigure(1, weight=0, minsize=BTN_W + 16)

		self.entries: dict[str, ctk.CTkEntry] = {}
		self.constraints: dict[str, tuple[float, float, str]] = {}

		# Titre du panneau
		ctk.CTkLabel(
			param_container,
			text="Paramètres",
			text_color=COLOR_TEXT_TITLE,
			font=FONT_SECTION,
		).grid(row=0, column=0, columnspan=2, pady=(10, 0))

		# Ligne activation + scores (dans le panneau)
		top_row = ctk.CTkFrame(param_container, fg_color="transparent")
		top_row.configure(width=CONTENT_WIDTH)
		top_row.grid(row=1, column=0, columnspan=2, padx=12, pady=(8, 6), sticky="ew")
		# Centrer le bloc (act + 2 scores) dans la frame
		top_row.grid_columnconfigure(0, weight=1)
		top_row.grid_columnconfigure(1, weight=0, minsize=ACT_W + 40)
		top_row.grid_columnconfigure(2, weight=0, minsize=SCORE_W + 40)
		top_row.grid_columnconfigure(3, weight=0, minsize=SCORE_W + 40)
		top_row.grid_columnconfigure(4, weight=1)

		ctk.CTkLabel(
			top_row,
			text="Fonction act.",
			text_color=COLOR_TEXT_MUTED,
			font=FONT_LABEL,
		).grid(row=0, column=1, padx=10, pady=(0, 4))

		ctk.CTkLabel(
			top_row,
			text="Score attendu",
			text_color=COLOR_TEXT_MUTED,
			font=FONT_LABEL,
		).grid(row=0, column=2, padx=10, pady=(0, 4))

		ctk.CTkLabel(
			top_row,
			text="Score obtenu",
			text_color=COLOR_TEXT_MUTED,
			font=FONT_LABEL,
		).grid(row=0, column=3, padx=10, pady=(0, 4))

		self.act_var = ctk.StringVar(value="sigmoïde")
		self.act_menu = ctk.CTkOptionMenu(
			top_row,
			values=["sigmoïde", "tan", "tanh", "gelu"],
			variable=self.act_var,
			fg_color=COLOR_ENTRY,
			button_color=COLOR_ENTRY,
			button_hover_color=COLOR_BORDER,
			text_color=COLOR_TEXT,
			dropdown_fg_color=COLOR_ENTRY,
			dropdown_text_color=COLOR_TEXT,
			dropdown_hover_color=COLOR_BORDER,
			font=FONT_ENTRY,
			dropdown_font=FONT_ENTRY,
			width=ACT_W,
			height=30,
		)
		self.act_menu.grid(row=1, column=1, padx=10)

		score_attendu = ctk.CTkEntry(
			top_row,
			width=SCORE_W,
			height=28,
			fg_color=COLOR_ENTRY,
			border_color=COLOR_BORDER,
			text_color=COLOR_TEXT,
			font=FONT_ENTRY,
		)
		score_attendu.grid(row=1, column=2, padx=10)

		score_obtenu = ctk.CTkEntry(
			top_row,
			width=SCORE_W,
			height=28,
			fg_color=COLOR_ENTRY,
			border_color=COLOR_BORDER,
			text_color=COLOR_TEXT,
			font=FONT_ENTRY,
		)
		score_obtenu.grid(row=1, column=3, padx=10)

		self.entries["score_attendu"] = score_attendu
		self.constraints["score_attendu"] = (0, 1, "float")
		self.entries["score_obtenu"] = score_obtenu
		self.constraints["score_obtenu"] = (0, 1, "float")

		# Grille principale des paramètres
		param_grid = ctk.CTkFrame(param_container, fg_color="transparent")
		# La grille doit être centrée par rapport à TOUTE la frame (incluant la zone des boutons)
		param_grid.grid(row=2, column=0, columnspan=2, padx=12, pady=(8, 6), sticky="ew")
		# Centrer les 5 cellules avec 2 colonnes "spacer" (gauche/droite)
		param_grid.grid_columnconfigure(0, weight=1)
		for col in range(1, 6):
			param_grid.grid_columnconfigure(col, weight=0, uniform="param_cols", minsize=CELL_W)
		param_grid.grid_columnconfigure(6, weight=1)

		self._add_field(param_grid, "nb d’entrées", "1 ≤ Xn ≤ 480", 0, 1, "nb_entrees", 1, 480, "int")
		self._add_field(param_grid, "nb couches\nCachées", "1 ≤ Cn ≤ 10", 0, 2, "nb_couches", 1, 10, "int")
		self._add_field(param_grid, "nb neurones/\nCou. cachées", "1 ≤ Nn ≤ 10", 0, 3, "nb_neurones", 1, 10, "int")
		self._add_field(param_grid, "nb de sorties", "1 ≤ Sn ≤ 10", 0, 4, "nb_sorties", 1, 10, "int")
		self._add_field(param_grid, "taux\nd’apprentissage", "0.1 ≤ η ≤ 1", 0, 5, "taux_apprentissage", 0.1, 1, "float")

		# Ligne basse (min/max + itérations/époques + boutons), centrée sur la frame
		bottom_row = ctk.CTkFrame(param_container, fg_color="transparent")
		bottom_row.grid(row=3, column=0, columnspan=2, padx=12, pady=(2, 12), sticky="ew")
		bottom_row.grid_columnconfigure(0, weight=1)
		bottom_row.grid_columnconfigure(1, weight=0)
		bottom_row.grid_columnconfigure(2, weight=0)
		bottom_row.grid_columnconfigure(3, weight=1)

		lower_grid = ctk.CTkFrame(bottom_row, fg_color="transparent")
		lower_grid.grid(row=0, column=1, sticky="w", padx=(0, 18))
		lower_grid.grid_columnconfigure(0, weight=0, minsize=34)
		lower_grid.grid_columnconfigure(1, weight=0, minsize=CELL_W)
		lower_grid.grid_columnconfigure(2, weight=0, minsize=CELL_W)
		lower_grid.grid_columnconfigure(3, weight=0, minsize=96)
		lower_grid.grid_columnconfigure(4, weight=0, minsize=CELL_W)

		ctk.CTkLabel(
			lower_grid,
			text="min",
			text_color=COLOR_TEXT_MUTED,
			font=FONT_LABEL,
		).grid(row=0, column=0, padx=(0, 8), pady=(6, 2), sticky="w")

		ctk.CTkLabel(
			lower_grid,
			text="max",
			text_color=COLOR_TEXT_MUTED,
			font=FONT_LABEL,
		).grid(row=1, column=0, padx=(0, 8), pady=(6, 2), sticky="w")

		bias_min = ctk.CTkEntry(
			lower_grid,
			width=ENTRY_W,
			height=28,
			fg_color=COLOR_ENTRY,
			border_color=COLOR_BORDER,
			text_color=COLOR_TEXT,
			font=FONT_ENTRY,
		)
		bias_min.grid(row=0, column=1, padx=(6, 10), pady=(4, 2), sticky="w")

		bias_max = ctk.CTkEntry(
			lower_grid,
			width=ENTRY_W,
			height=28,
			fg_color=COLOR_ENTRY,
			border_color=COLOR_BORDER,
			text_color=COLOR_TEXT,
			font=FONT_ENTRY,
		)
		bias_max.grid(row=1, column=1, padx=(6, 10), pady=(4, 2), sticky="w")

		poids_min = ctk.CTkEntry(
			lower_grid,
			width=ENTRY_W,
			height=28,
			fg_color=COLOR_ENTRY,
			border_color=COLOR_BORDER,
			text_color=COLOR_TEXT,
			font=FONT_ENTRY,
		)
		poids_min.grid(row=0, column=2, padx=(10, 10), pady=(4, 2), sticky="w")

		poids_max = ctk.CTkEntry(
			lower_grid,
			width=ENTRY_W,
			height=28,
			fg_color=COLOR_ENTRY,
			border_color=COLOR_BORDER,
			text_color=COLOR_TEXT,
			font=FONT_ENTRY,
		)
		poids_max.grid(row=1, column=2, padx=(10, 10), pady=(4, 2), sticky="w")

		ctk.CTkLabel(
			lower_grid,
			text="i itérations",
			text_color=COLOR_TEXT_MUTED,
			font=FONT_LABEL,
		).grid(row=0, column=3, padx=(12, 6), pady=(6, 2), sticky="e")

		iter_entry = ctk.CTkEntry(
			lower_grid,
			width=ENTRY_W,
			height=28,
			fg_color=COLOR_ENTRY,
			border_color=COLOR_BORDER,
			text_color=COLOR_TEXT,
			font=FONT_ENTRY,
		)
		iter_entry.grid(row=0, column=4, padx=(0, 6), pady=(4, 2), sticky="w")

		ctk.CTkLabel(
			lower_grid,
			text="k époques",
			text_color=COLOR_TEXT_MUTED,
			font=FONT_LABEL,
		).grid(row=1, column=3, padx=(12, 6), pady=(6, 2), sticky="e")

		epoch_entry = ctk.CTkEntry(
			lower_grid,
			width=ENTRY_W,
			height=28,
			fg_color=COLOR_ENTRY,
			border_color=COLOR_BORDER,
			text_color=COLOR_TEXT,
			font=FONT_ENTRY,
		)
		epoch_entry.grid(row=1, column=4, padx=(0, 6), pady=(4, 2), sticky="w")

		ctk.CTkLabel(
			lower_grid,
			text="1 ≤ biais ≤ 5",
			text_color=COLOR_TEXT_HINT,
			font=FONT_HINT,
		).grid(row=2, column=1, padx=(6, 0), pady=(4, 0), sticky="w")

		ctk.CTkLabel(
			lower_grid,
			text="-0.1 ≤ Wn ≤ 0.1",
			text_color=COLOR_TEXT_HINT,
			font=FONT_HINT,
		).grid(row=2, column=2, padx=(10, 0), pady=(4, 0), sticky="w")

		self.entries["biais_min"] = bias_min
		self.constraints["biais_min"] = (1, 5, "float")
		self.entries["biais_max"] = bias_max
		self.constraints["biais_max"] = (1, 5, "float")
		self.entries["poids_min"] = poids_min
		self.constraints["poids_min"] = (-0.1, 0.1, "float")
		self.entries["poids_max"] = poids_max
		self.constraints["poids_max"] = (-0.1, 0.1, "float")
		self.entries["iterations"] = iter_entry
		self.constraints["iterations"] = (1, 1000000, "int")
		self.entries["k_epoques"] = epoch_entry
		self.constraints["k_epoques"] = (1, 1000000, "int")

		# Colonne droite (boutons)
		right_col = ctk.CTkFrame(bottom_row, fg_color="transparent")
		right_col.grid(row=0, column=2, padx=(18, 0), pady=(4, 0), sticky="n")

		ctk.CTkButton(
			right_col,
			text="Get config",
			width=BTN_W,
			height=BTN_H,
			fg_color=COLOR_ACCENT,
			hover_color=COLOR_ACCENT_HOVER,
			text_color=COLOR_ON_ACCENT,
			font=FONT_BUTTON,
			command=self._on_get_config,
		).pack(pady=(8, 8))

		ctk.CTkButton(
			right_col,
			text="Set config",
			width=BTN_W,
			height=BTN_H,
			fg_color=COLOR_ACCENT,
			hover_color=COLOR_ACCENT_HOVER,
			text_color=COLOR_ON_ACCENT,
			font=FONT_BUTTON,
			command=self._on_set_config,
		).pack(pady=8)

		ctk.CTkButton(
			right_col,
			text="Exécuter",
			width=BTN_W,
			height=BTN_H,
			fg_color=COLOR_ACCENT,
			hover_color=COLOR_ACCENT_HOVER,
			text_color=COLOR_ON_ACCENT,
			font=FONT_BUTTON,
			command=self._on_execute,
		).pack(pady=8)

		# ---------- Tableau des paramètres ----------
		table_frame = ctk.CTkFrame(
			root_frame,
			fg_color="transparent",
			border_color=COLOR_BORDER,
			border_width=2,
			corner_radius=12,
		)
		table_frame.configure(width=CONTENT_WIDTH)
		table_frame.grid(row=2, column=0, padx=16, pady=(4, 14), sticky="nsew")
		table_frame.grid_columnconfigure(0, weight=1)
		table_frame.grid_rowconfigure(1, weight=1)

		ctk.CTkLabel(
			table_frame,
			text="Tableau des paramètres",
			text_color=COLOR_TEXT_TITLE,
			font=FONT_SECTION,
		).grid(row=0, column=0, pady=(6, 4))

		self.table_box = ctk.CTkTextbox(
			table_frame,
			width=CONTENT_WIDTH - 28,
			height=1,
			fg_color=COLOR_ENTRY,
			text_color=COLOR_TEXT,
			font=FONT_ENTRY,
			border_color=COLOR_BORDER,
			border_width=1,
		)
		self.table_box.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="nsew")
		self.table_box.configure(state="disabled")

	# ==================== _add_radio =========================
	def _add_radio(self, parent: ctk.CTkFrame, text: str) -> ctk.CTkRadioButton:
		"""Crée un bouton radio pour le choix du mode (Généralisation/Validation/Apprentissage)."""
		return ctk.CTkRadioButton(
			parent,
			text=text,
			variable=self.mode_var,
			value=text,
			command=self._update_mode_label,
			text_color=COLOR_TEXT_RADIO,
			fg_color=COLOR_ACCENT,
			hover_color=COLOR_ACCENT_HOVER,
			font=FONT_RADIO,
		)

	# ==================== _add_field =========================
	def _add_field(
		self,
		parent: ctk.CTkFrame,
		label: str,
		range_text: str,
		row: int,
		col: int,
		key: str,
		min_val: float,
		max_val: float,
		value_type: str,
		show_range: bool = True,
	) -> None:
		"""Ajoute une cellule de la grille (label + entry + indication de plage).

		- Enregistre l'entry dans `self.entries[key]`.
		- Enregistre la contrainte (min/max/type) dans `self.constraints[key]`.
		"""
		cell = ctk.CTkFrame(parent, fg_color="transparent", width=CELL_W, height=CELL_H)
		cell.grid(row=row, column=col, padx=6, pady=8, sticky="n")
		cell.grid_propagate(False)

		ctk.CTkLabel(
			cell,
			text=label,
			text_color=COLOR_TEXT_MUTED,
			font=FONT_LABEL,
		).pack(pady=(0, 6))

		entry = ctk.CTkEntry(
			cell,
			width=ENTRY_W,
			height=28,
			fg_color=COLOR_ENTRY,
			border_color=COLOR_BORDER,
			text_color=COLOR_TEXT,
			font=FONT_ENTRY,
		)
		entry.pack()

		if show_range and range_text:
			ctk.CTkLabel(
				cell,
				text=range_text,
				text_color=COLOR_TEXT_HINT,
				font=FONT_HINT,
			).pack(pady=(4, 0))

		self.entries[key] = entry
		self.constraints[key] = (min_val, max_val, value_type)

	# ==================== _update_mode_label =========================
	def _update_mode_label(self) -> None:
		"""Met à jour le texte "Mode sélectionné" selon le bouton radio choisi."""
		self.mode_label.configure(text=f"Mode sélectionné : {self.mode_var.get()}")

	# ==================== _on_get_config =========================
	def _on_get_config(self) -> None:
		"""Charge `config,json` et remplit les champs de l'interface."""
		if not CONFIG_PATH.exists():
			messagebox.showwarning("Config", f"Fichier introuvable: {CONFIG_PATH.name}")
			return
		try:
			data = json.loads(CONFIG_PATH.read_text(encoding="utf-8") or "{}")
		except Exception as exc:
			messagebox.showerror("Config", f"Config invalide: {exc}")
			return

		show_missing: list[str] = []
		for key, entry in self.entries.items():
			if key in data:
				entry.delete(0, "end")
				entry.insert(0, str(data[key]))
			else:
				show_missing.append(key)

		mode = data.get("mode")
		if isinstance(mode, str) and mode in {"Généralisation", "Validation", "Apprentissage"}:
			self.mode_var.set(mode)
			self._update_mode_label()

		act = data.get("activation")
		if isinstance(act, str) and act in {"sigmoïde", "tan", "tanh", "gelu"}:
			self.act_var.set(act)

		fichier = data.get("fichier")
		if isinstance(fichier, str):
			self.file_entry.delete(0, "end")
			self.file_entry.insert(0, fichier)

		# Pas de pop-up si tout va bien: on reste sobre.

	# ==================== _on_set_config =========================
	def _on_set_config(self) -> None:
		"""Valide les champs puis sauvegarde la configuration dans `config,json`."""
		try:
			values = self._validate_entries()
		except ValueError as exc:
			messagebox.showerror("Erreur", str(exc))
			return

		data: dict[str, object] = {
			"mode": self.mode_var.get(),
			"activation": self.act_var.get(),
			"fichier": self.file_entry.get().strip(),
			**{k: v for k, v in values.items()},
		}
		try:
			CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
		except Exception as exc:
			messagebox.showerror("Config", f"Impossible d'écrire la config: {exc}")

	# ==================== _choose_file =========================
	def _choose_file(self) -> None:
		"""Ouvre une boîte de dialogue et met le chemin choisi dans le champ fichier."""
		path = filedialog.askopenfilename(title="Choisir un fichier")
		if path:
			self.file_entry.delete(0, "end")
			self.file_entry.insert(0, path)

	# ==================== _parse_value =========================
	def _parse_value(self, key: str, raw: str) -> float | int:
		"""Convertit une valeur texte en nombre et vérifie sa plage.

		- Supporte la virgule comme séparateur décimal.
		- Lève une erreur si vide ou hors limites.
		"""
		min_val, max_val, value_type = self.constraints[key]
		text = raw.strip().replace(",", ".")
		if not text:
			raise ValueError("Champ vide")

		if value_type == "int":
			value = int(float(text))
		else:
			value = float(text)

		if value < min_val or value > max_val:
			raise ValueError(f"{min_val} à {max_val}")
		return value

	# ==================== _validate_entries =========================
	def _validate_entries(self) -> dict[str, float | int]:
		"""Valide tous les champs et retourne un dictionnaire de valeurs.

		Vérifie aussi les contraintes croisées (min <= max) pour biais et poids.
		"""
		values: dict[str, float | int] = {}
		for key, entry in self.entries.items():
			try:
				values[key] = self._parse_value(key, entry.get())
			except Exception as exc:
				raise ValueError(f"Valeur invalide pour {key} ({exc})") from exc

		# Contraintes croisées (min <= max)
		if float(values["biais_min"]) > float(values["biais_max"]):
			raise ValueError("biais_min doit être ≤ biais_max")
		if float(values["poids_min"]) > float(values["poids_max"]):
			raise ValueError("poids_min doit être ≤ poids_max")
		return values

	# ==================== _update_table =========================
	def _update_table(self, values: dict[str, float | int]) -> None:
		"""Met à jour le tableau récapitulatif des paramètres (zone texte)."""
		self.table_box.configure(state="normal")
		self.table_box.delete("1.0", "end")
		lines = [
			f"Mode: {self.mode_var.get()}",
			f"Fichier: {self.file_entry.get().strip() or '-'}",
			f"Fonction: {self.act_var.get()}",
			"",
			f"Entrées: {values['nb_entrees']}",
			f"Couches cachées: {values['nb_couches']}",
			f"Neurones/couche: {values['nb_neurones']}",
			f"Sorties: {values['nb_sorties']}",
			f"Taux d'apprentissage: {values['taux_apprentissage']}",
			f"Biais min/max: {values['biais_min']} / {values['biais_max']}",
			f"Poids min/max: {values['poids_min']} / {values['poids_max']}",
			f"i itérations: {values['iterations']}",
			f"k époques: {values['k_epoques']}",
			f"Score attendu: {values['score_attendu']}",
			f"Score obtenu: {values['score_obtenu']}",
		]
		self.table_box.insert("1.0", "\n".join(lines))
		self.table_box.configure(state="disabled")

	# ==================== _on_execute =========================
	def _on_execute(self) -> None:
		"""Valide, met à jour le tableau, puis lance `c2p59.py`.

		Les paramètres transmis au script:
		- `N_FCT` : index de la fonction d'activation.
		- `ETA`   : taux d'apprentissage.
		"""
		try:
			values = self._validate_entries()
		except ValueError as exc:
			messagebox.showerror("Erreur", str(exc))
			return

		self._update_table(values)

		act_map = {"sigmoïde": 1, "tan": 2, "tanh": 3, "gelu": 4}
		n_fct = act_map.get(self.act_var.get(), 1)
		eta = values.get("taux_apprentissage", 0.1)

		project_root = Path(__file__).resolve().parents[1]
		script_path = project_root / "neurone" / "c2p59.py"
		if not script_path.exists():
			messagebox.showerror("Erreur", "c2p59.py introuvable")
			return

		env = os.environ.copy()
		env["N_FCT"] = str(n_fct)
		env["ETA"] = str(eta)

		try:
			subprocess.Popen(
				[sys.executable, str(script_path)],
				cwd=str(project_root),
				env=env,
			)
		except Exception as exc:
			messagebox.showerror("Erreur", f"Impossible de lancer c2p59: {exc}")


# ==================== Point d'entrée =========================
if __name__ == "__main__":
	app = HMIApp()
	app.mainloop()
