from __future__ import annotations

"""Constantes et helpers de layout (UI) pour le lab1.

Ce module centralise **toutes** les données de mise en forme:
- thème (couleurs)
- typographie (polices)
- dimensions (fenêtre, tailles des widgets)
- espacements/paddings

Important:
- `init_fonts()` doit être appelé après la création de la fenêtre Tk/CTk.
- `apply_theme()` peut être appelé dès l'import (avant la création de la fenêtre).
"""

import ctypes
import sys

try:
	import customtkinter as ctk
except Exception:  # pragma: no cover
	ctk = None  # type: ignore[assignment]


# ========================
# Thème (couleurs)
# ========================
COLOR_BG = "#1B1D24"            # fond principal (moins sombre)
COLOR_PANEL = "#232633"         # panneau principal
COLOR_ENTRY = "#1E212B"         # champs / dropdown
COLOR_BORDER = "#3A3F4B"        # contours

# Couleur de texte unifiée (à appliquer partout sauf boutons et radios)
COLOR_TEXT_UNIFIED = "#DADAE6"

# Déclinaisons (conservées pour compatibilité dans le code)
COLOR_TEXT_TITLE = COLOR_TEXT_UNIFIED
COLOR_TEXT = COLOR_TEXT_UNIFIED
COLOR_TEXT_MUTED = COLOR_TEXT_UNIFIED
COLOR_TEXT_HINT = COLOR_TEXT_UNIFIED

# Exceptions: boutons et radios gardent leurs couleurs.
COLOR_TEXT_RADIO = COLOR_TEXT_UNIFIED

# Accent (violet)
COLOR_ACCENT = "#7B6CFF"
COLOR_ACCENT_HOVER = "#6B5BF3"
COLOR_ON_ACCENT = "#0C0C10"


# ========================
# Fenêtre
# ========================
WINDOW_W = 900
WINDOW_H = 900

# ========================
# Scaling (anti-zoom)
# ========================
# Certains environnements Windows appliquent un scaling DPI qui donne un rendu
# "zoomé". Forcer 1.0 rend l'UI à l'échelle neutre.
UI_SCALING = 1.0


# ========================
# Dimensions des widgets
# ========================
ENTRY_W = 90
SCORE_W = 120
ACT_W = 170
CELL_W = 100
CELL_H = 112
BTN_W = 120
BTN_H = 30


# ========================
# Espacements
# ========================
PAD_OUTER = 18
PAD_SECTION_X = 16
PAD_SECTION_Y = 10
PAD_INNER = 12
PAD_SMALL = 8
PAD_TINY = 2
PAD_GAP = 6
PAD_MED = 10

# Largeur utile pour les panneaux internes.
# Fenêtre non redimensionnable → on dérive des paddings pour éviter les débordements.
CONTENT_WIDTH = WINDOW_W - (2 * PAD_OUTER) - (2 * PAD_SECTION_X)


# ========================
# Typographie
# ========================
FONT_FAMILY = "Segoe UI"

FONT_TITLE = None
FONT_SECTION = None
FONT_LABEL = None
FONT_HINT = None
FONT_ENTRY = None
FONT_BUTTON = None
FONT_RADIO = None


def _enable_windows_dpi_awareness() -> None:
	"""Active le DPI-aware sous Windows pour éviter le texte flou."""
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


def apply_theme() -> None:
	"""Applique le thème global CustomTkinter.

	À appeler une seule fois. Peut être fait avant la création de la fenêtre.
	"""
	if ctk is None:  # pragma: no cover
		raise RuntimeError("customtkinter est requis pour l'interface (apply_theme)")
	ctk.set_appearance_mode("Dark")
	_enable_windows_dpi_awareness()


def init_fonts() -> None:
	"""Initialise les polices.

	Important: doit être appelé après création de la racine Tk/CTk.
	"""
	global FONT_TITLE, FONT_SECTION, FONT_LABEL, FONT_HINT, FONT_ENTRY, FONT_BUTTON, FONT_RADIO
	if ctk is None:  # pragma: no cover
		raise RuntimeError("customtkinter est requis pour l'interface (init_fonts)")
	if FONT_TITLE is not None:
		return

	FONT_TITLE = ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold")
	FONT_SECTION = ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold")
	FONT_LABEL = ctk.CTkFont(family=FONT_FAMILY, size=12)
	FONT_HINT = ctk.CTkFont(family=FONT_FAMILY, size=11)
	FONT_ENTRY = ctk.CTkFont(family=FONT_FAMILY, size=12)
	FONT_BUTTON = ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold")
	FONT_RADIO = ctk.CTkFont(family=FONT_FAMILY, size=12)


def center_window(window, width: int | None = None, height: int | None = None) -> None:
	"""Centre une fenêtre Tk/CTk sur l'écran.

	- Fonction volontairement indépendante de CustomTkinter.
	- Utilise `update_idletasks()` pour obtenir des dimensions fiables.
	"""
	try:
		window.update_idletasks()
	except Exception:
		pass

	try:
		w = int(width) if width is not None else int(window.winfo_width())
		h = int(height) if height is not None else int(window.winfo_height())
		# Fallback si la fenêtre n'est pas encore rendue.
		if w <= 1:
			w = int(width) if width is not None else int(window.winfo_reqwidth())
		if h <= 1:
			h = int(height) if height is not None else int(window.winfo_reqheight())

		sw = int(window.winfo_screenwidth())
		sh = int(window.winfo_screenheight())
		x = max(0, int((sw - w) / 2))
		y = max(0, int((sh - h) / 2))
		window.geometry(f"{w}x{h}+{x}+{y}")
	except Exception:
		# Ne doit jamais faire planter l'app si l'OS/WM refuse.
		return


# ========================
# Constantes applicatives (UI)
# ========================

# Entête du fichier parametres.txt
PARAMETRES_HEADER = (
	"[Focntion act.] [[nb d'entrées] [nb couches cachées] [nb neorones/cou.cachées] "
	"[nb de sorties] [eta]] [poid Wn_c] [biais Bn_c] [Score]"
)

# Modes + fichiers source par défaut
DEFAULT_SOURCE_FILES: dict[str, str] = {
	"Généralisation": "data_test.txt",
	"Validation": "data_vc.txt",
	"Apprentissage": "data_train.txt",
}

# Fonctions d'activation disponibles
ACTIVATION_VALUES = ["sigmoïde", "tan", "tanh", "gelu"]
ACTIVATION_MAP = {"sigmoïde": 1, "tan": 2, "tanh": 3, "gelu": 4}

# Tableau (Treeview)
TABLE_COLUMNS = ("fonction", "nb_entrees", "nb_couches", "nb_neurones", "nb_sorties", "eta", "score")
TABLE_HEADINGS: dict[str, str] = {
	"fonction": "[fonction act.]",
	"nb_entrees": "[nb d'entrées]",
	"nb_couches": "[nb couches cachées]",
	"nb_neurones": "[nb neurones couches cachées]",
	"nb_sorties": "[nb de sorties]",
	"eta": "[eta]",
	"score": "[Score]",
}

# Précision d'arrondi lors de la génération aléatoire
GEN_PRECISION = 6


# ========================
# Libellés UI (texte)
# ========================

UI_TITLE_TABLE = "Tableau des paramètres"
UI_LABEL_ACTIVATION = "Fonction act."
UI_LABEL_SCORE_EXPECTED = "Score attendu"
UI_LABEL_SCORE_OBSERVED = "Score obtenu"

UI_DIALOG_ERROR = "Erreur"
UI_ERROR_C2P59_MISSING = "c2p59.py introuvable"
