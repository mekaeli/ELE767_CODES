"""interface

Rôle
	Interface HMI du laboratoire 1 (CustomTkinter).

Fonctions principales
	- Construire l'interface (widgets + layout) via `HMIApp`.
	- Valider les champs utilisateur (contraintes min/max et cohérences).
	- Action "Exécuter" : construit un payload puis délègue à `lanceur.execute_payload`.

Flux global (simplifié)
	UI -> payload (dict) -> lanceur -> (reseau + backpp) -> affichage console

Note:
	- La fenêtre est volontairement non redimensionnable: on peut donc dériver des largeurs
	  de contenu à partir des paddings pour garder un layout stable.
"""

from __future__ import annotations

import random
from pathlib import Path
import sys
import tkinter.ttk as ttk
import tkinter.font as tkfont
import tkinter.messagebox as messagebox

import customtkinter as ctk

# Les constantes de mise en forme (couleurs, fonts, paddings, tailles) sont
# centralisées dans `layout.py` pour éviter la duplication entre fichiers.
# Le module est conçu pour être exécuté/importé comme package.
# Exécution recommandée: `python -m lab1.lanceur`.
from . import layout


layout.apply_theme()
# Racine locale du module lab1 (aucune dépendance au dossier parent).
LAB1_ROOT = Path(__file__).resolve().parent


class HMIApp(ctk.CTk):
	"""Fenêtre principale de l'interface."""
	SCORE_KEYS = {"score_attendu", "score_obtenu"}

	def _set_form_lock_for_mode(self, mode: str) -> None:
		"""Verrouille/déverrouille les champs selon le mode.

		Exigence option 2 (Test général):
		- toutes les cases sont verrouillées
		- modification seulement via sélection dans le tableau
		"""
		mode_s = str(mode or "").strip()
		lock = mode_s == "Test général"

		# Entrées: en Test général on désactive tout.
		# Hors Test général on réactive tout, sauf les champs explicitement read-only.
		always_disabled = {"iterations"}
		if hasattr(self, "entries"):
			for key, entry in self.entries.items():
				try:
					if lock or key in always_disabled:
						entry.configure(state="disabled")
					else:
						entry.configure(state="normal")
				except Exception:
					pass

		# Dropdown activation: verrouillé en Test général (doit venir du tableau).
		try:
			if hasattr(self, "act_menu"):
				self.act_menu.configure(state=("disabled" if lock else "normal"))
		except Exception:
			pass

	def _apply_startup_defaults(self) -> None:
		"""Applique les valeurs par défaut fournies au démarrage (si présentes)."""
		defaults = getattr(self, "startup_defaults", None) or {}

		# Mode au démarrage (1 choix parmi 3)
		startup_mode = defaults.get("mode")
		if startup_mode is not None and hasattr(self, "mode_var"):
			try:
				self.mode_var.set(str(startup_mode))
			except Exception:
				pass
		# score obtenu (format % accepté)
		score_ob = defaults.get("score_ob")
		if score_ob is not None:
			self._set_entry_text("score_obtenu", self._format_score_percent(score_ob))

		# bornes biais
		b = defaults.get("biais")
		if isinstance(b, (list, tuple)) and len(b) >= 2:
			self._set_entry_text("biais_min", str(b[0]))
			self._set_entry_text("biais_max", str(b[1]))

		# bornes poids (Wn)
		p = defaults.get("poids")
		if isinstance(p, (list, tuple)) and len(p) >= 2:
			self._set_entry_text("poids_min", str(p[0]))
			self._set_entry_text("poids_max", str(p[1]))

		# i / k
		if defaults.get("i") is not None:
			self._set_entry_text("iterations", str(defaults.get("i")))
		if defaults.get("k") is not None:
			self._set_entry_text("k_epoques", str(defaults.get("k")))
		# Applique les effets du mode (label + boutons)
		try:
			self._update_mode_label()
		except Exception:
			pass

	def _select_last_non_empty_table_row(self) -> None:
		"""Sélectionne et charge la dernière ligne non vide du tableau."""
		if not hasattr(self, "table_tree"):
			return
		children = list(self.table_tree.get_children(""))
		for item_id in reversed(children):
			vals = self.table_tree.item(item_id, "values")
			if not isinstance(vals, (list, tuple)):
				continue
			if not any(str(v).strip() not in ("", "-") for v in vals):
				continue
			try:
				self.table_tree.selection_set(item_id)
				self.table_tree.focus(item_id)
				self.table_tree.see(item_id)
			except Exception:
				pass
			# Recharge explicitement les champs.
			self._on_table_select()
			break

	def _set_entry_text(self, key: str, text: str) -> None:
		"""Met à jour un champ (CTkEntry) si présent."""
		if not hasattr(self, "entries"):
			return
		entry = self.entries.get(key)
		if entry is None:
			return
		# Certains champs peuvent être en lecture seule (state="disabled").
		# On les active temporairement pour permettre les mises à jour internes.
		try:
			prev_state = entry.cget("state")
			had_disabled = str(prev_state).lower() == "disabled"
			if had_disabled:
				entry.configure(state="normal")
		except Exception:
			had_disabled = False
		entry.delete(0, "end")
		entry.insert(0, text)
		if had_disabled:
			try:
				entry.configure(state="disabled")
			except Exception:
				pass

	def _format_score_percent(self, raw: object) -> str:
		"""Normalise un score pour affichage en pourcentage (ex: "60% ").

		Accepte:
		- "60%" -> "60%"
		- "60"  -> "60%"
		- 60 / 60.0 -> "60%"
		"""
		s = str(raw).strip() if raw is not None else ""
		if not s or s == "-":
			return "-"
		if s.endswith("%"):
			return s.replace(" ", "")
		try:
			val = float(s.replace(",", "."))
		except Exception:
			return s
		# Valeur numérique = pourcentage directement (0..100 typiquement).
		if abs(val - round(val)) < 1e-9:
			return f"{int(round(val))}%"
		# 2 décimales max, sans zéros inutiles
		formatted = f"{val:.2f}".rstrip("0").rstrip(".")
		return f"{formatted}%"

	def _load_table_row_into_form(self, values: tuple[str, ...]) -> None:
		"""Charge une ligne du tableau dans les champs de saisie."""
		if len(values) < 7:
			return
		fonction, nb_entrees, nb_couches, nb_neurones, nb_sorties, eta, score = values[:7]

		# Fonction d'activation (dropdown)
		if hasattr(self, "act_var") and fonction:
			try:
				# Ne change que si la valeur est valide dans le menu.
				menu_values = getattr(self, "act_menu")._values  # type: ignore[attr-defined]
				if fonction in menu_values:
					self.act_var.set(fonction)
			except Exception:
				# Fallback: tente de setter quand même.
				try:
					self.act_var.set(fonction)
				except Exception:
					pass

		# Champs numériques
		self._set_entry_text("nb_entrees", (nb_entrees or "").strip())
		self._set_entry_text("nb_couches", (nb_couches or "").strip())
		self._set_entry_text("nb_neurones", (nb_neurones or "").strip())
		self._set_entry_text("nb_sorties", (nb_sorties or "").strip())
		self._set_entry_text("taux_apprentissage", (eta or "").strip())
		# Exigence: au clic, le score de la ligne va dans "Score attendu".
		self._set_entry_text("score_attendu", self._format_score_percent(score))

	def _format_nb_list(self, value: object) -> str:
		"""Formate une liste de neurones (ex: [2, 1]) en "2, 1"."""
		if isinstance(value, (list, tuple)):
			try:
				return ", ".join(str(int(v)) for v in value)
			except Exception:
				return ", ".join(str(v) for v in value)
		return str(value)

	def _expected_layer_sizes(
		self,
		nb_entrees: int,
		nb_couches_cachees: int,
		nb_list: list[int],
		nb_sorties: int,
		*,
		n_s: int = 1,
	) -> list[int]:
		"""Retourne la liste N_b attendue et valide sa cohérence.

		Nouvelle règle (projet):
			- Le champ UI `nb_neurones` décrit UNIQUEMENT les couches cachées.
			- `nb_sorties` donne la taille de la couche de sortie.
			- Donc: N_b = nb_neurones + [nb_sorties]

		Important: aucune rétro-compatibilité. Le format ancien (incluant la sortie
		dans nb_neurones) est considéré invalide.
		"""
		if nb_couches_cachees < 0:
			raise ValueError("nb_couches doit être ≥ 0")
		if int(n_s) <= 0:
			raise ValueError("n_s doit être ≥ 1")
		if nb_sorties <= 0:
			raise ValueError("nb_sorties doit être ≥ 1")
		if nb_entrees <= 0:
			raise ValueError("nb_entrees doit être > 0")
		if len(nb_list) != nb_couches_cachees:
			raise ValueError("nb_neurones doit contenir nb_couches valeurs (couches cachées uniquement)")
		if any(int(v) <= 0 for v in nb_list):
			raise ValueError("nb_neurones: toutes les valeurs doivent être ≥ 1")

		# N_b contient toutes les couches (cachées + sortie)
		return [int(v) for v in nb_list] + [int(nb_sorties)]

	def _check_weights_bias_dimensions(
		self,
		*,
		nb_entrees: int,
		nb_list: list[int],
		Wn_c: object,
		Bn_c: object,
	) -> None:
		"""Vérifie que Wn_c et Bn_c ont des dimensions cohérentes avec la topologie."""
		if not isinstance(Wn_c, list) or not isinstance(Bn_c, list):
			raise ValueError("Wn_c et Bn_c doivent être des listes")
		if len(Wn_c) != len(nb_list) or len(Bn_c) != len(nb_list):
			raise ValueError("Wn_c et Bn_c doivent contenir une entrée par couche (incluant la sortie)")

		prev = int(nb_entrees)
		for idx, layer_size in enumerate(nb_list):
			expected_w = prev * int(layer_size)
			w_layer = Wn_c[idx]
			b_layer = Bn_c[idx]
			if not isinstance(w_layer, list) or len(w_layer) != expected_w:
				raise ValueError(
					f"Dimension invalide pour Wn_c[{idx}] (attendu {expected_w} poids pour {prev}->{layer_size})"
				)
			if not isinstance(b_layer, list) or len(b_layer) != int(layer_size):
				raise ValueError(
					f"Dimension invalide pour Bn_c[{idx}] (attendu {layer_size} biais)"
				)
			prev = int(layer_size)

	def _generate_weights_biases(
		self,
		*,
		nb_entrees: int,
		layer_sizes: list[int],
		poids_min: float,
		poids_max: float,
		biais_min: float,
		biais_max: float,
		precision: int = layout.GEN_PRECISION,
	) -> tuple[list[list[float]], list[list[float]]]:
		"""Génère Wn_c et Bn_c cohérents avec la topologie.

		- Wn_c: liste de couches, chaque couche est une liste aplatie row-major.
		- Bn_c: liste de couches, chaque couche est une liste de biais.
		"""
		Wn_c: list[list[float]] = []
		Bn_c: list[list[float]] = []
		prev = int(nb_entrees)
		for layer_size in layer_sizes:
			layer_size_i = int(layer_size)
			w_count = prev * layer_size_i
			w_layer = [round(random.uniform(poids_min, poids_max), precision) for _ in range(w_count)]
			b_layer = [round(random.uniform(biais_min, biais_max), precision) for _ in range(layer_size_i)]
			Wn_c.append(w_layer)
			Bn_c.append(b_layer)
			prev = layer_size_i
		return Wn_c, Bn_c

	def _on_table_select(self, event=None) -> None:
		"""Charge la ligne sélectionnée (clavier/souris) dans le formulaire."""
		if not hasattr(self, "table_tree"):
			return
		selected = self.table_tree.selection()
		if not selected:
			return
		item = selected[0]
		vals = self.table_tree.item(item, "values")
		if isinstance(vals, (list, tuple)):
			self._load_table_row_into_form(tuple(str(v) for v in vals))
			# Recharge aussi k_epoques depuis la ligne brute (non affichée dans le tableau).
			raw_line = getattr(self, "table_raw_lines", {}).get(item)
			parsed = self._parse_parametres_line(raw_line or "") if raw_line else None
			k_val = None if not parsed else (parsed.get("k_epoques") or "").strip()
			self._set_entry_text("k_epoques", str(k_val if k_val else 5))

	def _on_table_click(self, event) -> None:
		"""Supporte le clic simple même sur la même ligne (recharge toujours)."""
		if not hasattr(self, "table_tree"):
			return
		row_id = self.table_tree.identify_row(event.y)
		if not row_id:
			return
		try:
			self.table_tree.selection_set(row_id)
			self.table_tree.focus(row_id)
		except Exception:
			pass
		vals = self.table_tree.item(row_id, "values")
		if isinstance(vals, (list, tuple)):
			self._load_table_row_into_form(tuple(str(v) for v in vals))
			# Recharge aussi k_epoques depuis la ligne brute (non affichée dans le tableau).
			raw_line = getattr(self, "table_raw_lines", {}).get(row_id)
			parsed = self._parse_parametres_line(raw_line or "") if raw_line else None
			k_val = None if not parsed else (parsed.get("k_epoques") or "").strip()
			self._set_entry_text("k_epoques", str(k_val if k_val else 5))

	def _on_table_double_click(self, event) -> None:
		"""Double-clic: demande au lanceur de supprimer la ligne correspondante."""
		if not hasattr(self, "table_tree"):
			return
		row_id = self.table_tree.identify_row(event.y)
		if not row_id:
			return
		raw_line = getattr(self, "table_raw_lines", {}).get(row_id)
		if not raw_line:
			messagebox.showerror("Suppression", "Impossible d'identifier la ligne dans parametres.txt")
			return
		if callable(getattr(self, "on_delete_config", None)):
			try:
				ok, msg = self.on_delete_config({"raw_line": raw_line})
			except Exception as exc:
				messagebox.showerror("Suppression", f"Erreur (lanceur): {exc}")
				return
			if not ok and msg:
				messagebox.showwarning("Suppression", msg)
			return
		messagebox.showerror("Suppression", "Callback de suppression manquant")

	def _extract_bracket_groups(self, text: str) -> list[str]:
		"""Extrait les groupes `[...]` de premier niveau d'une ligne.

		Exemple:
		`[sigmoïde] [[2] [1] [2] [1] [0.1]] [[...]] [[...]] [60%]`
		-> ["sigmoïde", "[2] [1] [2] [1] [0.1]", "[...]", "[...]", "60%"]
		"""
		groups: list[str] = []
		buf: list[str] = []
		depth = 0
		in_group = False
		for ch in text:
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

	def _parse_parametres_line(self, line: str) -> dict[str, str] | None:
		"""Parse une ligne de `parametres.txt` et retourne les champs consommés.

	Format attendu (une ligne):
		[Fonction act.] [[nb d'entrées] [nb couches cachées] [nb neorones/cou.cachées] [nb de sorties] [eta]] [poid Wn_c] [biais Bn_c] [Score]

		Note: on ignore le poids/biais, et on n'utilise que:
			- Fonction act., nb d'entrées, nb couches cachées, nb neorones/cou.cachées, nb de sorties, eta, Score
		"""
		s = (line or "").strip()
		if not s:
			return None

		# Ligne d'en-tête éventuelle
		lower = s.lower()
		if "nb d'entr" in lower and "couches" in lower and "eta" in lower:
			return None

		groups = self._extract_bracket_groups(s)
		if len(groups) < 3:
			return None

		fonction = groups[0].strip()

		# groups[1] contient 5 valeurs encadrées de crochets.
		inner = self._extract_bracket_groups(groups[1])
		if len(inner) < 5:
			return None

		# Nouveau: la ligne peut se terminer par un groupe supplémentaire [k_epoques].
		# Dans ce cas, le score est l'avant-dernier groupe.
		k_epoques: str | None = None
		score = groups[-1].strip()
		if len(groups) >= 6:
			k_epoques = groups[-1].strip()
			score = groups[-2].strip()
		return {
			"fonction": fonction,
			"nb_entrees": inner[0].strip(),
			"nb_couches": inner[1].strip(),
			"nb_neurones": inner[2].strip(),
			"nb_sorties": inner[3].strip(),
			"eta": inner[4].strip(),
			"score": score,
			"k_epoques": (k_epoques or "").strip(),
		}

	def load_parametres_text(self, text: str) -> None:
		"""Charge toutes les lignes de `parametres.txt` dans le tableau."""
		self._clear_table_rows()
		# Validation stricte: seules les lignes au nouveau format sont affichées.
		from . import service
		invalid_count = 0
		invalid_examples: list[str] = []
		# Conserve la correspondance item_id -> ligne brute pour la suppression.
		self.table_raw_lines = {}
		for raw_line in (text or "").splitlines():
			parsed = self._parse_parametres_line(raw_line)
			if not parsed:
				continue
			# Ignore les lignes invalides (format non conforme, corruption, etc.)
			try:
				service.validate_parametres_line_new_format((raw_line or "").strip())
			except Exception:
				invalid_count += 1
				if len(invalid_examples) < 3:
					ex = (raw_line or "").strip()
					if len(ex) > 160:
						ex = ex[:157] + "..."
					invalid_examples.append(ex)
				continue
			item_id = self.table_tree.insert(
				"",
				"end",
				values=(
					parsed["fonction"],
					parsed["nb_entrees"],
					parsed["nb_couches"],
					parsed["nb_neurones"],
					parsed["nb_sorties"],
					parsed["eta"],
					parsed["score"],
				),
			)
			self.table_raw_lines[item_id] = (raw_line or "").strip()
		# Stocke pour que le lanceur (init) puisse notifier une seule fois.
		self._last_invalid_parametres_lines = int(invalid_count)
		self._last_invalid_parametres_examples = list(invalid_examples)

	# Les colonnes et libellés du tableau sont centralisés dans `layout.py`.

	def _init_table_style(self) -> None:
		"""Configure le style ttk du tableau (Treeview) pour matcher le thème sombre."""
		style = ttk.Style()
		try:
			style.theme_use("clam")
		except Exception:
			pass

		# Exigences:
		# - Police du contenu du tableau = police des radios (Validation, etc.)
		# - Police de l'entête = police du titre de frame (ex: "Tableau des paramètres")
		body_size = 12
		heading_size = 12
		try:
			if layout.FONT_RADIO is not None:
				body_size = int(layout.FONT_RADIO.cget("size"))
		except Exception:
			pass
		rowheight = max(22, body_size + 14)

		style.configure(
			"HMI.Treeview",
			background=layout.COLOR_ENTRY,
			fieldbackground=layout.COLOR_ENTRY,
			foreground=layout.COLOR_TEXT,
			bordercolor=layout.COLOR_BORDER,
			lightcolor=layout.COLOR_BORDER,
			darkcolor=layout.COLOR_BORDER,
			font=(layout.FONT_FAMILY, body_size),
			rowheight=rowheight,
		)
		style.map(
			"HMI.Treeview",
			background=[("selected", layout.COLOR_ACCENT)],
			foreground=[("selected", layout.COLOR_ON_ACCENT)],
		)

		style.configure(
			"HMI.Treeview.Heading",
			background=layout.COLOR_PANEL,
			foreground=layout.COLOR_TEXT,
			relief="flat",
			font=(layout.FONT_FAMILY, heading_size, "bold"),
			# Augmente la hauteur visuelle de l'entête.
			padding=(6, 10),
		)
		style.map(
			"HMI.Treeview.Heading",
			background=[("active", layout.COLOR_PANEL)],
			foreground=[("active", layout.COLOR_TEXT)],
		)

	def _autosize_table_columns(self) -> None:
		"""Ajuste la police du header + largeur des colonnes.

		Objectifs:
		- Pas de scroll horizontal (si possible).
		- Les colonnes occupent toute la largeur disponible (répartition propre).
		- Le texte du tableau utilise la même police que le reste.
		"""
		if not hasattr(self, "table_tree"):
			return

		# La largeur réelle n'est fiable qu'après le rendu.
		available = int(self.table_tree.winfo_width())
		if available <= 1:
			self.after(80, self._autosize_table_columns)
			return

		# Enlève un petit buffer pour la scrollbar verticale et les bordures.
		available = max(1, available - 26)

		# Exigence: entêtes jamais tronqués.
		# => On ne compresse jamais les colonnes en-dessous de la largeur requise par le texte.
		# Si la somme dépasse la largeur disponible, la scrollbar horizontale prendra le relais.
		base_size = 12
		pad_px = 26
		min_col_px = 70
		heading_font = tkfont.Font(family=layout.FONT_FAMILY, size=base_size, weight="bold")
		widths = [
			max(min_col_px, int(heading_font.measure(layout.TABLE_HEADINGS.get(c, c))) + pad_px)
			for c in layout.TABLE_COLUMNS
		]

		# Si on a de la place, on répartit l'espace restant pour éviter une zone vide à droite.
		total_width = sum(widths)
		if total_width > 0 and total_width < available:
			delta = available - total_width
			weight_total = sum(widths)
			adjusted = [w + int(delta * w / weight_total) for w in widths]
			final_delta = available - sum(adjusted)
			if final_delta != 0:
				adjusted[-1] = max(min_col_px, adjusted[-1] + final_delta)
			widths = [max(min_col_px, int(w)) for w in adjusted]

		# Applique la police d'en-tête (fixe) et fixe les colonnes.
		style = ttk.Style()
		style.configure(
			"HMI.Treeview.Heading",
			font=(layout.FONT_FAMILY, base_size, "bold"),
			padding=(6, 10),
		)
		for col, w in zip(layout.TABLE_COLUMNS, widths, strict=False):
			self.table_tree.column(col, width=int(w), minwidth=int(w), stretch=False, anchor="center")

	def _clear_table_rows(self) -> None:
		if not hasattr(self, "table_tree"):
			return
		for item in self.table_tree.get_children():
			self.table_tree.delete(item)
		# Nettoie la map des lignes brutes si présente.
		if hasattr(self, "table_raw_lines"):
			self.table_raw_lines = {}

	def _add_table_row(self, values: dict[str, object]) -> None:
		"""Ajoute une ligne au tableau (multi-lignes)."""
		if not hasattr(self, "table_tree"):
			return
		activation = self.act_var.get() if hasattr(self, "act_var") else "-"
		score_percent = self._format_score_percent(values.get("score_obtenu", "-"))
		nb_neurones_val = values.get("nb_neurones", "-")
		if isinstance(nb_neurones_val, (list, tuple)):
			nb_neurones_val = self._format_nb_list(nb_neurones_val)
		row = (
			str(activation),
			str(values.get("nb_entrees", "-")),
			str(values.get("nb_couches", "-")),
			str(nb_neurones_val),
			str(values.get("nb_sorties", "-")),
			str(values.get("eta", values.get("taux_apprentissage", "-"))),
			score_percent,
		)
		item_id = self.table_tree.insert("", "end", values=row)
		# Alternance de lignes (zébrage) pour distinguer visuellement.
		idx = len(self.table_tree.get_children())
		tag = "odd" if (idx % 2) else "even"
		self.table_tree.item(item_id, tags=(tag,))
		try:
			self.table_tree.selection_set(item_id)
			self.table_tree.see(item_id)
		except Exception:
			pass

	def _make_entry(self, parent: ctk.CTkFrame, *, width: int = layout.ENTRY_W, height: int = 28) -> ctk.CTkEntry:
		"""Crée un champ (Entry) avec le style unifié du thème."""
		return ctk.CTkEntry(
			parent,
			width=width,
			height=height,
			fg_color=layout.COLOR_ENTRY,
			border_color=layout.COLOR_BORDER,
			text_color=layout.COLOR_TEXT,
			font=layout.FONT_ENTRY,
		)

	def _make_button(
		self,
		parent: ctk.CTkFrame,
		*,
		text: str,
		command,
		width: int = layout.BTN_W,
		height: int = layout.BTN_H,
	) -> ctk.CTkButton:
		"""Crée un bouton avec le style unifié du thème."""
		return ctk.CTkButton(
			parent,
			text=text,
			width=width,
			height=height,
			fg_color=layout.COLOR_ACCENT,
			hover_color=layout.COLOR_ACCENT_HOVER,
			text_color=layout.COLOR_ON_ACCENT,
			font=layout.FONT_BUTTON,
			command=command,
		)

	# ==================== __init__ =========================
	def __init__(
		self,
		*,
		parametres_text: str | None = None,
		startup_defaults: dict[str, object] | None = None,
		on_set_config=None,
		on_delete_config=None,
		on_execute=None,
	) -> None:
		"""Crée la fenêtre et construit tous les widgets."""
		super().__init__()
		self.startup_defaults = startup_defaults or {}
		self.on_set_config = on_set_config
		self.on_delete_config = on_delete_config
		self.on_execute = on_execute
		# Neutralise un éventuel scaling DPI qui peut donner un rendu "zoomé".
		try:
			ctk.set_widget_scaling(float(getattr(layout, "UI_SCALING", 1.0)))
			ctk.set_window_scaling(float(getattr(layout, "UI_SCALING", 1.0)))
			self.tk.call("tk", "scaling", float(getattr(layout, "UI_SCALING", 1.0)))
		except Exception:
			pass
		layout.init_fonts()
		self.title("HMI - Réseaux de neurones")
		self.geometry(f"{layout.WINDOW_W}x{layout.WINDOW_H}")
		# Exigence: fenêtre principale centrée sur l'écran (au démarrage).
		# On centre 2 fois: immédiatement + après le premier rendu, car certains WM
		# peuvent repositionner la fenêtre au tout début.
		layout.center_window(self, layout.WINDOW_W, layout.WINDOW_H)
		try:
			self.after(0, lambda: layout.center_window(self, layout.WINDOW_W, layout.WINDOW_H))
		except Exception:
			pass
		self.minsize(layout.WINDOW_W, layout.WINDOW_H)
		self.resizable(False, False)
		self.configure(fg_color=layout.COLOR_BG)

		self._build_ui()
		self._update_mode_label()
		# Valeurs par défaut au démarrage (score/biais/poids/i/k)
		self._apply_startup_defaults()
		if parametres_text:
			self.load_parametres_text(parametres_text)
			# Notification unique si le fichier contient des lignes non conformes.
			invalid = int(getattr(self, "_last_invalid_parametres_lines", 0) or 0)
			if invalid > 0 and not bool(getattr(self, "_startup_invalid_notified", False)):
				self._startup_invalid_notified = True
				examples = getattr(self, "_last_invalid_parametres_examples", [])
				if not isinstance(examples, list):
					examples = []
				ex_msg = ""
				if examples:
					ex_msg = "\n\nExemples:\n- " + "\n- ".join(str(e) for e in examples)
				self.after(
					60,
					lambda: messagebox.showwarning(
						"parametres.txt",
						f"{invalid} ligne(s) invalide(s) ont été ignorée(s) (format non conforme).{ex_msg}",
					),
				)
			# Au démarrage: charger la dernière ligne non vide du tableau.
			self.after(140, self._select_last_non_empty_table_row)

		# Met à jour le nom de fichier affiché lorsque nb_entrees change.
		try:
			if hasattr(self, "entries") and "nb_entrees" in self.entries:
				self.entries["nb_entrees"].bind("<KeyRelease>", lambda _e: self._update_mode_label())
				self.entries["nb_entrees"].bind("<FocusOut>", lambda _e: self._update_mode_label())
		except Exception:
			pass

	# ==================== _build_ui =========================
	def _build_ui(self) -> None:
		"""Construit toute l'interface (layout + widgets)."""
		# Root panel (carte principale)
		root_frame = ctk.CTkFrame(
			self,
			fg_color=layout.COLOR_PANEL,
			border_color=layout.COLOR_BORDER,
			border_width=2,
			corner_radius=18,
		)
		root_frame.pack(fill="both", expand=True, padx=layout.PAD_OUTER, pady=layout.PAD_OUTER)

		root_frame.grid_columnconfigure(0, weight=1)
		# Répartir l'espace vertical: le tableau (bas) prend le reste.
		root_frame.grid_rowconfigure(0, weight=0)
		root_frame.grid_rowconfigure(1, weight=0)
		root_frame.grid_rowconfigure(2, weight=1)

		# ---------- En-tête (modes) ----------
		header_frame = ctk.CTkFrame(root_frame, fg_color="transparent")
		header_frame.grid(row=0, column=0, padx=layout.PAD_SECTION_X, pady=(14, 6), sticky="ew")

		self.mode_var = ctk.StringVar(value="Test unitaire")
		radio_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
		radio_frame.pack(pady=(layout.PAD_SMALL, 10))

		self._add_radio(radio_frame, "Test unitaire").pack(side="left", padx=16)
		self._add_radio(radio_frame, "Test général").pack(side="left", padx=16)
		self._add_radio(radio_frame, "Apprentissage").pack(side="left", padx=16)

		self.mode_label = ctk.CTkLabel(
			header_frame,
			text="Mode sélectionné : Test unitaire",
			text_color=layout.COLOR_TEXT_TITLE,
			font=layout.FONT_TITLE,
		)
		self.mode_label.pack(pady=(0, 10))

		self.mode_file_label = ctk.CTkLabel(
			header_frame,
			text="nom fichier : s/o",
			text_color=layout.COLOR_TEXT_MUTED,
			font=layout.FONT_LABEL,
		)
		self.mode_file_label.pack(pady=(0, 6))

		# Indicateur de test en cours (doit être visible pendant l'exécution).
		self.test_status_label = ctk.CTkLabel(
			header_frame,
			text="",
			text_color="#3FB950",
			font=layout.FONT_LABEL,
		)
		self.test_status_label.pack(pady=(0, 8))

		# Synchronise label + état des boutons avec le mode au démarrage.
		self._update_mode_label()

		# ---------- Zone paramètres (grille + boutons) ----------
		param_container = ctk.CTkFrame(
			root_frame,
			fg_color="transparent",
			border_color=layout.COLOR_BORDER,
			border_width=2,
			corner_radius=16,
		)
		param_container.configure(width=layout.CONTENT_WIDTH)
		param_container.grid(
			row=1,
			column=0,
			padx=layout.PAD_SECTION_X,
			pady=(layout.PAD_SECTION_Y, layout.PAD_SECTION_Y),
			sticky="ew",
		)

		param_container.grid_columnconfigure(0, weight=1, minsize=layout.CONTENT_WIDTH - (layout.BTN_W + 16))
		param_container.grid_columnconfigure(1, weight=0, minsize=layout.BTN_W + 16)

		self.entries: dict[str, ctk.CTkEntry] = {}
		self.constraints: dict[str, tuple[float, float, str]] = {}

		# Titre du panneau
		ctk.CTkLabel(
			param_container,
			text="Paramètres",
			text_color=layout.COLOR_TEXT_TITLE,
			font=layout.FONT_SECTION,
		).grid(row=0, column=0, columnspan=2, pady=(10, 0))

		# Ligne activation + scores (dans le panneau)
		top_row = ctk.CTkFrame(param_container, fg_color="transparent")
		top_row.configure(width=layout.CONTENT_WIDTH)
		top_row.grid(
			row=1,
			column=0,
			columnspan=2,
			padx=layout.PAD_INNER,
			pady=(layout.PAD_MED, layout.PAD_MED),
			sticky="ew",
		)
		# Centrer le bloc (act + 2 scores) dans la frame
		top_row.grid_columnconfigure(0, weight=1)
		top_row.grid_columnconfigure(1, weight=0, minsize=layout.ACT_W + 40)
		top_row.grid_columnconfigure(2, weight=0, minsize=layout.SCORE_W + 40)
		top_row.grid_columnconfigure(3, weight=0, minsize=layout.SCORE_W + 40)
		top_row.grid_columnconfigure(4, weight=1)

		# Mise en forme "pilule" (comme les choix de mode) pour:
		# Fonction act. / Score attendu / Score obtenu
		pill_kwargs = {
			"fg_color": layout.COLOR_ENTRY,
			"border_color": layout.COLOR_BORDER,
			"border_width": 2,
			"corner_radius": 18,
		}

		act_pill = ctk.CTkFrame(top_row, **pill_kwargs)
		act_pill.grid(row=0, column=1, padx=layout.PAD_MED, pady=(0, layout.PAD_GAP), sticky="n")
		ctk.CTkLabel(
			act_pill,
			text=layout.UI_LABEL_ACTIVATION,
			text_color=layout.COLOR_TEXT_MUTED,
			font=layout.FONT_LABEL,
		).grid(row=0, column=0, padx=layout.PAD_MED, pady=(layout.PAD_SMALL, 0))
		self.act_var = ctk.StringVar(value="sigmoïde")
		self.act_menu = ctk.CTkOptionMenu(
			act_pill,
			values=list(layout.ACTIVATION_VALUES),
			variable=self.act_var,
			fg_color=layout.COLOR_PANEL,
			button_color=layout.COLOR_PANEL,
			button_hover_color=layout.COLOR_BORDER,
			text_color=layout.COLOR_TEXT,
			dropdown_fg_color=layout.COLOR_ENTRY,
			dropdown_text_color=layout.COLOR_TEXT,
			dropdown_hover_color=layout.COLOR_BORDER,
			font=layout.FONT_ENTRY,
			dropdown_font=layout.FONT_ENTRY,
			width=layout.ACT_W,
			height=32,
			corner_radius=16,
		)
		self.act_menu.grid(row=1, column=0, padx=layout.PAD_MED, pady=(layout.PAD_SMALL, layout.PAD_MED))

		score_att_pill = ctk.CTkFrame(top_row, **pill_kwargs)
		score_att_pill.grid(row=0, column=2, padx=layout.PAD_MED, pady=(0, layout.PAD_GAP), sticky="n")
		ctk.CTkLabel(
			score_att_pill,
			text=layout.UI_LABEL_SCORE_EXPECTED,
			text_color=layout.COLOR_TEXT_MUTED,
			font=layout.FONT_LABEL,
		).grid(row=0, column=0, padx=layout.PAD_MED, pady=(layout.PAD_SMALL, 0))
		score_attendu = self._make_entry(score_att_pill, width=layout.SCORE_W, height=32)
		score_attendu.grid(row=1, column=0, padx=layout.PAD_MED, pady=(layout.PAD_SMALL, layout.PAD_MED))
		try:
			score_attendu.configure(corner_radius=16, border_width=2)
		except Exception:
			pass

		score_ob_pill = ctk.CTkFrame(top_row, **pill_kwargs)
		score_ob_pill.grid(row=0, column=3, padx=layout.PAD_MED, pady=(0, layout.PAD_GAP), sticky="n")
		ctk.CTkLabel(
			score_ob_pill,
			text=layout.UI_LABEL_SCORE_OBSERVED,
			text_color=layout.COLOR_TEXT_MUTED,
			font=layout.FONT_LABEL,
		).grid(row=0, column=0, padx=layout.PAD_MED, pady=(layout.PAD_SMALL, 0))
		score_obtenu = self._make_entry(score_ob_pill, width=layout.SCORE_W, height=32)
		score_obtenu.grid(row=1, column=0, padx=layout.PAD_MED, pady=(layout.PAD_SMALL, layout.PAD_MED))
		try:
			score_obtenu.configure(corner_radius=16, border_width=2)
		except Exception:
			pass

		self.entries["score_attendu"] = score_attendu
		# Scores en pourcentage.
		self.constraints["score_attendu"] = (0, 100, "float")
		self.entries["score_obtenu"] = score_obtenu
		self.constraints["score_obtenu"] = (0, 100, "float")

		# Grille principale des paramètres
		param_grid = ctk.CTkFrame(param_container, fg_color="transparent")
		# La grille doit être centrée par rapport à TOUTE la frame (incluant la zone des boutons)
		param_grid.grid(
			row=2,
			column=0,
			columnspan=2,
			padx=layout.PAD_INNER,
			pady=(layout.PAD_MED, layout.PAD_MED),
			sticky="ew",
		)
		# Centrer les 5 cellules avec 2 colonnes "spacer" (gauche/droite)
		param_grid.grid_columnconfigure(0, weight=1)
		for col in range(1, 6):
			param_grid.grid_columnconfigure(col, weight=0, uniform="param_cols", minsize=layout.CELL_W)
		param_grid.grid_columnconfigure(6, weight=1)

		# Le fichier parametres.txt fourni contient des valeurs comme nb_entrees=2.
		self._add_field(param_grid, "nb d’entrées", "1 ≤ Xn ≤ 480", 0, 1, "nb_entrees", 1, 480, "int")
		self._add_field(param_grid, "nb couches\nCachées", "1 ≤ Cn ≤ 10", 0, 2, "nb_couches", 1, 10, "int")
		# nb_neurones = liste des couches cachées uniquement, ex: "2" (1 couche), "2, 3" (2 couches).
		self._add_field(param_grid, "nb neurones/\nCachées", "1 ≤ Nh ≤ 200 (ex: 2, 3)", 0, 3, "nb_neurones", 1, 200, "list_int")
		self._add_field(param_grid, "nb de sorties", "1 ≤ Sn ≤ 10", 0, 4, "nb_sorties", 1, 10, "int")
		self._add_field(param_grid, "taux\nd’apprentissage", "0.1 ≤ η ≤ 1", 0, 5, "taux_apprentissage", 0.1, 1, "float")

		# Ligne basse (min/max + itérations/époques + boutons), centrée sur la frame
		bottom_row = ctk.CTkFrame(param_container, fg_color="transparent")
		bottom_row.grid(
			row=3,
			column=0,
			columnspan=2,
			padx=layout.PAD_INNER,
			pady=(2, layout.PAD_INNER),
			sticky="ew",
		)
		bottom_row.grid_columnconfigure(0, weight=1)
		bottom_row.grid_columnconfigure(1, weight=0)
		bottom_row.grid_columnconfigure(2, weight=0)
		bottom_row.grid_columnconfigure(3, weight=1)

		lower_grid = ctk.CTkFrame(bottom_row, fg_color="transparent")
		lower_grid.grid(row=0, column=1, sticky="w", padx=(0, layout.PAD_INNER))
		lower_grid.grid_columnconfigure(0, weight=0, minsize=34)
		lower_grid.grid_columnconfigure(1, weight=0, minsize=layout.CELL_W)
		lower_grid.grid_columnconfigure(2, weight=0, minsize=layout.CELL_W)
		lower_grid.grid_columnconfigure(3, weight=0, minsize=96)
		lower_grid.grid_columnconfigure(4, weight=0, minsize=layout.CELL_W)

		ctk.CTkLabel(
			lower_grid,
			text="min",
			text_color=layout.COLOR_TEXT_MUTED,
			font=layout.FONT_LABEL,
		).grid(
			row=0,
			column=0,
			padx=(0, layout.PAD_SMALL),
			pady=(layout.PAD_GAP, layout.PAD_TINY),
			sticky="w",
		)

		ctk.CTkLabel(
			lower_grid,
			text="max",
			text_color=layout.COLOR_TEXT_MUTED,
			font=layout.FONT_LABEL,
		).grid(
			row=1,
			column=0,
			padx=(0, layout.PAD_SMALL),
			pady=(layout.PAD_GAP, layout.PAD_TINY),
			sticky="w",
		)

		bias_min = self._make_entry(lower_grid)
		bias_min.grid(
			row=0,
			column=1,
			padx=(layout.PAD_GAP, layout.PAD_MED),
			pady=(layout.PAD_TINY * 2, layout.PAD_TINY),
			sticky="w",
		)

		bias_max = self._make_entry(lower_grid)
		bias_max.grid(
			row=1,
			column=1,
			padx=(layout.PAD_GAP, layout.PAD_MED),
			pady=(layout.PAD_TINY * 2, layout.PAD_TINY),
			sticky="w",
		)

		poids_min = self._make_entry(lower_grid)
		poids_min.grid(
			row=0,
			column=2,
			padx=(layout.PAD_MED, layout.PAD_MED),
			pady=(layout.PAD_TINY * 2, layout.PAD_TINY),
			sticky="w",
		)

		poids_max = self._make_entry(lower_grid)
		poids_max.grid(
			row=1,
			column=2,
			padx=(layout.PAD_MED, layout.PAD_MED),
			pady=(layout.PAD_TINY * 2, layout.PAD_TINY),
			sticky="w",
		)

		ctk.CTkLabel(
			lower_grid,
			text="i itérations",
			text_color=layout.COLOR_TEXT_MUTED,
			font=layout.FONT_LABEL,
		).grid(
			row=0,
			column=3,
			padx=(layout.PAD_INNER, layout.PAD_GAP),
			pady=(layout.PAD_GAP, layout.PAD_TINY),
			sticky="e",
		)

		iter_entry = self._make_entry(lower_grid)
		# Exigence: i (iterations) ne doit pas être modifiable manuellement.
		try:
			iter_entry.configure(state="disabled")
		except Exception:
			pass
		iter_entry.grid(
			row=0,
			column=4,
			padx=(0, layout.PAD_GAP),
			pady=(layout.PAD_TINY * 2, layout.PAD_TINY),
			sticky="w",
		)

		ctk.CTkLabel(
			lower_grid,
			text="k époques",
			text_color=layout.COLOR_TEXT_MUTED,
			font=layout.FONT_LABEL,
		).grid(
			row=1,
			column=3,
			padx=(layout.PAD_INNER, layout.PAD_GAP),
			pady=(layout.PAD_GAP, layout.PAD_TINY),
			sticky="e",
		)

		epoch_entry = self._make_entry(lower_grid)
		epoch_entry.grid(
			row=1,
			column=4,
			padx=(0, layout.PAD_GAP),
			pady=(layout.PAD_TINY * 2, layout.PAD_TINY),
			sticky="w",
		)

		ctk.CTkLabel(
			lower_grid,
			text="1 ≤ biais ≤ 5",
			text_color=layout.COLOR_TEXT_HINT,
			font=layout.FONT_HINT,
		).grid(
			row=2,
			column=1,
			padx=(layout.PAD_GAP, 0),
			pady=(layout.PAD_TINY * 2, 0),
			sticky="w",
		)

		ctk.CTkLabel(
			lower_grid,
			text="-0.1 ≤ Wn ≤ 0.1",
			text_color=layout.COLOR_TEXT_HINT,
			font=layout.FONT_HINT,
		).grid(
			row=2,
			column=2,
			padx=(layout.PAD_MED, 0),
			pady=(layout.PAD_TINY * 2, 0),
			sticky="w",
		)

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
		right_col.grid(row=0, column=2, padx=(layout.PAD_INNER, 0), pady=(0, 0), sticky="nw")
		right_col.grid_columnconfigure(0, weight=1)

		self.set_config_btn = self._make_button(right_col, text="Set config", command=self._on_set_config)
		self.set_config_btn.grid(
			row=0, column=0, pady=(layout.PAD_TINY * 2, layout.PAD_TINY), sticky="w"
		)

		self._make_button(right_col, text="Exécuter", command=self._on_execute).grid(
			row=1, column=0, pady=(layout.PAD_TINY * 2, layout.PAD_TINY), sticky="w"
		)

		# ---------- Tableau des paramètres (scroll vertical + horizontal) ----------
		table_frame = ctk.CTkFrame(
			root_frame,
			fg_color="transparent",
			border_color=layout.COLOR_BORDER,
			border_width=2,
			corner_radius=12,
		)
		table_frame.configure(width=layout.CONTENT_WIDTH)
		table_frame.grid(row=2, column=0, padx=layout.PAD_SECTION_X, pady=(6, 14), sticky="nsew")
		table_frame.grid_columnconfigure(0, weight=1)
		table_frame.grid_rowconfigure(1, weight=1)

		ctk.CTkLabel(
			table_frame,
			text=layout.UI_TITLE_TABLE,
			text_color=layout.COLOR_TEXT_TITLE,
			font=layout.FONT_SECTION,
		).grid(row=0, column=0, pady=(6, 4))

		# Tableau multi-lignes sélectionnable (Treeview)
		self._init_table_style()
		table_container = ctk.CTkFrame(
			table_frame,
			fg_color=layout.COLOR_ENTRY,
			border_color=layout.COLOR_BORDER,
			border_width=1,
			corner_radius=10,
		)
		table_container.grid(row=1, column=0, padx=layout.PAD_INNER, pady=(0, layout.PAD_INNER), sticky="nsew")
		table_container.grid_columnconfigure(0, weight=1)
		table_container.grid_rowconfigure(0, weight=1)

		self.table_tree = ttk.Treeview(
			table_container,
			columns=layout.TABLE_COLUMNS,
			show="headings",
			selectmode="browse",
			style="HMI.Treeview",
		)
		# Un clic sur une ligne recharge les champs correspondants.
		self.table_tree.bind("<<TreeviewSelect>>", self._on_table_select)
		self.table_tree.bind("<ButtonRelease-1>", self._on_table_click)
		self.table_tree.bind("<Double-1>", self._on_table_double_click)
		for col in layout.TABLE_COLUMNS:
			self.table_tree.heading(col, text=layout.TABLE_HEADINGS.get(col, col), anchor="center")
			# largeur calculée ensuite pour afficher les entêtes au complet
			self.table_tree.column(col, anchor="center", stretch=False)
		# Ajuste police + colonnes après rendu pour tout voir sans scroll horizontal.
		self.after(80, self._autosize_table_columns)

		# Zébrage des lignes
		try:
			self.table_tree.tag_configure("odd", background=layout.COLOR_PANEL)
			self.table_tree.tag_configure("even", background=layout.COLOR_ENTRY)
		except Exception:
			pass

		vsb = ttk.Scrollbar(table_container, orient="vertical", command=self.table_tree.yview)
		hsb = ttk.Scrollbar(table_container, orient="horizontal", command=self.table_tree.xview)
		self.table_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

		self.table_tree.grid(row=0, column=0, sticky="nsew")
		vsb.grid(row=0, column=1, sticky="ns")
		hsb.grid(row=1, column=0, sticky="ew")

	# ==================== _add_radio =========================
	def _add_radio(self, parent: ctk.CTkFrame, text: str) -> ctk.CTkRadioButton:
		"""Crée un bouton radio pour le choix du mode (Test général/Apprentissage)."""
		return ctk.CTkRadioButton(
			parent,
			text=text,
			variable=self.mode_var,
			value=text,
			command=self._update_mode_label,
			text_color=layout.COLOR_TEXT_RADIO,
			fg_color=layout.COLOR_ACCENT,
			hover_color=layout.COLOR_ACCENT_HOVER,
			font=layout.FONT_RADIO,
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
		cell = ctk.CTkFrame(parent, fg_color="transparent", width=layout.CELL_W, height=layout.CELL_H)
		cell.grid(row=row, column=col, padx=layout.PAD_GAP, pady=layout.PAD_SMALL, sticky="n")
		cell.grid_propagate(False)

		ctk.CTkLabel(
			cell,
			text=label,
			text_color=layout.COLOR_TEXT_MUTED,
			font=layout.FONT_LABEL,
		).pack(pady=(0, 6))

		entry = ctk.CTkEntry(
			cell,
			width=layout.ENTRY_W,
			height=28,
			fg_color=layout.COLOR_ENTRY,
			border_color=layout.COLOR_BORDER,
			text_color=layout.COLOR_TEXT,
			font=layout.FONT_ENTRY,
		)
		entry.pack()

		if show_range and range_text:
			ctk.CTkLabel(
				cell,
				text=range_text,
				text_color=layout.COLOR_TEXT_HINT,
				font=layout.FONT_HINT,
			).pack(pady=(4, 0))

		self.entries[key] = entry
		self.constraints[key] = (min_val, max_val, value_type)

	# ==================== _update_mode_label =========================
	def _update_mode_label(self) -> None:
		"""Met à jour le texte "Mode sélectionné" selon le bouton radio choisi."""
		self.mode_label.configure(text=f"Mode sélectionné : {self.mode_var.get()}")
		try:
			self._update_mode_file_label()
		except Exception:
			pass
		# Exigence: bouton Set config actif uniquement en mode Apprentissage.
		try:
			if hasattr(self, "set_config_btn"):
				state = "normal" if (self.mode_var.get() == "Apprentissage") else "disabled"
				self.set_config_btn.configure(state=state)
		except Exception:
			pass

		# Exigence option 2: verrouillage des champs en mode Test général.
		try:
			self._set_form_lock_for_mode(self.mode_var.get())
		except Exception:
			pass

	def _update_mode_file_label(self) -> None:
		"""Affiche le nom de fichier (dataset) sous le mode, si valide.

		- Si le fichier attendu existe dans `lab1/data/`, affiche le nom.
		- Sinon, affiche "nom fichier : s/o".
		"""
		if not hasattr(self, "mode_file_label"):
			return
		mode = self.mode_var.get() if hasattr(self, "mode_var") else ""
		if mode == "Test unitaire":
			self.mode_file_label.configure(text="nom fichier : s/o")
			return

		# Sélecteur 40/50/60 depuis le champ nb_entrees.
		n_in: int | None = None
		try:
			entry = getattr(self, "entries", {}).get("nb_entrees")
			if entry is not None:
				raw = str(entry.get() or "").strip()
				if raw:
					n_in = int(float(raw.replace(",", ".")))
		except Exception:
			n_in = None

		if n_in not in {40, 50, 60}:
			self.mode_file_label.configure(text="nom fichier : s/o")
			return

		suffix: str | None = None
		if mode == "Apprentissage":
			suffix = "data_train"
		elif mode == "Test général":
			suffix = "data_test"
		else:
			suffix = None

		if not suffix:
			self.mode_file_label.configure(text="nom fichier : s/o")
			return

		file_name = f"{int(n_in)}_{suffix}.txt"
		try:
			file_path = LAB1_ROOT / "data" / file_name
			exists = bool(file_path.exists())
		except Exception:
			exists = False

		if exists:
			self.mode_file_label.configure(text=f"nom fichier : {file_name}")
		else:
			self.mode_file_label.configure(text="nom fichier : s/o")

	def _set_test_status(self, message: str) -> None:
		"""Affiche/efface le statut de test en cours (label vert)."""
		try:
			if hasattr(self, "test_status_label"):
				self.test_status_label.configure(text=str(message or ""))
				# Force le rendu avant de lancer un calcul potentiellement long.
				self.update_idletasks()
		except Exception:
			pass

	# ==================== _on_set_config =========================
	def _on_set_config(self) -> None:
		"""Mode Apprentissage: valide + formalise, puis délègue l'écriture à `service` via le lanceur."""
		if self.mode_var.get() != "Apprentissage":
			messagebox.showinfo("Set config", "Disponible uniquement en mode Apprentissage")
			return
		try:
			values = self._validate_entries()
		except ValueError as exc:
			messagebox.showerror(layout.UI_DIALOG_ERROR, str(exc))
			return

		try:
			nb_entrees = int(values["nb_entrees"])
			nb_couches = int(values["nb_couches"])
			nb_sorties = int(values["nb_sorties"])
			n_s = 1
			hidden_list = values["nb_neurones"]
			if not isinstance(hidden_list, list):
				raise ValueError("nb_neurones doit être une liste")
			hidden_list = [int(v) for v in hidden_list]
			N_b_full = self._expected_layer_sizes(nb_entrees, nb_couches, hidden_list, nb_sorties, n_s=n_s)
			# Normalise pour l'écriture: `nb_neurones` = couches cachées uniquement.
			hidden_list = N_b_full[:-1]
		except Exception as exc:
			messagebox.showerror(layout.UI_DIALOG_ERROR, f"Topologie invalide ({exc})")
			return

		# Génère Wn_c / Bn_c selon les bornes min/max et la topologie.
		try:
			poids_min = float(values["poids_min"])
			poids_max = float(values["poids_max"])
			biais_min = float(values["biais_min"])
			biais_max = float(values["biais_max"])
			Wn_c, Bn_c = self._generate_weights_biases(
				nb_entrees=nb_entrees,
				layer_sizes=N_b_full,
				poids_min=poids_min,
				poids_max=poids_max,
				biais_min=biais_min,
				biais_max=biais_max,
			)
		except Exception as exc:
			messagebox.showerror(layout.UI_DIALOG_ERROR, f"Impossible de générer Wn_c/Bn_c ({exc})")
			return
		# Sanity check: dimensions
		try:
			self._check_weights_bias_dimensions(nb_entrees=nb_entrees, nb_list=N_b_full, Wn_c=Wn_c, Bn_c=Bn_c)
		except Exception as exc:
			messagebox.showerror(layout.UI_DIALOG_ERROR, str(exc))
			return
		# Mémorise comme valeurs courantes
		self.startup_defaults["Wn_c"] = Wn_c
		self.startup_defaults["Bn_c"] = Bn_c

		activation = self.act_var.get() if hasattr(self, "act_var") else "-"
		eta = values.get("taux_apprentissage", 0.1)
		score = self._format_score_percent(values.get("score_attendu", "-"))
		# Exigence: k_epoques doit être enregistré à la fin de la ligne, pour être rechargé.
		try:
			k_epoques = int(values.get("k_epoques", 5))
		except Exception:
			k_epoques = 5
		# IMPORTANT: Wn_c et Bn_c sont déjà des listes Python (ex: [[...],[...]]),
		# donc str(...) produit exactement le format attendu dans parametres.txt.
		formatted_line = (
			f"[{activation}] [[{nb_entrees}] [{nb_couches}] [{self._format_nb_list(hidden_list)}] [{nb_sorties}] [{eta}]] "
			f"{Wn_c} {Bn_c} [{score}] [{k_epoques}]"
		)

		# Délégation au lanceur (callback) pour: écrire (anti-doublon) + refresh tableau.
		if callable(getattr(self, "on_set_config", None)):
			try:
				ok, msg = self.on_set_config(
					{
						"formatted_line": formatted_line,
						"values": values,
						"activation": activation,
						"N_b": N_b_full,
						"Wn_c": Wn_c,
						"Bn_c": Bn_c,
					}
				)
			except Exception as exc:
				messagebox.showerror("Set config", f"Erreur (lanceur): {exc}")
				return
			if ok:
				messagebox.showinfo("Set config", "Écriture réussie dans parametres.txt")
			else:
				messagebox.showwarning("Set config", msg or "Refus d'écriture (doublon)")
			return

		# Fallback: si aucun callback n'est fourni, on tente d'écrire directement via service.
		from . import service
		ok, msg = service.add_parametres_line(formatted_line)
		if ok:
			self.load_parametres_text(service.read_parametres_text())
			self.after(80, self._select_last_non_empty_table_row)
			messagebox.showinfo("Set config", "Écriture réussie dans parametres.txt")
		else:
			messagebox.showwarning("Set config", msg or "Refus d'écriture (doublon)")


	# ==================== _parse_value =========================
	def _parse_value(self, key: str, raw: str):
		"""Convertit une valeur texte en nombre et vérifie sa plage.

		- Supporte la virgule comme séparateur décimal.
		- Lève une erreur si vide ou hors limites.
		"""
		min_val, max_val, value_type = self.constraints[key]
		raw_text = (raw or "").strip()
		if value_type == "list_int":
			# Accepte: "2,1" / "2, 1" / "[2, 1]" / "2 1".
			clean = raw_text.replace("[", "").replace("]", "").strip()
			if not clean:
				raise ValueError("Liste vide")
			if "," in clean:
				parts = [p.strip() for p in clean.split(",") if p.strip()]
			else:
				parts = [p.strip() for p in clean.split() if p.strip()]
			if not parts:
				raise ValueError("Liste vide")
			values_list: list[int] = []
			for part in parts:
				v = int(float(part.replace(",", ".")))
				if v < min_val or v > max_val:
					raise ValueError(f"{min_val} à {max_val}")
				values_list.append(v)
			return values_list

		text = raw_text.replace(",", ".")
		if key in self.SCORE_KEYS:
			text = text.replace(" ", "")
			if text.endswith("%"):
				text = text[:-1]
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
	def _validate_entries(self) -> dict[str, object]:
		"""Valide tous les champs et retourne un dictionnaire de valeurs.

		Vérifie aussi les contraintes croisées (min <= max) pour biais et poids.
		"""
		values: dict[str, object] = {}
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
		"""Ajoute une ligne au tableau (multi-lignes)."""
		self._add_table_row(values)

	# ==================== _on_execute =========================
	def _on_execute(self) -> None:
		"""Valide les champs puis délègue l'exécution au lanceur."""
		mode_selected = self.mode_var.get() if hasattr(self, "mode_var") else "-"
		selected_raw_line = None
		try:
			if hasattr(self, "table_tree"):
				selected = self.table_tree.selection()
				if selected:
					selected_raw_line = getattr(self, "table_raw_lines", {}).get(selected[0])
		except Exception:
			selected_raw_line = None

		# Option 2: si Test général, l'utilisateur doit choisir une config du tableau.
		if mode_selected == "Test général" and not (selected_raw_line and str(selected_raw_line).strip()):
			messagebox.showerror(layout.UI_DIALOG_ERROR, "Veuillez sélectionner une configuration dans le tableau")
			return

		try:
			values = self._validate_entries()
		except ValueError as exc:
			messagebox.showerror(layout.UI_DIALOG_ERROR, str(exc))
			return

		# Validation de topologie (N_b)
		try:
			nb_entrees = int(values["nb_entrees"])
			nb_couches = int(values["nb_couches"])
			nb_sorties = int(values["nb_sorties"])
			n_s = 1
			hidden_list = values["nb_neurones"]
			if not isinstance(hidden_list, list):
				raise ValueError("nb_neurones doit être une liste")
			hidden_list = [int(v) for v in hidden_list]
			N_b_full = self._expected_layer_sizes(nb_entrees, nb_couches, hidden_list, nb_sorties, n_s=n_s)
			# Normalise côté UI (pas obligatoire mais cohérent avec l'affichage).
			hidden_list = N_b_full[:-1]
		except Exception as exc:
			messagebox.showerror(layout.UI_DIALOG_ERROR, f"Topologie invalide ({exc})")
			return

		activation = self.act_var.get() if hasattr(self, "act_var") else "-"
		payload = {
			"mode": mode_selected,
			"activation": activation,
			"values": values,
			"N_b": N_b_full,
			"test_unitaire": bool(mode_selected == "Test unitaire"),
			# Permet au lanceur de retrouver Wn_c/Bn_c exacts dans parametres.txt.
			"selected_raw_line": (str(selected_raw_line).strip() if selected_raw_line else ""),
		}

		if callable(getattr(self, "on_execute", None)):
			self._set_test_status("Veuillez patienter, test en cours")
			try:
				ok, msg = self.on_execute(payload)
			except Exception as exc:
				self._set_test_status("")
				messagebox.showerror("Exécuter", f"Erreur (lanceur): {exc}")
				return
			finally:
				self._set_test_status("")
			if ok:
				messagebox.showinfo("Exécuter", msg or "OK")
			else:
				messagebox.showwarning("Exécuter", msg or "Paramètres refusés")
			return

		# Fallback: aucun lanceur branché
		messagebox.showinfo("Exécuter", "Aucun lanceur connecté (callback on_execute manquant)")
