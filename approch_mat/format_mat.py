"""Outils de formatage pour vecteurs/matrices.

Ce module centralise l'affichage des matrices/vecteurs pour les scripts
matriciels (approch_mat/*).

Fonctions:
  - matrix_lines(M): retourne une liste de lignes (str) pour une matrice
  - vector_lines(v): idem pour un vecteur (colonne ou ligne)
  - shape_str(x): affiche la forme (nxm) en chaîne
"""


# ==================== _format_float =========================
def _format_float(x: float, precision: int) -> str:
	"""Formate un nombre pour affichage aligné.

	Règle:
		- fixe si "raisonnable"
		- scientifique si très petit/grand
	"""

	try:
		xf = float(x)
	except Exception:
		return str(x)

	ax = abs(xf)
	if ax != 0.0 and (ax < 10 ** (-(precision - 1)) or ax >= 10 ** (precision + 1)):
		return f"{xf:.{precision}e}"
	return f"{xf:.{precision}f}"


# ==================== matrix_lines =========================
def matrix_lines(M, precision: int = 6):
	"""Retourne une liste de lignes (str) pour afficher une matrice proprement.

	M: liste de listes (n_lignes x n_colonnes)
	"""

	if M is None:
		return ["<None>"]
	if len(M) == 0:
		return ["[ ]"]

	rows = [[_format_float(v, precision) for v in row] for row in M]
	n_cols = max((len(r) for r in rows), default=0)
	if n_cols == 0:
		return ["[ ]"]

	col_widths = [0] * n_cols
	for r in rows:
		for j in range(n_cols):
			s = r[j] if j < len(r) else ""
			col_widths[j] = max(col_widths[j], len(s))

	out = ["["]
	for r in rows:
		padded = []
		for j in range(n_cols):
			s = r[j] if j < len(r) else ""
			padded.append(s.rjust(col_widths[j]))
		out.append("  [ " + "  ".join(padded) + " ]")
	out.append("]")
	return out


# ==================== vector_lines =========================
def vector_lines(v, precision: int = 6, as_column: bool = True):
	"""Retourne des lignes (str) pour afficher un vecteur en colonne ou en ligne."""

	if v is None:
		return ["<None>"]

	vals = list(v)
	if as_column:
		return matrix_lines([[x] for x in vals], precision=precision)
	return matrix_lines([vals], precision=precision)


# ==================== shape_str =========================
def shape_str(M) -> str:
	"""Retourne une chaîne "(lignes x colonnes)" pour vecteurs/matrices."""

	if M is None:
		return "(?x?)"
	if isinstance(M, (list, tuple)) and (len(M) == 0 or not isinstance(M[0], (list, tuple))):
		# vecteur
		return f"({len(M)}x1)"
	if isinstance(M, (list, tuple)):
		rows = len(M)
		cols = len(M[0]) if rows > 0 else 0
		return f"({rows}x{cols})"
	return "(?x?)"

