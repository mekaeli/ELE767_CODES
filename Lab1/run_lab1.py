"""Script de test (Lab1).

But:
	Démo minimale pour vérifier que les imports fonctionnent et que la console
	est nettoyée.
"""

# import librairy standard
import sys
import os

# import modules locales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import generic_func


# ==================== main =========================
def main():
	"""Point d'entrée du script Lab1."""

	# Nettoyer la console (Windows et Unix)
	generic_func.clear_console()


if __name__ == "__main__":
	main()