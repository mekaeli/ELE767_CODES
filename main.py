"""Point d'entrée minimal du projet.

But:
	Exemple simple qui montre l'utilisation de clear_console.
"""

import os

# ==================== clear_console =========================
def clear_console() -> None:
	"""Nettoie la console.

	- Windows : commande `cls`
	- Linux/Mac : commande `clear`
	"""
	os.system("cls" if os.name == "nt" else "clear")
	print("Console nettoyée.")


# ==================== main =========================
def main():
	"""Nettoie la console (démo minimale)."""

	clear_console()


if __name__ == "__main__":
	main()

