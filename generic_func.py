"""Fonctions utilitaires communes.

Ce fichier contient des petites fonctions réutilisées dans plusieurs scripts.
"""

# import librairy standard
import os
import math

# import modules locales


# ==================== clear_console =========================
def clear_console():
    """Nettoie la console.

    But:
        Effacer l'écran pour rendre les sorties plus lisibles.

    Fonctionnement:
        - Windows : commande `cls`
        - Linux/Mac : commande `clear`
    """

    os.system("cls" if os.name == "nt" else "clear")
    print("")
    print("Console nettoyée.")
    print("")




