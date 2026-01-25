# import librairy standard
import sys
import os

# import modules locales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import generic_func

# Nettoyer la console (Windows et Unix)
generic_func.clear_console()