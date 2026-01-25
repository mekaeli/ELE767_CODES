"""Compatibilité: build_neurone.py

Le module a été renommé vers `backpp_neurone.py`.
On conserve ce fichier comme shim pour ne pas casser les anciens imports.
"""

from neurone.backpp_neurone import Neurone  # noqa: F401
