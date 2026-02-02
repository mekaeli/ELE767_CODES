LABORATOIRE 1 (LAB1/) - GUIDE D'UTILISATION
===========================================

1) Objet
--------
Ce dossier contient le laboratoire 1 (paquet Python `lab1/`) :
	- un réseau de neurones de type MLP (perceptron multicouche) entièrement connecté
	- une propagation avant et une itération de rétropropagation du gradient
	- une interface graphique (CustomTkinter) optionnelle
	- un affichage console à visée pédagogique (détaillé / synthétique)
	- une chaîne de traitement pour la lecture de jeux de données au format "label: x1 x2 ..." (apprentissage/validation/test)

Pour une description structurée (architecture + enchaînements), consulter `structure.md`.


2) Dépendances
--------------
Dépendance externe (pip):
	- customtkinter (interface graphique uniquement)

Bibliothèque standard (Python):
	- tkinter (interface graphique), dataclasses, pathlib, typing
	- functools (lru_cache), collections.abc
	- random, math, argparse, tempfile
	- os, sys, ast, copy, unicodedata
	- ctypes (ajustements Windows/DPI dans layout.py)


3) Fichiers importants
---------------------
Orchestration / exécution:
	- lanceur.py
		Point d'entrée applicatif. Valide la charge utile (dictionnaire de paramètres),
		construit struct_reso via reseau.py, puis exécute selon le mode:
			- Test unitaire (affichage détaillé)
			- Test général (option 2) sur un fichier de test
			- Apprentissage (option 3) sur un fichier d'entraînement + validation + ajout de configuration
			- Validation (option 4) via la fonction test_validation (utilisée notamment en apprentissage)

Interface graphique:
	- interface.py
		Interface CustomTkinter. Construit une charge utile puis appelle `lanceur.execute_payload`.
	- layout.py
		Constantes d'interface + fonctions utilitaires (thème, polices, centrage, correspondances).

Coeur MLP:
	- reseau.py
		Construit/manipule la structure de réseau (struct_reso: X/W/B/D) + affichage des paramètres.
	- backpp.py
		Coeur numérique (propagation avant, rétropropagation, mise à jour) + affichage console.
	- fct_activation.py
		Fonctions d’activation (Fi) et dérivées (Fp).

Données:
	- loader.py
		Lecture des jeux de données "label: x1 x2 ..." (encodage indicateur via convert_label),
		accès séquentiel/par ligne, mélange aléatoire, mémoire cache des lignes.
	- data/file_concerter.py
		Outil en ligne de commande pour convertir des fichiers "audio trames concaténées" en vecteurs fixes (40/50/60).

Services:
	- service.py
		Gestion de `parametres.txt` + validation de format + fonction_max (indice du maximum -> vecteur indicateur).
	- smoke_test.py
		Tests rapides de validation (sans interface graphique).
	- structure.md
		Architecture + enchaînements détaillés (à jour).


4) Exécution
------------
A) Lancer l'interface graphique
	python -m lab1.lanceur

B) Lancer les tests rapides de validation
	python -m lab1.smoke_test

C) Générer des fichiers de données convertis (40_*, 50_*, 60_*)
	python -m lab1.data.file_concerter


5) Données et configuration
---------------------------
Configurations:
	- parametres.txt
		Table de configurations sauvegardées (lue/écrite via service.py).

Jeux de données (lus par lanceur.py via loader.py):
	- lab1/data/{40,50,60}_data_train.txt  (apprentissage)
	- lab1/data/{40,50,60}_data_vc.txt     (validation)
	- lab1/data/{40,50,60}_data_test.txt   (test général)

Format de fichier attendu:
	- Une ligne = "label: x1 x2 x3 ..."
	- label est encodé en vecteur indicateur (encodage unitaire) via convert_label(label, nb_sorties)

Effets de bord (apprentissage):
	- erreur.txt (dans lab1/)
		Réinitialisé à chaque exécution d'Apprentissage et rempli avec 1 ligne par époque.


6) Exemple pratique (programme Python)
-------------------------------------
Objectif: exécuter le mode "Test unitaire" en appelant directement `execute_payload`.

Créer un fichier `tmp_demo_execute_payload.py` à la racine du projet, puis exécuter:
	python tmp_demo_execute_payload.py

Contenu minimal:

---
from lab1.lanceur import execute_payload

payload = {
	"mode": "Test unitaire",
	"test_unitaire": True,
	"activation": "sigmoide",
	# N_b = tailles des couches (ex: [cachée, sortie])
	"N_b": [2, 3],
	"values": {
		"nb_entrees": 2,
		"nb_couches": 1,
		"nb_sorties": 3,
		"biais_min": 0,
		"biais_max": 0,
		"poids_min": 0,
		"poids_max": 0,
		"iterations": 1,
		"k_epoques": 1,
		"eta": 0.1,
		"score_obtenu": "-",
	}
}

ok, msg = execute_payload(payload)
print(ok, msg)
---

Résultat attendu:
	- affichage console détaillé (sections intermédiaires + résumé)
	- puis "True OK (mode=Test unitaire, test_unitaire=True)"


7) Démarche - Test général (option 2)
------------------------------------
Objectif:
	- Évaluer une configuration (poids/biais) sur un fichier de test
	- Obtenir un taux de réussite (OK/NOK)

Prérequis:
	- Disposer du fichier: lab1/data/{40,50,60}_data_test.txt
	- Disposer d'au moins une configuration dans parametres.txt (Wn_c/Bn_c)

Procédure (via l'interface graphique)
	1) Lancer:
		python -m lab1.lanceur
	2) Dans le tableau des configurations (parametres.txt), sélectionner une ligne.
	3) Choisir le mode "Test général".
	4) Vérifier la cohérence des paramètres (nb_entrees=40/50/60, nb_sorties, activation).
	5) Cliquer "Exécuter".
	6) Lire la console: "OK=... | NOK=... | taille=... | Score=..%".


8) Démarche - Apprentissage (option 3)
-------------------------------------
Objectif:
	- Ajuster les poids/biais sur le fichier d'entraînement, puis valider sur le fichier de validation
	- Produire un fichier erreur.txt (1 ligne par époque)
	- Ajouter automatiquement la configuration dans parametres.txt si le score atteint le seuil

Prérequis:
	- Disposer des fichiers:
		- lab1/data/{40,50,60}_data_train.txt
		- lab1/data/{40,50,60}_data_vc.txt
	- Choisir un nombre d'entrées parmi {40,50,60}

Procédure (via l'interface graphique)
	1) Lancer:
		python -m lab1.lanceur
	2) Choisir le mode "Apprentissage".
	3) Renseigner:
		- nb_entrees: 40/50/60
		- nb_couches + nb_neurones (couches cachées)
		- nb_sorties (taille de la couche de sortie)
		- eta (taux d'apprentissage)
		- k_epoques (nombre d'époques par bloc)
		- score_attendu (seuil en %)
	4) Cliquer "Exécuter".
	5) Consulter:
		- la console (retour de validation + score)
		- lab1/erreur.txt (1 ligne par époque)
		- parametres.txt (configuration ajoutée si score >= seuil)


9) Affichage console (exigence du laboratoire)
----------------------------------------------
Sections standard:
	1) Paramètres
	2) Résumé
	3) struct_reso

Comportement:
	- Test unitaire: affichage détaillé (inclut des étapes intermédiaires)
	- Test général / Validation / Apprentissage: affichage synthétique (uniquement Paramètres/Résumé/struct_reso)

Remarque:
	- struct_reso est affiché avec le même style que "Paramètres" (via Affiche_reso).

