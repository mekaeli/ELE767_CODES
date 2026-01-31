LABORATOIRE 1 (LAB1/) - README
==============================

1) Objet
--------
Ce dossier contient le laboratoire 1:
	- un réseau MLP (feed-forward) entièrement connecté
	- une itération de rétropropagation (backprop)
	- une interface graphique (CustomTkinter)
	- un affichage console pédagogique (détaillé / résumé)


2) Fichiers importants
---------------------
	- interface.py
		Interface graphique (UI). Construit un payload puis appelle lanceur.execute_payload.

	- lanceur.py
		Orchestrateur: valide le payload, crée struct_reso via reseau.py, exécute via backpp.py.

	- reseau.py
		Construit et affiche la structure de réseau (struct_reso: X/W/B/D).

	- backpp.py
		Coeur numérique (forward/backprop/mise à jour) + affichage console.

	- service.py
		Gestion de parametres.txt (lecture/ajout/suppression de configs).

	- smoke_test.py
		Tests rapides sans GUI.

	- structure.md
		Description de la structure du projet (mise à jour).


3) Exécution
------------
A) Lancer l'interface graphique
	- Depuis la racine du workspace:
		python -m lab1.lanceur
	
	- Alternativement:
		python lab1/lanceur.py

B) Lancer les smoke tests (sans GUI)
	- Depuis la racine du workspace:
		python -m lab1.smoke_test


4) Données et configuration
---------------------------
	- parametres.txt
		Table de configurations sauvegardées (gérée par service.py).

	- config,json
		Configuration locale de l'UI (créée/consommée par interface.py).

	- data_train.txt / data_vc.txt / data_test.txt
		Fichiers de données (modes 2-4). La logique complète est à venir.


5) Affichage console (exigence du laboratoire)
----------------------------------------------
Les sections d'affichage pertinentes sont:
	1) Paramètres
	2) Résumé
	3) struct_reso

Comportement actuel:
	- Option 1 (Test unitaire): affichage détaillé (étapes intermédiaires + résumé).
	- Options != 1 (Généralisation / Validation / Apprentissage):
		UNIQUEMENT ces trois sections doivent être affichées (pas de reste).

Note sur le style:
	- struct_reso est affiché avec le même style que "Paramètres" (via Affiche_reso).

