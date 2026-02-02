STRUCTURE DU PROJET (LAB1/)
===========================

1) Vue générale
---------------
Ce dossier contient l’implémentation du laboratoire 1:
  - un réseau MLP (feed-forward) entièrement connecté
  - une itération de rétropropagation (backprop)
  - une interface graphique (CustomTkinter)
  - un affichage console (pédagogique + résumé)

Calcul (approche matricielle, par couche L):
  - z_L = a_{L-1} @ W_L + b_L
  - a_L = Fi(z_L)
  - fp_L = Fp(z_L)

Backpropagation (convention du laboratoire):
  - delta_sortie = (d - y) ⊙ fp_sortie
  - delta_cachee = (W_suivante @ delta_suivante) ⊙ fp_courante
  - dW = eta * outer(a_precedente, delta)
  - db = eta * delta
  - W <- W + dW
  - b <- b + db


2) Architecture
---------------
Vue “couches” (du plus haut niveau au plus bas niveau) :

  UI (optionnelle)
    interface.py  +  layout.py
        |
        | payload (dict)
        v
  Orchestration / règles labo
    lanceur.py  +  service.py
        |
        | struct_reso (dictionnaire X/W/B/D)
        v
  Coeur numérique MLP
    reseau.py  -> construit struct_reso
    backpp.py  -> forward/backprop/update sur struct_reso
    fct_activation.py -> Fi/Fp (activation)

  Données (optionnelles selon le mode)
    data/file_concerter.py -> génère data_*.txt
    loader.py              -> lit data_*.txt (label: x1 x2 ...)

Flux principaux:
  - Mode GUI: interface.py -> lanceur.execute_payload -> (reseau/backpp) -> affichage
  - Mode CLI tests: python -m lab1.smoke_test
  - Modes dataset: lanceur utilise loader.py pour lire train/validation/test

Mini “diagrammes de séquence” (modes principaux)

Option 2: Test général (mode "Test général")
  interface.py (ou appel direct) -> lanceur.execute_payload(payload)
    -> lit la ligne sélectionnée dans parametres.txt (service.read_parametres_text + parse)
    -> crée struct_reso initial (reseau.mon_reso(...).cree_reso())
    -> lance test_general_option2(struct_reso, Wn_c_cfg, Bn_c_cfg, n_in)
      -> ouvre {n_in}_data_test.txt et calcule taille (loader.count_non_empty_lines)
      -> pour chaque ligne i:
           loader.get_validation_sample_seq(...) -> (D_onehot, X)
           reseau.mon_reso.set_reso(struct_reso, X, Wn_c_cfg, Bn_c_cfg, D)
           backpp.forward(...) -> y
           service.fonction_max(y) -> D_pred (one-hot)
           compare D_pred vs D
      -> imprime OK/NOK/Score et retourne %

Option 3: Apprentissage (mode "Apprentissage")
  interface.py (ou appel direct) -> lanceur.execute_payload(payload)
    -> crée struct_reso initial (reseau.mon_reso(...).cree_reso())
    -> lance run_apprentissage_option3(payload, r, struct_reso, eta)
      -> ouvre {n_in}_data_train.txt et fixe STATE.i = taille (loader.count_non_empty_lines)
      -> boucle (tant que l'utilisateur continue):
           pour epoch = 1..k_epoques:
             D_shuffled, X_shuffled = loader.get_validation_samples_random(train_path, ...)
             pour chaque (D, X):
               struct_reso <- reseau.mon_reso.set_reso(struct_reso, X, W_actuels, B_actuels, D)
               bp = backpp(struct_reso, ...)
               bp.backprop_update(...)  -> met à jour W/B dans struct_reso
             écrit la moyenne des deltas de sortie dans lab1/erreur.txt (1 ligne/epoch)
           validation finale via test_validation(...) sur {n_in}_data_vc.txt
           si score >= score_ref: ajoute la config dans parametres.txt (service.add_parametres_line)
           sinon: demande à l'utilisateur (messagebox) si on continue

Option 4: Validation (fonction test_validation)
  lanceur.test_validation(struct_reso, Wn_c, Bn_c, n_in)
    -> ouvre {n_in}_data_vc.txt et calcule taille (loader.count_non_empty_lines)
    -> pour chaque ligne i:
         loader.get_validation_sample_seq(...) -> (D_onehot, X)
         reseau.mon_reso.set_reso(struct_reso, X, Wn_c, Bn_c, D)
         backpp.forward(...) -> y
         service.fonction_max(y) -> D_pred
         compare D_pred vs D
    -> imprime OK/NOK/Score et retourne %

Fichiers lus / écrits (effets de bord)
-------------------------------------
Lecture (datasets, selon le mode):
  - lab1/data/{40,50,60}_data_train.txt  (apprentissage)
  - lab1/data/{40,50,60}_data_vc.txt     (validation)
  - lab1/data/{40,50,60}_data_test.txt   (test général)

Écriture (apprentissage):
  - lab1/erreur.txt
    - Réinitialisé à chaque exécution d'Apprentissage
    - 1 ligne par époque: moyenne des deltas de sortie

Configurations (UI / persistance):
  - lab1/parametres.txt
    - lu via service.read_parametres_text()
    - écrit/maj via service.add_parametres_line()
    - suppression via service.delete_parametres_line()

Outils de génération de données:
  - lab1/data/file_concerter.py lit des fichiers data_*.txt “audio trames concaténées”
    et écrit des fichiers convertis (ex: 40_*, 50_*, 60_*) au format "label: x1 x2 ...".


3) Modules et responsabilités (lab1/)
------------------------------------
A) Modules “coeur”
  - reseau.py
    - Construit et manipule la structure `struct_reso` (X/W/B/D).
    - Fournit l’affichage des paramètres (style du laboratoire).
    - Ne fait pas l’apprentissage: représentation + affichage.

  - backpp.py
    - Coeur numérique + affichage console.
    - NeuroneMat: forward / deltas / correcteurs / mises à jour (listes Python).
    - backpp: façade autour de struct_reso (conversion struct -> matrices -> struct).

  - fct_activation.py
    - Fonctions d’activation (sigmoïde, tan, tanh, gelu) + dérivées.

B) Modules “application / orchestration”
  - lanceur.py
    - Point d’entrée “application”.
    - Reçoit un payload (depuis l’UI), valide, crée struct_reso, puis exécute.
    - Option 1 (Test unitaire): affichage détaillé.
    - Option 2: test généralisation (lecture data_test via loader.py).
    - Option 3: apprentissage (boucle epochs/itérations) + validation + ajout config si score atteint.
    - Option 4: validation (lecture data_vc via loader.py).

  - interface.py
    - Interface CustomTkinter.
    - Construit un payload puis appelle lanceur.execute_payload.

C) Modules “support”
  - service.py
    - Lecture/écriture/suppression de lignes de configuration dans parametres.txt.
    - Validation du format de ligne (nouveau format avec groupes [...]).
    - Utilitaire fonction_max (argmax -> one-hot).

  - layout.py
    - Constantes UI (thème, couleurs, tailles).
    - Mapping des modes -> fichiers data_*.txt.

  - smoke_test.py
    - Smoke tests sans GUI (service + backprop + exécution d’un payload).

  - loader.py
    - Lecture des datasets "label: x1 x2 ...".
    - convert_label (one-hot), accès séquentiel (par ligne) et aléatoire (shuffle), cache des lignes.

  - __init__.py
    - Package Python.


4) Format de struct_reso
------------------------
struct_reso est un dictionnaire représentant le réseau sous forme “nommée”.

Clés typiques:
  - X{n_in}      : entrées, liste (nom, valeur) ex: ('x1', 1.0)
  - D{n_out}     : cibles,  liste (nom, valeur) ex: ('d1', 0)
  - W{n_layers}  : poids,   liste de couches, chaque couche = liste (nom, valeur)
  - B{n_layers}  : biais,   liste de couches, chaque couche = liste (nom, valeur)

Convention de nommage (couche k en 1..n_layers):
  - poids: w{i}{j}_{k} = poids de l’entrée i vers neurone j (couche k)
  - biais : b{j}_{k}   = biais du neurone j (couche k)


5) Flux d’exécution
------------------
A) Via interface graphique
  1. interface.py valide les champs et construit un payload (dict).
  2. lanceur.execute_payload(payload):
    - valide + met à jour l’état
    - crée struct_reso via reseau.py
    - exécute via backpp.py (affichage console)

B) Via console (tests)
  - Exécuter les smoke tests:
    python -m lab1.smoke_test

C) Via console (application)
  - Lancer l'application (GUI):
    python -m lab1.lanceur

D) Via console (génération de datasets)
  - Générer des fichiers 40_*, 50_*, 60_* dans lab1/data/:
    python -m lab1.data.file_concerter


6) Affichage console (exigence du laboratoire)
----------------------------------------------
Sections “standard”:
  - Paramètres
  - Résumé
  - struct_reso (affiché avec le même style que “Paramètres”, via Affiche_reso)

Comportement actuel:
  - Option 1 (Test unitaire): affichage détaillé (étapes intermédiaires + résumé).
  - Option différent de 1 (Généralisation / Validation / Apprentissage):
    UNIQUEMENT ces trois sections doivent être affichées:
      1) Paramètres
      2) Résumé
      3) struct_reso
    (les sections Forward/Deltas/Correcteurs/Mises à jour ne sont pas imprimées)


7) Dépendances
--------------
Externe (pip):
  - customtkinter (GUI uniquement)

Standard library Python (principales utilisées dans lab1):
  - tkinter (ttk/font/messagebox) (GUI)
  - dataclasses, pathlib, typing
  - functools (lru_cache), collections.abc
  - random, math
  - argparse, tempfile
  - os, sys, ast, copy, unicodedata
  - ctypes (ajustements Windows/DPI dans layout.py)


8) Données et configuration
---------------------------
  - lab1/parametres.txt : configurations sauvegardées (service.py)
  - lab1/data/*.txt : fichiers de données (train/vc/test + variantes 40_/50_/60_)

Format dataset attendu (loader.py):
  - Une ligne = "label: x1 x2 x3 ..."
  - label est converti en one-hot via convert_label(label, nb_sorties)

