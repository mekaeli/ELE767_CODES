STRUCTURE DU PROJET (ETAT ACTUEL - LAB1/)
=========================================

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


2) Modules et responsabilités (lab1/)
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
    - Options 2–4 (Généralisation / Validation / Apprentissage): logique complète à venir,
      mais l’aperçu console minimal est déjà branché.

  - interface.py
    - Interface CustomTkinter.
    - Construit un payload puis appelle lanceur.execute_payload.

C) Modules “support”
  - service.py
    - Lecture/écriture/suppression de lignes de configuration dans parametres.txt.

  - layout.py
    - Constantes UI (thème, couleurs, tailles).
    - Mapping des modes -> fichiers data_*.txt.

  - smoke_test.py
    - Smoke tests sans GUI (service + backprop + exécution d’un payload).

  - loader.py
    - Réservé / extension (placeholder).

  - __init__.py
    - Package Python.


3) Format de struct_reso
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


4) Flux d’exécution
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


5) Affichage console (exigence du laboratoire)
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


6) Données et configuration
---------------------------
  - parametres.txt : configurations sauvegardées (service.py)
  - config,json    : configuration locale UI (interface.py)
  - data_train.txt / data_vc.txt / data_test.txt : fichiers de données (modes 2–4; logique à venir)

