
STRUCTURE DU SYSTEME (selon lab1/learn.py)
=========================================

1) Idée générale
---------------
On construit un réseau de neurones entièrement connecté (feed-forward) avec:
	- une couche d'entrée de taille nx
	- (nc-2) couches cachées
	- une couche de sortie de taille ncs

Dans learn.py, les tailles des couches sont construites comme suit:

	Couche 0 (entrée)        : nx
	Couche 1 (cachée 1)       : nsc1
	Couche 2 (cachée 2)       : nsc2
	Couche (nc-1) (sortie)    : ncs
	Toutes les autres couches : ncs
		(ex: si nc > 4, alors les couches 3..(nc-2) ont ncs neurones)

Donc, la liste des tailles est:
	layer_sizes = [taille_couche0, taille_couche1, ..., taille_couche(nc-1)]


2) Vecteurs et paramètres générés
--------------------------------
Entrée:
	vecX = [x1_1, x1_2, ..., x1_nx]
	- taille: nx
	- valeurs par défaut: aléatoires 0 ou 1

Sortie désirée:
	vecD = [d(nc-1)_1, d(nc-1)_2, ..., d(nc-1)_ncs]
	- taille: ncs
	- valeurs par défaut: aléatoires 0 ou 1

Poids:
	vecW = [W1, W2, ..., W(nc-1)]
	où chaque W_L est une matrice (n_in x n_out)

	Pour une couche L (L=1..nc-1):
		W_L a la taille: (layer_sizes[L-1] x layer_sizes[L])

Biais:
	vecB = [b1, b2, ..., b(nc-1)]
	où chaque b_L est un vecteur de taille layer_sizes[L]


3) Convention de nommage (variables explicites)
----------------------------------------------
Dans learn.py, on associe aussi des noms pour faciliter l'identification:

	Entrées:
		x1_1, x1_2, ..., x1_nx

	Poids:
		wL_i_j = poids qui relie:
			- le neurone i de la couche (L-1)
			- au neurone j de la couche L
		Exemple: w2_3_7 = de couche 1 neurone 3 -> couche 2 neurone 7

	Biais:
		bL_j = biais du neurone j dans la couche L

	Sorties désirées:
		d(nc-1)_j = cible de la sortie j


4) Schéma (exemple avec les valeurs par défaut)
----------------------------------------------
Paramètres par défaut:
	nx=5, nc=4, nsc1=15, nsc2=15, ncs=10

Tailles des couches:
	couche 0:  5   (entrée)
	couche 1: 15   (cachée 1)
	couche 2: 15   (cachée 2)
	couche 3: 10   (sortie)

Schéma:

	vecX (5)
		|
		|  W1: (5 x 15)   + b1: (15)
		v
	couche 1 (15)
		|
		|  W2: (15 x 15)  + b2: (15)
		v
	couche 2 (15)
		|
		|  W3: (15 x 10)  + b3: (10)
		v
	sortie y (10)   (comparée à vecD (10))


5) Calcul (approche matricielle)
-------------------------------
Pour chaque couche L:
	zL = a(L-1) @ W_L + b_L
	aL = Fi(zL)   (fonction d'activation n_fct)
	fpL = Fp(zL)  (dérivée)

Puis backpropagation (NeuroneMat):
	delta_sortie = (d - a_sortie) * fp_sortie
	delta_cachee = (delta_suivante @ W_suivante^T) * fp_courante
	dW = eta * outer(a_precedente, delta_courante)
	db = eta * delta_courante
	W <- W + dW
	b <- b + db

