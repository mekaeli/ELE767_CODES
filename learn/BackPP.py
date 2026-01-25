# Fonction maj pour la mise à jour d'un poids (Backpropagation classique)
def maj(w, delta_w):
	"""
	Calcule la nouvelle valeur d'un poids après correction selon la méthode Backpropagation classique.
	Args:
		w (float): ancienne valeur du poids
		delta_w (float): facteur de correction à appliquer
	Returns:
		float: nouvelle valeur du poids
	"""
	# Formule classique : nouveau poids = ancien poids + correction
	return w + delta_w

# =============================================================================================
# Fonction Correcteur pour le calcul du facteur de correction d'une neurone (Backpropagation classique)
def Correcteur(eta, x, delta):
	"""
	Calcule le facteur de correction pour la mise à jour du poids d'une neurone selon la méthode Backpropagation classique.
	Args:
		eta (float): taux d'apprentissage (learning rate)
		x (float): entrée de la neurone (ou sortie de la couche précédente)
		delta (float): signal d'erreur de la neurone
	Returns:
		float: facteur de correction à appliquer au poids
	"""
	# Formule classique : correction = eta * x * delta
	correction = eta * x * delta
	return correction

# =============================================================================================
# Fonction Delta_cache pour le calcul du signal d'erreur d'une neurone cachée (Backpropagation classique)
def Delta_cache(delta_w_list, Fp):
	"""
	Calcule le signal d'erreur (delta) pour une neurone cachée selon la méthode Backpropagation classique.
	Args:
		delta_w_list (list of tuple): liste de paires (delta, w) où
			delta (float) : signal d'erreur de la couche suivante
			w (float) : poids de connexion entre la neurone cachée et la neurone de la couche suivante
		Fp (float): dérivée de la fonction d'activation de la neurone cachée
	Returns:
		float: signal d'erreur delta pour la neurone cachée
	"""
	# Somme des produits delta * w pour chaque connexion sortante
	somme = sum(delta * w for delta, w in delta_w_list)
	# Multiplication par la dérivée de la fonction d'activation
	delta_cache = somme * Fp
	return delta_cache

# =============================================================================================
# Fonction Delta pour le calcul du signal d'erreur d'une neurone de sortie (Backpropagation classique)
def Delta(d, Fi, Fp):
	"""
	Calcule le signal d'erreur (delta) pour une neurone de sortie selon la méthode Backpropagation classique.
	Args:
		d (float): valeur désirée (cible)
		Fi (float): sortie réelle de la neurone (après activation)
		Fp (float): dérivée de la fonction d'activation
	Returns:
		float: signal d'erreur delta
	"""
	# Formule classique : delta = (d - Fi) * Fp
	delta = (d - Fi) * Fp
	return delta
