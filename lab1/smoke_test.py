"""smoke_test

Rôle
	Tests rapides (sans GUI) pour valider les éléments critiques de `lab1/`.

Objectifs
	- Vérifier que `service.py` lit/écrit/supprime correctement des lignes dans
	  `parametres.txt` (incluant la conservation de l'en-tête).
	- Vérifier que `backpp.backprop_update()` modifie les poids/biais et réduit
	  la perte dans un cas simple.
	- Simuler l'exécution d'un payload (comme si on cliquait sur "Exécuter")
	  sans lancer l'interface CustomTkinter.

Exécution
	- `python -m lab1.smoke_test`
"""

from __future__ import annotations

from pathlib import Path
import tempfile


# ==================== _loss_mse_half =========================
def _loss_mse_half(y: list[float], d: list[float]) -> float:
	"""Calcule la perte $0.5 * \sum (d - y)^2$.

	Paramètres
		y : sorties obtenues
		d : sorties désirées
	"""
	if len(y) != len(d):
		raise ValueError("y et d doivent avoir la même taille")
	return 0.5 * sum((di - yi) ** 2 for di, yi in zip(d, y))


# ==================== run_backpp =========================
def run_backpp() -> None:
	"""Smoke test: une itération de backprop_update() doit réduire la perte."""
	from .reseau import mon_reso
	from .backpp import backpp

	# Réseau minimal: 2 entrées -> 1 sortie (une couche)
	r = mon_reso(
		n_in=2,
		n_c=0,
		n_s=1,
		N_b=[1],
		biais=[0, 0],
		poids=[0, 0],
		X=[1],
		D=[1],
	)
	struct = r.cree_reso()

	# Paramètres déterministes
	struct["X2"] = [("x1", 1.0), ("x2", 1.0)]
	struct["D1"] = [("d1", 1.0)]
	struct["W1"] = [[("w11_1", 0.0), ("w21_1", 0.0)]]
	struct["B1"] = [[("b1_1", 0.0)]]

	bp = backpp(struct, n_fct=1, eta=0.5)
	y0 = bp.forward()
	L0 = _loss_mse_half(y0, [1.0])

	bp.backprop_update()
	y1 = bp.forward()
	L1 = _loss_mse_half(y1, [1.0])

	assert L1 < L0, f"Perte non décroissante: L0={L0}, L1={L1}"

	# Vérifie que des paramètres ont bien été mis à jour
	W1 = bp.struct_reso["W1"]
	B1 = bp.struct_reso["B1"]
	assert W1 != [[("w11_1", 0.0), ("w21_1", 0.0)]], "W1 n'a pas changé"
	assert B1 != [[("b1_1", 0.0)]], "B1 n'a pas changé"

	print("OK: smoke_test backpp")


# ==================== run_execute_payload =========================
def run_execute_payload() -> None:
	"""Smoke test: simule le clic sur 'Exécuter' sans lancer le GUI."""
	from . import lanceur

	payload = {
		"mode": "Test unitaire",
		"activation": "sigmoide",
		# Doit correspondre au varEntryManuel courant dans lab1/lanceur.py:
		# 2 entrées -> 2 neurones cachés -> 3 sorties
		"N_b": [2, 3],
		"values": {
			"nb_entrees": 2,
			"nb_couches": 1,
			# NB: ce champ n'est pas utilisé par execute_payload en Test unitaire,
			# mais on le garde cohérent.
			"nb_neurones": [2],
			"nb_sorties": 3,
			"biais_min": 1,
			"biais_max": 5,
			"poids_min": -0.1,
			"poids_max": 0.1,
			"iterations": 4,
			"k_epoques": 40,
			"eta": 0.1,
			"score_obtenu": "0%",
		},
	}

	ok, msg = lanceur.execute_payload(payload)
	assert ok, f"execute_payload a échoué: {msg}"
	assert "OK" in msg, f"message inattendu: {msg}"

	# Cas invalide: incohérence nb_sorties vs N_b[-1]
	bad = {
		**payload,
		"values": {**payload["values"], "nb_sorties": 2},
	}
	ok, msg = lanceur.execute_payload(bad)
	assert not ok, "execute_payload devrait échouer si nb_sorties != N_b[-1]"
	assert "nb_sorties" in msg.lower(), f"message d'erreur inattendu: {msg}"
	print("OK: smoke_test execute_payload")


# ==================== run =========================
def run() -> None:
	"""Point d'entrée du smoke test (sans interface graphique)."""
	from . import layout
	from . import service
	from . import lanceur
	from . import loader

	# Smoke test: règle MAX (one-hot)
	assert service.fonction_max([0.12, -0.32, 0.52, 0.42]) == [0, 0, 1, 0]
	assert service.fonction_max([]) == []
	# En cas d'égalité, le premier max gagne (argmax)
	assert service.fonction_max([1.0, 1.0, 0.0]) == [1, 0, 0]

	# Mini-tests: convert_label() (selon l'énoncé)
	assert loader.convert_label(1, 10) == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	assert loader.convert_label(2, 10) == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
	assert loader.convert_label(9, 10) == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
	assert loader.convert_label(0, 10) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
	# Cas générique (0-based)
	assert loader.convert_label(0, 3) == [1, 0, 0]
	assert loader.convert_label(2, 3) == [0, 0, 1]
	try:
		loader.convert_label(10, 10)
		raise AssertionError("convert_label(10, 10) devait lever ValueError")
	except ValueError:
		pass

	# Test simple de lecture/écriture de parametres.txt (dans un dossier temporaire).
	line = "[sigmoïde] [[2] [1] [2] [1] [0.1]] [[0.1,0.2]] [[0.0]] [60%]"

	with tempfile.TemporaryDirectory() as tmp:
		p = Path(tmp) / "parametres.txt"

		ok, msg = service.add_parametres_line(line, p)
		assert ok, f"add_parametres_line failed: {msg}"

		# Duplicate should be rejected.
		ok, msg = service.add_parametres_line(line, p)
		assert not ok, "duplicate add should be rejected"
		assert "doublon" in msg.lower(), f"unexpected duplicate message: {msg}"

		# Delete should remove exactly one line.
		ok, msg = service.delete_parametres_line(line, p)
		assert ok, f"delete_parametres_line failed: {msg}"

		content = p.read_text(encoding="utf-8")
		assert line not in content, "line should be removed"
		assert layout.PARAMETRES_HEADER in content, "header should be preserved"

	print("OK: smoke_test")
	run_execute_payload()
	run_backpp()


if __name__ == "__main__":
	run()
