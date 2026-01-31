
from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path
import random


def _parse_dataset_line(
	line: str,
	*,
	n_in: int,
	nb_sorties: int,
	convert_label: Callable[[int, int], list[int]],
) -> tuple[list[int], list[float]]:
	"""Parse une ligne de dataset `label: x1 x2 ...` -> (Dn_s, Xn)."""
	use_n_in = int(n_in)
	use_nb_sorties = int(nb_sorties)
	if use_n_in <= 0:
		raise ValueError("n_in doit être >= 1")
	if use_nb_sorties <= 0:
		raise ValueError("nb_sorties doit être >= 1")

	s = (line or "").strip()
	if not s:
		raise ValueError("Ligne invalide (vide)")
	if ":" not in s:
		raise ValueError(f"Ligne invalide (pas de ':'): {s[:80]!r}")

	left, _sep, right = s.partition(":")
	label_str = left.strip()
	values_str = right.strip()

	Dn_s = convert_label(int(label_str), use_nb_sorties)
	tokens = values_str.split()
	if not tokens:
		raise ValueError(f"Ligne invalide (aucune valeur X): {s[:80]!r}")
	try:
		X_all = [float(t.replace(",", ".")) for t in tokens]
	except Exception as exc:
		raise ValueError(f"Valeur X invalide sur la ligne: {s[:80]!r}") from exc

	if len(X_all) < use_n_in:
		raise ValueError(f"Taille Xn insuffisante: {len(X_all)} (attendu >= {use_n_in})")
	Xn = X_all[:use_n_in]
	return Dn_s, Xn


def resolve_validation_file(
	n_in: int,
	*,
	root_dir: Path,
	fichier: Path | None = None,
) -> Path:
	"""Retourne le chemin du fichier de validation.

	- Par défaut: `root_dir / f"{n_in}_data_vc.txt"`
	- Fallback: `root_dir / "data_vc.txt"`
	- Si `fichier` est fourni, il est utilisé tel quel.
	"""
	if fichier is not None:
		file_path = Path(fichier)
		if not file_path.exists():
			raise FileNotFoundError(f"Fichier introuvable: {file_path}")
		return file_path

	use_n_in = int(n_in)
	file_path = root_dir / f"{use_n_in}_data_vc.txt"
	if file_path.exists():
		return file_path

	fallback = root_dir / "data_vc.txt"
	if fallback.exists():
		return fallback

	raise FileNotFoundError(f"Fichier de validation introuvable: {file_path}")


@lru_cache(maxsize=32)
def _read_non_empty_lines_cached(file_path_str: str) -> tuple[str, ...]:
	"""Lit et met en cache les lignes non vides d'un fichier.

	Le cache est basé sur le chemin (string). Si le fichier change sur disque,
	il faut vider le cache avec `_read_non_empty_lines_cached.cache_clear()`.
	"""
	file_path = Path(file_path_str)
	lines: list[str] = []
	with open(file_path, "r", encoding="utf-8") as f:
		for raw in f:
			line = raw.strip()
			if line:
				lines.append(line)
	return tuple(lines)


def count_non_empty_lines(file_path: Path) -> int:
	"""Compte le nombre de lignes non vides (strip) dans un fichier."""
	return len(_read_non_empty_lines_cached(str(Path(file_path))))


def get_validation_sample_at(
	file_path: Path,
	*,
	i: int,
	n_in: int,
	nb_sorties: int,
	convert_label: Callable[[int, int], list[int]],
) -> tuple[bool, list[int] | None, list[float] | None]:
	"""Retourne l'exemple (Dn_s, Xn) à l'index `i` (0-based).

	- Si `i` existe: retourne (True, Dn_s, Xn)
	- Sinon: retourne (False, None, None)

	Les lignes vides sont ignorées (donc `i` est l'index parmi les lignes non vides).
	"""
	idx = int(i)
	if idx < 0:
		return False, None, None

	lines = _read_non_empty_lines_cached(str(Path(file_path)))
	if idx >= len(lines):
		return False, None, None

	line = lines[idx]
	Dn_s, Xn = _parse_dataset_line(
		line,
		n_in=n_in,
		nb_sorties=nb_sorties,
		convert_label=convert_label,
	)
	return True, Dn_s, Xn


def iter_validation_samples_cached(
	file_path: Path,
	*,
	n_in: int,
	nb_sorties: int,
	convert_label: Callable[[int, int], list[int]],
) -> Iterator[tuple[list[int], list[float]]]:
	"""Itère sur les exemples (Dn_s, Xn) via le cache mémoire (lignes non vides).

	Utile quand on veut faire plusieurs passes sur le même fichier sans le relire.
	"""
	lines = _read_non_empty_lines_cached(str(Path(file_path)))
	for line in lines:
		yield _parse_dataset_line(
			line,
			n_in=n_in,
			nb_sorties=nb_sorties,
			convert_label=convert_label,
		)


def iter_validation_samples(
	file_path: Path,
	*,
	n_in: int,
	nb_sorties: int,
	convert_label: Callable[[int, int], list[int]],
) -> Iterator[tuple[list[int], list[float]]]:
	"""Itère sur les exemples (Dn_s, Xn) d'un fichier `*_data_vc.txt`.

	Format de ligne attendu: `label: x1 x2 ...`.
	- `Dn_s` est un one-hot obtenu via `convert_label(label, nb_sorties)`
	- `Xn` est une liste de floats, tronquée à `n_in`
	"""
	with open(file_path, "r", encoding="utf-8") as f:
		for raw_line in f:
			line = (raw_line or "").strip()
			if not line:
				continue
			yield _parse_dataset_line(
				line,
				n_in=n_in,
				nb_sorties=nb_sorties,
				convert_label=convert_label,
			)


def load_validation_samples(
	file_path: Path,
	*,
	n_in: int,
	nb_sorties: int,
	convert_label: Callable[[int, int], list[int]],
) -> tuple[int, list[list[int]], list[list[float]]]:
	"""Lit séquentiellement le fichier et retourne (n_lignes, D_list, X_list)."""
	D_list: list[list[int]] = []
	X_list: list[list[float]] = []
	for Dn_s, Xn in iter_validation_samples(
		file_path,
		n_in=n_in,
		nb_sorties=nb_sorties,
		convert_label=convert_label,
	):
		D_list.append(Dn_s)
		X_list.append(Xn)
	return len(D_list), D_list, X_list


def load_validation_samples_random(
	file_path: Path,
	*,
	n_in: int,
	nb_sorties: int,
	convert_label: Callable[[int, int], list[int]],
	seed: int | None = None,
) -> tuple[int, list[list[int]], list[list[float]]]:
	"""Lit les mêmes données que `load_validation_samples`, mais en ordre aléatoire.

	Propriété importante
	- Contenu identique à la lecture séquentielle: les couples (Dn_s, Xn) sont
	  strictement les mêmes, seul l'ordre change.
	- Si `seed` est fourni, l'ordre est déterministe/reproductible.
	"""
	n, D_list, X_list = load_validation_samples(
		file_path,
		n_in=n_in,
		nb_sorties=nb_sorties,
		convert_label=convert_label,
	)
	indices = list(range(n))
	rng = random.Random(seed)
	rng.shuffle(indices)
	D_shuffled = [D_list[i] for i in indices]
	X_shuffled = [X_list[i] for i in indices]
	return n, D_shuffled, X_shuffled
