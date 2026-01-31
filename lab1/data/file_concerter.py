"""file_concerter

Contient la logique de conversion des fichiers de données vers une taille fixe.

Ce module provient de l'ancien `loader.py` (déplacé ici pour mieux refléter son rôle).

Usage:
    python lab1/data/file_concerter.py

Options:
  --train/--vc/--test : chemins des fichiers d'entrée
  --n_keep 40 50 60   : tailles N à générer
  --energy Es         : énergie utilisée pour la sélection
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _is_label_token(token: str) -> bool:
    return bool(token) and token.endswith(":") and token[:-1].isdigit()


def _detect_coeffs_per_frame(n_values: int) -> int | None:
    """Tente de déduire C (coeffs par trame)."""
    for c in (39, 26, 13):
        if n_values % c == 0:
            return c
    return None


def _energy_index(c: int, energy_kind: str) -> int | None:
    """Retourne l'index de l'énergie dans une trame (0-based), selon C."""
    kind = (energy_kind or "").strip().lower()
    if c == 39:
        mapping = {"es": 12, "ed": 25, "edd": 38}
        return mapping.get(kind)
    if c == 26:
        mapping = {"es": 12, "ed": 25}
        return mapping.get(kind)
    if c == 13:
        return 12 if kind == "es" else None
    return None


def _parse_dataset_line(line: str) -> tuple[int, list[float]] | None:
    s = (line or "").strip()
    if not s:
        return None
    if s[0].isalpha():
        return None

    parts = s.split()
    if not parts or not _is_label_token(parts[0]):
        return None
    label = int(parts[0][:-1])
    if len(parts) == 1:
        return None
    try:
        values = [float(v) for v in parts[1:]]
    except Exception:
        return None
    return label, values


def _select_frames_by_energy(values: list[float], *, n_keep: int, energy_kind: str) -> list[list[float]]:
    """Convertit une ligne (flatten) en trames, sélectionne n_keep par énergie."""
    c = _detect_coeffs_per_frame(len(values))
    if c is None:
        raise ValueError(f"Impossible de déduire C: len(values)={len(values)} (attendu multiple de 39/26/13)")

    t = len(values) // c
    frames = [values[i * c : (i + 1) * c] for i in range(t)]

    e_idx = _energy_index(c, energy_kind)
    if e_idx is None:
        raise ValueError(f"Énergie '{energy_kind}' non supportée pour C={c}")

    energies = [(abs(fr[e_idx]), i) for i, fr in enumerate(frames)]
    if t >= n_keep:
        # Top N par énergie puis on conserve l'ordre temporel.
        idx = [i for _, i in sorted(energies, key=lambda x: x[0])[-n_keep:]]
        idx.sort()
        selected = [frames[i] for i in idx]
    else:
        selected = frames[:]

    # Pad si besoin
    if len(selected) < n_keep:
        pad = [0.0] * c
        selected.extend([pad] * (n_keep - len(selected)))

    return selected


def convert_line_to_static40(values: list[float], *, n_keep: int = 40, energy_kind: str = "Es") -> list[float]:
    """Pipeline: selection énergie → N trames → garder 12 statiques → flatten (N*12)."""
    frames = _select_frames_by_energy(values, n_keep=n_keep, energy_kind=energy_kind)
    static = [fr[:12] for fr in frames]
    out: list[float] = []
    for row in static:
        out.extend(row)
    return out


def convert_file(
    in_path: str | Path,
    *,
    out_path: str | Path,
    n_keep: int = 40,
    energy_kind: str = "Es",
) -> tuple[int, int]:
    """Convertit un fichier source -> fichier 'N_*'. Retourne (nb_lignes, nb_cols)."""
    in_path = Path(in_path)
    out_path = Path(out_path)

    n_rows = 0
    n_cols = 0

    with open(in_path, "r", encoding="utf-8", errors="ignore") as fin, open(
        out_path, "w", encoding="utf-8", newline="\n"
    ) as fout:
        for raw in fin:
            parsed = _parse_dataset_line(raw)
            if parsed is None:
                continue
            label, values = parsed
            vec = convert_line_to_static40(values, n_keep=n_keep, energy_kind=energy_kind)
            if n_cols == 0:
                n_cols = len(vec)

            fout.write(f"{label}: ")
            fout.write(" ".join(f"{v:.10g}" for v in vec))
            fout.write("\n")
            n_rows += 1

    return n_rows, n_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Convertit data_*.txt en N trames (sélection par énergie) + 12 statiques")
    # Par défaut, on cherche les fichiers dans le même dossier que ce script.
    parser.add_argument("--train", default="data_train.txt")
    parser.add_argument("--vc", default="data_vc.txt")
    parser.add_argument("--test", default="data_test.txt")
    parser.add_argument(
        "--n_keep",
        type=int,
        nargs="+",
        default=[40, 50, 60],
        help="Liste des N à générer (ex: --n_keep 40 50 60)",
    )
    parser.add_argument("--energy", choices=["Es", "Ed", "Edd"], default="Es")
    args = parser.parse_args()

    n_values = [int(n) for n in args.n_keep]
    if not n_values:
        print("Aucune valeur N fournie via --n_keep")
        return

    base_dir = Path(__file__).resolve().parent

    for p in (args.train, args.vc, args.test):
        src = Path(p)
        if not src.is_absolute() and not src.exists():
            src = base_dir / src

        if not src.exists():
            print(f"Fichier introuvable: {src}")
            continue

        for n_keep in n_values:
            # Dépose la destination dans le même répertoire que la source.
            dst = src.with_name(f"{n_keep}_{src.name}")
            rows, cols = convert_file(src, out_path=dst, n_keep=n_keep, energy_kind=args.energy)
            print(f"{src.name} -> {dst.name} (lignes={rows} colonnes={cols})")


if __name__ == "__main__":
    main()
