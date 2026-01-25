"""Vérification de correspondance (alg vs mat) pour c2p59 / c2p69.

But:
  - Calculer les mêmes valeurs intermédiaires dans les deux approches
  - Vérifier qu'elles correspondent numériquement (tolérances)

Usage:
  python verify_correspondance.py
  python verify_correspondance.py --case c2p59
  python verify_correspondance.py --case c2p69

Notes:
  - Les scripts *_alg.py et *_mat.py ont été rendus importables via un return dict
    dans resolution_reseau_exemple(), sans changer leur affichage.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from generic_func import clear_console


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _flatten_numbers(x: Any) -> List[float]:
    if _is_number(x):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        out: List[float] = []
        for xi in x:
            out.extend(_flatten_numbers(xi))
        return out
    raise TypeError(f"Type non supporté pour flatten: {type(x).__name__}")


def _max_abs_diff(a: Any, b: Any) -> float:
    aa = _flatten_numbers(a)
    bb = _flatten_numbers(b)
    if len(aa) != len(bb):
        raise ValueError(f"Tailles différentes: {len(aa)} vs {len(bb)}")
    return max(abs(x - y) for x, y in zip(aa, bb)) if aa else 0.0


def assert_close(name: str, a: Any, b: Any, *, rel_tol: float, abs_tol: float) -> None:
    if _is_number(a) and _is_number(b):
        if not math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol):
            raise AssertionError(f"{name}: {a} != {b} (abs diff={abs(float(a)-float(b))})")
        return

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"{name}: longueurs différentes {len(a)} vs {len(b)}")
        for i, (ai, bi) in enumerate(zip(a, b)):
            assert_close(f"{name}[{i}]", ai, bi, rel_tol=rel_tol, abs_tol=abs_tol)
        return

    raise TypeError(f"{name}: types incompatibles {type(a).__name__} vs {type(b).__name__}")


def compare_dicts(
    title: str,
    ref: Dict[str, Any],
    test: Dict[str, Any],
    *,
    rel_tol: float,
    abs_tol: float,
    keys: Sequence[str],
) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    for key in keys:
        if key not in ref:
            errors.append(f"{key}: absent du dictionnaire 'alg'")
            continue
        if key not in test:
            errors.append(f"{key}: absent du dictionnaire 'mat'")
            continue

        try:
            assert_close(key, ref[key], test[key], rel_tol=rel_tol, abs_tol=abs_tol)
        except Exception as e:  # noqa: BLE001
            try:
                diff = _max_abs_diff(ref[key], test[key])
                errors.append(f"{key}: mismatch ({e}) ; max_abs_diff={diff:g}")
            except Exception:
                errors.append(f"{key}: mismatch ({e})")

    ok = len(errors) == 0
    return ok, errors


def run_case(case: str, *, rel_tol: float, abs_tol: float) -> int:
    def _silent_call(fn, *args, **kwargs):
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            return fn(*args, **kwargs)

    if case == "c2p59":
        from approche_alg.c2p59_alg import resolution_reseau_exemple as alg
        from approch_mat.c2p59_mat import resolution_reseau_exemple as mat

        alg_res = _silent_call(alg, 1)
        mat_res = _silent_call(mat, 1)

        keys = [
            "X",
            "W1",
            "b1",
            "W2",
            "b2",
            "d",
            "eta",
            "z1",
            "a1",
            "fp1",
            "z2",
            "a2",
            "fp2",
            "delta2",
            "delta1",
            "dW2",
            "db2",
            "dW1",
            "db1",
            "new_W2",
            "new_b2",
            "new_W1",
            "new_b1",
        ]

        ok, errors = compare_dicts("c2p59", alg_res, mat_res, rel_tol=rel_tol, abs_tol=abs_tol, keys=keys)
        print("=== VERIF c2p59 (alg vs mat) ===")
        if ok:
            print(f"OK: correspondance (rel_tol={rel_tol}, abs_tol={abs_tol})")
            return 0
        print(f"ECHEC: {len(errors)} différences (rel_tol={rel_tol}, abs_tol={abs_tol})")
        for line in errors:
            print("- " + line)
        return 1

    if case == "c2p69":
        from approche_alg.c2p69_alg import resolution_reseau_exemple as alg
        from approch_mat.c2p69_mat import resolution_reseau_exemple as mat

        alg_res = _silent_call(alg, 1)
        mat_res = _silent_call(mat, 1)

        keys = [
            "X",
            "W1",
            "b1",
            "W2",
            "b2",
            "d",
            "eta",
            "z1",
            "a1",
            "fp1",
            "z2",
            "a2",
            "fp2",
            "delta2",
            "delta1",
            "dW2",
            "db2",
            "dW1",
            "db1",
            "new_W2",
            "new_b2",
            "new_W1",
            "new_b1",
        ]

        ok, errors = compare_dicts("c2p69", alg_res, mat_res, rel_tol=rel_tol, abs_tol=abs_tol, keys=keys)
        print("=== VERIF c2p69 (alg vs mat) ===")
        if ok:
            print(f"OK: correspondance (rel_tol={rel_tol}, abs_tol={abs_tol})")
            return 0
        print(f"ECHEC: {len(errors)} différences (rel_tol={rel_tol}, abs_tol={abs_tol})")
        for line in errors:
            print("- " + line)
        return 1

    raise ValueError("case doit être c2p59 ou c2p69")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["c2p59", "c2p69", "all"], default="all")
    parser.add_argument("--rel", type=float, default=1e-9, help="relative tolerance")
    parser.add_argument("--abs", dest="abs_", type=float, default=1e-12, help="absolute tolerance")
    args = parser.parse_args()

    clear_console()

    rc = 0
    if args.case in ("c2p59", "all"):
        rc |= run_case("c2p59", rel_tol=args.rel, abs_tol=args.abs_)
    if args.case in ("c2p69", "all"):
        rc |= run_case("c2p69", rel_tol=args.rel, abs_tol=args.abs_)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
