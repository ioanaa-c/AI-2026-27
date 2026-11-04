"""intrebarile despre evolutia unui automat

intrebarea 1: state_after      intrebarea 4: count_after
intrebarea 2: first_stable     intrebarea 5: find_cycle   (doar pentru automate deterministe)

"""
from __future__ import annotations

import hashlib

import numpy as np

from src.engine.simulator import step
from src.parser.schema import AutomatonSpec


def _state_hash(grid: np.ndarray) -> str:
    """Amprenta unei stări (formula din contract); grile egale au aceeași amprentă."""
    return hashlib.sha1(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _require_deterministic(spec: AutomatonSpec, question: str) -> None:
    if spec.is_stochastic:
        raise ValueError(
            f"{question} nu se poate decide pentru un automat cu probabilități: "
            "o grilă care reapare nu garantează același viitor, pentru că pasul "
            "următor folosește alte numere aleatoare."
        )


def state_after(grid: np.ndarray, spec: AutomatonSpec, n: int,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Întrebarea 1: starea după n iterații."""
    if n < 0:
        raise ValueError("Numărul de iterații trebuie să fie cel puțin 0.")
    current = grid.astype(np.int8, copy=True)
    for _ in range(n):
        current = step(current, spec, rng)
    return current


def count_cells(grid: np.ndarray, spec: AutomatonSpec, state_name: str) -> int:
    """Câte celule au starea `state_name` în grila dată."""
    return int(np.count_nonzero(grid == spec.state_code(state_name)))


def count_after(grid: np.ndarray, spec: AutomatonSpec, n: int, state_name: str,
                rng: np.random.Generator | None = None) -> int:
    """Întrebarea 4: câte celule de tipul `state_name` există după n iterații."""
    code = spec.state_code(state_name)   # validăm numele înainte de a simula
    return int(np.count_nonzero(state_after(grid, spec, n, rng) == code))


def first_stable(grid: np.ndarray, spec: AutomatonSpec, max_steps: int) -> int | None:
    """Întrebarea 2: cel mai mic k cu generația k == generația k+1; None dacă nu apare în max_steps pași."""
    _require_deterministic(spec, "Stabilitatea")
    current = grid.astype(np.int8, copy=True)
    for k in range(max_steps):
        nxt = step(current, spec)
        if np.array_equal(current, nxt):
            return k
        current = nxt
    return None


def find_cycle(grid: np.ndarray, spec: AutomatonSpec, max_steps: int) -> tuple[int, int] | None:
    """Întrebarea 5: (start, perioadă) a primului ciclu; perioada 1 = stare stabilă.

    start = prima generație care face parte din ciclu. None dacă nicio stare nu se
    repetă în primii max_steps pași.
    """
    _require_deterministic(spec, "Ciclul")
    seen: dict[str, int] = {}               # amprenta stării -> generația la care a apărut prima dată
    current = grid.astype(np.int8, copy=True)
    for k in range(max_steps + 1):
        h = _state_hash(current)
        if h in seen:
            return seen[h], k - seen[h]
        seen[h] = k
        if k < max_steps:
            current = step(current, spec)
    return None
