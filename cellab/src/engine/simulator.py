"""
motorul: calculez generatia urmatoare pentru automatele de tip neighbour_count
"""
from __future__ import annotations

import numpy as np

from src.parser.schema import AutomatonSpec, Condition

# Deplasările (rând, coloană) către fiecare vecin.
MOORE_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
VON_NEUMANN_OFFSETS = [(-1, 0), (0, -1), (0, 1), (1, 0)]


def _offsets(neighborhood: str) -> list[tuple[int, int]]:
    if neighborhood == "moore":
        return MOORE_OFFSETS
    if neighborhood == "von_neumann":
        return VON_NEUMANN_OFFSETS
    raise ValueError(f"Vecinătate necunoscută: '{neighborhood}'. Valori permise: moore, von_neumann.")


def count_neighbours(grid: np.ndarray, code: int,
                     neighborhood: str, boundary: str) -> np.ndarray:
    """Pentru fiecare celulă, câți vecini au starea `code`; matrice de aceeași formă ca grila."""
    mask = (grid == code).astype(np.int8)          # 1 unde celula are starea căutată, 0 în rest
    rows, cols = mask.shape
    counts = np.zeros_like(mask)

    if boundary == "toroidal":
        for dr, dc in _offsets(neighborhood):
            # rolled[r, c] = mask[r - dr, c - dc]: vecinul din direcția (-dr, -dc),
            # cu marginile lipite. Suma pe toate direcțiile dă numărul de vecini.
            counts += np.roll(mask, shift=(dr, dc), axis=(0, 1))
    elif boundary == "fixed":
        # În afara grilei se consideră starea 0, deci la numărarea stării 0
        # bordura contează ca vecin, iar la celelalte stări nu.
        border = 1 if code == 0 else 0
        padded = np.pad(mask, 1, constant_values=border)
        for dr, dc in _offsets(neighborhood):
            # Vecinul lui (r, c) e (r+dr, c+dc), adică padded[r+dr+1, c+dc+1].
            counts += padded[1 + dr: 1 + dr + rows, 1 + dc: 1 + dc + cols]
    else:
        raise ValueError(f"Margine necunoscută: '{boundary}'. Valori permise: toroidal, fixed.")
    return counts


def _condition_mask(cond: Condition, n: np.ndarray) -> np.ndarray:
    """Unde este adevărată condiția, știind n = numărul de vecini de tipul cond.count."""
    if cond.in_ is not None:
        return np.isin(n, cond.in_)
    if cond.not_in is not None:
        return ~np.isin(n, cond.not_in)
    result = np.ones(n.shape, dtype=bool)
    if cond.min is not None:
        result &= n >= cond.min
    if cond.max is not None:
        result &= n <= cond.max
    return result


def _check_grid(grid: np.ndarray, spec: AutomatonSpec) -> None:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        raise ValueError("Grila trebuie să fie o matrice numpy 2D.")
    if grid.size and (grid.min() < 0 or grid.max() >= len(spec.states)):
        raise ValueError(
            f"Grila conține coduri în afara intervalului 0..{len(spec.states) - 1} "
            f"(automatul are {len(spec.states)} stări)."
        )


def step(grid: np.ndarray, spec: AutomatonSpec,
         rng: np.random.Generator | None = None) -> np.ndarray:
    """Calculează generația următoare. Nu modifică grila primită."""
    _check_grid(grid, spec)
    if spec.is_stochastic and rng is None:
        raise ValueError("Automatul are reguli cu probabilitate; dați un seed "
                         "(rng = np.random.default_rng(seed)).")

    new = grid.astype(np.int8, copy=True)           # generația nouă, separată de cea veche
    assigned = np.zeros(grid.shape, dtype=bool)     # celule la care s-a aplicat deja o regulă
    counts_cache: dict[int, np.ndarray] = {}        # numărările se fac o singură dată per stare

    for rule in spec.rules:
        src = spec.state_code(rule.from_)
        dst = spec.state_code(rule.to)

        # Condițiile se evaluează pe generația VECHE (grid), deci actualizarea e simultană.
        eligible = (grid == src) & ~assigned
        if rule.if_ is not None:
            code = spec.state_code(rule.if_.count)
            if code not in counts_cache:
                counts_cache[code] = count_neighbours(grid, code, spec.neighborhood, spec.grid.boundary)
            eligible &= _condition_mask(rule.if_, counts_cache[code])
        if rule.probability < 1.0:
            # Tragerea se face pe toată grila, ca rezultatul să depindă doar de seed.
            # Celulele care „pică” tragerea rămân neatribuite și trec la regula următoare.
            eligible &= rng.random(grid.shape) < rule.probability

        new[eligible] = dst
        assigned |= eligible

    # otherwise = "unchanged": celulele neatribuite au rămas cu valoarea copiată din grid.
    return new


def run(grid: np.ndarray, spec: AutomatonSpec, n: int,
        rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Rulează n pași; întoarce n+1 generații, prima fiind o copie a grilei inițiale."""
    if n < 0:
        raise ValueError("Numărul de generații trebuie să fie cel puțin 0.")
    _check_grid(grid, spec)
    generations = [grid.astype(np.int8, copy=True)]
    for _ in range(n):
        generations.append(step(generations[-1], spec, rng))
    return generations
