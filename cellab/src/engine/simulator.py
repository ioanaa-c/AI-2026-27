"""
motorul: calculez generatia urmatoare pentru automatele de tip neighbour_count
"""
from __future__ import annotations

import numpy as np

from src.parser.schema import AutomatonSpec, Condition

# deplasarile pe rand si coloana catre fiecare vecin
MOORE_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
VON_NEUMANN_OFFSETS = [(-1, 0), (0, -1), (0, 1), (1, 0)]


def _offsets(neighborhood: str) -> list[tuple[int, int]]:
    if neighborhood == "moore":
        return MOORE_OFFSETS
    if neighborhood == "von_neumann":
        return VON_NEUMANN_OFFSETS
    raise ValueError(f"vecinatate necunoscuta: '{neighborhood}'. Valori permise: moore, von_neumann.")


def count_neighbours(grid: np.ndarray, code: int,
                     neighborhood: str, boundary: str) -> np.ndarray:
    
    """Pentru fiecare celula, cati vecini au starea `code`"""
    mask = (grid == code).astype(np.int8)          # da sau nu transform in 1 sau 0
    rows, cols = mask.shape
    counts = np.zeros_like(mask) #matricea de counturi 0 init

    if boundary == "toroidal":

        for dr, dc in _offsets(neighborhood):
            counts += np.roll(mask, shift=(dr, dc), axis=(0, 1)) #mut masca

    elif boundary == "fixed":

        #celulele din afara au starea pe care o numar acum?
        border = 1 if code == 0 else 0
        padded = np.pad(mask, 1, constant_values=border)

        for dr, dc in _offsets(neighborhood):
            #vecinul lui (r, c) e (r+dr, c+dc), adica padded[r+dr+1, c+dc+1].
            counts += padded[1 + dr: 1 + dr + rows, 1 + dc: 1 + dc + cols]
    else:
        raise ValueError(f"Margine necunoscuta: '{boundary}'. Valori permise: toroidal, fixed.")
    return counts

# unde este adevarata conditia, stiind n = numarul de vecini de tipul cond.count
def _condition_mask(cond: Condition, n: np.ndarray) -> np.ndarray:
    
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


#ma asigur ca grila grila e valida
def _check_grid(grid: np.ndarray, spec: AutomatonSpec) -> None:

    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        raise ValueError("grila trebuie sa fie o matrice numpy 2D")
    
    if grid.size and (grid.min() < 0 or grid.max() >= len(spec.states)):
        raise ValueError(
            f"grila contine coduri in afara intervalului 0, {len(spec.states) - 1} "
            f"(automatul are {len(spec.states)} stari)."
        )

"""calculeaza generatia urmatoare fara sa modific grila primita"""
def step(grid: np.ndarray, spec: AutomatonSpec,
         rng: np.random.Generator | None = None) -> np.ndarray:
    
    _check_grid(grid, spec) #verific ca grila sa fie valida
    if spec.is_stochastic and rng is None:
        raise ValueError("automatul are reguli cu probabilitate; introdu un seed"
                         "(rng = np.random.default_rng(seed)).")

    new = grid.astype(np.int8, copy=True)           # copie a grilei care va fi noua generatie
    assigned = np.zeros(grid.shape, dtype=bool)     # celule la care s-a aplicat deja o regula
    counts_cache: dict[int, np.ndarray] = {}        # numararile se fac o singura data per stare

    for rule in spec.rules: #pentru fiecare regula in parte
        
        src = spec.state_code(rule.from_)
        dst = spec.state_code(rule.to)

        eligible = (grid == src) & ~assigned #celulele care nu au primit inca regula
        
        if rule.if_ is not None: #daca regula contine if
            code = spec.state_code(rule.if_.count)
            
            if code not in counts_cache:
                counts_cache[code] = count_neighbours(grid, code, spec.neighborhood, spec.grid.boundary)

            #conditia din json rule.if_ si cati numaratorile din cache
            #raspunde pt fiecare celula cu da/nu
            eligible &= _condition_mask(rule.if_, counts_cache[code]) 
        
        if rule.probability < 1.0:
            #se trage cate un nr aleator pt fiecare celula
            #cele care pica sunt scoase din eligible
            eligible &= rng.random(grid.shape) < rule.probability

        new[eligible] = dst #scriu starea noua doar unde eligible e DA
        assigned |= eligible #marchez ca folosit

    #otherwise = "unchanged"
    return new


#intoarce n+1 generatii, prima e copia grilei initiale
#n nr gen dorite
def run(grid: np.ndarray, spec: AutomatonSpec, n: int,
        rng: np.random.Generator | None = None) -> list[np.ndarray]:
    
    if n < 0:
        raise ValueError("numarul de generatii trebuie sa fie cel putin 0")
    
    _check_grid(grid, spec)

    generations = [grid.astype(np.int8, copy=True)] #lista cu copia gridului

    for _ in range(n):
        generations.append(step(generations[-1], spec, rng)) #fiecare gen in list

    return generations
