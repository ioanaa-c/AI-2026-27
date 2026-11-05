"""intrebarile despre evolutia unui automat

intrebarea 1: state_after      
intrebarea 2: first_stable  
intrebarea 4: count_after  
intrebarea 5: find_cycle   (pt auto deterministe)

"""
from __future__ import annotations

import hashlib

import numpy as np

from src.engine.simulator import step
from src.parser.schema import AutomatonSpec


"""amprenta unei stari
grilele egale au aceeasi amprenta"""
def _state_hash(grid: np.ndarray) -> str:
    
    return hashlib.sha1(grid.tobytes() + str(grid.shape).encode()).hexdigest() #adaug si dimensiunea in hash


def _require_deterministic(spec: AutomatonSpec, question: str) -> None:

    if spec.is_stochastic: #true daca vreo regula are probabilitate sub 1, nu pot decide
        raise ValueError(
            f"{question} nu se poate decide pentru un automat cu probabilitati: "
            "o grila care reapare nu garanteaza acelasi viitor, pentru ca pasul "
            "urmator foloseste alte numere aleatoare."
        )

"""1: starea dupa n iteratii?"""
def state_after(grid: np.ndarray, spec: AutomatonSpec, n: int,
                rng: np.random.Generator | None = None) -> np.ndarray:
    
    if n < 0:
        raise ValueError("Numarul de iteratii trebuie sa fie cel putin 0")
    
    current = grid.astype(np.int8, copy=True)

    for _ in range(n):
        current = step(current, spec, rng) #current e mereu inlocuit, nu mai salvez istoricul
        #ma intereseaza starea finala

    return current


"""Cate celule au starea `state_name` in grila data"""
def count_cells(grid: np.ndarray, spec: AutomatonSpec, state_name: str) -> int:

    #spec.state_code(ALIVE) e 1 de exemplu
    #grid == 1 fac o masca care raspunde la intrebarea daca celula e ALIVE sau nu
    #count_nonzero: cate raspunsuri de DA sunt 
    return int(np.count_nonzero(grid == spec.state_code(state_name)))


"""4: cate celule de tipul `state_name` exista dupa n iteratii?"""
def count_after(grid: np.ndarray, spec: AutomatonSpec, n: int, state_name: str,
                rng: np.random.Generator | None = None) -> int:
    
    code = spec.state_code(state_name)   # transform numele starii in cod
    #iau tabla dupa cele n iteratii
    #fac masca cu == code
    #count_nonzero: cate celule de acel tip sunt adica numar DA urile
    return int(np.count_nonzero(state_after(grid, spec, n, rng) == code))


"""2: starea stabila: cel mai mic k cu generatia k == generatia k+1; None daca nu apare in max_steps pasi"""
def first_stable(grid: np.ndarray, spec: AutomatonSpec, max_steps: int) -> int | None:
    
    _require_deterministic(spec, "Stabilitatea") #nu vreau automat cu probab
    current = grid.astype(np.int8, copy=True)   #copie

    for k in range(max_steps): #parcurg pasii
        nxt = step(current, spec) #caclulez iteratia viitoare
        if np.array_equal(current, nxt):   #e egala cu cea curenta?
            return k
        current = nxt #merg mai departe
    return None


"""5: (start, perioada) a primului ciclu; perioada 1 = stare stabila.

    start = prima generatie care face parte din ciclu. None daca nicio stare nu se
    repeta in primii max_steps pasi.
    """
def find_cycle(grid: np.ndarray, spec: AutomatonSpec, max_steps: int) -> tuple[int, int] | None:
    
    _require_deterministic(spec, "Ciclul")
    seen: dict[str, int] = {}    # amprenta starii o pastrez intr un fel de dictionar
    current = grid.astype(np.int8, copy=True)

    for k in range(max_steps + 1):
        h = _state_hash(current) #amprenta tablei de acum
        if h in seen: #daca am mai vazut amprenta
            return seen[h], k - seen[h] #unde a inceput ciclul
        seen[h] = k #daca nu, o notez
        if k < max_steps:
            current = step(current, spec) #next gen
    return None
