"""Teste pentru src/analysis/questions.py."""
import numpy as np
import pytest

from src.analysis.questions import (count_after, count_cells, find_cycle,
                                    first_stable, state_after)
from src.engine.simulator import run
from src.parser.schema import parse_spec

GOL = parse_spec({
    "name": "Game of Life", "kind": "neighbour_count",
    "grid": {"rows": 8, "cols": 8, "boundary": "toroidal"},
    "states": ["DEAD", "ALIVE"], "neighborhood": "moore",
    "rules": [
        {"from": "DEAD", "to": "ALIVE", "if": {"count": "ALIVE", "in": [3]}},
        {"from": "ALIVE", "to": "DEAD", "if": {"count": "ALIVE", "not_in": [2, 3]}},
    ],
})
FOREST = parse_spec({
    "name": "Forest Fire", "kind": "neighbour_count",
    "grid": {"rows": 1, "cols": 5, "boundary": "fixed"},
    "states": ["EMPTY", "TREE", "FIRE"], "neighborhood": "moore",
    "rules": [
        {"from": "FIRE", "to": "EMPTY"},
        {"from": "TREE", "to": "FIRE", "if": {"count": "FIRE", "min": 1}},
    ],
})
RANDOM = parse_spec({
    "name": "Random", "kind": "neighbour_count",
    "grid": {"rows": 5, "cols": 5, "boundary": "toroidal"},
    "states": ["DEAD", "ALIVE"], "neighborhood": "moore",
    "rules": [{"from": "DEAD", "to": "ALIVE", "probability": 0.3}],
})


def gol_grid(cells, size=8):
    g = np.zeros((size, size), dtype=np.int8)
    for r, c in cells:
        g[r, c] = 1
    return g


GLIDER = gol_grid([(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)])
BLINKER = gol_grid([(3, 2), (3, 3), (3, 4)])
BLOCK = gol_grid([(3, 3), (3, 4), (4, 3), (4, 4)])


# ---------- întrebările 1 și 4 ----------

def test_state_after_matches_run():
    assert np.array_equal(state_after(GLIDER, GOL, 7), run(GLIDER, GOL, 7)[-1])
    assert np.array_equal(state_after(GLIDER, GOL, 0), GLIDER)


def test_count_cells_and_count_after():
    assert count_cells(GLIDER, GOL, "ALIVE") == 5
    assert count_cells(GLIDER, GOL, "DEAD") == 64 - 5
    assert count_after(GLIDER, GOL, 8, "ALIVE") == 5     # glider-ul își păstrează forma


def test_forest_fire_counts():
    g = np.array([[1, 1, 2, 1, 0]], dtype=np.int8)       # T T F T .
    # pasul 1: T F . F .   pasul 2: F . . . .   pasul 3: . . . . .
    assert count_after(g, FOREST, 1, "FIRE") == 2
    assert count_after(g, FOREST, 2, "FIRE") == 1
    assert count_after(g, FOREST, 3, "TREE") == 0


def test_unknown_state_name_raises():
    with pytest.raises(Exception, match="nu există"):
        count_cells(GLIDER, GOL, "ALIV")


# ---------- întrebările 2 și 5 (bonus) ----------

def test_first_stable():
    assert first_stable(BLOCK, GOL, 10) == 0
    g = np.array([[1, 1, 2, 1, 0]], dtype=np.int8)
    assert first_stable(g, FOREST, 10) == 3              # de la generația 3 totul e EMPTY
    assert first_stable(BLINKER, GOL, 50) is None        # oscilează, nu se oprește


def test_find_cycle():
    assert find_cycle(BLINKER, GOL, 10) == (0, 2)
    assert find_cycle(BLOCK, GOL, 10) == (0, 1)
    assert find_cycle(GLIDER, GOL, 40) == (0, 32)        # pe grila 8x8 revine după 32 de pași
    assert find_cycle(GLIDER, GOL, 31) is None           # nu are destui pași ca să-l vadă


def test_find_cycle_with_transient():
    g = np.array([[1, 1, 2, 1, 0]], dtype=np.int8)
    assert find_cycle(g, FOREST, 10) == (3, 1)           # 3 generații până la starea stabilă


def test_stochastic_questions_raise():
    g = np.zeros((5, 5), dtype=np.int8)
    with pytest.raises(ValueError, match="probabilitati"):
        first_stable(g, RANDOM, 10)
    with pytest.raises(ValueError, match="probabilitati"):
        find_cycle(g, RANDOM, 10)
