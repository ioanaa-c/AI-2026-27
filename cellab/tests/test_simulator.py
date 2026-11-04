"""Teste pentru src/engine/simulator.py.

Rezultatele așteptate sunt scrise explicit ca text și au fost verificate de mână.
"""
import numpy as np
import pytest

from src.engine.simulator import count_neighbours, run, step
from src.parser.schema import parse_spec

GOL_RULES = [
    {"from": "DEAD", "to": "ALIVE", "if": {"count": "ALIVE", "in": [3]}},
    {"from": "ALIVE", "to": "DEAD", "if": {"count": "ALIVE", "not_in": [2, 3]}},
]
BRIAN_RULES = [
    {"from": "FIRING", "to": "DYING"},
    {"from": "DYING", "to": "OFF"},
    {"from": "OFF", "to": "FIRING", "if": {"count": "FIRING", "in": [2]}},
]


def make_spec(states, rules, neighborhood="moore", boundary="toroidal", rows=5, cols=5):
    return parse_spec({
        "name": "test", "kind": "neighbour_count",
        "grid": {"rows": rows, "cols": cols, "boundary": boundary},
        "states": states, "neighborhood": neighborhood, "rules": rules,
    })


def gol(boundary="toroidal"):
    return make_spec(["DEAD", "ALIVE"], GOL_RULES, boundary=boundary)


def grid(text, spec):
    """Grilă din text, cu simbolurile spec-ului (doar pentru teste)."""
    sym = {v: spec.state_code(k) for k, v in spec.symbol_map().items()}
    lines = [ln.replace(" ", "") for ln in text.strip().splitlines()]
    return np.array([[sym[ch] for ch in ln] for ln in lines], dtype=np.int8)


# ---------- numărarea vecinilor ----------

def test_count_neighbours_moore_toroidal_corner_wraps():
    g = np.zeros((3, 4), dtype=np.int8)
    g[0, 0] = 1
    c = count_neighbours(g, 1, "moore", "toroidal")
    # celula (0,0) e vecină, prin lipirea marginilor, cu colțul opus (2,3)
    assert c[2, 3] == 1
    assert c[0, 0] == 0                    # celula nu se numără pe ea însăși
    assert c.sum() == 8                    # o celulă vie are exact 8 vecini


def test_count_neighbours_fixed_does_not_wrap():
    g = np.zeros((3, 4), dtype=np.int8)
    g[0, 0] = 1
    c = count_neighbours(g, 1, "moore", "fixed")
    assert c[2, 3] == 0
    assert c.sum() == 3                    # la colț are doar 3 vecini în grilă


def test_count_neighbours_fixed_outside_counts_as_state_0():
    g = np.zeros((3, 3), dtype=np.int8)
    c = count_neighbours(g, 0, "moore", "fixed")
    assert np.all(c == 8)                  # în afara grilei e tot starea 0


def test_count_neighbours_von_neumann():
    g = np.zeros((3, 3), dtype=np.int8)
    g[1, 1] = 1
    c = count_neighbours(g, 1, "von_neumann", "toroidal")
    expected = np.array([[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]])
    assert np.array_equal(c, expected)


# ---------- Game of Life ----------

def test_blinker_oscillates_with_period_2():
    spec = gol()
    horizontal = grid("""
        .....
        .....
        .AAA.
        .....
        .....""", spec)
    vertical = grid("""
        .....
        ..A..
        ..A..
        ..A..
        .....""", spec)
    g1 = step(horizontal, spec)
    assert np.array_equal(g1, vertical)
    assert np.array_equal(step(g1, spec), horizontal)


def test_block_is_stable():
    spec = gol()
    block = grid("""
        ....
        .AA.
        .AA.
        ....""", spec)
    assert np.array_equal(step(block, spec), block)


def test_glider_moves_one_cell_diagonally_every_4_steps():
    spec = gol()
    glider = grid("""
        .A......
        ..A.....
        AAA.....
        ........
        ........
        ........
        ........
        ........""", spec)
    gens = run(glider, spec, 4)
    assert np.array_equal(gens[4], np.roll(glider, shift=(1, 1), axis=(0, 1)))
    # și după 32 de pași (8 deplasări) revine în același loc pe grila 8x8
    assert np.array_equal(run(glider, spec, 32)[-1], glider)


def test_toroidal_vs_fixed_at_the_edge():
    start_text = """
        .AAA.
        .....
        .....
        .....
        ....."""
    toroidal = gol("toroidal")
    fixed = gol("fixed")
    # toroidal: rândul 0 e vecin cu rândul 4, deci se naște o celulă și jos
    assert np.array_equal(step(grid(start_text, toroidal), toroidal), grid("""
        ..A..
        ..A..
        .....
        .....
        ..A..""", toroidal))
    # fixed: deasupra rândului 0 e doar DEAD
    assert np.array_equal(step(grid(start_text, fixed), fixed), grid("""
        ..A..
        ..A..
        .....
        .....
        .....""", fixed))


# ---------- Brian's Brain ----------

def test_brians_brain_firing_dying_off():
    spec = make_spec(["OFF", "FIRING", "DYING"], BRIAN_RULES, rows=3, cols=3)
    g0 = grid("""
        ...
        .F.
        ...""", spec)
    g1 = step(g0, spec)
    assert np.array_equal(g1, grid("""
        ...
        .D.
        ...""", spec))
    assert np.array_equal(step(g1, spec), np.zeros((3, 3), dtype=np.int8))


def test_brians_brain_example_from_statement():
    spec = make_spec(["OFF", "FIRING", "DYING"], BRIAN_RULES, boundary="fixed", rows=3, cols=5)
    g0 = grid("""
        . . F . .
        . F . F .
        . . . . .""", spec)
    # calculat de mână: celulele OFF cu exact 2 vecini FIRING devin FIRING
    expected = grid("""
        . F D F .
        . D . D .
        . . F . .""", spec)
    assert np.array_equal(step(g0, spec), expected)


# ---------- von Neumann (Crystal Growth) ----------

def test_crystal_growth_von_neumann_makes_a_plus():
    spec = make_spec(["EMPTY", "CRYSTAL"],
                     [{"from": "EMPTY", "to": "CRYSTAL", "if": {"count": "CRYSTAL", "in": [1]}}],
                     neighborhood="von_neumann")
    g0 = grid("""
        .....
        .....
        ..C..
        .....
        .....""", spec)
    assert np.array_equal(step(g0, spec), grid("""
        .....
        ..C..
        .CCC.
        ..C..
        .....""", spec))


# ---------- ordinea regulilor și probabilități ----------

def test_first_matching_rule_wins():
    spec = make_spec(["A", "B", "C"], [
        {"from": "A", "to": "B", "if": {"count": "A", "min": 1}},
        {"from": "A", "to": "C"},
    ], rows=1, cols=3)
    g0 = np.array([[0, 0, 1]], dtype=np.int8)   # A A B
    # ambele celule A au cel puțin un vecin A (toroidal), deci se aplică prima regulă
    assert np.array_equal(step(g0, spec), np.array([[1, 1, 1]], dtype=np.int8))


def test_failed_probability_falls_through_to_next_rule():
    spec = make_spec(["A", "B", "C"], [
        {"from": "A", "to": "B", "probability": 0.0},
        {"from": "A", "to": "C"},
    ])
    g0 = np.zeros((5, 5), dtype=np.int8)
    result = step(g0, spec, np.random.default_rng(0))
    assert np.all(result == 2)                   # toate trec la a doua regulă: C


def test_same_seed_gives_same_result():
    spec = make_spec(["DEAD", "ALIVE"],
                     [{"from": "DEAD", "to": "ALIVE", "probability": 0.5}], rows=20, cols=20)
    g0 = np.zeros((20, 20), dtype=np.int8)
    a = run(g0, spec, 3, np.random.default_rng(42))
    b = run(g0, spec, 3, np.random.default_rng(42))
    c = run(g0, spec, 3, np.random.default_rng(7))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[1], c[1])


def test_stochastic_without_rng_raises():
    spec = make_spec(["DEAD", "ALIVE"], [{"from": "DEAD", "to": "ALIVE", "probability": 0.5}])
    with pytest.raises(ValueError, match="seed"):
        step(np.zeros((5, 5), dtype=np.int8), spec)


# ---------- contractul funcțiilor ----------

def test_step_does_not_modify_input_and_returns_int8():
    spec = gol()
    g0 = grid("""
        .....
        .AAA.
        .....""", spec)
    before = g0.copy()
    g1 = step(g0, spec)
    assert np.array_equal(g0, before)
    assert g1 is not g0
    assert g1.dtype == np.int8


def test_run_returns_n_plus_1_generations_starting_with_a_copy():
    spec = gol()
    g0 = grid("""
        .....
        .AAA.
        .....""", spec)
    gens = run(g0, spec, 5)
    assert len(gens) == 6
    assert np.array_equal(gens[0], g0)
    assert gens[0] is not g0
    assert len(run(g0, spec, 0)) == 1


def test_invalid_codes_in_grid_raise():
    spec = gol()
    with pytest.raises(ValueError, match="coduri"):
        step(np.full((3, 3), 5, dtype=np.int8), spec)
