"""Teste de capăt la capăt pe 3 automate: parser -> world -> engine.

Toate grilele așteptate sunt scrise de mână, ca text, și NU sunt calculate cu
motorul. Calculul de mână e explicat în comentarii, lângă fiecare grilă.

Convenții de verificare
-----------------------
* Fișierele din specs/ se încarcă din JSON. Exemplele sunt mici (4x5, 6x6...),
  deci suprascriem DOAR `grid.boundary` cu "fixed" (în afara grilei = starea 0),
  altfel marginile s-ar lega și rezultatele de mână ar fi altele. Regulile,
  vecinătatea și stările rămân exact cele din fișier. Excepția e testul de
  planor pe torus, care folosește boundary-ul din fișier.
* Forest Fire: exemplele sunt deterministe. Dacă specs/forest_fire.json are și
  reguli cu "probability" < 1 (fulger, creștere), le scoatem din copia folosită
  în test; regula de propagare trebuie să fie în fișier și să aibă probabilitate 1.
  Rezultatele depind de vecinătatea din fișier; ambele variante sunt calculate de mână.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.engine.simulator import run, step
from src.parser.schema import parse_spec
from src.world.initial import from_text, to_text

SPECS = Path(__file__).resolve().parent.parent / "specs"


def load(name, **grid_overrides):
    raw = json.loads((SPECS / name).read_text(encoding="utf-8"))
    raw["grid"] = {**raw["grid"], **grid_overrides}
    return parse_spec(raw)


def load_deterministic(name, **grid_overrides):
    """Ca `load`, dar fără regulile probabilistice (probability < 1)."""
    raw = json.loads((SPECS / name).read_text(encoding="utf-8"))
    raw["grid"] = {**raw["grid"], **grid_overrides}
    raw["rules"] = [r for r in raw["rules"] if (r.get("probability") or 1.0) >= 1.0]
    return parse_spec(raw)


def assert_grid(actual, expected_text, spec, label=""):
    expected = from_text(expected_text, spec)
    assert actual.dtype == np.int8
    assert np.array_equal(actual, expected), (
        f"{label}\n--- obținut ---\n{to_text(actual, spec)}"
        f"\n--- așteptat ---\n{to_text(expected, spec)}"
    )


def count(grid, spec, state):
    return int((grid == spec.state_code(state)).sum())


# =========================================================================== #
# GAME OF LIFE  (Moore, B3/S23)
# =========================================================================== #
@pytest.fixture(scope="module")
def life():
    return load("game_of_life.json", boundary="fixed")


# Blinker. Celule vii gen0: (2,1) (2,2) (2,3).
#  - (1,2) și (3,2) văd exact 3 vii -> se nasc;
#  - (2,2) vede 2 vii (stânga+dreapta) -> supraviețuiește;
#  - (2,1) și (2,3) văd câte 1 vie -> mor;
#  - orice altă celulă moartă vede cel mult 2 vii -> rămâne moartă.
# => bară verticală. Același raționament, rotit cu 90°, o readuce în gen0.
BLINKER = [
    """
    . . . . .
    . . . . .
    . A A A .
    . . . . .
    . . . . .
    """,
    """
    . . . . .
    . . A . .
    . . A . .
    . . A . .
    . . . . .
    """,
]


def test_gol_blinker_oscillates_with_period_2(life):
    g = from_text(BLINKER[0], life)
    for gen in range(1, 7):
        g = step(g, life)
        assert_grid(g, BLINKER[gen % 2], life, f"blinker gen {gen}")


# Planor, grila 6x6, celule vii (rând, col). Număr de vecini vii calculat pe celulă.
#
# gen0: (0,1) (1,2) (2,0) (2,1) (2,2)
#   (0,1): 1 vecin -> moare      (1,0): 3 -> se naște     (1,1): 5 -> rămâne moartă
#   (1,2): 3 -> trăiește         (2,0): 1 -> moare        (2,1): 3 -> trăiește
#   (2,2): 2 -> trăiește         (3,1): 3 -> se naște     restul: <= 2 -> neschimbat
# gen1: (1,0) (1,2) (2,1) (2,2) (3,1)
#   (1,0): 1 -> moare   (1,2): 2 -> trăiește   (2,0): 3 -> se naște
#   (2,1): 4 -> moare   (2,2): 3 -> trăiește   (3,1): 2 -> trăiește   (3,2): 3 -> se naște
# gen2: (1,2) (2,0) (2,2) (3,1) (3,2)
#   (1,1): 3 -> se naște   (1,2): 1 -> moare   (2,0): 1 -> moare   (2,2): 3 -> trăiește
#   (2,3): 3 -> se naște   (3,1): 3 -> trăiește   (3,2): 2 -> trăiește   (2,1): 5 -> rămâne moartă
# gen3: (1,1) (2,2) (2,3) (3,1) (3,2)
#   (1,1): 1 -> moare   (1,2): 3 -> se naște   (2,2): 4 -> moare   (2,3): 2 -> trăiește
#   (3,1): 2 -> trăiește   (3,2): 3 -> trăiește   (3,3): 3 -> se naște
# gen4: (1,2) (2,3) (3,1) (3,2) (3,3)  = gen0 deplasat cu (+1,+1)
GLIDER = [
    """
    . A . . . .
    . . A . . .
    A A A . . .
    . . . . . .
    . . . . . .
    . . . . . .
    """,
    """
    . . . . . .
    A . A . . .
    . A A . . .
    . A . . . .
    . . . . . .
    . . . . . .
    """,
    """
    . . . . . .
    . . A . . .
    A . A . . .
    . A A . . .
    . . . . . .
    . . . . . .
    """,
    """
    . . . . . .
    . A . . . .
    . . A A . .
    . A A . . .
    . . . . . .
    . . . . . .
    """,
    """
    . . . . . .
    . . A . . .
    . . . A . .
    . A A A . .
    . . . . . .
    . . . . . .
    """,
]


def test_gol_glider_first_four_generations(life):
    g = from_text(GLIDER[0], life)
    for gen in range(1, 5):
        g = step(g, life)
        assert_grid(g, GLIDER[gen], life, f"glider gen {gen}")


def test_gol_glider_run_matches_hand_computed_generations(life):
    gens = run(from_text(GLIDER[0], life), life, 4)
    assert len(gens) == 5
    for i, expected in enumerate(GLIDER):
        assert_grid(gens[i], expected, life, f"run: glider gen {i}")


def test_gol_glider_shifts_by_one_diagonal_every_4_generations(life):
    # gen4 e gen0 mutat cu (+1,+1): ultimul pas din comentariul de mai sus.
    g = from_text(GLIDER[0], life)
    for _ in range(4):
        g = step(g, life)
    assert np.argwhere(g == 1).tolist() == [[1, 2], [2, 3], [3, 1], [3, 2], [3, 3]]


def test_gol_glider_on_torus_returns_after_4n_generations():
    # boundary "toroidal" luat din fișier. Planorul înaintează (+1,+1) la fiecare
    # 4 generații (verificat mai sus), deci pe un torus 6x6 revine după 4*6 = 24.
    # Un torus 6x6 e destul de mare ca planorul (3x3) să nu interacționeze cu el însuși.
    spec = load("game_of_life.json")
    assert spec.grid.boundary == "toroidal"
    start = from_text(GLIDER[0], spec)
    g = start
    for gen in range(1, 25):
        g = step(g, spec)
        if gen < 24:
            assert not np.array_equal(g, start), f"s-a repetat prea devreme la gen {gen}"
    assert_grid(g, GLIDER[0], spec, "glider după 24 de generații pe torus")


# =========================================================================== #
# BRIAN'S BRAIN  (Moore; READY->FIRING dacă exact 2 vecini FIRING;
#                 FIRING->DYING; DYING->READY)
# =========================================================================== #
@pytest.fixture(scope="module")
def brain():
    return load("brians_brain.json", boundary="fixed")


# gen0 (FIRING în A=(0,2), B=(1,1), C=(1,3)):
#   . . F . .
#   . F . F .
#   . . . . .
# Pas 0 -> 1:
#   FIRING -> DYING:  A, B, C devin D.
#   Pentru celulele READY numărăm vecinii FIRING (Moore, în afara grilei = 0):
#     (0,1): A,B = 2 -> F        (0,3): A,C = 2 -> F        (2,2): B,C = 2 -> F
#     (1,2): A,B,C = 3 -> rămâne READY (nu e EXACT 2)
#     (0,0),(1,0),(2,0),(2,1): doar B = 1 ; (0,4),(1,4),(2,3),(2,4): doar C = 1 -> rămân READY
# gen1:
#   . F D F .
#   . D . D .
#   . . F . .
# Pas 1 -> 2:
#   FIRING -> DYING: (0,1),(0,3),(2,2) devin D.   DYING -> READY: (0,2),(1,1),(1,3) devin '.'.
#   READY cu vecini FIRING din gen1 (P=(0,1), Q=(0,3), R=(2,2)):
#     (1,2): P,Q,R = 3 -> rămâne READY ; (0,0),(1,0),(2,1): doar P (sau R) = 1 ;
#     (0,4),(1,4),(2,3): doar Q (sau R) = 1 ; (2,0),(2,4): 0 -> nimeni nu se aprinde.
# gen2:
#   . D . D .
#   . . . . .
#   . . D . .
# Pas 2 -> 3 (în plus): nu mai există FIRING => nimic nu se aprinde, iar DYING -> READY.
# gen3: totul READY.
BRAIN = [
    """
    . . F . .
    . F . F .
    . . . . .
    """,
    """
    . F D F .
    . D . D .
    . . F . .
    """,
    """
    . D . D .
    . . . . .
    . . D . .
    """,
    """
    . . . . .
    . . . . .
    . . . . .
    """,
]


def test_brians_brain_example_two_generations(brain):
    g = from_text(BRAIN[0], brain)
    for gen in (1, 2):
        g = step(g, brain)
        assert_grid(g, BRAIN[gen], brain, f"brain gen {gen}")


def test_brians_brain_dies_out_and_stays_empty(brain):
    g = from_text(BRAIN[2], brain)
    g = step(g, brain)
    assert_grid(g, BRAIN[3], brain, "brain gen 3")
    for _ in range(5):
        g = step(g, brain)
        assert_grid(g, BRAIN[3], brain, "brain rămâne gol")


def test_brians_brain_run_matches_hand_computed(brain):
    gens = run(from_text(BRAIN[0], brain), brain, 3)
    for i in range(4):
        assert_grid(gens[i], BRAIN[i], brain, f"run: brain gen {i}")


# =========================================================================== #
# FOREST FIRE  (FIRE->EMPTY; TREE->FIRE dacă are >= 1 vecin FIRE)
# =========================================================================== #
# Rezultatul depinde de vecinătatea din specs/forest_fire.json, deci am calculat
# de mână AMBELE variante (moore și von_neumann); testul o alege pe cea din fișier.
@pytest.fixture(scope="module")
def forest():
    spec = load_deterministic("forest_fire.json", boundary="fixed")
    assert spec.neighborhood in ("moore", "von_neumann")
    return spec


# Exemplul din enunț (4 rânduri x 5 coloane). Focul pornește în (2,2).
# Arbori inițial: 4 + 4 + 3 + 5 = 16.
FOREST_START = """
    T T T T .
    T T . T T
    . T F T T
    T T T T T
"""

# ---- MOORE (8 vecini) ----
# gen0 -> gen1: (2,2) devine '.'. Vecinii ei Moore sunt (1,1) (1,2) (1,3) (2,1) (2,3)
#   (3,1) (3,2) (3,3); (1,2) e gol, ceilalți 7 sunt T -> se aprind.
# gen1 -> gen2: cele 7 focuri devin '.'. Fiecare dintre cei 9 arbori rămași are un
#   vecin Moore în foc: (0,0),(0,1) de la (1,1); (0,2) de la (1,1)/(1,3); (0,3) de la (1,3);
#   (1,0) de la (1,1); (1,4) de la (1,3); (2,4) de la (1,3),(2,3),(3,3);
#   (3,0) de la (2,1),(3,1); (3,4) de la (2,3),(3,3). Se aprind toți 9.
# gen2 -> gen3: cele 9 focuri devin '.'; nu mai există arbori, deci nu se aprinde nimic.
FOREST_MOORE = [
    FOREST_START,
    """
    T T T T .
    T F . F T
    . F . F T
    T F F F T
    """,
    """
    F F F F .
    F . . . F
    . . . . F
    F . . . F
    """,
    """
    . . . . .
    . . . . .
    . . . . .
    . . . . .
    """,
]

# ---- VON NEUMANN (4 vecini) ----
# gen0 -> gen1: vecinii lui (2,2) sunt (1,2)=gol, (2,1),(2,3),(3,2)=T -> se aprind.
# gen1 -> gen2: se aprind (1,1),(3,1) de la (2,1); (1,3),(3,3),(2,4) de la (2,3).
# gen2 -> gen3: se aprind (0,1),(1,0) de la (1,1); (0,3) de la (1,3); (1,4) de la
#   (1,3)/(2,4); (3,0) de la (3,1); (3,4) de la (3,3)/(2,4). (0,0) și (0,2) nu ating foc.
# gen3 -> gen4: (0,0) are vecinul (0,1) în foc, (0,2) are (0,1),(0,3) -> se aprind.
# gen4 -> gen5: cele 2 focuri devin '.', nu mai e niciun arbore.
FOREST_VON_NEUMANN = [
    FOREST_START,
    """
    T T T T .
    T T . T T
    . F . F T
    T T F T T
    """,
    """
    T T T T .
    T F . F T
    . . . . F
    T F . F T
    """,
    """
    T F T F .
    F . . . F
    . . . . .
    F . . . F
    """,
    """
    F . F . .
    . . . . .
    . . . . .
    . . . . .
    """,
    """
    . . . . .
    . . . . .
    . . . . .
    . . . . .
    """,
]

# Numărători scrise de mână (T și F numărate pe grilele de mai sus).
FOREST_EXPECTED = {
    "moore": {"grids": FOREST_MOORE, "trees": [16, 9, 0, 0], "fires": [1, 7, 9, 0]},
    "von_neumann": {
        "grids": FOREST_VON_NEUMANN,
        "trees": [16, 13, 8, 2, 0, 0],
        "fires": [1, 3, 5, 6, 2, 0],
    },
}
# În ambele variante toți arborii sunt conectați de celula (2,2), deci toată
# pădurea arde: la final rămân 0 arbori.


def run_until_no_fire(grid, spec, limit=50):
    gens = [grid]
    while count(gens[-1], spec, "FIRE") > 0:
        assert len(gens) <= limit, "focul nu se stinge"
        gens.append(step(gens[-1], spec))
    return gens


def test_forest_fire_example_step_by_step(forest):
    e = FOREST_EXPECTED[forest.neighborhood]
    gens = run_until_no_fire(from_text(FOREST_START, forest), forest)
    assert len(gens) == len(e["grids"]), "focul trebuie să se stingă la generația așteptată"
    for i, expected in enumerate(e["grids"]):
        assert_grid(gens[i], expected, forest, f"forest ({forest.neighborhood}) gen {i}")
        assert count(gens[i], forest, "TREE") == e["trees"][i], f"TREE la gen {i}"
        assert count(gens[i], forest, "FIRE") == e["fires"][i], f"FIRE la gen {i}"


def test_forest_fire_example_final_state_is_stable(forest):
    e = FOREST_EXPECTED[forest.neighborhood]
    gens = run_until_no_fire(from_text(FOREST_START, forest), forest)
    assert count(gens[-1], forest, "TREE") == 0
    g = gens[-1]
    for _ in range(10):
        g = step(g, forest)
        assert_grid(g, e["grids"][-1], forest, "starea finală trebuie să fie stabilă")


def test_forest_fire_run_matches_hand_computed(forest):
    e = FOREST_EXPECTED[forest.neighborhood]
    gens = run(from_text(FOREST_START, forest), forest, len(e["grids"]) - 1)
    for i, expected in enumerate(e["grids"]):
        assert_grid(gens[i], expected, forest, f"run: forest gen {i}")


def test_forest_fire_trees_never_increase_without_growth(forest):
    # Fără regula de creștere, numărul de TREE e monoton necrescător.
    trees = FOREST_EXPECTED[forest.neighborhood]["trees"]
    assert trees == sorted(trees, reverse=True)


# Al doilea exemplu, în care ARBORI RĂMÂN: coloana 2 goală taie propagarea.
FOREST_ISOLATED_START = """
    F T . T
    T T . T
"""
# MOORE: gen0->1: (0,1),(1,0),(1,1) au (0,0) ca vecin (și diagonala) -> se aprind.
#   (0,3),(1,3) nu au niciun vecin în foc (coloana 2 e goală). gen1->2: cele 3 focuri
#   devin '.'; coloana 3 tot nu are vecini în foc. Rămân 2 arbori.
# VON NEUMANN: gen0->1: (0,1),(1,0) se aprind; (1,1) nu (vecinii ei sunt T).
#   gen1->2: (1,1) se aprinde de la (0,1)/(1,0). gen2->3: (1,1) devine '.'. Rămân 2 arbori.
FOREST_ISOLATED = {
    "moore": [
        FOREST_ISOLATED_START,
        """
        . F . T
        F F . T
        """,
        """
        . . . T
        . . . T
        """,
    ],
    "von_neumann": [
        FOREST_ISOLATED_START,
        """
        . F . T
        F T . T
        """,
        """
        . . . T
        . F . T
        """,
        """
        . . . T
        . . . T
        """,
    ],
}


def test_forest_fire_isolated_trees_survive_and_are_stable(forest):
    expected_grids = FOREST_ISOLATED[forest.neighborhood]
    gens = run_until_no_fire(from_text(FOREST_ISOLATED_START, forest), forest)
    assert len(gens) == len(expected_grids)
    for i, expected in enumerate(expected_grids):
        assert_grid(gens[i], expected, forest, f"forest izolat ({forest.neighborhood}) gen {i}")
    assert count(gens[-1], forest, "TREE") == 2
    g = gens[-1]
    for _ in range(10):
        g = step(g, forest)
        assert_grid(g, expected_grids[-1], forest, "cei 2 arbori rămân stabili")