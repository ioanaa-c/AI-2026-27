"""Teste pentru src/world/initial.py (nu depind de fișierele din specs/)."""

import numpy as np
import pytest

from src.parser.schema import parse_spec
from src.world.initial import (
    WorldError,
    default_symbols,
    from_file,
    from_text,
    predefined,
    random_world,
    to_text,
)


def _spec(states, neighborhood="moore", rows=5, cols=7, symbols=None):
    """Spec minim valid; regulile nu contează pentru lumea inițială."""
    data = (
        {
            "name": "t",
            "kind": "neighbour_count",
            "grid": {"rows": rows, "cols": cols, "boundary": "fixed"},
            "states": states,
            "neighborhood": neighborhood,
            "rules": [{"from": states[1], "to": states[0]}],
            "otherwise": "unchanged",
        }
    )
    if symbols is not None:
        data["symbols"] = symbols
    return parse_spec(data)


@pytest.fixture
def life():  # 5 x 7, neneprătrat, ca să prindem confuziile rows/cols
    return _spec(["DEAD", "ALIVE"])


@pytest.fixture
def brain():
    return _spec(["READY", "FIRING", "DYING"])


@pytest.fixture
def custom():
    # FIRE și FIRING ar avea amândouă 'F'; parserul cere ca spec-ul să le separe prin "symbols"
    return _spec(["EMPTY", "FIRE", "FIRING"], symbols={"FIRING": "G"})


# ------------------------------------------------------------------ simboluri
def test_default_symbols(life, brain):
    assert default_symbols(life) == {"DEAD": ".", "ALIVE": "A"}
    assert default_symbols(brain) == {"READY": ".", "FIRING": "F", "DYING": "D"}


def test_default_symbols_come_from_spec_symbols(custom):
    assert default_symbols(custom) == {"EMPTY": ".", "FIRE": "F", "FIRING": "G"}
    g = from_text(". F G", custom)
    assert g.tolist() == [[0, 1, 2]]
    assert to_text(g, custom) == ". F G"


def test_argument_symbols_override_spec_symbols(custom):
    g = from_text(". F X", custom, symbols={"FIRING": "X"})
    assert g.tolist() == [[0, 1, 2]]
    assert to_text(g, custom, symbols={"FIRING": "X"}) == ". F X"
    with pytest.raises(WorldError, match="'G'"):  # simbolul din spec nu mai e valid
        from_text(". F G", custom, symbols={"FIRING": "X"})


def test_collision_caused_by_override_is_clear_error(brain):
    # FIRING -> 'D' s-ar ciocni cu simbolul implicit al lui DYING
    with pytest.raises(WorldError, match="același simbol 'D'"):
        from_text(". F D", brain, symbols={"FIRING": "D"})
    with pytest.raises(WorldError, match="symbols"):
        to_text(np.zeros((1, 1), dtype=np.int8), brain, symbols={"FIRING": "D"})


def test_collision_resolved_by_overriding_both_states(brain):
    sym = {"FIRING": "D", "DYING": "F"}  # simbolurile schimbate între ele
    assert from_text(". D F", brain, symbols=sym).tolist() == [[0, 1, 2]]


# -------------------------------------------------------------------- from_text
def test_from_text_basic(life):
    g = from_text(". A .\nA A .", life)
    assert g.dtype == np.int8
    assert g.shape == (2, 3)
    assert g.tolist() == [[0, 1, 0], [1, 1, 0]]


def test_from_text_spaces_ignored_and_blank_lines_skipped(life):
    compact = from_text(".A.\nAA.", life)
    spaced = from_text("\n   .  A  .  \n\tA A .\n\n", life)
    assert np.array_equal(compact, spaced)


def test_from_text_three_states(brain):
    g = from_text(". . F . .\n. F . D .", brain)
    assert g.shape == (2, 5)
    assert g.tolist() == [[0, 0, 1, 0, 0], [0, 1, 0, 2, 0]]


def test_from_text_custom_symbols(life):
    g = from_text("_ # _\n# # _", life, symbols={"DEAD": "_", "ALIVE": "#"})
    assert g.tolist() == [[0, 1, 0], [1, 1, 0]]


def test_from_text_partial_symbols_keep_defaults_for_the_rest(brain):
    g = from_text(". # D", brain, symbols={"FIRING": "#"})
    assert g.tolist() == [[0, 1, 2]]


def test_from_text_ragged_rows(life):
    with pytest.raises(WorldError, match="rândul 2 are 2"):
        from_text(". A .\n. A", life)


def test_from_text_unknown_symbol_reports_row_and_column(life):
    with pytest.raises(WorldError) as e:
        from_text(". A .\n. A X", life)
    assert "rândul 2" in str(e.value) and "coloana 3" in str(e.value)
    assert "'X'" in str(e.value)


def test_from_text_unknown_symbol_column_ignores_spaces(life):
    with pytest.raises(WorldError, match="rândul 1, coloana 2"):
        from_text(".    X", life)


def test_from_text_is_case_sensitive(life):
    with pytest.raises(WorldError, match="'a'"):
        from_text(". a", life)


def test_from_text_empty_text(life):
    for t in ("", "   \n\n  "):
        with pytest.raises(WorldError, match="niciun rând"):
            from_text(t, life)


def test_bad_symbols_argument(life, brain):
    with pytest.raises(WorldError, match="necunoscute"):
        from_text(". A", life, symbols={"GHOST": "G"})
    with pytest.raises(WorldError, match="un singur"):
        from_text(". A", life, symbols={"ALIVE": "AA"})
    with pytest.raises(WorldError, match="un singur"):
        from_text(". A", life, symbols={"ALIVE": " "})
    with pytest.raises(WorldError, match="ambele"):
        from_text(". A", brain, symbols={"FIRING": "x", "DYING": "x"})


# -------------------------------------------------------------------- from_file
def test_from_file_same_as_from_text(life, tmp_path):
    text = ". A .\nA A .\n"
    p = tmp_path / "world.txt"
    p.write_text(text, encoding="utf-8")
    assert np.array_equal(from_file(p, life), from_text(text, life))
    assert np.array_equal(from_file(str(p), life), from_text(text, life))


def test_from_file_custom_symbols_and_errors(life, tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("_ #\n# _", encoding="utf-8")
    g = from_file(p, life, symbols={"DEAD": "_", "ALIVE": "#"})
    assert g.tolist() == [[0, 1], [1, 0]]
    p.write_text(". A\n. ?", encoding="utf-8")
    with pytest.raises(WorldError, match="rândul 2, coloana 2"):
        from_file(p, life)


def test_from_file_missing_file(life, tmp_path):
    with pytest.raises(FileNotFoundError):
        from_file(tmp_path / "nu_exista.txt", life)


# ------------------------------------------------------------------ random_world
def test_random_world_shape_dtype_defaults_from_spec(life):
    g = random_world(life, {"ALIVE": 0.3}, seed=1)
    assert g.shape == (5, 7)  # rows x cols din spec.grid
    assert g.dtype == np.int8
    assert set(np.unique(g)) <= {0, 1}


def test_random_world_rows_cols_override(life):
    assert random_world(life, {"ALIVE": 0.3}, seed=1, rows=3, cols=9).shape == (3, 9)
    assert random_world(life, {"ALIVE": 0.3}, seed=1, rows=4).shape == (4, 7)
    assert random_world(life, {"ALIVE": 0.3}, seed=1, cols=2).shape == (5, 2)


def test_random_world_same_seed_same_world(life):
    a = random_world(life, {"ALIVE": 0.4}, seed=42, rows=30, cols=30)
    b = random_world(life, {"ALIVE": 0.4}, seed=42, rows=30, cols=30)
    c = random_world(life, {"ALIVE": 0.4}, seed=43, rows=30, cols=30)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_world_density_extremes(life):
    assert not random_world(life, {"ALIVE": 0.0}, seed=3, rows=20, cols=20).any()
    assert not random_world(life, {}, seed=3, rows=20, cols=20).any()
    assert (random_world(life, {"ALIVE": 1.0}, seed=3, rows=20, cols=20) == 1).all()


def test_random_world_density_is_approximately_respected(life):
    g = random_world(life, {"ALIVE": 0.3}, seed=7, rows=200, cols=200)
    assert abs((g == 1).mean() - 0.3) < 0.02


def test_random_world_multiple_states_rest_is_state_zero(brain):
    g = random_world(brain, {"FIRING": 0.2, "DYING": 0.3}, seed=5, rows=200, cols=200)
    assert abs((g == 1).mean() - 0.2) < 0.02
    assert abs((g == 2).mean() - 0.3) < 0.02
    assert abs((g == 0).mean() - 0.5) < 0.02


def test_random_world_dict_order_does_not_matter(brain):
    a = random_world(brain, {"FIRING": 0.2, "DYING": 0.3}, seed=5, rows=40, cols=40)
    b = random_world(brain, {"DYING": 0.3, "FIRING": 0.2}, seed=5, rows=40, cols=40)
    assert np.array_equal(a, b)


def test_random_world_sum_over_one(brain):
    with pytest.raises(WorldError, match="> 1"):
        random_world(brain, {"FIRING": 0.6, "DYING": 0.5}, seed=0)


def test_random_world_sum_exactly_one_is_ok(brain):
    g = random_world(brain, {"FIRING": 0.5, "DYING": 0.5}, seed=0, rows=50, cols=50)
    assert not (g == 0).any()


def test_random_world_unknown_state(life):
    with pytest.raises(WorldError, match="GHOST"):
        random_world(life, {"GHOST": 0.1}, seed=0)


def test_random_world_invalid_density_values(life):
    with pytest.raises(WorldError, match="\\[0, 1\\]"):
        random_world(life, {"ALIVE": -0.1}, seed=0)
    with pytest.raises(WorldError, match="\\[0, 1\\]"):
        random_world(life, {"ALIVE": 1.5}, seed=0)


def test_random_world_density_for_state_zero_rejected(life):
    with pytest.raises(WorldError, match="Starea 0"):
        random_world(life, {"DEAD": 0.5}, seed=0)


def test_random_world_invalid_dims(life):
    for bad in (0, -3, 2.5):
        with pytest.raises(WorldError, match="rows"):
            random_world(life, {"ALIVE": 0.1}, seed=0, rows=bad)


# -------------------------------------------------------------------- predefined
def test_predefined_empty(life):
    g = predefined("empty", life)
    assert g.shape == (5, 7) and g.dtype == np.int8 and not g.any()


def test_predefined_single_center_odd_and_even(life):
    g = predefined("single_center", life, rows=5, cols=5)
    assert to_text(g, life) == "\n".join(
        [". . . . .", ". . . . .", ". . A . .", ". . . . .", ". . . . ."]
    )
    g = predefined("single_center", life, rows=4, cols=6)  # (rows//2, cols//2) = (2, 3)
    assert np.argwhere(g == 1).tolist() == [[2, 3]]
    assert g.sum() == 1


def test_predefined_blinker(life):
    g = predefined("blinker", life, rows=5, cols=5)
    assert to_text(g, life) == "\n".join(
        [". . . . .", ". . . . .", ". A A A .", ". . . . .", ". . . . ."]
    )
    assert predefined("blinker", life).sum() == 3  # 5 x 7 din spec


def test_predefined_glider(life):
    g = predefined("glider", life, rows=6, cols=6)
    assert to_text(g, life) == "\n".join(
        [
            ". . . . . .",
            ". . A . . .",
            ". . . A . .",
            ". A A A . .",
            ". . . . . .",
            ". . . . . .",
        ]
    )


def test_predefined_glider_tiny_grid_has_no_margin(life):
    g = predefined("glider", life, rows=3, cols=3)
    assert to_text(g, life) == ". A .\n. . A\nA A A"


def test_predefined_defaults_come_from_spec(life):
    assert predefined("glider", life).shape == (5, 7)
    assert predefined("single_center", life, rows=9, cols=3).shape == (9, 3)


def test_predefined_name_is_normalised(life):
    assert np.array_equal(predefined("  Blinker ", life), predefined("blinker", life))


def test_predefined_unknown_name(life):
    with pytest.raises(WorldError, match="necunoscută"):
        predefined("pulsar", life)


def test_predefined_too_small(life):
    with pytest.raises(WorldError, match="glider"):
        predefined("glider", life, rows=2, cols=5)
    with pytest.raises(WorldError, match="blinker"):
        predefined("blinker", life, rows=5, cols=2)


def test_predefined_uses_code_one_in_multi_state_spec(brain):
    g = predefined("single_center", brain, rows=3, cols=3)
    assert g[1, 1] == 1 and g.sum() == 1  # FIRING


# ----------------------------------------------------------------------- to_text
def test_to_text_exact_output(brain):
    g = np.array([[0, 0, 1], [2, 0, 0]], dtype=np.int8)
    assert to_text(g, brain) == ". . F\nD . ."


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_roundtrip_random_world(brain, seed):
    g = random_world(brain, {"FIRING": 0.3, "DYING": 0.3}, seed=seed, rows=9, cols=13)
    assert np.array_equal(from_text(to_text(g, brain), brain), g)


def test_roundtrip_with_custom_symbols(brain):
    sym = {"READY": "_", "FIRING": "#", "DYING": "~"}
    g = random_world(brain, {"FIRING": 0.3, "DYING": 0.3}, seed=9, rows=6, cols=6)
    text = to_text(g, brain, symbols=sym)
    assert set(text) <= {"_", "#", "~", " ", "\n"}
    assert np.array_equal(from_text(text, brain, symbols=sym), g)


def test_roundtrip_predefined_worlds(life):
    for name in ("empty", "single_center", "blinker", "glider"):
        g = predefined(name, life, rows=8, cols=8)
        assert np.array_equal(from_text(to_text(g, life), life), g)


def test_to_text_roundtrip_keeps_dtype_and_shape(life):
    g = from_text(". A\nA .", life)
    back = from_text(to_text(g, life), life)
    assert back.dtype == np.int8 and back.shape == g.shape


def test_to_text_rejects_invalid_grids(life):
    with pytest.raises(WorldError, match="Cod de stare invalid 5 la rândul 1, coloana 2"):
        to_text(np.array([[0, 5]], dtype=np.int8), life)
    with pytest.raises(WorldError, match="Cod de stare invalid -1"):
        to_text(np.array([[0, -1]], dtype=np.int8), life)
    with pytest.raises(WorldError, match="2D"):
        to_text(np.array([0, 1, 0], dtype=np.int8), life)
    with pytest.raises(WorldError, match="întregi"):
        to_text(np.array([[0.0, 1.0]]), life)