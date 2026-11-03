"""Teste pentru src/parser/schema.py; depind doar de schema.py."""

from pathlib import Path

import pytest

from src.parser.schema import (AutomatonSpec, SpecError, lint_spec, load_spec,
                               parse_spec, save_spec)

SPECS_DIR = Path(__file__).resolve().parents[1] / "specs"
SPEC_FILES = sorted(SPECS_DIR.glob("*.json"))
DELETE = object()


def _gol() -> dict:
    return {
        "name": "Game of Life",
        "description": "Conway's Game of Life",
        "kind": "neighbour_count",
        "grid": {"rows": 50, "cols": 50, "boundary": "toroidal"},
        "states": ["DEAD", "ALIVE"],
        "symbols": {"DEAD": ".", "ALIVE": "A"},
        "neighborhood": "moore",
        "rules": [
            {"from": "DEAD", "to": "ALIVE", "if": {"count": "ALIVE", "in": [3]}},
            {"from": "ALIVE", "to": "DEAD",
             "if": {"count": "ALIVE", "not_in": [2, 3]}},
        ],
        "otherwise": "unchanged",
    }


def _with(*changes) -> dict:
    # aplică schimbări (cale, valoare) pe o copie a spec-ului de bază
    data = _gol()
    for path, value in changes:
        target = data
        for key in path[:-1]:
            target = target[key]
        if value is DELETE:
            del target[path[-1]]
        else:
            target[path[-1]] = value
    return data


def _errors(data) -> list[str]:
    with pytest.raises(SpecError) as info:
        parse_spec(data)
    return info.value.errors


# spec-urile din specs/

def test_exista_cele_cinci_spec_uri():
    assert {p.stem for p in SPEC_FILES} == {
        "game_of_life", "brians_brain", "forest_fire", "cyclic_ca",
        "crystal_growth"}


@pytest.mark.parametrize("path", SPEC_FILES, ids=lambda p: p.stem)
def test_spec_se_incarca_fara_erori(path):
    spec = load_spec(path)
    assert parse_spec(spec.to_dict()) == spec
    assert lint_spec(spec) == []
    assert not spec.is_stochastic


def test_proprietatile_game_of_life():
    spec = load_spec(SPECS_DIR / "game_of_life.json")
    assert spec.state_code("ALIVE") == 1
    assert spec.state_name(0) == "DEAD"
    assert spec.symbol_map() == {"DEAD": ".", "ALIVE": "A"}
    assert spec.max_neighbours == 8


def test_cyclic_are_simbolurile_cifre():
    spec = load_spec(SPECS_DIR / "cyclic_ca.json")
    assert spec.symbol_map() == {"S0": "0", "S1": "1", "S2": "2", "S3": "3"}


def test_simboluri_implicite_cand_lipsesc():
    spec = parse_spec(_with((("symbols",), DELETE)))
    assert spec.symbol_map() == {"DEAD": ".", "ALIVE": "A"}


def test_von_neumann_are_4_vecini():
    assert parse_spec(_with((("neighborhood",), "von_neumann"))).max_neighbours == 4


def test_spec_stocastic():
    spec = parse_spec(_with((("rules", 0, "probability"), 0.5)))
    assert spec.is_stochastic


def test_automat_static_cu_reguli_goale():
    assert parse_spec(_with((("rules",), []))).rules == []


def test_to_dict_foloseste_aliasuri_fara_none():
    rule = load_spec(SPECS_DIR / "brians_brain.json").to_dict()["rules"][0]
    assert rule == {"from": "FIRING", "to": "DYING", "probability": 1.0}


def test_state_code_necunoscut():
    spec = parse_spec(_gol())
    with pytest.raises(SpecError):
        spec.state_code("ZOMBIE")


def test_save_si_load_pastreaza_spec_ul(tmp_path):
    spec = parse_spec(_gol())
    save_spec(spec, tmp_path / "gol.json")
    assert load_spec(tmp_path / "gol.json") == spec


# spec-uri greșite: (id, schimbări, mesaj așteptat)

BAD_CASES = [
    ("stare_necunoscuta_in_to", [(("rules", 1, "to"), "ALIV")],
     "Regula 2: starea 'ALIV' nu există. Stări cunoscute: DEAD, ALIVE."),
    ("stare_necunoscuta_in_from", [(("rules", 0, "from"), "DED")],
     "Regula 1: starea 'DED' nu există."),
    ("count_necunoscut", [(("rules", 0, "if", "count"), "LIVE")],
     "Regula 1: starea 'LIVE' din 'count' nu există."),
    ("stari_duplicate", [(("states",), ["DEAD", "DEAD"])],
     "Stări duplicate în 'states': DEAD."),
    ("o_singura_stare", [(("states",), ["DEAD"])], "cel puțin 2"),
    ("nume_stare_invalid", [(("states",), ["dead", "ALIVE"])],
     "Numele stării 'dead' este invalid"),
    ("conditie_fara_forma", [(("rules", 0, "if", "in"), DELETE)],
     "Regula 1: condiția nu are nicio formă"),
    ("conditie_cu_doua_forme", [(("rules", 0, "if", "min"), 2)],
     "Regula 1: condiția combină in, min"),
    ("valoare_peste_maxim_moore", [(("rules", 0, "if", "in"), [3, 9])],
     "Regula 1: în 'in', valorile [9] sunt în afara intervalului 0…8"),
    ("valoare_peste_maxim_von_neumann",
     [(("neighborhood",), "von_neumann"), (("rules", 0, "if", "in"), [5])],
     "afara intervalului 0…4"),
    ("valoare_negativa", [(("rules", 1, "if", "not_in"), [-1])],
     "Regula 2: în 'not_in', valorile [-1]"),
    ("min_mai_mare_decat_max",
     [(("rules", 0, "if"), {"count": "ALIVE", "min": 5, "max": 2})],
     "Regula 1: 'min' (5) este mai mare decât 'max' (2)."),
    ("probabilitate_peste_1", [(("rules", 0, "probability"), 1.5)],
     "Regula 1: 'probability' este 1.5; trebuie să fie între 0 și 1."),
    ("probabilitate_negativa", [(("rules", 0, "probability"), -0.1)],
     "Regula 1: 'probability' este -0.1"),
    ("rows_zero", [(("grid", "rows"), 0)],
     "grid.rows este 0; trebuie să fie între 1 și 1000."),
    ("cols_prea_mare", [(("grid", "cols"), 1001)], "grid.cols este 1001"),
    ("kind_invalid", [(("kind",), "ltl")], "'kind' are valoarea 'ltl'"),
    ("boundary_invalid", [(("grid", "boundary"), "wrap")],
     "grid.boundary are valoarea 'wrap'"),
    ("neighborhood_invalid", [(("neighborhood",), "hex")],
     "'neighborhood' are valoarea 'hex'"),
    ("simboluri_in_conflict", [(("symbols", "DEAD"), "A")],
     "Simbolul 'A' este folosit de stările DEAD, ALIVE."),
    ("simbol_cu_spatiu", [(("symbols", "DEAD"), " ")],
     "'symbols': simbolul pentru 'DEAD'"),
    ("simbol_pentru_stare_inexistenta", [(("symbols", "ZOMBIE"), "Z")],
     "'symbols': starea 'ZOMBIE' nu există."),
    ("camp_obligatoriu_lipsa", [(("rules",), DELETE)],
     "Câmpul obligatoriu 'rules' lipsește."),
    ("camp_necunoscut", [(("neigborhood",), "moore")],
     "Câmpul 'neigborhood' nu este recunoscut."),
]


@pytest.mark.parametrize("changes, expected",
                         [c[1:] for c in BAD_CASES],
                         ids=[c[0] for c in BAD_CASES])
def test_spec_gresit_da_mesajul_asteptat(changes, expected):
    errors = _errors(_with(*changes))
    assert any(expected in e for e in errors), errors


def test_simboluri_implicite_in_conflict():
    data = _with((("states",), ["OFF", "FIRING", "FLAME"]),
                 (("symbols",), DELETE), (("rules",), []))
    assert any("Simbolul 'F' este folosit de stările FIRING, FLAME" in e
               for e in _errors(data))


def test_se_strang_toate_erorile_odata():
    errors = _errors(_with((("rules", 1, "to"), "ALIV"),
                           (("grid", "rows"), 0),
                           (("rules", 0, "probability"), 2),
                           (("kind",), "x")))
    assert len(errors) == 4


def test_spec_care_nu_e_dictionar():
    assert len(_errors([])) == 1


def test_modelul_singur_refuza_spec_invalid_semantic():
    with pytest.raises(SpecError) as info:
        AutomatonSpec.model_validate(_with((("rules", 1, "to"), "ALIV")))
    assert "Regula 2" in info.value.errors[0]


def test_load_spec_cu_json_stricat(tmp_path):
    path = tmp_path / "prost.json"
    path.write_text("{ nu e json", encoding="utf-8")
    with pytest.raises(SpecError):
        load_spec(path)


# avertismente (lint)

def test_lint_regula_acoperita_de_una_necondizionata():
    rules = [{"from": "ALIVE", "to": "DEAD"},
             {"from": "ALIVE", "to": "ALIVE",
              "if": {"count": "ALIVE", "in": [2]}}]
    warnings = lint_spec(parse_spec(_with((("rules",), rules))))
    assert len(warnings) == 1
    assert "Regula 2: nu se aplică niciodată" in warnings[0]
    assert "regula 1" in warnings[0]


def test_lint_regula_acoperita_de_o_conditie_mai_larga():
    rules = [{"from": "ALIVE", "to": "DEAD",
              "if": {"count": "ALIVE", "min": 2}},
             {"from": "ALIVE", "to": "ALIVE",
              "if": {"count": "ALIVE", "in": [3, 4]}}]
    assert len(lint_spec(parse_spec(_with((("rules",), rules))))) == 1


def test_lint_nu_avertizeaza_cand_regula_anterioara_e_probabilista():
    rules = [{"from": "ALIVE", "to": "DEAD", "probability": 0.5},
             {"from": "ALIVE", "to": "ALIVE"}]
    assert lint_spec(parse_spec(_with((("rules",), rules)))) == []


def test_lint_nu_avertizeaza_la_conditii_pe_stari_diferite():
    rules = [{"from": "ALIVE", "to": "DEAD",
              "if": {"count": "DEAD", "min": 2}},
             {"from": "ALIVE", "to": "ALIVE",
              "if": {"count": "ALIVE", "in": [2]}}]
    assert lint_spec(parse_spec(_with((("rules",), rules)))) == []


def test_lint_nu_avertizeaza_la_stari_from_diferite():
    rules = [{"from": "DEAD", "to": "ALIVE"}, {"from": "ALIVE", "to": "DEAD"}]
    assert lint_spec(parse_spec(_with((("rules",), rules)))) == []
