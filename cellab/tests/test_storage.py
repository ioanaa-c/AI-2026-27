"""Teste pentru src/storage/experiments.py (salvare / încărcare experimente)."""
import json
import sqlite3

import numpy as np
import pytest
from pydantic import ValidationError

from src.engine.simulator import run
from src.parser.schema import load_spec, parse_spec
from src.storage.experiments import (
    ExperimentConfig, list_experiments, load_experiment, save_experiment,
)
from src.world.initial import random_world

SPECS = "specs"


@pytest.fixture
def gol():
    return load_spec(f"{SPECS}/game_of_life.json")


def _config(**kw):
    base = dict(generations=10, seed=1, store_intermediate=True,
                store_every=1, init_method="random", init_params={"ALIVE": 0.3})
    base.update(kw)
    return ExperimentConfig(**base)


def _random_run(spec, seed=1, n=10):
    g0 = random_world(spec, {"ALIVE": 0.3}, seed=seed, rows=12, cols=12)
    return g0, run(g0, spec, n)


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #
def test_round_trip_identic(gol, tmp_path):
    g0, gens = _random_run(gol)
    exp_id = save_experiment(gol, g0, _config(), gens, base_dir=str(tmp_path))

    loaded = load_experiment(exp_id, base_dir=str(tmp_path))
    assert loaded.indices == list(range(11))
    assert len(loaded.generations) == len(gens)
    for orig, back in zip(gens, loaded.generations):
        assert np.array_equal(orig, back)
        assert back.dtype == np.int8
    assert loaded.spec.to_dict() == gol.to_dict()
    assert loaded.config == _config()


def test_fisierele_din_folder(gol, tmp_path):
    g0, gens = _random_run(gol)
    exp_id = save_experiment(gol, g0, _config(), gens, base_dir=str(tmp_path))
    folder = tmp_path / exp_id
    assert (folder / "spec.json").is_file()
    assert (folder / "config.json").is_file()
    assert (folder / "generations.npz").is_file()
    assert (tmp_path / "index.db").is_file()
    assert json.loads((folder / "spec.json").read_text(encoding="utf-8"))["name"] == "Game of Life"


def test_round_trip_brians_brain(tmp_path):
    spec = load_spec(f"{SPECS}/brians_brain.json")
    g0 = np.zeros((6, 6), dtype=np.int8)
    g0[2, 2] = g0[2, 3] = spec.state_code("FIRING")
    gens = run(g0, spec, 5)
    exp_id = save_experiment(spec, g0, _config(generations=5, init_method="manual"),
                             gens, base_dir=str(tmp_path))
    loaded = load_experiment(exp_id, base_dir=str(tmp_path))
    assert loaded.spec.states == ["OFF", "FIRING", "DYING"]
    assert all(np.array_equal(a, b) for a, b in zip(gens, loaded.generations))


def test_round_trip_spec_cu_probabilitate(tmp_path):
    spec = parse_spec({
        "name": "Prob", "kind": "neighbour_count",
        "grid": {"rows": 5, "cols": 5, "boundary": "toroidal"},
        "states": ["A", "B"], "neighborhood": "moore",
        "rules": [{"from": "A", "to": "B", "probability": 0.5}],
    })
    g0 = np.zeros((5, 5), dtype=np.int8)
    gens = run(g0, spec, 3, rng=np.random.default_rng(7))
    exp_id = save_experiment(spec, g0, _config(generations=3, seed=7), gens, base_dir=str(tmp_path))
    loaded = load_experiment(exp_id, base_dir=str(tmp_path))
    assert loaded.spec.rules[0].probability == 0.5
    assert all(np.array_equal(a, b) for a, b in zip(gens, loaded.generations))


# --------------------------------------------------------------------------- #
# store_every / store_intermediate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("every, expected", [
    (1, list(range(11))),
    (2, [0, 2, 4, 6, 8, 10]),
    (3, [0, 3, 6, 9, 10]),      # ultima generație se păstrează mereu
    (4, [0, 4, 8, 10]),
    (50, [0, 10]),
])
def test_store_every(gol, tmp_path, every, expected):
    g0, gens = _random_run(gol)
    exp_id = save_experiment(gol, g0, _config(store_every=every), gens, base_dir=str(tmp_path))
    loaded = load_experiment(exp_id, base_dir=str(tmp_path))
    assert loaded.indices == expected
    for idx, grid in zip(loaded.indices, loaded.generations):
        assert np.array_equal(grid, gens[idx])


def test_fara_intermediare_doar_prima_si_ultima(gol, tmp_path):
    g0, gens = _random_run(gol)
    exp_id = save_experiment(gol, g0, _config(store_intermediate=False, store_every=1),
                             gens, base_dir=str(tmp_path))
    loaded = load_experiment(exp_id, base_dir=str(tmp_path))
    assert loaded.indices == [0, 10]
    assert np.array_equal(loaded.generations[0], gens[0])
    assert np.array_equal(loaded.generations[1], gens[-1])


def test_o_singura_generatie(gol, tmp_path):
    g0, gens = _random_run(gol, n=0)
    exp_id = save_experiment(gol, g0, _config(generations=0), gens, base_dir=str(tmp_path))
    assert load_experiment(exp_id, base_dir=str(tmp_path)).indices == [0]


# --------------------------------------------------------------------------- #
# Reproductibilitate cu seed
# --------------------------------------------------------------------------- #
def test_acelasi_seed_aceleasi_generatii(gol, tmp_path):
    g0, gens = _random_run(gol, seed=123)
    exp_id = save_experiment(gol, g0, _config(seed=123, init_params={"ALIVE": 0.3}),
                             gens, base_dir=str(tmp_path))

    # Re-rulăm doar din ce s-a salvat: spec + config (seed, densități)
    loaded = load_experiment(exp_id, base_dir=str(tmp_path))
    g0_again = random_world(loaded.spec, loaded.config.init_params,
                            seed=loaded.config.seed, rows=12, cols=12)
    gens_again = run(g0_again, loaded.spec, loaded.config.generations)

    assert len(gens_again) == len(loaded.generations)
    for a, b in zip(gens_again, loaded.generations):
        assert np.array_equal(a, b)


def test_seed_diferit_lume_diferita(gol):
    g1, _ = _random_run(gol, seed=1)
    g2, _ = _random_run(gol, seed=2)
    assert not np.array_equal(g1, g2)


# --------------------------------------------------------------------------- #
# SQLite / list_experiments
# --------------------------------------------------------------------------- #
def test_list_experiments_cele_mai_noi_primele(gol, tmp_path):
    g0, gens = _random_run(gol)
    ids = [save_experiment(gol, g0, _config(seed=s), gens, base_dir=str(tmp_path))
           for s in (1, 2, 3)]
    rows = list_experiments(str(tmp_path))
    assert [r["id"] for r in rows] == ids[::-1]
    assert rows[0]["name"] == "Game of Life"
    assert rows[0]["seed"] == 3
    assert rows[0]["generations"] == 10


def test_randul_din_sqlite(gol, tmp_path):
    g0, gens = _random_run(gol)
    exp_id = save_experiment(gol, g0, _config(seed=None), gens, base_dir=str(tmp_path))
    with sqlite3.connect(tmp_path / "index.db") as conn:
        row = conn.execute("SELECT name, generations, seed, path FROM experiments WHERE id=?",
                           (exp_id,)).fetchone()
    assert row[0] == "Game of Life" and row[1] == 10 and row[2] is None
    assert row[3].endswith(exp_id)


def test_list_experiments_folder_gol(tmp_path):
    assert list_experiments(str(tmp_path / "nu_exista")) == []


# --------------------------------------------------------------------------- #
# Erori
# --------------------------------------------------------------------------- #
def test_experiment_inexistent(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiment("nu-exista", base_dir=str(tmp_path))


@pytest.mark.parametrize("bad", [
    {"store_every": 0},
    {"store_every": -2},
    {"generations": -1},
    {"init_method": "magic"},
])
def test_config_invalid(bad):
    with pytest.raises(ValidationError):
        _config(**bad)
