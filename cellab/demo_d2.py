"""Demo pentru Livrabilul 2.

Pentru Game of Life, Brian's Brain și Forest Fire: încarcă spec-ul din specs/,
creează lumea inițială (predefined sau from_text), rulează 10 pași, afișează
primele 3 generații, răspunde la „câte celule de tip X după N pași”, salvează
experimentul, îl reîncarcă și verifică dacă e identic.

Rulare (din folderul cellab/):  python demo_d2.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # ca importurile `src.` să meargă din orice folder

from src.parser.schema import load_spec
from src.world.initial import from_text, predefined, to_text
from src.engine.simulator import run
from src.analysis.questions import count_cells
from src.storage.experiments import (
    ExperimentConfig, save_experiment, load_experiment, list_experiments,
)

SPECS_DIR = ROOT / "specs"
EXPERIMENTS_DIR = ROOT / "experiments"
STEPS = 10
SEED = 42

# Brian's Brain: două perechi de celule FIRING alăturate (grilă 8x8, toroidală)
BRIANS_BRAIN_TEXT = """
. . . . . . . .
. . . . . . . .
. . . F . . . .
. . . F . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
"""

# Forest Fire: exemplul din enunț (grilă 4x5, margini fixe)
FOREST_FIRE_TEXT = """
T T T T .
T T . T T
. T F T T
T T T T T
"""


def demo_automaton(spec_file, state_to_count, make_world, init_method, init_params):
    print(f"\n{'=' * 50}\n DEMO: {spec_file}\n{'=' * 50}")

    # 1. Spec-ul
    spec = load_spec(SPECS_DIR / spec_file)
    print(f"Specificație încărcată: {spec.name}")

    # 2. Lumea inițială
    initial_grid = make_world(spec)
    print(f"Lume inițială ({init_method}): {initial_grid.shape[0]}x{initial_grid.shape[1]}")

    # 3. Rulare
    rng = np.random.default_rng(SEED)
    generations = run(initial_grid, spec, STEPS, rng=rng)

    # 4. Primele 3 generații
    for i in range(min(3, len(generations))):
        print(f"\nGenerația {i}:")
        print(to_text(generations[i], spec))

    # 5. Întrebarea 4: câte celule de tip X după N pași
    n_cells = count_cells(generations[-1], spec, state_to_count)
    print(f"\nQ: Câte celule '{state_to_count}' există după {STEPS} pași?")
    print(f"A: {n_cells}")

    # 6. Salvare
    config = ExperimentConfig(
        generations=STEPS,
        seed=SEED,
        store_intermediate=True,
        store_every=1,
        init_method=init_method,
        init_params=init_params,
    )
    exp_id = save_experiment(spec, initial_grid, config, generations,
                             base_dir=str(EXPERIMENTS_DIR))
    print(f"\nExperiment salvat: {exp_id}")

    # 7. Reîncărcare și verificare
    loaded = load_experiment(exp_id, base_dir=str(EXPERIMENTS_DIR))
    identical = (
        loaded.spec.to_dict() == spec.to_dict()
        and loaded.indices == list(range(len(generations)))
        and all(np.array_equal(a, b) for a, b in zip(generations, loaded.generations))
    )
    print("Reîncărcare OK" if identical else "EROARE: experimentul reîncărcat diferă!")
    return identical


def main():
    results = [
        demo_automaton(
            "game_of_life.json", "ALIVE",
            lambda spec: predefined("glider", spec, rows=8, cols=8),
            "predefined", {"name": "glider", "rows": 8, "cols": 8},
        ),
        demo_automaton(
            "brians_brain.json", "FIRING",
            lambda spec: from_text(BRIANS_BRAIN_TEXT, spec),
            "manual", {"text": BRIANS_BRAIN_TEXT.strip()},
        ),
        demo_automaton(
            "forest_fire.json", "TREE",
            lambda spec: from_text(FOREST_FIRE_TEXT, spec),
            "manual", {"text": FOREST_FIRE_TEXT.strip()},
        ),
    ]

    print(f"\n{'=' * 50}\n Ultimele experimente din index.db\n{'=' * 50}")
    for row in list_experiments(str(EXPERIMENTS_DIR))[:3]:
        print(f"{row['date'][:19]}  {row['name']:<15} gen={row['generations']}  seed={row['seed']}  id={row['id']}")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
