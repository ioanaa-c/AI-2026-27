import os
import numpy as np
from pathlib import Path

# Importuri din modulele proiectului (asumând structura Cerinței 2)
from src.parser.schema import load_spec
from src.world.initial import predefined, to_text
from src.engine.simulator import run
from src.analysis.questions import count_cells
from src.storage.experiments import ExperimentConfig, save_experiment, load_experiment

def demo_automaton(spec_filename: str, state_to_count: str, init_name: str = "glider"):
    print(f"\n{'='*50}")
    print(f" DEMO: {spec_filename}")
    print(f"{'='*50}")
    
    spec_path = Path("specs") / spec_filename
    if not spec_path.exists():
        print(f"Fișierul {spec_path} lipsește! Creează-l pentru a rula demo-ul.")
        return

    # 1. Încărcare spec
    spec = load_spec(str(spec_path))
    print(f"Specificație încărcată cu succes: {spec.name}")
    
    # 2. Lume inițială
    # Pentru simplicitate apelăm "predefined"; fallback pe creare aleatoare dacă nu e definit
    try:
        initial_grid = predefined(init_name, spec, rows=spec.grid.rows, cols=spec.grid.cols)
    except Exception:
        # Fallback (în caz că 'predefined' nu are acest pattern implementat)
        initial_grid = np.zeros((spec.grid.rows, spec.grid.cols), dtype=np.int8)
        initial_grid[1, 1] = spec.state_code(spec.states[1])
    
    # Configurare experiment
    steps = 10
    config = ExperimentConfig(
        generations=steps,
        store_intermediate=True,
        store_every=1,
        init_method="predefined",
        init_params={"name": init_name}
    )
    
    # 3. Rulare
    rng = np.random.default_rng(42)  # Seed fix pentru determinism în demo
    generations = run(initial_grid, spec, steps, rng=rng)
    
    # 4. Afișare primele 3 generații
    print("\n--- Primele 3 generații ---")
    for i in range(min(3, len(generations))):
        print(f"Generația {i}:")
        print(to_text(generations[i], spec))
        print("-" * 20)
        
    # 5. Întrebare analitică
    count_after_10 = count_cells(generations[-1], spec, state_to_count)
    print(f"Q: Câte celule '{state_to_count}' există după {steps} pași?")
    print(f"A: {count_after_10}")
    
    # 6. Salvare experiment
    print("\n--- Salvare experiment ---")
    exp_id = save_experiment(spec, initial_grid, config, generations)
    print(f"Experiment salvat cu ID: {exp_id}")
    
    # 7. Încărcare și verificare (Round-trip)
    print("\n--- Încărcare experiment ---")
    loaded_data = load_experiment(exp_id)
    
    is_identical = True
    if len(generations) != len(loaded_data.generations):
        is_identical = False
    else:
        for orig, loaded in zip(generations, loaded_data.generations):
            if not np.array_equal(orig, loaded):
                is_identical = False
                break
                
    if is_identical:
        print("Reîncărcare OK: Grilele sunt identice!")
    else:
        print("EROARE: Grilele încărcate diferă de cele originale.")

if __name__ == "__main__":
    # Asigurăm folderul specs
    os.makedirs("specs", exist_ok=True)
    
    # Exemplu de rulări:
    # Va trebui să existe fișierele corespunzătoare în folderul "specs"
    demo_automaton("game_of_life.json", state_to_count="ALIVE", init_name="glider")
    demo_automaton("brians_brain.json", state_to_count="FIRING", init_name="oscillator")
    demo_automaton("forest_fire.json", state_to_count="FIRE", init_name="random_forest")