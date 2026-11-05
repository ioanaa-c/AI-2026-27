import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
import numpy as np

# Se presupune existența parserului conform specificațiilor
from src.parser.schema import AutomatonSpec, parse_spec

class ExperimentConfig(BaseModel):
    generations: int = Field(..., ge=0)
    seed: Optional[int] = None
    store_intermediate: bool = True
    store_every: int = Field(default=1, ge=1)
    init_method: str = Field(..., pattern="^(manual|random|predefined)$")
    init_params: Dict[str, Any] = Field(default_factory=dict)

class ExperimentData(BaseModel):
    spec: AutomatonSpec
    config: ExperimentConfig
    generations: List[np.ndarray]
    indices: List[int]
    
    # Sintaxa actualizată pentru Pydantic v2
    model_config = ConfigDict(arbitrary_types_allowed=True)

def _get_db_connection(base_dir: Path) -> sqlite3.Connection:
    db_path = base_dir / "index.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT,
            date TEXT,
            generations INTEGER,
            seed INTEGER,
            path TEXT
        )
    """)
    conn.commit()
    return conn

def save_experiment(
    spec: AutomatonSpec, 
    initial_grid: np.ndarray, 
    config: ExperimentConfig, 
    generations: List[np.ndarray], 
    base_dir: str = "experiments"
) -> str:
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    exp_id = str(uuid.uuid4())
    exp_dir = base_path / exp_id
    exp_dir.mkdir(exist_ok=True)
    
    # 1. Salvare spec.json
    with open(exp_dir / "spec.json", "w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, indent=2)
        
    # 2. Salvare config.json
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)
        
    # 3. Salvare generations.npz
    indices = []
    arrays_to_save = {}
    total_gens = len(generations)
    
    for i, grid in enumerate(generations):
        # Determinăm dacă salvăm indexul curent
        save_this = False
        if i == 0 or i == total_gens - 1:
            save_this = True
        elif config.store_intermediate and i % config.store_every == 0:
            save_this = True
            
        if save_this and i not in indices:
            indices.append(i)
            arrays_to_save[f"gen_{i}"] = grid

    arrays_to_save["indices"] = np.array(indices, dtype=np.int32)
    np.savez_compressed(exp_dir / "generations.npz", **arrays_to_save)
    
    # 4. Inserare în baza de date SQLite
    conn = _get_db_connection(base_path)
    date_str = datetime.now().isoformat()
    try:
        conn.execute(
            "INSERT INTO experiments (id, name, date, generations, seed, path) VALUES (?, ?, ?, ?, ?, ?)",
            (exp_id, spec.name, date_str, config.generations, config.seed, str(exp_dir))
        )
        conn.commit()
    finally:
        conn.close()
        
    return exp_id

def load_experiment(exp_id: str, base_dir: str = "experiments") -> ExperimentData:
    exp_dir = Path(base_dir) / exp_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experimentul {exp_id} nu a fost găsit.")
        
    # Încărcare spec
    with open(exp_dir / "spec.json", "r", encoding="utf-8") as f:
        spec_data = json.load(f)
        spec = parse_spec(spec_data)
        
    # Încărcare config
    with open(exp_dir / "config.json", "r", encoding="utf-8") as f:
        config_data = json.load(f)
        config = ExperimentConfig(**config_data)
        
    # Încărcare matrice
    # `with` închide fișierul .npz (altfel, pe Windows, folderul nu poate fi șters)
    with np.load(exp_dir / "generations.npz") as npz_data:
        indices = npz_data["indices"].tolist()
        loaded_generations = [npz_data[f"gen_{idx}"] for idx in indices]
        
    return ExperimentData(
        spec=spec,
        config=config,
        generations=loaded_generations,
        indices=indices
    )

def list_experiments(base_dir: str = "experiments") -> List[Dict[str, Any]]:
    base_path = Path(base_dir)
    if not (base_path / "index.db").exists():
        return []
        
    conn = _get_db_connection(base_path)
    try:
        cursor = conn.execute("SELECT id, name, date, generations, seed, path FROM experiments ORDER BY date DESC")
        columns = [col[0] for col in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return results
    finally:
        conn.close()