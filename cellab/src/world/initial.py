"""Lumea inițială a unui automat: din text, din fișier, aleatoare sau predefinită.

Convenții
---------
* O grilă este un ``np.ndarray`` 2D, ``dtype=np.int8``, shape ``(rows, cols)``,
  iar valorile sunt CODURI de stare (indexul stării în ``spec.states``).
* ``symbols`` este un dicționar ``{nume_stare: caracter}``, ca în spec. Stările
  care lipsesc din el iau simbolul din ``spec.symbols``, iar dacă nici acolo nu
  apar, simbolul implicit din ``spec.symbol_map()``: ``'.'`` pentru starea 0 și
  prima literă a numelui pentru restul (ALIVE -> 'A', FIRING -> 'F').
* Toate erorile de acest modul sunt ``WorldError`` (subclasă de ``ValueError``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from src.parser.schema import AutomatonSpec

__all__ = [
    "WorldError",
    "PREDEFINED_NAMES",
    "default_symbols",
    "from_text",
    "from_file",
    "random_world",
    "predefined",
    "to_text",
]

EMPTY_SYMBOL = "."
MAX_STATES = 127  # np.int8
PREDEFINED_NAMES = ("single_center", "glider", "blinker", "empty")

# Planor standard, desenat cu starea 1 (1 = celulă "vie"):
#   . A .
#   . . A
#   A A A
_GLIDER = ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2))


class WorldError(ValueError):
    """Eroare la construirea sau afișarea unei lumi inițiale."""


# --------------------------------------------------------------------------- #
# Simboluri
# --------------------------------------------------------------------------- #
def _symbol_list(spec: AutomatonSpec, symbols: Optional[Mapping[str, str]]) -> list[str]:
    """Returnează lista de caractere, indexată după codul stării.

    Ordinea de prioritate: ``symbols`` primit ca argument, apoi ``symbols`` din
    spec, apoi simbolurile implicite (``spec.symbol_map()``).
    """
    names = list(spec.states)
    if len(names) > MAX_STATES:
        raise WorldError(f"Automatul are {len(names)} stări; maximum {MAX_STATES} (int8).")

    explicit = dict(symbols) if symbols else {}
    unknown = [n for n in explicit if n not in names]
    if unknown:
        raise WorldError(
            f"symbols conține stări necunoscute: {unknown}. Stări valide: {names}."
        )

    base = spec.symbol_map()  # spec.symbols + implicite; fără ciocniri (verifică parserul)
    chars: list[str] = []
    owner: dict[str, str] = {}
    for name in names:
        if name in explicit:
            ch = explicit[name]
            if not isinstance(ch, str) or len(ch) != 1 or ch.isspace():
                raise WorldError(
                    f"Simbolul pentru starea {name!r} trebuie să fie un singur "
                    f"caracter care nu e spațiu; am primit {ch!r}."
                )
        else:
            ch = base[name]

        if ch in owner:
            other = owner[ch]
            if name in explicit and other in explicit:
                raise WorldError(
                    f"În symbols, simbolul {ch!r} e folosit pentru ambele stări "
                    f"{other!r} și {name!r}."
                )
            raise WorldError(
                f"Stările {other!r} și {name!r} au același simbol {ch!r}; "
                f"transmite symbols explicit pentru cel puțin una dintre ele "
                f"(ex.: symbols={{{name!r}: 'x'}})."
            )
        owner[ch] = name
        chars.append(ch)
    return chars


def default_symbols(spec: AutomatonSpec) -> dict[str, str]:
    """Simbolurile folosite fără argument ``symbols``: ``{nume_stare: caracter}``."""
    return dict(zip(spec.states, _symbol_list(spec, None)))


# --------------------------------------------------------------------------- #
# Text <-> grilă
# --------------------------------------------------------------------------- #
def from_text(
    text: str, spec: AutomatonSpec, symbols: Optional[Mapping[str, str]] = None
) -> np.ndarray:
    """Construiește grila dintr-un text: un rând de text = un rând de grilă.

    Spațiile (și tab-urile) sunt ignorate, la fel și rândurile complet goale.
    În mesajele de eroare, rândul și coloana sunt numărate de la 1, rândul printre
    rândurile ne-goale, coloana printre simbolurile rândului (fără spații).
    """
    chars = _symbol_list(spec, symbols)
    by_char = {ch: code for code, ch in enumerate(chars)}
    valid = ", ".join(f"{ch!r}={name}" for ch, name in zip(chars, spec.states))

    rows: list[list[int]] = []
    for line in text.splitlines():
        cells = [c for c in line if not c.isspace()]
        if not cells:
            continue
        r = len(rows) + 1
        codes = []
        for col, ch in enumerate(cells, start=1):
            if ch not in by_char:
                raise WorldError(
                    f"Simbol necunoscut {ch!r} la rândul {r}, coloana {col}. "
                    f"Simboluri valide: {valid}."
                )
            codes.append(by_char[ch])
        rows.append(codes)

    if not rows:
        raise WorldError("Textul nu conține niciun rând de grilă.")

    width = len(rows[0])
    for r, row in enumerate(rows, start=1):
        if len(row) != width:
            raise WorldError(
                f"Rânduri de lungimi diferite: rândul 1 are {width} celule, "
                f"dar rândul {r} are {len(row)}."
            )
    return np.array(rows, dtype=np.int8)


def from_file(
    path, spec: AutomatonSpec, symbols: Optional[Mapping[str, str]] = None
) -> np.ndarray:
    """Citește un fișier text în același format ca ``from_text``."""
    return from_text(Path(path).read_text(encoding="utf-8"), spec, symbols)


def to_text(
    grid: np.ndarray, spec: AutomatonSpec, symbols: Optional[Mapping[str, str]] = None
) -> str:
    """Inversa lui ``from_text``: simboluri separate prin spațiu, rânduri prin '\\n'."""
    chars = _symbol_list(spec, symbols)
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.size == 0:
        raise WorldError(f"Grila trebuie să fie 2D și nevidă; are shape {grid.shape}.")
    if not np.issubdtype(grid.dtype, np.integer):
        raise WorldError(f"Grila trebuie să conțină coduri întregi; dtype={grid.dtype}.")
    bad = (grid < 0) | (grid >= len(chars))
    if bad.any():
        r, c = (int(i) for i in np.argwhere(bad)[0])
        raise WorldError(
            f"Cod de stare invalid {int(grid[r, c])} la rândul {r + 1}, coloana {c + 1} "
            f"(coduri valide: 0..{len(chars) - 1})."
        )
    return "\n".join(" ".join(chars[int(v)] for v in row) for row in grid)


# --------------------------------------------------------------------------- #
# Dimensiuni
# --------------------------------------------------------------------------- #
def _dims(spec: AutomatonSpec, rows: Optional[int], cols: Optional[int]) -> tuple[int, int]:
    rows = spec.grid.rows if rows is None else rows
    cols = spec.grid.cols if cols is None else cols
    for label, v in (("rows", rows), ("cols", cols)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
            raise WorldError(f"{label} trebuie să fie un întreg >= 1; am primit {v!r}.")
    return int(rows), int(cols)


# --------------------------------------------------------------------------- #
# Lume aleatoare
# --------------------------------------------------------------------------- #
def random_world(
    spec: AutomatonSpec,
    densities: Mapping[str, float],
    seed,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
) -> np.ndarray:
    """Lume aleatoare: fiecare celulă ia independent starea S cu probabilitatea
    ``densities[S]``; restul probabilității devine starea cu codul 0.

    Aceeași ``seed`` (și aceleași dimensiuni/densități) dă aceeași lume. Ordinea
    cheilor din ``densities`` nu contează: stările se iau în ordinea din spec.
    Densitățile sunt probabilități pe celulă, deci procentul efectiv e aproximativ.
    """
    rows, cols = _dims(spec, rows, cols)
    names = list(spec.states)

    unknown = [n for n in densities if n not in names]
    if unknown:
        raise WorldError(f"Stări necunoscute în densities: {unknown}. Stări valide: {names}.")
    if names[0] in densities:
        raise WorldError(
            f"Starea 0 ({names[0]!r}) nu are densitate: primește automat restul "
            f"(1 - suma celorlalte densități)."
        )
    for n, d in densities.items():
        if not isinstance(d, (int, float, np.integer, np.floating)) or not 0.0 <= d <= 1.0:
            raise WorldError(f"Densitatea pentru {n!r} trebuie să fie în [0, 1]; am primit {d!r}.")
    total = float(sum(densities.values()))
    if total > 1.0 + 1e-9:
        raise WorldError(f"Suma densităților este {total:g} > 1.")

    ordered = [(code, names[code]) for code in range(1, len(names)) if names[code] in densities]
    edges = np.cumsum([float(densities[n]) for _, n in ordered])
    codes = np.array([c for c, _ in ordered], dtype=np.int8)

    rng = np.random.default_rng(seed)
    u = rng.random((rows, cols))
    grid = np.zeros((rows, cols), dtype=np.int8)
    if len(ordered):
        idx = np.searchsorted(edges, u, side="right")  # idx == len(ordered) -> starea 0
        hit = idx < len(ordered)
        grid[hit] = codes[idx[hit]]
    return grid


# --------------------------------------------------------------------------- #
# Lumi predefinite
# --------------------------------------------------------------------------- #
def predefined(
    name: str,
    spec: AutomatonSpec,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
) -> np.ndarray:
    """Lumi predefinite (celulele "vii" au codul 1):

    * ``empty``: totul pe starea 0;
    * ``single_center``: o celulă în (rows//2, cols//2);
    * ``blinker``: 3 celule orizontale, cea din mijloc în (rows//2, cols//2);
    * ``glider``: planor standard în colțul din stânga-sus, cu margine de 1
      celulă când grila are >= 5 rânduri/coloane.
    """
    key = str(name).strip().lower()
    if key not in PREDEFINED_NAMES:
        raise WorldError(f"Lume predefinită necunoscută {name!r}. Valide: {list(PREDEFINED_NAMES)}.")
    rows, cols = _dims(spec, rows, cols)
    grid = np.zeros((rows, cols), dtype=np.int8)
    if key == "empty":
        return grid

    if len(spec.states) < 2:
        raise WorldError(f"{key!r} are nevoie de cel puțin 2 stări (folosește starea cu codul 1).")

    if key == "single_center":
        grid[rows // 2, cols // 2] = 1
    elif key == "blinker":
        if cols < 3:
            raise WorldError(f"'blinker' cere cel puțin 3 coloane; grila are {cols}.")
        c = cols // 2
        grid[rows // 2, c - 1 : c + 2] = 1
    else:  # glider
        if rows < 3 or cols < 3:
            raise WorldError(f"'glider' cere cel puțin 3x3; grila are {rows}x{cols}.")
        r0 = 1 if rows >= 5 else 0
        c0 = 1 if cols >= 5 else 0
        for dr, dc in _GLIDER:
            grid[r0 + dr, c0 + dc] = 1
    return grid