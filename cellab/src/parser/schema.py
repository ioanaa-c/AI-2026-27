"""Modelul și validarea spec-urilor de automate celulare (CelLab, D2)."""

import json
import re
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (BaseModel, ConfigDict, Field, ValidationError,
                      model_validator)

_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_MAX_NEIGHBOURS = {"moore": 8, "von_neumann": 4}
_KINDS = ("neighbour_count",)
_BOUNDARIES = ("toroidal", "fixed")
_FORMS = ("in", "not_in", "min", "max")
_VALID_FORMS = ({"in"}, {"not_in"}, {"min"}, {"max"}, {"min", "max"})
_TOP_KEYS = ("name", "description", "kind", "grid", "states", "symbols",
             "neighborhood", "rules", "otherwise")
_REQUIRED_TOP = ("name", "kind", "grid", "states", "neighborhood", "rules")
_GRID_MAX = 1000


class SpecError(Exception):
    """Eroare de spec; `errors` conține toate problemele găsite."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


class GridSize(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    rows: int
    cols: int
    boundary: Literal["toroidal", "fixed"]


class Condition(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    count: str
    in_: list[int] | None = Field(None, alias="in")
    not_in: list[int] | None = None
    min: int | None = None
    max: int | None = None


class Rule(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    from_: str = Field(alias="from")
    to: str
    if_: Condition | None = Field(None, alias="if")
    probability: float = 1.0


class AutomatonSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    name: str
    description: str = ""
    kind: Literal["neighbour_count"]
    grid: GridSize
    states: list[str]
    symbols: dict[str, str] = {}
    neighborhood: Literal["moore", "von_neumann"]
    rules: list[Rule]
    otherwise: Literal["unchanged"] = "unchanged"

    @model_validator(mode="after")
    def _validate_semantics(self) -> Self:
        # aceeași validare ca în parse_spec, ca să nu existe un spec invalid
        errors = _check_raw(self.to_dict())
        if errors:
            raise SpecError(errors)
        return self

    def state_code(self, name: str) -> int:
        """Codul (indexul) unei stări; SpecError dacă starea nu există."""
        if name not in self.states:
            raise SpecError([f"Starea '{name}' nu există. "
                             f"Stări cunoscute: {', '.join(self.states)}."])
        return self.states.index(name)

    def state_name(self, code: int) -> str:
        """Numele stării cu codul dat; ValueError dacă codul e în afara listei."""
        if not 0 <= code < len(self.states):
            raise ValueError(f"Codul {code} nu există; codurile valide sunt "
                             f"între 0 și {len(self.states) - 1}.")
        return self.states[code]

    def symbol_map(self) -> dict[str, str]:
        """Starea -> simbol, complet, cu simbolurile implicite."""
        return _symbol_map(self.states, self.symbols)

    @property
    def is_stochastic(self) -> bool:
        """Adevărat dacă există cel puțin o regulă cu probability < 1."""
        return any(r.probability < 1 for r in self.rules)

    @property
    def max_neighbours(self) -> int:
        """8 la moore, 4 la von_neumann."""
        return _MAX_NEIGHBOURS[self.neighborhood]

    def to_dict(self) -> dict:
        """Dicționar cu aliasuri ("from", "if", "in"), fără valori None."""
        return self.model_dump(by_alias=True, exclude_none=True)


def _symbol_map(states: list[str], symbols: dict[str, str]) -> dict[str, str]:
    # starea 0 -> ".", celelalte -> prima literă, dacă nu e dat explicit
    result = {}
    for code, name in enumerate(states):
        default = "." if code == 0 else name[:1]
        result[name] = symbols.get(name, default)
    return result


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _state_error(prefix: str, value: Any, where: str,
                 states: list[str] | None) -> list[str]:
    # fără listă de stări validă nu avem față de ce să verificăm
    if states is None or (isinstance(value, str) and value in states):
        return []
    return [f"{prefix}starea '{value}'{where} nu există. "
            f"Stări cunoscute: {', '.join(states)}."]


def _check_condition(prefix: str, cond: Any, states: list[str] | None,
                     max_n: int | None, neighborhood: Any) -> list[str]:
    if not isinstance(cond, dict):
        return [prefix + "'if' trebuie să fie un obiect cu 'count' și o "
                "formă de condiție."]
    errors = []
    if "count" not in cond:
        errors.append(prefix + "câmpul obligatoriu 'count' lipsește din 'if'.")
    else:
        errors += _state_error(prefix, cond["count"], " din 'count'", states)
    for key in cond:
        if key != "count" and key not in _FORMS:
            errors.append(prefix + f"câmpul '{key}' din 'if' nu este "
                          "recunoscut.")
    forms = [k for k in cond if k in _FORMS]
    if not forms:
        errors.append(prefix + "condiția nu are nicio formă; folosiți exact "
                      "una dintre 'in', 'not_in', 'min', 'max' sau perechea "
                      "'min' + 'max'.")
    elif set(forms) not in _VALID_FORMS:
        errors.append(prefix + f"condiția combină {', '.join(forms)}; "
                      "folosiți exact una dintre 'in', 'not_in', 'min', 'max' "
                      "sau perechea 'min' + 'max'.")
    for key in forms:
        value = cond[key]
        if key in ("in", "not_in"):
            if not isinstance(value, list) or not all(_is_int(x) for x in value):
                errors.append(prefix + f"'{key}' trebuie să fie o listă de "
                              "numere întregi.")
                continue
            if not value:
                errors.append(prefix + f"lista '{key}' este goală.")
                continue
            values = value
        else:
            if not _is_int(value):
                errors.append(prefix + f"'{key}' trebuie să fie un număr "
                              "întreg.")
                continue
            values = [value]
        if max_n is not None:
            bad = [x for x in values if not 0 <= x <= max_n]
            if bad:
                errors.append(prefix + f"în '{key}', valorile {bad} sunt în "
                              f"afara intervalului 0…{max_n} (vecinătatea "
                              f"'{neighborhood}' are {max_n} vecini).")
    if ("min" in forms and "max" in forms
            and _is_int(cond["min"]) and _is_int(cond["max"])
            and cond["min"] > cond["max"]):
        errors.append(prefix + f"'min' ({cond['min']}) este mai mare decât "
                      f"'max' ({cond['max']}).")
    return errors


def _check_rule(index: int, rule: Any, states: list[str] | None,
                max_n: int | None, neighborhood: Any) -> list[str]:
    prefix = f"Regula {index}: "
    if not isinstance(rule, dict):
        return [prefix + "trebuie să fie un obiect cu 'from', 'to' și, "
                "opțional, 'if' și 'probability'."]
    errors = []
    for key in rule:
        if key not in ("from", "to", "if", "probability"):
            errors.append(prefix + f"câmpul '{key}' nu este recunoscut.")
    for key in ("from", "to"):
        if key not in rule:
            errors.append(prefix + f"câmpul obligatoriu '{key}' lipsește.")
        else:
            errors += _state_error(prefix, rule[key], "", states)
    if "probability" in rule:
        p = rule["probability"]
        if not _is_number(p):
            errors.append(prefix + "'probability' trebuie să fie un număr.")
        elif not 0 <= p <= 1:
            errors.append(prefix + f"'probability' este {p}; trebuie să fie "
                          "între 0 și 1.")
    if "if" in rule:
        errors += _check_condition(prefix, rule["if"], states, max_n,
                                   neighborhood)
    return errors


def _check_raw(data: Any) -> list[str]:
    """Validează un dicționar de spec și întoarce toate erorile (în română)."""
    if not isinstance(data, dict):
        return ["Spec-ul trebuie să fie un obiect JSON (dicționar)."]
    errors: list[str] = []

    for key in data:
        if key not in _TOP_KEYS:
            errors.append(f"Câmpul '{key}' nu este recunoscut. "
                          f"Câmpuri permise: {', '.join(_TOP_KEYS)}.")
    for key in _REQUIRED_TOP:
        if key not in data:
            errors.append(f"Câmpul obligatoriu '{key}' lipsește.")

    if "name" in data and (not isinstance(data["name"], str)
                           or not data["name"].strip()):
        errors.append("'name' trebuie să fie un text nevid.")
    if "description" in data and not isinstance(data["description"], str):
        errors.append("'description' trebuie să fie un text.")
    if "kind" in data and data["kind"] not in _KINDS:
        errors.append(f"'kind' are valoarea '{data['kind']}'; singura valoare "
                      "permisă este 'neighbour_count'.")
    if "otherwise" in data and data["otherwise"] != "unchanged":
        errors.append(f"'otherwise' are valoarea '{data['otherwise']}'; "
                      "singura valoare permisă este 'unchanged'.")

    # grila
    if "grid" in data:
        grid = data["grid"]
        if not isinstance(grid, dict):
            errors.append("'grid' trebuie să fie un obiect cu 'rows', 'cols' "
                          "și 'boundary'.")
        else:
            for key in grid:
                if key not in ("rows", "cols", "boundary"):
                    errors.append(f"Câmpul 'grid.{key}' nu este recunoscut.")
            for dim in ("rows", "cols"):
                if dim not in grid:
                    errors.append(f"Câmpul obligatoriu 'grid.{dim}' lipsește.")
                elif not _is_int(grid[dim]):
                    errors.append(f"grid.{dim} trebuie să fie un număr întreg.")
                elif not 1 <= grid[dim] <= _GRID_MAX:
                    errors.append(f"grid.{dim} este {grid[dim]}; trebuie să "
                                  f"fie între 1 și {_GRID_MAX}.")
            if "boundary" not in grid:
                errors.append("Câmpul obligatoriu 'grid.boundary' lipsește.")
            elif grid["boundary"] not in _BOUNDARIES:
                errors.append(f"grid.boundary are valoarea "
                              f"'{grid['boundary']}'; valorile permise sunt "
                              "'toroidal' și 'fixed'.")

    # stările
    states = None
    states_ok = False
    if "states" in data:
        raw = data["states"]
        if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
            errors.append("'states' trebuie să fie o listă de texte.")
        else:
            states = raw
            if len(raw) < 2:
                errors.append(f"'states' are {len(raw)} stări; trebuie să "
                              "fie cel puțin 2.")
            dups = [s for s in dict.fromkeys(raw) if raw.count(s) > 1]
            if dups:
                errors.append(f"Stări duplicate în 'states': "
                              f"{', '.join(dups)}. Numele trebuie să fie "
                              "unice.")
            bad_names = [s for s in dict.fromkeys(raw) if not _NAME_RE.match(s)]
            for s in bad_names:
                errors.append(f"Numele stării '{s}' este invalid; folosiți "
                              "litere mari, cifre și '_', începând cu o "
                              "literă (ex. ALIVE, S0).")
            states_ok = bool(raw) and not dups

    # simbolurile
    if "symbols" in data:
        symbols = data["symbols"]
        if not isinstance(symbols, dict):
            errors.append("'symbols' trebuie să fie un obiect stare -> "
                          "caracter.")
        else:
            symbols_ok = True
            for key, value in symbols.items():
                if states is not None and key not in states:
                    errors.append(f"'symbols': starea '{key}' nu există. "
                                  f"Stări cunoscute: {', '.join(states)}.")
                    symbols_ok = False
                if (not isinstance(value, str) or len(value) != 1
                        or value.isspace()):
                    errors.append(f"'symbols': simbolul pentru '{key}' "
                                  "trebuie să fie un singur caracter, fără "
                                  f"spațiu (primit: {value!r}).")
                    symbols_ok = False
            if states_ok and symbols_ok:
                _check_symbol_collisions(states, symbols, errors)
    elif states_ok:
        _check_symbol_collisions(states, {}, errors)

    # vecinătatea
    max_n = None
    neighborhood = data.get("neighborhood")
    if "neighborhood" in data:
        if isinstance(neighborhood, str) and neighborhood in _MAX_NEIGHBOURS:
            max_n = _MAX_NEIGHBOURS[neighborhood]
        else:
            errors.append(f"'neighborhood' are valoarea '{neighborhood}'; "
                          "valorile permise sunt 'moore' și 'von_neumann'.")

    # regulile
    if "rules" in data:
        rules = data["rules"]
        if not isinstance(rules, list):
            errors.append("'rules' trebuie să fie o listă (poate fi goală).")
        else:
            for i, rule in enumerate(rules, start=1):
                errors += _check_rule(i, rule, states, max_n, neighborhood)
    return errors


def _check_symbol_collisions(states: list[str], symbols: dict[str, str],
                             errors: list[str]) -> None:
    by_symbol: dict[str, list[str]] = {}
    for name, symbol in _symbol_map(states, symbols).items():
        by_symbol.setdefault(symbol, []).append(name)
    for symbol, names in by_symbol.items():
        if len(names) > 1:
            errors.append(f"Simbolul '{symbol}' este folosit de stările "
                          f"{', '.join(names)}. Definiți 'symbols' cu un "
                          "simbol distinct pentru fiecare stare.")


def parse_spec(data: dict) -> AutomatonSpec:
    """Validează `data` și întoarce spec-ul; SpecError cu toate erorile."""
    errors = _check_raw(data)
    if errors:
        raise SpecError(errors)
    try:
        return AutomatonSpec.model_validate(data)
    except ValidationError as exc:
        # nu ar trebui să se ajungă aici; plasă de siguranță
        raise SpecError([f"Câmpul '{'.'.join(map(str, e['loc']))}': {e['msg']}"
                         for e in exc.errors()]) from exc


def load_spec(path: str | Path) -> AutomatonSpec:
    """Citește un fișier JSON și îl validează."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise SpecError([f"Fișierul '{path}' nu este JSON valid: {exc}."]) from exc
    return parse_spec(data)


def save_spec(spec: AutomatonSpec, path: str | Path) -> None:
    """Scrie spec-ul ca JSON indentat."""
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")


def _true_set(cond: Condition, max_n: int) -> set[int]:
    # numerele de vecini pentru care condiția e adevărată
    full = set(range(max_n + 1))
    if cond.in_ is not None:
        return set(cond.in_)
    if cond.not_in is not None:
        return full - set(cond.not_in)
    low = cond.min if cond.min is not None else 0
    high = cond.max if cond.max is not None else max_n
    return set(range(low, high + 1))


def _covers(earlier: Rule, later: Rule, max_n: int) -> bool:
    # o regulă cu probabilitate < 1 poate să nu se aplice, deci nu acoperă
    if earlier.probability < 1:
        return False
    if earlier.if_ is None:
        return True
    earlier_set = _true_set(earlier.if_, max_n)
    if earlier_set == set(range(max_n + 1)):
        return True
    if later.if_ is None or later.if_.count != earlier.if_.count:
        return False
    return _true_set(later.if_, max_n) <= earlier_set


def lint_spec(spec: AutomatonSpec) -> list[str]:
    """Avertismente (nu erori), ex. reguli care nu se aplică niciodată."""
    warnings = []
    for j, later in enumerate(spec.rules):
        for i in range(j):
            earlier = spec.rules[i]
            if (earlier.from_ == later.from_
                    and _covers(earlier, later, spec.max_neighbours)):
                warnings.append(f"Regula {j + 1}: nu se aplică niciodată, "
                                f"pentru că regula {i + 1} (aceeași stare "
                                f"'from': '{later.from_}') o acoperă complet.")
                break
    return warnings