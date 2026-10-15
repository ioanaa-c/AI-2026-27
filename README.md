# CelLab

Aplicație care primește descrierea unui automat celular în limbaj natural, o validează, simulează generațiile următoare și răspunde la întrebări despre evoluția lumii.

> Stadiu: Deliverable 1 – echipa și detalii

## Echipa

| Membru | Arie | Responsabilități |
| --- | --- | --- |
| Fichioș Alexia | Motor + analiză | simulator, tipuri de reguli, cele 5 întrebări |
| Căprariu Ioana | Parsare NL + validare | schema JSON, LLM / parser, mesaje de eroare |
| Basarab Roxana-Ioana | GUI + stocare | Streamlit, vizualizare, salvare / încărcare |
| Rendiuc Ioana | Dataset + NN + documentație | experimentul ML, ghidul de instalare, log-urile cu asistenții AI |

**De ce așa:** Fiecare deliverable are un „responsabil principal”, dar toți membrii vor lucra împreună la fiecare deliverable.

## Arhitectură

```
descriere text ──▶ [Parser NL] ──▶ spec JSON ──▶ [Motor] ──▶ generații ──▶ [Analiză] ──▶ răspunsuri
                                                                 │
                                                          [Interfață + Stocare]
```

Componentele comunică doar prin **spec-ul JSON** (descrierea automatului) și prin **lista de generații**.

**De ce așa:** fiecare componentă poate fi scrisă și testată separat. De exemplu, motorul poate fi testat cu un spec scris de mână înainte să existe parserul.

## Decizii tehnice

### Limbaj: Python 3.11+

**De ce:** are cele mai bune biblioteci pentru tot ce ne trebuie (calcul numeric, interfață, rețele neuronale), iar toate membrele echipei îl cunoaștem. Nu avem nevoie de un al doilea limbaj.

### Grila: matrice `numpy`

- Fiecare tip de celulă primește un cod întreg (de ex. `0 = DEAD`, `1 = ALIVE`).
- Un automat 1D (Rule 30) este o grilă cu un singur rând; generațiile puse una sub alta dau imaginea 2D.

**De ce:** operațiile pe toată matricea deodată sunt mult mai rapide decât buclele Python, ceea ce contează când generăm multe generații pentru dataset. Numărarea celulelor de un tip devine o singură linie: `np.count_nonzero(grid == X)`.

### Descrierea automatului: spec JSON

Un automat este descris prin stări, vecinătate, comportament la margine și reguli. Exemplu pentru Game of Life:

```json
{
  "name": "Game of Life",
  "grid": { "rows": 50, "cols": 50, "boundary": "toroidal" },
  "states": ["DEAD", "ALIVE"],
  "neighborhood": "moore",
  "rules": [
    { "from": "DEAD",  "to": "ALIVE", "if": { "count": "ALIVE", "in": [3] } },
    { "from": "ALIVE", "to": "DEAD",  "if": { "count": "ALIVE", "not_in": [2, 3] } }
  ],
  "otherwise": "unchanged"
}
```

Tipuri de reguli pe care le vom acoperi treptat:

- **pe baza numărului de vecini**: Game of Life, Brian's Brain, Forest Fire, Cyclic CA;
- **tabele de tranziție**: Rule 30 și celelalte automate elementare;
- **cu agenți**: Langton's Ant;
- **cu probabilități**: rulate cu seed fix.

**De ce JSON:** e ușor de citit, ușor de validat automat și ușor de salvat. Parserul NL are o „țintă” clară: trebuie doar să producă acest JSON. Pentru un automat nou din aceeași familie nu scriem cod nou, doar un spec nou.

**De ce seed fix la automatele cu probabilități:** altfel aceeași simulare ar da rezultate diferite la fiecare rulare, iar întrebări ca „intră în ciclu?” n-ar avea un răspuns unic.

### Parsarea limbajului natural: hibrid (LLM + validare)

| Variantă | Avantaje | Dezavantaje |
| --- | --- | --- |
| Parser pe reguli (regex / gramatică) | determinist, explicabil | acoperă doar formulările prevăzute |
| Doar LLM → JSON | înțelege formulări variate | poate genera JSON greșit, fără verificare |
| **Hibrid (ales)** | LLM generează JSON-ul, iar `pydantic` îl validează; la eroare, utilizatorul primește un mesaj clar | puțin mai mult de implementat |

**De ce:** un parser doar pe reguli ar eșua la orice formulare la care nu ne-am gândit, iar un LLM singur poate inventa reguli. Validarea cu `pydantic` prinde greșelile ambelor variante și ne dă mesaje de eroare clare (stare necunoscută, regulă lipsă), cerute explicit în enunț.

### Răspunsul la întrebări

Le acoperim pe toate 5, nu doar minimul de 3:

| Întrebare | Cum o rezolvăm |
| --- | --- |
| Starea după N iterații | se rulează N pași |
| Când ajunge într-o stare stabilă | prima generație identică cu cea anterioară |
| Apare o anumită configurație | comparăm fiecare generație cu configurația (toată grila sau o fereastră glisantă) |
| Câte celule de tip X după N iterații | `np.count_nonzero(grid == X)` pe generația N |
| Intră într-un ciclu | salvăm un hash al fiecărei generații; un hash care reapare înseamnă ciclu, iar diferența de generații dă lungimea lui |

**De ce hash-uri:** comparăm un număr scurt în loc de grile întregi, deci detectarea ciclurilor rămâne rapidă și la simulări lungi. Pe o grilă finită și deterministă sistemul ajunge sigur, la un moment dat, într-o stare deja văzută, deci întrebarea are mereu răspuns.

### Stocare: JSON + `.npz` + SQLite

- spec-ul și parametrii simulării: JSON;
- generațiile: un fișier `.npz` comprimat (opțional doar fiecare a k-a generație);
- lista experimentelor (automat, dată, parametri, fișier): SQLite.

**De ce:** `.npz` salvează direct matricile `numpy`, fără conversii, și ocupă puțin. Am renunțat la HDF5 pentru că e mai complicat decât avem nevoie, iar CSV ar fi mare și lent. SQLite e inclus în Python, nu necesită server și ne permite să regăsim experimentele vechi.

### Interfață: Streamlit

| Opțiune | Comentariu |
| --- | --- |
| **Streamlit (ales)** | cea mai rapidă de făcut, scrisă integral în Python |
| PySide6 / PyQt | aplicație desktop, mult mai mult cod pentru layout |
| FastAPI + frontend JS | cel mai frumos rezultat, dar încă un limbaj și un frontend separat |

**De ce:** avem nevoie de o casetă de text, câțiva parametri, o grilă afișată și un slider prin generații, iar Streamlit le oferă gata făcute. Timpul economisit merge în motor și parser, care contează mai mult la notare. Dezavantaj acceptat: animațiile în timp real sunt limitate, așa că navigăm generațiile cu un slider.

### Rețea neuronală (Deliverable 3): PyTorch

Motorul generează un dataset de perechi `(generația t, generația t+1)`, iar o rețea convoluțională mică învață să prezică generația următoare. Începem cu Game of Life.

**De ce:** regula unui automat celular e locală (depinde doar de vecini), exact ce modelează o rețea convoluțională. Game of Life e un punct de pornire bun pentru că se știe că poate fi învățat. PyTorch primește direct array-urile `numpy`, fără pas de conversie.

## Ce construim (scope)

| Prioritate | Ce |
| --- | --- |
| Obligatoriu | automate pe baza numărului de vecini (Game of Life, Brian's Brain, Forest Fire, Cyclic, Crystal Growth), toate cele 5 întrebări, interfață, salvare / încărcare |
| Dacă avem timp | Rule 30, Langton's Ant, variante cu probabilități |
| Nu facem | lumi 3D, ecosistemul complet (iarbă + iepuri + vulpi) |

**De ce:** automatele din prima linie folosesc același tip de regulă, deci un singur interpretor le acoperă pe toate. Pe celelalte le adăugăm doar după ce baza funcționează.

## Calendar

| Laborator | Ce avem gata |
| --- | --- |
| L3 | Deliverable 1: echipa, deciziile din acest README, repo-ul |
| L6 | Deliverable 2: spec JSON, motor, salvare / încărcare, primele întrebări |
| L8 | interfață funcțională, toate cele 5 întrebări |
| L11 | Deliverable 3: dataset + rețea neuronală |
| L13 | Deliverable 4: parsarea limbajului natural |
| L14 | prezentarea finală, documentație completă |

## Mod de lucru

- Cod pe GitHub: un branch per sarcină, unite în `main` prin pull request revizuit de altă colegă/alte colege.
- Toate conversațiile cu asistenții AI se salvează în `ai_conversations/`.

**De ce:** istoricul din Git arată contribuția fiecăreia (notarea poate fi individuală), iar review-ul prinde erorile înainte să ajungă în `main`.

## Structura repository-ului (planificată)

```
cellab/
├── README.md
├── requirements.txt
├── src/
│   ├── engine/        # grila și aplicarea regulilor
│   ├── analysis/      # cele 5 întrebări
│   ├── parser/        # text → spec JSON, validare
│   ├── storage/       # salvare / încărcare experimente
│   └── app/           # interfața Streamlit
├── specs/             # automate predefinite (JSON)
├── ml/                # dataset și rețea neuronală
├── tests/
└── ai_conversations/  # interacțiunile cu asistenții AI
```
