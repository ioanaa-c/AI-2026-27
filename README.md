# CelLab

Aplicație care primește descrierea unui automat celular în limbaj natural, o validează, simulează generațiile următoare și răspunde la întrebări despre evoluția lumii.

> Stadiu: Deliverable 1 – echipa și direcția tehnică.

## Echipa

| Membru | Arie | Responsabilități |
| --- | --- | --- |
| Fichioș Alexia | Motor + analiză | simulator, interpretorul de reguli, cele 5 întrebări |
| Căprariu Ioana | Parsare NL + validare | schema spec-ului, LLM, validare, mesaje de eroare |
| Basarab Roxana-Ioana | GUI + stocare | Streamlit, vizualizare, salvare / încărcare experimente |
| Rendiuc Ioana | Lumi inițiale + automate 1D + ML | generarea lumii inițiale, Rule 30, dataset, rețeaua neuronală |

**De ce așa:** fiecare arie corespunde unei componente separate din arhitectură, deci putem lucra în paralel fără să ne blocăm una pe alta. Fiecare colegă are cod de scris încă de la Deliverable 2 (lumile inițiale și automatele 1D sunt necesare devreme), deci încărcarea e echilibrată, important pentru că notarea poate fi individuală.

## Arhitectură

```
descriere text ──▶ [Parser NL] ──▶ spec JSON ──┐
                                               ├──▶ [Motor] ──▶ generații ──▶ [Analiză] ──▶ răspunsuri
lume inițială (manuală / aleatoare) ───────────┘
                         [Interfață + Stocare] leagă totul și salvează experimentele
```

Componentele comunică doar prin **spec-ul JSON** (descrierea automatului), **lumea inițială** și **lista de generații**.

**De ce așa:** fiecare componentă poate fi scrisă și testată separat. De exemplu, motorul poate fi testat cu un spec scris de mână înainte să existe parserul.

## Decizii tehnice

### Limbaj: Python 3.11+

**De ce:** are cele mai bune biblioteci pentru tot ce ne trebuie (calcul numeric, interfață, rețele neuronale), iar toate membrele echipei îl cunoaștem. Nu avem nevoie de un al doilea limbaj.

### Grila: matrice `numpy`

- Fiecare tip de celulă primește un cod întreg (de ex. `0 = DEAD`, `1 = ALIVE`).
- Lumea este mereu o matrice 2D. Un automat 1D (Rule 30) este o matrice cu un singur rând (1 × W); istoricul generațiilor, pus rând sub rând, formează imaginea 2D afișată.

**De ce:** operațiile pe toată matricea deodată sunt mult mai rapide decât buclele Python, ceea ce contează când generăm multe generații pentru dataset. Numărarea celulelor de un tip devine o singură linie: `np.count_nonzero(grid == X)`. Tratând 1D ca un caz particular de 2D, motorul, stocarea și analiza rămân aceleași.

### Lumea inițială

Utilizatorul alege una din variante:

- **manuală**: desenată în interfață, lipită ca text (un caracter per celulă) sau încărcată dintr-un fișier;
- **aleatoare**: o densitate pentru fiecare tip de celulă și un seed (de ex. 30% `ALIVE`, restul `DEAD`, seed 42);
- **predefinită**: tipare simple, de exemplu o singură celulă în centru (Crystal Growth, Rule 30).

**De ce:** enunțul cere ca lumea inițială să poată fi „setată sau generată după reguli”. Seed-ul face ca o lume aleatoare să poată fi regenerată identic, deci un experiment salvat se poate repeta exact.

### Descrierea automatului: spec JSON

Un automat este descris prin stări, vecinătate, comportament la margine și reguli. Exemplu pentru Game of Life:

```json
{
  "name": "Game of Life",
  "kind": "neighbour_count",
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

Câmpul `kind` spune ce tip de reguli are automatul. Fiecare tip are propriul format de reguli, verificat separat:

| `kind` | Automate | Ce conțin regulile |
| --- | --- | --- |
| `neighbour_count` | Game of Life, Brian's Brain, Forest Fire, Cyclic CA, Crystal Growth | „celula de tip X devine Y dacă numărul vecinilor de tip Z este în mulțimea …”, opțional cu o probabilitate |
| `lookup_1d` | Rule 30 și celelalte automate elementare | tabelul celor 8 combinații (stânga, centru, dreapta) → noua valoare |
| `agent` | Langton's Ant (dacă avem timp) | poziția și direcția agentului + ce face pe fiecare culoare de celulă (întoarce, schimbă culoarea, avansează) |

**De ce JSON:** e ușor de citit de om, ușor de validat automat și ușor de salvat. Parserul NL are o „țintă” clară: trebuie doar să producă acest JSON. Pentru un automat nou din aceeași familie nu scriem cod nou, doar un spec nou.

**De ce câmpul `kind`:** un agent cu poziție și direcție nu poate fi exprimat ca „numără vecinii”, iar un tabel 1D nu are nevoie de vecinătate 2D. În loc să forțăm totul într-un singur format, fiecare familie are formatul ei, iar validatorul știe ce să ceară pentru fiecare.

**Automatele cu probabilități** (de ex. Forest Fire cu aprindere aleatoare) rulează cu un **seed fix**, ca o rulare să poată fi repetată exact. Vezi mai jos ce întrebări au sens pentru ele.

### Parsarea limbajului natural: hibrid (LLM local + validare)

| Variantă | Avantaje | Dezavantaje |
| --- | --- | --- |
| Parser pe reguli (regex / gramatică) | determinist, explicabil | acoperă doar formulările prevăzute |
| Doar LLM → JSON | înțelege formulări variate | poate genera JSON greșit, fără verificare |
| **Hibrid (ales)** | LLM-ul generează JSON-ul, iar `pydantic` îl validează; la eroare, utilizatorul primește un mesaj clar | puțin mai mult de implementat |

**Ce LLM:** un model open-source mic, rulat **local** cu [Ollama](https://ollama.com). Modelul exact îl alegem la prototip (L8), comparând câteva modele mici pe exemplele noastre.

**Dacă LLM-ul nu e disponibil** (laptop prea slab, model neinstalat): utilizatorul poate încărca un automat din biblioteca de spec-uri predefinite sau poate scrie spec-ul JSON direct. Restul aplicației funcționează normal.

**De ce:** un parser doar pe reguli ar eșua la orice formulare la care nu ne-am gândit, iar un LLM singur poate inventa reguli. Validarea cu `pydantic` prinde greșelile și dă mesaje clare (stare necunoscută, regulă lipsă), cerute explicit în enunț. Un model local nu are nevoie de internet, cont sau cheie API, deci demonstrația de la prezentare nu depinde de rețeaua din sală.

### Răspunsul la întrebări

| Întrebare | Cum o rezolvăm |
| --- | --- |
| Starea după N iterații | se rulează N pași |
| Când ajunge într-o stare stabilă | prima generație identică cu cea anterioară |
| Apare o anumită configurație | potrivire exactă cu toată grila sau cu un tipar mai mic căutat în orice poziție (fereastră glisantă) |
| Câte celule de tip X după N iterații | `np.count_nonzero(grid == X)` pe generația N |
| Intră într-un ciclu | salvăm un hash al fiecărei stări; un hash care reapare înseamnă ciclu, iar diferența de generații dă lungimea lui |

Precizări:

- **Ce înseamnă „stare”.** Pentru automatele obișnuite, starea este grila. Pentru `agent`, starea este grila plus poziția și direcția agentului, iar hash-ul le include pe toate. Altfel o grilă identică cu furnica în alt loc ar fi luată greșit drept ciclu.
- **Stabilitate și cicluri doar la automatele deterministe.** La un automat cu probabilități, o grilă care reapare nu garantează un ciclu, pentru că pasul următor folosește alte numere aleatoare. Din același motiv, o generație neschimbată nu garantează stabilitatea. Pentru ele răspundem la întrebările 1, 3 și 4 (pentru seed-ul dat), iar la 2 și 5 aplicația explică de ce nu se pot decide.
- **Automate 1D.** Starea este rândul curent. Stabil = rândul nu se mai schimbă; ciclu = un rând deja văzut reapare. Pe un rând finit (margine fixă sau circulară) numărul de rânduri posibile e finit, deci și aici ciclul apare garantat.
- **Configurații:** căutăm doar potrivire exactă, în orice poziție (translații). Rotirile și oglindirile nu sunt căutate.

**De ce hash-uri:** comparăm un număr scurt în loc de grile întregi, deci detectarea ciclurilor rămâne rapidă și la simulări lungi. Un automat determinist pe o grilă finită are un număr finit de stări posibile, deci ajunge sigur, la un moment dat, într-o stare deja văzută: întrebarea despre ciclu are mereu răspuns.

**De ce fără rotiri:** enunțul întreabă dacă „o anumită configurație” apare, iar o configurație rotită e alta. Căutarea pe translații acoperă cazul util (un tipar care se deplasează pe grilă) fără să complice analiza.

### Stocare: JSON + `.npz` + SQLite

- spec-ul, lumea inițială (sau densitățile + seed-ul) și parametrii simulării: JSON;
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

| Prioritate | Automate | De ce |
| --- | --- | --- |
| Obligatoriu | Game of Life, Brian's Brain, Forest Fire (determinist), Cyclic CA, Crystal Growth | toate sunt `neighbour_count`, deci un singur interpretor le acoperă |
| Dacă avem timp | Rule 30 și automatele elementare (`lookup_1d`), Forest Fire cu probabilități, Langton's Ant (`agent`) | fiecare cere un tip de spec în plus; le adăugăm după ce baza funcționează |
| Nu facem | Sandpile, Diffusion | celulele au valori numerice, nu tipuri, iar regulile sunt calcule (prăbușire, medie), nu „numără vecinii” |
| Nu facem | Traffic cu viteze, Predator–Prey, Schelling, Falling Sand, ecosistemul artificial | agenți care se mută, au energie sau viteză; ar cere un motor separat |
| Nu facem | Epidemic (I → R după N ture) | fiecare celulă ar trebui să țină minte de câte ture e infectată |
| Nu facem | lumi 3D | în afara cerințelor |

Pentru toate automatele obligatorii acoperim toate cele 5 întrebări, plus interfața, lumea inițială și salvarea / încărcarea.

## Calendar

| Laborator | Ce avem gata |
| --- | --- |
| L3 | Deliverable 1: echipa, deciziile din acest README, repo-ul |
| L6 | Deliverable 2: spec JSON + validator, motor `neighbour_count`, lume inițială manuală / aleatoare, salvare / încărcare, întrebările 1 și 4 |
| L8 | interfață funcțională, toate cele 5 întrebări, **prototip de parser NL** (LLM local + validare) pentru propoziții simple de tip „numără vecinii” |
| L11 | Deliverable 3: dataset + rețea neuronală; Rule 30 |
| L13 | Deliverable 4: parser complet (formulări variate, mesaje de eroare, testat pe un set de descrieri) |
| L14 | prezentarea finală, documentație completă |

**De ce prototipul de parser la L8:** descrierea în limbaj natural e cerința centrală a proiectului. Dacă am începe-o abia după L11 și ceva ar întârzia, am risca exact partea esențială. Cu un prototip devreme aflăm din timp dacă modelul local e suficient de bun.

## Mod de lucru

- Cod pe GitHub: un branch per sarcină, unite în `main` prin pull request revizuit de altă colegă.
- Toate conversațiile cu asistenții AI se salvează în `ai_conversations/`, cerință obligatorie la predare.

## Structura repository-ului (planificată)

```
cellab/
├── README.md
├── requirements.txt
├── src/
│   ├── engine/        # grila și aplicarea regulilor
│   ├── world/         # lumea inițială: manuală, aleatoare, predefinită
│   ├── analysis/      # cele 5 întrebări
│   ├── parser/        # text → spec JSON (LLM local), validare
│   ├── storage/       # salvare / încărcare experimente
│   └── app/           # interfața Streamlit
├── specs/             # automate predefinite (JSON)
├── ml/                # dataset și rețea neuronală
├── tests/
└── ai_conversations/  # interacțiunile cu asistenții AI
```

## Instalare și utilizare

Se completează pe măsură ce apare codul (`pip install -r requirements.txt`, instalarea Ollama și a modelului ales, apoi `streamlit run src/app/main.py`).
