# HeuriLab — Algorithm Provenance

Original publication for every algorithm in `ALL_ALGORITHMS` (102 classes, 6 categories).

**Verification.** Every DOI below was resolved against the Crossref REST API and the returned
title, authors, year and venue checked against the entry. DOIs were not taken from memory or
from search snippets. Entries that could not be verified are marked **UNVERIFIED** rather than
guessed — see [Needs your attention](#needs-your-attention) before using this file in a paper.

**Year convention.** Where a paper has an online-first year earlier than its print issue, the
issue year is given with the online year in parentheses. Pick one convention for your
manuscript and apply it consistently.

---

## Swarm Intelligence (20)

| Acronym | Algorithm | Authors | Year | Venue | DOI |
|---|---|---|---|---|---|
| PSO | Particle Swarm Optimization | Kennedy & Eberhart | 1995 | Proc. ICNN'95, 4, 1942–1948 | 10.1109/ICNN.1995.488968 |
| GWO | Grey Wolf Optimizer | Mirjalili, Mirjalili & Lewis | 2014 | Advances in Engineering Software, 69, 46–61 | 10.1016/j.advengsoft.2013.12.007 |
| WOA | Whale Optimization Algorithm | Mirjalili & Lewis | 2016 | Advances in Engineering Software, 95, 51–67 | 10.1016/j.advengsoft.2016.01.008 |
| MFO | Moth-Flame Optimization | Mirjalili | 2015 | Knowledge-Based Systems, 89, 228–249 | 10.1016/j.knosys.2015.07.006 |
| SSA | Salp Swarm Algorithm | Mirjalili et al. | 2017 | Advances in Engineering Software, 114, 163–191 | 10.1016/j.advengsoft.2017.07.002 |
| HHO | Harris Hawks Optimization | Heidari et al. | 2019 | Future Generation Computer Systems, 97, 849–872 | 10.1016/j.future.2019.02.028 |
| MPA | Marine Predators Algorithm | Faramarzi et al. | 2020 | Expert Systems with Applications, 152, 113377 | 10.1016/j.eswa.2020.113377 |
| BA | Bat Algorithm | Yang | 2010 | NICSO 2010, SCI 284, 65–74 | 10.1007/978-3-642-12538-6_6 |
| CS | Cuckoo Search via Lévy Flights | Yang & Deb | 2009 | Proc. NaBIC 2009, 210–214 | 10.1109/NABIC.2009.5393690 |
| FPA | Flower Pollination Algorithm | Yang | 2012 | UCNC 2012, LNCS 7445, 240–249 | 10.1007/978-3-642-32894-7_27 |
| DA | Dragonfly Algorithm | Mirjalili | 2016 (online 2015) | Neural Computing and Applications, 27, 1053–1073 | 10.1007/s00521-015-1920-1 |
| GOA | Grasshopper Optimisation Algorithm | Saremi, Mirjalili & Lewis | 2017 | Advances in Engineering Software, 105, 30–47 | 10.1016/j.advengsoft.2017.01.004 |
| ALO | Ant Lion Optimizer | Mirjalili | 2015 | Advances in Engineering Software, 83, 80–98 | 10.1016/j.advengsoft.2015.01.010 |
| SHO | Spotted Hyena Optimizer | Dhiman & Kumar | 2017 | Advances in Engineering Software, 114, 48–70 | 10.1016/j.advengsoft.2017.05.014 |
| **DO** | **"Dolphin Optimizer"** | — | — | — | **UNVERIFIED — see note D1** |
| EHO | Elephant Herding Optimization | Wang, Deb & Coelho | 2015 | Proc. ISCBI 2015, 1–5 | 10.1109/ISCBI.2015.8 |
| AO | Aquila Optimizer | Abualigah et al. | 2021 | Computers & Industrial Engineering, 157, 107250 | 10.1016/j.cie.2021.107250 |
| HGS | Hunger Games Search | Yang et al. | 2021 | Expert Systems with Applications, 177, 114864 | 10.1016/j.eswa.2021.114864 |
| GTO | Artificial Gorilla Troops Optimizer | Abdollahzadeh et al. | 2021 | Int. Journal of Intelligent Systems, 36(10), 5887–5958 | 10.1002/int.22535 |
| RUN | RUNge Kutta Optimizer | Ahmadianfar, Heidari, Gandomi, Chu & Chen | 2021 | Expert Systems with Applications, 181, 115079 | 10.1016/j.eswa.2021.115079 |

## Evolutionary (15)

| Acronym | Algorithm | Authors | Year | Venue | DOI |
|---|---|---|---|---|---|
| GA | Genetic Algorithm (*Adaptation in Natural and Artificial Systems*) | Holland | 1975 / 1992 ed. | Univ. Michigan Press; MIT Press | 10.7551/mitpress/1090.001.0001 (1992 ed.) |
| DE | Differential Evolution | Storn & Price | 1997 | Journal of Global Optimization, 11(4), 341–359 | 10.1023/A:1008202821328 |
| ES | Evolution Strategies (*Evolutionsstrategie*) | Rechenberg | 1973 | Frommann-Holzboog (monograph) | No DOI. Citable alternative: Beyer & Schwefel, *Natural Computing* 1, 3–52 (2002), 10.1023/A:1015059928466 |
| EP | Evolutionary Programming (*Artificial Intelligence through Simulated Evolution*) | Fogel, Owens & Walsh | 1966 | John Wiley & Sons (monograph) | No DOI (ISBN 9780471265160) |
| CMA | CMA-ES — Completely Derandomized Self-Adaptation | Hansen & Ostermeier | 2001 | Evolutionary Computation, 9(2), 159–195 | 10.1162/106365601750190398 |
| BBO | Biogeography-Based Optimization | Simon | 2008 | IEEE Trans. Evolutionary Computation, 12(6), 702–713 | 10.1109/TEVC.2008.919004 |
| SHADE | Success-History Based Adaptive DE | Tanabe & Fukunaga | 2013 | Proc. IEEE CEC 2013, 71–78 | 10.1109/CEC.2013.6557555 |
| **TLGO** | Teaching–Learning-Based Optimization | Rao, Savsani & Vakharia | 2011 | Computer-Aided Design, 43(3), 303–315 | 10.1016/j.cad.2010.12.015 — **see note D4** |
| CoDE | DE with Composite Trial Vector Generation | Wang, Cai & Zhang | 2011 | IEEE Trans. Evolutionary Computation, 15(1), 55–66 | 10.1109/TEVC.2010.2087271 |
| SaDE | Self-adaptive Differential Evolution | Qin, Huang & Suganthan | 2009 | IEEE Trans. Evolutionary Computation, 13(2), 398–417 | 10.1109/TEVC.2008.927706 |
| OXDE | Orthogonal Crossover Differential Evolution | Wang, Cai & Zhang | 2012 | Information Sciences, 185(1), 153–177 | 10.1016/j.ins.2011.09.001 |
| AGDE | Adaptive Guided Differential Evolution | Mohamed & Mohamed | 2019 (online 2017) | Int. J. Machine Learning and Cybernetics, 10, 253–277 | 10.1007/s13042-017-0711-7 |
| LSHADE | SHADE with Linear Population Size Reduction | Tanabe & Fukunaga | **2014** | Proc. IEEE CEC 2014, 1658–1665 | 10.1109/CEC.2014.6900380 — **docstring says 2020, see note D3** |
| EBOwithCMAR | Effective Butterfly Optimizer with CMA Retreat | Kumar, Misra & Singh | 2017 | Proc. IEEE CEC 2017, 1835–1842 | 10.1109/CEC.2017.7969524 |
| IMODE | Improved Multi-operator Differential Evolution | Sallam et al. | 2020 | Proc. IEEE CEC 2020, 1–8 | 10.1109/CEC48606.2020.9185577 |

## Physics-based (16)

| Acronym | Algorithm | Authors | Year | Venue | DOI |
|---|---|---|---|---|---|
| GSA | Gravitational Search Algorithm | Rashedi, Nezamabadi-pour & Saryazdi | 2009 | Information Sciences, 179(13), 2232–2248 | 10.1016/j.ins.2009.03.004 |
| MVO | Multi-Verse Optimizer | Mirjalili, Mirjalili & Hatamlou | 2016 (online 2015) | Neural Computing and Applications, 27, 495–513 | 10.1007/s00521-015-1870-7 |
| SCA | Sine Cosine Algorithm | Mirjalili | 2016 | Knowledge-Based Systems, 96, 120–133 | 10.1016/j.knosys.2015.12.022 |
| AOA | Arithmetic Optimization Algorithm | Abualigah et al. | 2021 | Computer Methods in Applied Mechanics and Engineering, 376, 113609 | 10.1016/j.cma.2020.113609 |
| SA | Simulated Annealing | Kirkpatrick, Gelatt & Vecchi | 1983 | Science, 220(4598), 671–680 | 10.1126/science.220.4598.671 |
| EO | Equilibrium Optimizer | Faramarzi et al. | 2020 | Knowledge-Based Systems, 191, 105190 | 10.1016/j.knosys.2019.105190 |
| WDO | Wind Driven Optimization | Bayraktar, Komurcu, Bossard & Werner | 2013 (conf. 2010) | IEEE Trans. Antennas and Propagation, 61(5), 2745–2757 | 10.1109/TAP.2013.2238654 (conf.: 10.1109/APS.2010.5562213) |
| HGSO | Henry Gas Solubility Optimization | Hashim et al. | 2019 | Future Generation Computer Systems, 101, 646–667 | 10.1016/j.future.2019.07.015 |
| CSS | Charged System Search | Kaveh & Talatahari | 2010 | Acta Mechanica, 213, 267–289 | 10.1007/s00707-009-0270-4 |
| CFO | Central Force Optimization | Formato | 2007 | Progress In Electromagnetics Research, 77, 425–491 | 10.2528/PIER07082403 |
| TWO | Tug of War Optimization | Kaveh & Zolghadr | 2016 | Int. J. Optimization in Civil Engineering, 6(4), 469–492 | No DOI — journal does not register DOIs |
| ASO | Atom Search Optimization | Zhao, Wang & Zhang | 2019 | Knowledge-Based Systems, 163, 283–304 | 10.1016/j.knosys.2018.08.030 |
| RIME | RIME: A Physics-Based Optimization | Su et al. | 2023 | Neurocomputing, 532, 183–214 | 10.1016/j.neucom.2023.02.010 |
| AEO | Artificial Ecosystem-based Optimization | Zhao, Wang & Zhang | 2020 (online 2019) | Neural Computing and Applications, 32(13), 9383–9425 | 10.1007/s00521-019-04452-x |
| GBO | Gradient-Based Optimizer | Ahmadianfar, Bozorg-Haddad & Chu | 2020 | Information Sciences, 540, 131–159 | 10.1016/j.ins.2020.06.037 |
| TSO | Transient Search Optimization | Qais, Hasanien & Alghuwainem | 2020 | Applied Intelligence, 50, 3926–3941 | 10.1007/s10489-020-01727-y |

## Human / Social (16)

| Acronym | Algorithm | Authors | Year | Venue | DOI |
|---|---|---|---|---|---|
| TLBO | Teaching–Learning-Based Optimization | Rao, Savsani & Vakharia | 2011 | Computer-Aided Design, 43(3), 303–315 | 10.1016/j.cad.2010.12.015 |
| JA | Jaya Algorithm | Rao | 2016 | Int. J. Industrial Engineering Computations, 7(1), 19–34 | 10.5267/j.ijiec.2015.8.004 |
| HS | Harmony Search | Geem, Kim & Loganathan | 2001 | SIMULATION, 76(2), 60–68 | 10.1177/003754970107600201 |
| ICA | Imperialist Competitive Algorithm | Atashpaz-Gargari & Lucas | 2007 | Proc. IEEE CEC 2007, 4661–4667 | 10.1109/CEC.2007.4425083 |
| CA | Cultural Algorithms | Reynolds | 1994 | Proc. 3rd Conf. Evolutionary Programming, 131–139 | No chapter DOI (volume: 10.1142/9789814534116) |
| BSO | Brain Storm Optimization | Shi | 2011 | ICSI 2011, LNCS 6728, 303–309 | 10.1007/978-3-642-21515-5_36 |
| **SOS_H** | **"Social Optimization Search"** | — | — | — | **UNVERIFIED — see note D2** |
| **QLA** | **Q-Learning-based Algorithm** | — | — | — | **UNVERIFIED — see note D2** |
| INFO | weIghted meaN oF vectOrs | Ahmadianfar et al. | 2022 | Expert Systems with Applications, 195, 116516 | 10.1016/j.eswa.2022.116516 |
| HBO | Heap-Based Optimizer | Askari, Saeed & Younas | 2020 | Expert Systems with Applications, 161, 113702 | 10.1016/j.eswa.2020.113702 |
| AOArch | Archimedes Optimization Algorithm | Hashim et al. | 2021 (online 2020) | Applied Intelligence, 51, 1531–1551 | 10.1007/s10489-020-01893-z — **miscategorised, note D5** |
| CHIO | Coronavirus Herd Immunity Optimizer | Al-Betar et al. | 2021 (online 2020) | Neural Computing and Applications, 33(10), 5011–5042 | 10.1007/s00521-020-05296-6 |
| SSOA | **Sparrow Search Algorithm** (not Social Spider) | Xue & Shen | 2020 | Systems Science & Control Engineering, 8(1), 22–34 | 10.1080/21642583.2019.1708830 — **miscategorised, note D5** |
| POA | **Political Optimizer** (not Pelican) | Askari, Younas & Saeed | 2020 | Knowledge-Based Systems, 195, 105709 | 10.1016/j.knosys.2020.105709 |
| ED | Enterprise Development metaheuristic | Truong & Chou | 2024 | Engineering Structures, 318, 118679 | 10.1016/j.engstruct.2024.118679 |
| **AMO** | **Auction Market Optimizer** | *original to HeuriLab* | — | — | **No prior publication — see note D2** |

## Bio-inspired (15)

| Acronym | Algorithm | Authors | Year | Venue | DOI |
|---|---|---|---|---|---|
| ABC | Artificial Bee Colony | Karaboga & Basturk | 2007 | Journal of Global Optimization, 39(3), 459–471 | 10.1007/s10898-007-9149-x |
| FA | Firefly Algorithm | Yang | 2009 | SAGA 2009, LNCS 5792, 169–178 | 10.1007/978-3-642-04944-6_14 |
| SOS | Symbiotic Organisms Search | Cheng & Prayogo | 2014 | Computers & Structures, 139, 98–112 | 10.1016/j.compstruc.2014.03.007 |
| BFO | Bacterial Foraging Optimization | Passino | 2002 | IEEE Control Systems Magazine, 22(3), 52–67 | 10.1109/MCS.2002.1004010 |
| CSA | Crow Search Algorithm | Askarzadeh | 2016 | Computers & Structures, 169, 1–12 | 10.1016/j.compstruc.2016.03.001 |
| BOA | Butterfly Optimization Algorithm | Arora & Singh | 2019 (online 2018) | Soft Computing, 23, 715–734 | 10.1007/s00500-018-3102-4 |
| TSA | Tunicate Swarm Algorithm | Kaur et al. | 2020 | Engineering Applications of AI, 90, 103541 | 10.1016/j.engappai.2020.103541 |
| WHO | Wild Horse Optimizer | Naruei & Keynia | 2022 (online 2021) | Engineering with Computers, 38, 3025–3056 | 10.1007/s00366-021-01438-z |
| SBO | Satin Bowerbird Optimizer | Moosavi & Bardsiri | 2017 | Engineering Applications of AI, 60, 1–15 | 10.1016/j.engappai.2017.01.006 |
| MBO | Monarch Butterfly Optimization | Wang, Deb & Cui | 2019 (online 2015) | Neural Computing and Applications, 31(7), 1995–2014 | 10.1007/s00521-015-1923-y |
| EPO | Emperor Penguin Optimizer | Dhiman & Kumar | 2018 | Knowledge-Based Systems, 159, 20–50 | 10.1016/j.knosys.2018.06.001 |
| SMA | Slime Mould Algorithm | Li et al. | 2020 | Future Generation Computer Systems, 111, 300–323 | 10.1016/j.future.2020.03.055 |
| HBA | Honey Badger Algorithm | Hashim et al. | 2022 | Mathematics and Computers in Simulation, 192, 84–110 | 10.1016/j.matcom.2021.08.013 |
| RSA | Reptile Search Algorithm | Abualigah et al. | 2022 | Expert Systems with Applications, 191, 116158 | 10.1016/j.eswa.2021.116158 |
| GJO | Golden Jackal Optimization | Chopra & Ansari | 2022 | Expert Systems with Applications, 198, 116924 | 10.1016/j.eswa.2022.116924 |

## Modern 2022–2025 (20)

| Acronym | Algorithm | Authors | Year | Venue | DOI |
|---|---|---|---|---|---|
| AVOA | African Vultures Optimization Algorithm | Abdollahzadeh, Gharehchopogh & Mirjalili | 2021 | Computers & Industrial Engineering, 158, 107408 | 10.1016/j.cie.2021.107408 |
| DMO | Dwarf Mongoose Optimization | Agushaka, Ezugwu & Abualigah | 2022 | Computer Methods in Applied Mechanics and Engineering, 391, 114570 | 10.1016/j.cma.2022.114570 |
| MGO | Mountain Gazelle Optimizer | Abdollahzadeh et al. | 2022 | Advances in Engineering Software, 174, 103282 | 10.1016/j.advengsoft.2022.103282 |
| DBO | Dung Beetle Optimizer | Xue & Shen | 2023 (online 2022) | The Journal of Supercomputing, 79, 7305–7336 | 10.1007/s11227-022-04959-6 |
| COA | **Coati** Optimization Algorithm (confirmed from docstring) | Dehghani et al. | 2023 | Knowledge-Based Systems, 259, 110011 | 10.1016/j.knosys.2022.110011 |
| OOA | Osprey Optimization Algorithm | Dehghani & Trojovský | 2023 | Frontiers in Mechanical Engineering, 8, 1126450 | 10.3389/fmech.2022.1126450 |
| NOA | Nutcracker Optimizer | Abdel-Basset et al. | 2023 | Knowledge-Based Systems, 262, 110248 | 10.1016/j.knosys.2022.110248 |
| SAO | Snow Ablation Optimizer | Deng & Liu | 2023 | Expert Systems with Applications, 225, 120069 | 10.1016/j.eswa.2023.120069 |
| FLA | Fick's Law Algorithm | Hashim et al. | 2023 | Knowledge-Based Systems, 260, 110146 | 10.1016/j.knosys.2022.110146 |
| EVO | Energy Valley Optimizer | Azizi, Aickelin, Khorshidi & Shishehgarkhaneh | 2023 | Scientific Reports, 13, 226 | 10.1038/s41598-022-27344-y |
| EDO | Exponential Distribution Optimizer | Abdel-Basset et al. | 2023 | Artificial Intelligence Review, 56(9), 9329–9400 | 10.1007/s10462-023-10403-9 |
| MOA | Mayfly Optimization Algorithm | Zervoudakis & Tsafarakis | 2020 | Computers & Industrial Engineering, 145, 106559 | 10.1016/j.cie.2020.106559 |
| CPO | Crested Porcupine Optimizer | Abdel-Basset, Mohamed & Abouhawwash | 2024 | Knowledge-Based Systems, 284, 111257 | 10.1016/j.knosys.2023.111257 |
| PO | **Parrot** Optimizer (confirmed from docstring) | Lian et al. | 2024 | Computers in Biology and Medicine, 172, 108064 | 10.1016/j.compbiomed.2024.108064 |
| FO | **FOX** Optimizer | Mohammed & Rashid | **2023** (online 2022) | Applied Intelligence, 53(1), 1030–1050 | 10.1007/s10489-022-03533-0 — **docstring says 2024, note D3; fidelity concern, note D6** |
| HO | Hippopotamus Optimization Algorithm | Amiri et al. | 2024 | Scientific Reports, 14, 5032 | 10.1038/s41598-024-54910-3 |
| KOA | Kepler Optimization Algorithm | Abdel-Basset et al. | 2023 | Knowledge-Based Systems, 268, 110454 | 10.1016/j.knosys.2023.110454 |
| SBOA | Secretary Bird Optimization Algorithm | Fu, Liu, Chen & He | 2024 | Artificial Intelligence Review, 57(5), 123 | 10.1007/s10462-024-10729-y |
| GMO | Geometric Mean Optimizer | Rezaei, Safavi, Abd Elaziz & Mirjalili | 2023 | Soft Computing, 27(15), 10571–10606 | 10.1007/s00500-023-08202-z |
| FFO | Fennec Fox Optimization | Trojovská, Dehghani & Trojovský | 2022 | IEEE Access, 10, 84417–84443 | 10.1109/ACCESS.2022.3197745 |

---

## Needs your attention

These must be resolved before the table goes into a manuscript. Each is a claim a reviewer
could check and find wrong.

**D1 — `DO` is not a dolphin algorithm; it is WOA.**
`swarm/do.py` implements two update rules: the spiral bubble-net move
`D * exp(b·l) * cos(2πl) + best` and the encircling move `best − A·D` with `A = 2a·r − a`,
`C = 2r`, `a` decaying linearly from 2. Those are Whale Optimization Algorithm equations
(Mirjalili & Lewis 2016) with WOA's random-agent exploration branch removed and greedy
selection added. Separately, a Crossref sweep found **no metaheuristic published under the
name "Dolphin Optimizer"** at all — the six real dolphin algorithms are Dolphin Partner
Optimization (2009), Dolphin Echolocation (Kaveh & Farhoudi 2013), Dolphin Swarm Algorithm
(2016), Dolphin Swarm Optimization (2016), and Dolphin Pod Optimization (2017), none of which
this code matches. Counting `DO` and `WOA` as two distinct algorithms overstates the registry.
Options: relabel it as a WOA variant, reimplement against a chosen dolphin reference, or drop
it (registry becomes 101).

**D2 — Three algorithms have no traceable original publication.**
`AMO` (Auction Market Optimizer), `QLA` (Q-Learning-based Algorithm) and `SOS_H` (Social
Optimization Search) returned nothing across Crossref, Scite and the open web. AMO's docstring
includes a "Notes on the specification" section, which reads as an original design — if it is
yours, say so explicitly in the paper rather than leaving it uncited. QLA is a generic
ε-greedy Q-table over four hand-written move operators; there is no canonical "QLA" paper, so
cite Watkins & Dayan (*Machine Learning*, 1992) for the underlying method if you cite anything.
`SOS_H` is also distinct from `bio/sos.py` (Symbiotic Organisms Search) — do not conflate them.

**D3 — Two wrong years in the source docstrings.**
`evolutionary/lshade.py` says "Tanabe & Fukunaga, 2020"; L-SHADE is CEC **2014**.
`modern/fo.py` says "Mohammed & Rashid, 2024"; FOX is Applied Intelligence **2023** (online
April 2022). There is no 2024 FOX paper by those authors.

**D4 — `TLGO` and `TLBO` resolve to the same paper.**
Both point to Rao, Savsani & Vakharia (2011). If `evolutionary/tlgo.py` is a distinct variant,
it needs its own citation; if it is a duplicate implementation, the registry count is affected.

**D5 — Two algorithms are in the wrong category.**
`SSOA` (Sparrow Search) is a swarm algorithm and `AOArch` (Archimedes) is physics-inspired,
yet both sit under `human`. Harmless for results, but a reviewer reading the category table
will notice.

**D6 — `FO` may not implement FOX.**
The real FOX uses sound travel time, `Dist = sp · time / 2`, a ballistic jump `0.5·9.81·t²`,
and the `c1 = 0.18 / c2 = 0.82` branch constants. None appear in `modern/fo.py`, which uses a
generic exploration/exploitation split with ad-hoc "stalking"/"pouncing" rules. Verify before
citing, or the citation misrepresents the code.

**D7 — Monographs without DOIs.**
`GA` (Holland 1975), `ES` (Rechenberg 1973), `EP` (Fogel et al. 1966), `CA` (Reynolds 1994)
and `TWO` (IJOCE) predate or fall outside DOI registration. Cite them as books/proceedings, or
use the DOI-bearing alternatives noted in the tables.

**D8 — Conference-versus-journal choices.**
`SaDE` has an earlier CEC 2005 version (10.1109/CEC.2005.1554904); `WDO` has a 2010 conference
paper and a 2013 journal paper; `ABC`'s true first description is Karaboga's 2005 technical
report TR06 (no DOI); `FA` is sometimes cited to Yang's 2008 book instead of the 2009 LNCS
paper. Pick a convention and state it.
