"""
CEC 2017 benchmark dict for use with enhance(benchmarks=...).
"""

from heurilab.core.cec2017 import (
    CEC17_F1, CEC17_F3, CEC17_F4, CEC17_F5, CEC17_F6, CEC17_F7,
    CEC17_F8, CEC17_F9, CEC17_F10, CEC17_F11, CEC17_F12, CEC17_F13,
    CEC17_F14, CEC17_F15, CEC17_F16, CEC17_F17, CEC17_F18, CEC17_F19,
    CEC17_F20, CEC17_F21, CEC17_F22, CEC17_F23, CEC17_F24, CEC17_F25,
    CEC17_F26, CEC17_F27, CEC17_F28, CEC17_F29, CEC17_F30,
)


def cec2017_benchmarks(dim=30):
    """Return a dict of all 29 CEC 2017 functions for enhance()."""
    _ALL = [
        ("F1",  "F1 Bent Cigar (Unimodal)",         CEC17_F1,  100),
        ("F3",  "F3 Zakharov (Unimodal)",            CEC17_F3,  300),
        ("F4",  "F4 Rosenbrock (Multimodal)",        CEC17_F4,  400),
        ("F5",  "F5 Rastrigin (Multimodal)",         CEC17_F5,  500),
        ("F6",  "F6 Exp. Scaffer (Multimodal)",      CEC17_F6,  600),
        ("F7",  "F7 Lunacek BiRast. (Multimodal)",   CEC17_F7,  700),
        ("F8",  "F8 NonCont Rastrigin (Multimodal)",  CEC17_F8,  800),
        ("F9",  "F9 Levy (Multimodal)",              CEC17_F9,  900),
        ("F10", "F10 Schwefel (Multimodal)",         CEC17_F10, 1000),
        ("F11", "F11 Hybrid 1",                      CEC17_F11, 1100),
        ("F12", "F12 Hybrid 2",                      CEC17_F12, 1200),
        ("F13", "F13 Hybrid 3",                      CEC17_F13, 1300),
        ("F14", "F14 Hybrid 4",                      CEC17_F14, 1400),
        ("F15", "F15 Hybrid 5",                      CEC17_F15, 1500),
        ("F16", "F16 Hybrid 6",                      CEC17_F16, 1600),
        ("F17", "F17 Hybrid 7",                      CEC17_F17, 1700),
        ("F18", "F18 Hybrid 8",                      CEC17_F18, 1800),
        ("F19", "F19 Hybrid 9",                      CEC17_F19, 1900),
        ("F20", "F20 Hybrid 10",                     CEC17_F20, 2000),
        ("F21", "F21 Composition 1",                 CEC17_F21, 2100),
        ("F22", "F22 Composition 2",                 CEC17_F22, 2200),
        ("F23", "F23 Composition 3",                 CEC17_F23, 2300),
        ("F24", "F24 Composition 4",                 CEC17_F24, 2400),
        ("F25", "F25 Composition 5",                 CEC17_F25, 2500),
        ("F26", "F26 Composition 6",                 CEC17_F26, 2600),
        ("F27", "F27 Composition 7",                 CEC17_F27, 2700),
        ("F28", "F28 Composition 8",                 CEC17_F28, 2800),
        ("F29", "F29 Composition 9",                 CEC17_F29, 2900),
        ("F30", "F30 Composition 10",                CEC17_F30, 3000),
    ]

    benchmarks = {}
    for key, name, func, optimum in _ALL:
        benchmarks[key] = {
            "name": name,
            "func": func,
            "lb": -100,
            "ub": 100,
            "dim": dim,
            "optimum": optimum,
        }
    return benchmarks
