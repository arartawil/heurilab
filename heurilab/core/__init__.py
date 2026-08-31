"""
Core — experiment runner, benchmark configuration, and test functions.
"""

from heurilab.core.benchmarks import BenchmarkConfig, BenchmarkSuite
from heurilab.core.runner import run_experiment
from heurilab.core.functions import (
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
    F11, F12, F13, F14, F15, F16, F17, F18, F19, F20,
    F21, F22, F23,
    CLASSICAL_FUNCTIONS,
    get_classical_suite, get_unimodal_suite,
    get_multimodal_suite, get_fixeddim_suite,
)
from heurilab.core.cec2017 import (
    CEC17_F1, CEC17_F3,
    CEC17_F4, CEC17_F5, CEC17_F6, CEC17_F7, CEC17_F8, CEC17_F9, CEC17_F10,
    CEC17_F11, CEC17_F12, CEC17_F13, CEC17_F14, CEC17_F15,
    CEC17_F16, CEC17_F17, CEC17_F18, CEC17_F19, CEC17_F20,
    CEC17_F21, CEC17_F22, CEC17_F23, CEC17_F24, CEC17_F25,
    CEC17_F26, CEC17_F27, CEC17_F28, CEC17_F29, CEC17_F30,
    CEC2017_FUNCTIONS,
    get_cec2017_suite, get_cec2017_unimodal_suite,
    get_cec2017_multimodal_suite, get_cec2017_hybrid_suite,
    get_cec2017_composition_suite,
)
from heurilab.core.cec_official import (
    OfficialCEC, official_bias, supported_dimensions,
    CEC2017_FUNCTION_NUMBERS, CEC2022_FUNCTION_NUMBERS,
)
from heurilab.core.cec2022_fixed import (
    get_cec2022_suite, get_cec2022_function, get_cec2022_optimum,
    cec2022_supported_dimensions,
)
from heurilab.core.cec2017_fixed import (
    get_cec2017_official_suite, get_cec2017_official_function,
    get_cec2017_official_optimum, cec2017_supported_dimensions,
)
from heurilab.core.cec2020 import (
    CEC20_F1, CEC20_F2, CEC20_F3, CEC20_F4, CEC20_F5,
    CEC20_F6, CEC20_F7, CEC20_F8, CEC20_F9, CEC20_F10,
    CEC2020_FUNCTIONS,
    get_cec2020_suite, get_cec2020_unimodal_suite,
    get_cec2020_multimodal_suite, get_cec2020_hybrid_suite,
    get_cec2020_composition_suite,
)

__all__ = [
    # Official CEC 2017 / CEC 2022 (verified against the organisers' C code)
    "OfficialCEC", "official_bias", "supported_dimensions",
    "CEC2017_FUNCTION_NUMBERS", "CEC2022_FUNCTION_NUMBERS",
    "get_cec2022_suite", "get_cec2022_function", "get_cec2022_optimum",
    "cec2022_supported_dimensions",
    "get_cec2017_official_suite", "get_cec2017_official_function",
    "get_cec2017_official_optimum", "cec2017_supported_dimensions",

    "BenchmarkConfig", "BenchmarkSuite", "run_experiment",
    # Classical F1–F23
    "F1", "F2", "F3", "F4", "F5", "F6", "F7",
    "F8", "F9", "F10", "F11", "F12", "F13",
    "F14", "F15", "F16", "F17", "F18", "F19", "F20",
    "F21", "F22", "F23",
    "CLASSICAL_FUNCTIONS",
    "get_classical_suite", "get_unimodal_suite",
    "get_multimodal_suite", "get_fixeddim_suite",
    # CEC 2017
    "CEC17_F1", "CEC17_F3",
    "CEC17_F4", "CEC17_F5", "CEC17_F6", "CEC17_F7", "CEC17_F8", "CEC17_F9", "CEC17_F10",
    "CEC17_F11", "CEC17_F12", "CEC17_F13", "CEC17_F14", "CEC17_F15",
    "CEC17_F16", "CEC17_F17", "CEC17_F18", "CEC17_F19", "CEC17_F20",
    "CEC17_F21", "CEC17_F22", "CEC17_F23", "CEC17_F24", "CEC17_F25",
    "CEC17_F26", "CEC17_F27", "CEC17_F28", "CEC17_F29", "CEC17_F30",
    "CEC2017_FUNCTIONS",
    "get_cec2017_suite", "get_cec2017_unimodal_suite",
    "get_cec2017_multimodal_suite", "get_cec2017_hybrid_suite",
    "get_cec2017_composition_suite",
    # CEC 2020
    "CEC20_F1", "CEC20_F2", "CEC20_F3", "CEC20_F4", "CEC20_F5",
    "CEC20_F6", "CEC20_F7", "CEC20_F8", "CEC20_F9", "CEC20_F10",
    "CEC2020_FUNCTIONS",
    "get_cec2020_suite", "get_cec2020_unimodal_suite",
    "get_cec2020_multimodal_suite", "get_cec2020_hybrid_suite",
    "get_cec2020_composition_suite",
]
