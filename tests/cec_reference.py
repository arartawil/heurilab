"""
Build (or locate) the CEC organisers' reference C code and evaluate through it.

``tests/test_cec_validity.py`` checks HeuriLab's official CEC 2017 / CEC 2022
suites against the code the competitions were actually run with:

* CEC 2017 - ``cec17_test_func.cpp``, from
  https://github.com/P-N-Suganthan/CEC2017-BoundContrained
* CEC 2022 - ``cec22_test_func.cpp``, from
  https://github.com/P-N-Suganthan/2022-SO-BO

Neither is redistributable inside this repository, so this module fetches the
organisers' archives on first use, compiles a tiny stdin/stdout driver around
each with whatever C++ compiler it can find, and caches the result.

The driver protocol is deliberately dumb: write ``func_num ndim npoints``
followed by ``npoints * ndim`` doubles, read back one ``%.17g`` per line.

Configuration
-------------
``HEURILAB_CEC_REFERENCE_DIR``
    Use a prepared reference tree instead of downloading one.  Must contain
    ``cec2017/`` and ``cec2022/`` sub-directories, each holding the compiled
    ``oracle`` binary next to the organisers' ``input_data/``.
``HEURILAB_CEC_REQUIRE_REFERENCE=1``
    Turn "could not prepare the reference" from a skip into a hard failure.
    Set this in CI so a broken toolchain cannot quietly disable the strongest
    check in the suite.
``HEURILAB_CEC_REFERENCE_CACHE``
    Where to keep the downloaded/compiled tree.  Defaults to
    ``<tempdir>/heurilab-cec-reference``.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

_ARCHIVES = {
    2017: (
        "https://github.com/P-N-Suganthan/CEC2017-BoundContrained/raw/master/"
        "CEC17_fast_pow-C%2B%2B.zip",
        "cec17_test_func.cpp",          # the unmodified reference, not fast_pow
        "cec17_test_func",
    ),
    2022: (
        "https://github.com/P-N-Suganthan/2022-SO-BO/raw/main/CEC2022.zip",
        "cec22_test_func.cpp",
        "cec22_test_func",
    ),
}

_DRIVER = r"""
/* stdin: func_num ndim npoints, then npoints*ndim doubles.
   stdout: one %.17g per line. */
#include <stdio.h>
#include <stdlib.h>
#include "SOURCE_FILE"

double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;

int main(void) {
    int func_num, nx, np;
    if (scanf("%d %d %d", &func_num, &nx, &np) != 3) return 2;
    double *x = (double *)malloc(sizeof(double) * (size_t)np * nx);
    double *f = (double *)malloc(sizeof(double) * (size_t)np);
    if (!x || !f) return 4;
    for (long i = 0; i < (long)np * nx; i++)
        if (scanf("%lf", &x[i]) != 1) return 3;
    ENTRY_POINT(x, f, nx, np, func_num);
    for (int i = 0; i < np; i++) printf("%.17g\n", f[i]);
    return 0;
}
"""


class ReferenceUnavailable(RuntimeError):
    """The organisers' code could not be downloaded or compiled here."""


def _cache_root():
    override = os.environ.get("HEURILAB_CEC_REFERENCE_CACHE")
    if override:
        return override
    return os.path.join(tempfile.gettempdir(), "heurilab-cec-reference")


def _binary_name():
    return "oracle.exe" if sys.platform == "win32" else "oracle"


def _compilers():
    """Candidate C++ compilers, most convenient first.

    ``python -m ziglang c++`` is listed first because it needs no system
    toolchain - ``pip install ziglang`` is enough, which makes this check
    reproducible on a bare CI runner.
    """
    out = []
    try:
        import ziglang  # noqa: F401
        out.append([sys.executable, "-m", "ziglang", "c++"])
    except ImportError:
        pass
    for name in ("g++", "clang++", "c++"):
        found = shutil.which(name)
        if found:
            out.append([found])
    return out


def _download(url, dest):
    import urllib.request
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    with open(dest, "wb") as handle:
        handle.write(data)


def _extract(archive, workdir, wanted_source):
    """Pull the reference .cpp and the organisers' input_data out of the zip."""
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        source = next((n for n in names
                       if n.replace("\\", "/").endswith("/" + wanted_source)
                       or n.replace("\\", "/") == wanted_source), None)
        if source is None:
            raise ReferenceUnavailable(f"{wanted_source} not found in {archive}")
        base = source.replace("\\", "/").rsplit("/", 1)[0]
        prefix = (base + "/input_data/") if base else "input_data/"

        with zf.open(source) as handle:
            text = handle.read().decode("latin-1")
        # <WINDOWS.H> is pulled in for no reason and is not portable.
        text = text.replace("#include <WINDOWS.H>", "/* windows.h removed */")
        text = text.replace("#include <windows.h>", "/* windows.h removed */")
        with open(os.path.join(workdir, wanted_source), "w",
                  encoding="latin-1") as handle:
            handle.write(text)

        data_dir = os.path.join(workdir, "input_data")
        os.makedirs(data_dir, exist_ok=True)
        members = [n for n in names if n.replace("\\", "/").startswith(prefix)
                   and not n.endswith("/")]
        if not members:
            raise ReferenceUnavailable(f"no input_data/ inside {archive}")
        for member in members:
            leaf = member.replace("\\", "/").rsplit("/", 1)[-1]
            with zf.open(member) as src, \
                    open(os.path.join(data_dir, leaf), "wb") as dst:
                shutil.copyfileobj(src, dst)


def _build(year, workdir):
    url, source_name, entry = _ARCHIVES[year]
    archive = os.path.join(workdir, f"cec{year}.zip")
    if not os.path.isfile(archive):
        _download(url, archive)
    if not os.path.isfile(os.path.join(workdir, source_name)):
        _extract(archive, workdir, source_name)

    driver = os.path.join(workdir, "driver.cpp")
    with open(driver, "w", encoding="ascii") as handle:
        handle.write(_DRIVER.replace("SOURCE_FILE", source_name)
                            .replace("ENTRY_POINT", entry))

    binary = os.path.join(workdir, _binary_name())
    errors = []
    for compiler in _compilers():
        cmd = compiler + ["-O2", "-w", "-o", binary, driver]
        result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        if result.returncode == 0 and os.path.isfile(binary):
            return binary
        errors.append(f"{' '.join(compiler)}: {result.stderr[-400:]}")
    raise ReferenceUnavailable(
        "no working C++ compiler (try `pip install ziglang`).\n" + "\n".join(errors))


_PREPARED = {}


def prepare(year):
    """Directory holding a runnable reference oracle for ``year``.

    Raises :class:`ReferenceUnavailable` if it cannot be produced here.
    """
    year = int(year)
    if year in _PREPARED:
        return _PREPARED[year]

    override = os.environ.get("HEURILAB_CEC_REFERENCE_DIR")
    if override:
        workdir = os.path.join(override, f"cec{year}")
        binary = os.path.join(workdir, _binary_name())
        if not os.path.isfile(binary):
            raise ReferenceUnavailable(f"{binary} does not exist")
        if not os.path.isdir(os.path.join(workdir, "input_data")):
            raise ReferenceUnavailable(f"{workdir}/input_data does not exist")
        _PREPARED[year] = workdir
        return workdir

    workdir = os.path.join(_cache_root(), f"cec{year}")
    os.makedirs(workdir, exist_ok=True)
    binary = os.path.join(workdir, _binary_name())
    if not os.path.isfile(binary):
        try:
            _build(year, workdir)
        except ReferenceUnavailable:
            raise
        except Exception as exc:                      # network, zip, disk...
            raise ReferenceUnavailable(
                f"could not prepare the CEC {year} reference: "
                f"{type(exc).__name__}: {exc}") from exc
    _PREPARED[year] = workdir
    return workdir


def reference_values(year, func_num, ndim, points):
    """``f(x)`` from the organisers' C, for every row of ``points``."""
    workdir = prepare(year)
    points = np.asarray(points, dtype=float)
    payload = "%d %d %d\n" % (func_num, ndim, len(points))
    payload += "\n".join(" ".join("%.17g" % v for v in row) for row in points)
    result = subprocess.run([os.path.join(workdir, _binary_name())],
                            input=payload, capture_output=True, text=True,
                            cwd=workdir)
    if result.returncode != 0:
        raise ReferenceUnavailable(
            f"reference oracle failed for CEC{year} F{func_num} D{ndim}: "
            f"{result.stderr[-400:]}")
    values = np.array([float(v) for v in result.stdout.split()], dtype=float)
    if values.size != len(points):
        raise ReferenceUnavailable(
            f"reference oracle returned {values.size} values for "
            f"{len(points)} points (CEC{year} F{func_num} D{ndim}); "
            f"stderr: {result.stderr[-400:]}")
    return values


def reference_supports(year, func_num, ndim):
    """Whether the organisers ship the data this (function, dimension) needs."""
    try:
        workdir = prepare(year)
    except ReferenceUnavailable:
        return False
    data = os.path.join(workdir, "input_data")
    if not os.path.isfile(os.path.join(data, f"M_{func_num}_D{ndim}.txt")):
        return False
    needs_shuffle = ((year == 2017 and (11 <= func_num <= 20
                                        or func_num in (29, 30)))
                     or (year == 2022 and 6 <= func_num <= 8))
    if needs_shuffle and not os.path.isfile(
            os.path.join(data, f"shuffle_data_{func_num}_D{ndim}.txt")):
        return False
    return True
