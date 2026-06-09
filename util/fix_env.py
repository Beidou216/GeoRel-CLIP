import os


def fix_openmp_thread_env(default: str = "1") -> None:
    """Normalize invalid OpenMP-related env vars (e.g. OMP_NUM_THREADS=0)."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        val = os.environ.get(var)
        try:
            if val is not None and int(val) > 0:
                continue
        except ValueError:
            pass
        os.environ[var] = default
