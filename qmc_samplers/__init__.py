from __future__ import annotations

# Ensure the installation of numpy and scipy is done
try:
    import numpy as np
    from scipy.stats import qmc
except ImportError as e:
    raise ImportError(
        "This package requires numpy and scipy. "
        "Please install them via pip: pip install numpy scipy"
    ) from e


from .halton import halton_wrapper 
from .lhs import lhs_wrapper 
from .monte_carlo import monte_carlo_wrapper 
from .sobol import sobol_wrapper 


__all__ = ["halton", "lhs", "monte_carlo", "sobol"]

def __getattr__(name: str):
    if name in {"halton", "lhs", "monte_carlo", "sobol"}:
        return globals()[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__():
    return ["halton", "lhs", "monte_carlo", "sobol"]

def get_sampler(name: str):
    """Get the sampler function by name.

    Args:
        name (str): Name of the sampler. Options are 'halton', 'lhs', 'monte_carlo', 'sobol'.

    Returns:
        Callable: Corresponding sampler function.

    Raises:
        ValueError: If the sampler name is not recognized.
    """
    samplers = {
        "halton": halton_wrapper,
        "lhs": lhs_wrapper,
        "monte_carlo": monte_carlo_wrapper,
        "sobol": sobol_wrapper,
    }
    if name not in samplers:
        raise ValueError(f"Unknown sampler name: {name}. Available samplers are: {list(samplers.keys())}")
    return samplers[name]




