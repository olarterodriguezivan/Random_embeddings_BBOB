from typing import Optional
import numpy as np

def monte_carlo_wrapper(dim:int,
        n_samples:int,
        random_seed:Optional[int] = 43,
        **kwargs)->np.ndarray:
    
    """Generate a Monte Carlo Sampler"""

    # Set random seed
    rng = np.random.default_rng(seed=random_seed)
    data = rng.uniform(low=0.0, high=1.0, size=(n_samples, dim))

    return data






