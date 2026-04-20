from typing import Optional
import numpy as np
from scipy.stats import qmc

def halton_wrapper(dim:int, 
        n_samples:int,
        random_seed:Optional[int] = 42, 
        **kwargs)->np.ndarray:
    """Generate a Halton Sample using scipy.stats.qmc.

    Args:
        dim (int): Number of dimensions
        n_samples (int): Number of samples
        random_seed (int, optional): Random seed for reproducibility

    Returns:
        np.ndarray: Halton Sample
    """

    # Instantiate random number generator
    rng = np.random.default_rng(seed=random_seed)

    # Extract optional arguments
    criterion:str = kwargs.get('criterion', "random-cd")

    sampler = qmc.Halton(d=dim, 
                        scramble=True, 
                        seed=random_seed,
                        optimization=criterion)
    
    data = sampler.random(n=n_samples)
    
    return data