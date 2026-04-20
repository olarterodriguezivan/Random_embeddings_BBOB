from typing import Optional
import numpy as np
from scipy.stats import qmc

def sobol_wrapper(dim:int, 
        n_samples:int,
        random_seed:Optional[int] = 42, 
        **kwargs)->np.ndarray:
    """Generate a Sobol Sample using scipy.stats.qmc.

    Args:
        dim (int): Number of dimensions
        n_samples (int): Number of samples
        random_seed (int, optional): Random seed for reproducibility

    Returns:
        np.ndarray: Sobol Sample
    """
    # Initialize random generator
    rng = np.random.default_rng(seed=random_seed)

    # Extract optional arguments
    criterion:str = kwargs.get('criterion', "random-cd")

    sampler = qmc.Sobol(d=dim, 
                        scramble=True, 
                        seed=random_seed,
                        optimization=criterion)
    
    # Sobol requires n_samples to be a power of 2
    n_samples_mod = 2 ** int(np.ceil(np.log2(n_samples)))
    data = sampler.random(n=n_samples_mod)
    
    return data