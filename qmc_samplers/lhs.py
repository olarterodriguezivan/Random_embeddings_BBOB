from typing import Optional
import numpy as np
from scipy.stats import qmc

def lhs_wrapper(dim:int, 
        n_samples:int,
        random_seed:Optional[int] = 42, 
        **kwargs)->np.ndarray:
    """Generate a Latin Hypercube Sample for pyDOE2.

    Args:
        dim (int): Number of dimensions
        n_samples (int): Number of samples
        random_seed (int, optional): Random seed for reproducibility

    Returns:
        np.ndarray: Latin Hypercube Sample
    """

    # Extract optional arguments
    criterion:str = kwargs.get('criterion', "random-cd")
    strength:int = kwargs.get('strength', 1)

    sampler = qmc.LatinHypercube(d=dim,
                                 scramble=True,
                                 seed=random_seed,
                                 optimization=criterion,
                                 strength=strength)
    
    data = sampler.random(n=n_samples)
    
    return data