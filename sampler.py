r"""This is just a module to sample BBOB functions
"""

# Standard library imports
import os, sys
from typing import List, Tuple, Optional, Union
# Third party imports
from pathlib import Path # For handling file paths

import numpy as np
from ioh import get_problem # Use IOH library to get BBOB problems

# Import the logger
from ioh.iohcpp.logger import Analyzer
from ioh.iohcpp.logger.trigger import ALWAYS
from ioh.iohcpp.logger.property import TRANSFORMEDY

# Import samplers
from scipy.stats import qmc  # For Sobol and Halton sequences


# Local application imports
from argparse import ArgumentParser, Namespace 



def parse_args(args:List[str]) -> Namespace:
    r"""Parse command line arguments

    Args:
        args (List[str]): Command line arguments
    Returns:
        Namespace: Parsed command line arguments
    """

    parser = ArgumentParser(description="Sample BBOB functions",
                            add_help=True,
                            exit_on_error=True)
    parser.add_argument(
        "--problem-id",
        type=int,
        default=1,
        choices=list(range(1, 25)),
        help="BBOB problem ID to sample from",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        help="Dimension of the BBOB problem",
    )
    parser.add_argument(
        "--instance",
        type=int,
        default=1,
        choices=list(range(1, 16)),
        help="Instance ID of the BBOB problem",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=25,
        help="Multiplier to set the number of samples to generate based on dimension",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--sampler",
        type=str,
        default="lhs",
        choices=["monte-carlo","lhs","sobol","halton"],
        help="Sampler to use for generating samples",
    )

    # parser.add_argument(
    #     "--lhs-criterion",
    #     type=str,
    #     choices= [
    #         "center",
    #         "maximin",
    #         "centermaximin",
    #         "correlation",
    #     ],
    #     default="maximin",
    #     help="Criterion for Latin Hypercube Sampling Optimization",
    # )

    parser.add_argument(
        "--quasi-random-criterion",
        type=str,
        choices= [
            "random-cd",
            "lloyd"],
        default="random-cd",
        help="Criterion for Quasi-Random Sampling Optimization",
    )

    parser.add_argument(
        "--lhs-strength",
        type=int,
        default=1,
        choices=[1, 2],
        help="Strength for Latin Hypercube Sampling Optimization",
    )

    parser.add_argument(
        "--folder-name",
        type=str,
        default="",
        help="Output file to save the samples",
    )
    return parser.parse_args(args)

def main() -> None:
    """Main function to sample BBOB functions
    """

    # Parse command line arguments
    args = parse_args(sys.argv[1:])

    # Call the BBOB problem from IOH
    problem = get_problem(
        fid=args.problem_id,
        dimension=args.dimension,
        instance=args.instance
    )

    print(f"Sampling BBOB problem: {problem.meta_data.name}, Dimension: {args.dimension}, Instance: {problem.meta_data.instance}, Sampler: {args.sampler}")

    # Compute number of samples
    num_samples = args.dimension * args.multiplier

    # Generate samples based on the selected sampler
    if args.sampler == "monte-carlo":
        samples = monte_carlo_wrapper(
            dim=args.dimension,
            n_samples=num_samples,
            random_seed=args.random_seed,
        )
    elif args.sampler == "lhs":
        samples = lhs_wrapper(
            dim=args.dimension,
            n_samples=num_samples,
            random_seed=args.random_seed,
            criterion=args.quasi_random_criterion,
            strength=args.lhs_strength,
        )
    elif args.sampler == "sobol":
        samples = sobol_wrapper(
            dim=args.dimension,
            n_samples=num_samples,
            random_seed=args.random_seed,
            criterion=args.quasi_random_criterion,
        )
    elif args.sampler == "halton":
        samples = halton_wrapper(
            dim=args.dimension,
            n_samples=num_samples,
            random_seed=args.random_seed,
            criterion=args.quasi_random_criterion,
        )
    
    else:
        raise ValueError(f"Sampler {args.sampler} not recognized.")
    
    # Get problem bounds
    lb = problem.bounds.lb
    ub = problem.bounds.ub


    # Scale samples to the problem bounds
    samples = qmc.scale(samples, lb, ub)

    # Generate a logger to log the samples
    triggers = [ALWAYS]
    additional_properties = [TRANSFORMEDY]
    logger = Analyzer(triggers=triggers,
                      additional_properties=additional_properties,
                      root=str(Path(__file__).parent.joinpath(f"data/{args.sampler}/{args.problem_id}_{problem.meta_data.name}/Dim_{args.dimension}/Instance_{args.instance}").resolve()),
                      folder_name=f"{args.multiplier}_{args.random_seed}",
                      algorithm_name=args.sampler, # Set algorithm name to the sampler name
                      algorithm_info=f"Sampling BBOB functions, quasi-random criterion: {args.quasi_random_criterion}, random seed: {args.random_seed}, multiplier: {args.multiplier}",
                      store_positions=True)

    # Attach logger to the problem
    problem.attach_logger(logger)

    # Evaluate samples on the BBOB problem
    fitness_values = np.array([problem(x) for x in samples])

    # Detach logger from the problem
    problem.reset()
    problem.detach_logger()

    # Save samples to output file
    #np.save(args.output_file, samples)
    print(f"Saved {num_samples} of {problem.meta_data.name} problem, with multiplier {args.multiplier} samples to {logger.output_directory}")


def monte_carlo_wrapper(dim:int,
        n_samples:int,
        random_seed:Optional[int] = 43,
        **kwargs)->np.ndarray:
    
    """Generate a Monte Carlo Sampler"""

    # Set random seed
    rng = np.random.default_rng(seed=random_seed)
    data = rng.uniform(low=0.0, high=1.0, size=(n_samples, dim))

    return data

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
                                 rng=np.random.default_rng(seed=random_seed),
                                 optimization=criterion,
                                 strength=strength)
    
    data = sampler.random(n=n_samples)
    
    return data


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
                        rng=rng, 
                        optimization=criterion)
    
    # Sobol requires n_samples to be a power of 2
    n_samples_mod = 2 ** int(np.ceil(np.log2(n_samples)))
    data = sampler.random(n=n_samples_mod)
    
    return data

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
                        rng=rng,
                        optimization=criterion)
    
    data = sampler.random(n=n_samples)
    
    return data


if __name__ == "__main__":
    r"""Main function to sample BBOB functions
        """
    
    # Execute the main function
    main()
