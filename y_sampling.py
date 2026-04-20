from pathlib import Path
import pandas as pd
import numpy as np
#from cocoex import Suite, Problem
from ioh import get_problem
from ioh.iohcpp.problem import BBOB
from ioh import ProblemClass
from typing import List, Tuple, Optional
import os, sys
#from argparse import ArgumentParser, Namespace



def read_csv(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file and return its contents as a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The contents of the CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def save_csv(data: pd.DataFrame, file_path: str) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Args:
        data (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to the output CSV file.
    """
    data.to_csv(file_path, index=False)


def read_x_samples(file_path: str) -> np.ndarray:
    """
    Read X samples from a CSV file and return them as a numpy array.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        np.ndarray: The X samples as a numpy array.
    """
    df = pd.read_csv(file_path)
    return df.values


def get_x_sample_filelist(directory: str) -> List[Path]:
    r"""
    Read multiple X sample files from a directory and return a list of paths.

    Args:
        directory (str): The path to the directory containing CSV files.
    Returns:
        List[Path]: A list of paths to the CSV files in the directory.
    """ 

    dir_path = Path(directory)
    file_list = list(dir_path.rglob("*samples.csv"))
    return file_list

def distill_x_sample_list(file_list: List[Path]) -> List[Tuple[int, int, int, str]]:
    r"""
    Distill a list of X sample file paths into a list of tuples containing
    (dimension, seed, number of samples).

    Args:
    ------------
        file_list (List[Path]): A list of paths to the CSV files.

    Outputs:
    ------------
        - A list[Tuple[int, int, int, str]]: A list of tuples (dimension, seed, n_samples, objective_type).
    """ 
    distilled_list = []
    for file_path in file_list:
        parts = file_path.parts
        try:
            dim_part = [p for p in parts if p.startswith("Dimension_")][0]
            seed_part = [p for p in parts if p.startswith("seed_")][0]
            samples_part = [p for p in parts if p.startswith("Samples_")][0]
            objective_type_part = [p for p in parts if p in ["ELA_extraction","reduction"]][0]

            dim = int(dim_part.split("_")[1])
            seed = int(seed_part.split("_")[1])
            n_samples = int(samples_part.split("_")[1])
            objective_type = objective_type_part

            distilled_list.append((dim, seed, n_samples, objective_type))
        except (IndexError, ValueError):
            print(f"Warning: Could not parse file path '{file_path}'. Skipping.")
            continue
    return distilled_list


def evaluate_bbob_problem(prob: BBOB, X: pd.DataFrame) -> np.ndarray:
    """
    Evaluate a BBOB problem at given input points.

    Args:
        prob (Problem): A cocoex Problem object.
        X (pd.DataFrame): Input points of shape (n_samples, dim).

    Returns:
        np.ndarray: Function values at the input points.
    """

    assert X.shape[1] == prob.meta_data.n_variables, "Input dimension does not match problem dimension."

    fX = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        x = X.iloc[i].values
        f_val = prob(x)
        fX[i] = f_val


    return fX


def main():
    # Get the list of X sample files in the specified directory
    #directory = Path(os.getcwd())  # You can change this to any directory you want
    directory = Path("x_samples/ELA_extraction/Dimension_20")
    file_list = get_x_sample_filelist(directory)
    
    # Distill the file list to extract dimension, seed, and number of samples
    distilled_list = distill_x_sample_list(file_list)


    for ii, file in enumerate(file_list):
        # Get the distilled info
        dim, seed, n_samples, objective_type = distilled_list[ii]
        df_read = read_csv(file)

        
            
        for prob_id in range(1, 25):  # BBOB functions 1 to 24
            for instance in range(15):  # Instances 0 to 14
                
                prob:BBOB = get_problem(fid=prob_id, instance=instance, dimension=dim, problem_class=ProblemClass.REAL)


                #prob.free()
                fX = evaluate_bbob_problem(prob, df_read)

                # Delete the problem to free memory
                del prob

                # Save the results
                save_path = directory.joinpath("bbob_evaluations",objective_type,f"Dimension_{dim}",f"seed_{seed}",f"Samples_{n_samples}",f"f_{prob_id}",f"id_{instance}")
                if save_path.exists() is False:
                    save_path.mkdir(parents=True, exist_ok=True)
                else:
                    print(f"Evaluations for Function {prob_id}, Instance {instance} already exist at {save_path}, skipping.")
                    continue
                out_file = save_path.joinpath("evaluations.csv")
                df_out = pd.DataFrame(fX, columns=["fX"])
                save_csv(df_out, out_file)
                print(f"Saved evaluations for Function {prob_id}, Instance {instance} to {out_file}")
        


if __name__ == "__main__":
    main()



    