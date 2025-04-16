import pickle
import os

import numpy as np
import multiprocessing as mp
from functools import partial
from datetime import datetime
from tqdm.auto import tqdm

from fit6 import fit6

"""This script processes a set of time steps and pixels in parallel using multiprocessing.
It uses the `fit6` function to perform fits on the data and saves the results to disk.
The script is designed to handle large datasets by processing each pixel independently, saving intermediate results to reduce memory usage.
"""



def process_pixel(px, tms, lam, siiv, siiv_err):
    """Process all time steps for a single pixel"""
    pixel_results = {}
    
    for tm in tqdm(tms, desc=f"Processing pixel {px}", leave=True):
        try:
            result = fit6(tm, px, lam, siiv, siiv_err)
            
            # Store the full result object
            pixel_results[tm] = result
            
        except Exception as e:
            print(f"Error fitting pixel {px}, time {tm}: {e}")
            pixel_results[tm] = None
    
    # Save intermediate results to reduce memory usage
    save_dir = "fit_results"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/pixel_{px}_results.pkl", "wb") as f:
        pickle.dump(pixel_results, f)
    
    return px


def parallel_fits(tms, pxs, lam, siiv, siiv_err, n_processes=None):
    """Run fits in parallel across multiple CPU cores"""
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)  # Use all cores except one
    
    print(f"Starting parallel processing with {n_processes} processes")
    start_time = datetime.now()
    
    # Create a partial function with fixed parameters
    process_func = partial(process_pixel, tms=tms, lam=lam, siiv=siiv, siiv_err=siiv_err)
    
    # Run the processing in parallel
    with mp.Pool(processes=n_processes) as pool:
        # Use imap to get a progress indicator
        for _ in tqdm(pool.imap(process_func, pxs), total=len(pxs)):
            pass
    
    end_time = datetime.now()
    print(f"Parallel processing completed in {end_time - start_time}")
    
    # Function to load all the saved results
    def load_results_dict():
        results_dict = {}
        save_dir = "fit_results"
        for px in pxs:
            try:
                with open(f"{save_dir}/pixel_{px}_results.pkl", "rb") as f:
                    results_dict[px] = pickle.load(f)
            except FileNotFoundError:
                print(f"Warning: No results found for pixel {px}")
        return results_dict
    
    return load_results_dict

'''
if __name__ == "__main__":
    # Example usage
    # Replace these with your actual data
    tms = np.arange(0, 100)  # Time steps
    pxs = np.arange(0, 10)   # Pixels
    lam = np.random.rand(100, 10)  # Example data for lambda
    siiv = np.random.rand(100, 10)  # Example data for siiv
    siiv_err = np.random.rand(100, 10)  # Example data for siiv error

    results_loader = parallel_fits(tms, pxs, lam, siiv, siiv_err)
    print("All results loaded.")
    
    # and load the resuts 
    results_dictionary = results_loader()
'''