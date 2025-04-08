import numpy as np
from lmfit import Model, Parameters
from itertools import combinations
import copy

import matplotlib.pyplot as plt
from datetime import datetime

# Define the Gaussian function
def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x-center)**2 / (2*sigma**2))


# Define the spectral model with multiple Gaussians
# Gaussian 1 is always included, others are optional
# Note: Gaussian 3, 4, and 5 are negative Gaussians
def spectral_model(x, amp1, cen1, sig1, amp2=0, cen2=0, sig2=0, amp3=0, cen3=0, sig3=0, 
                   amp4=0, cen4=0, sig4=0, amp5=0, cen5=0, sig5=0, amp6=0, cen6=0, sig6=0, 
                   use_g1=True, use_g2=False, use_g3=False, use_g4=False, use_g5=False, use_g6=False):

    result = 0
    
    # Gaussian 1 is always included
    if use_g1:
        result += gaussian(x, amp1, cen1, sig1)
    
    # Optional Gaussians
    if use_g2:
        result += gaussian(x, amp2, cen2, sig2)
    if use_g3:
        result -= gaussian(x, amp3, cen3, sig3)  # Note: Gaussian 3 is negative
    if use_g4:
        result -= gaussian(x, amp4, cen4, sig4)  # Note: Gaussian 4 is negative
    if use_g5:
        result -= gaussian(x, amp5, cen5, sig5)  # Note: Gaussian 5 is negative
    if use_g6:
        result += gaussian(x, amp6, cen6, sig6)
        
    return result


def check_overlapping_gaussians(params, combo):
    # Only relevant if both Gaussians 5 and 6 are in the combo
    if 5 in combo and 6 in combo:
        cen5 = params['cen5'].value
        cen6 = params['cen6'].value
        
        # Define threshold for "overlapping" centers (in wavelength units)
        overlap_threshold = 0.1  # adjust as needed
        
        if abs(cen5 - cen6) < overlap_threshold:
            amp5 = abs(params['amp5'].value)  # Absolute value since G5 is negative
            amp6 = params['amp6'].value
            
            # new combo excluding the smaller one
            new_combo = list(combo)
            if amp5 < amp6:
                new_combo.remove(5)
                #print(f"Removed Gaussian 5 due to overlap with Gaussian 6 (centers: {cen5:.4f}, {cen6:.4f})")
            else:
                new_combo.remove(6)
                #print(f"Removed Gaussian 6 due to overlap with Gaussian 5 (centers: {cen5:.4f}, {cen6:.4f})")
            
            return tuple(new_combo)
    
    return combo


def post_fit_check_overlap(result, combo):
    # Only check if both Gaussians 5 and 6 are in the fit result
    if 5 in combo and 6 in combo:
        cen5 = result.params['cen5'].value
        cen6 = result.params['cen6'].value
        
        overlap_threshold = 0.1  # Adjust as needed
        
        if abs(cen5 - cen6) < overlap_threshold:
           
            amp5 = abs(result.params['amp5'].value)  # Absolute since G5 is negative
            amp6 = result.params['amp6'].value
            # Check if the amplitudes are significantly different
            # Define a threshold for the ratio of amplitudes
            amp_ratio_threshold = 0.3  # Adjust as needed
            
            if amp5 < amp_ratio_threshold * amp6:
                #print(f"Post-fit: Gaussian 5 has much smaller amplitude than Gaussian 6 in overlap")
                return False
            elif amp6 < amp_ratio_threshold * amp5:
                #print(f"Post-fit: Gaussian 6 has much smaller amplitude than Gaussian 5 in overlap")
                return False
    
    # If no issues found, this fit is valid
    return True


# Function to fit the best combination of Gaussians
# This function will try all combinations of Gaussians 2-6 and return the best one
# Gaussian 1 is always included
def fit_best_combination(tm, px, lam, spec_cube, err_cube):

    spec = spec_cube[px, tm]
    spec_err = err_cube[px, tm]
    
    spec_max = np.max(spec)
    imax = np.argmax(spec)
    
    tmt = tm - 290
    
    # Create base model with Gaussian 1 always present
    base_params = Parameters()
    
    # Si IV --- Positive (Gaussian 1 - always included)
    base_params.add('amp1', value=spec_max, min=0.45*spec_max) 
    base_params.add('cen1', value=lam[imax], min=1402.75, max=1403.21) 
    base_params.add('sig1', value=0.15, vary=True, min=0.06, max=0.35)
    base_params.add('use_g1', value=True, vary=False)
    
    # Si IV --- Red (Gaussian 2)
    base_params.add('delta_rb', value=0.2, vary=True, min=0.05, max=0.45)
    base_params.add('amp2', expr='amp1 * delta_rb')
    base_params.add('lamb_rb', value=0.45, vary=True, min=0.35, max=0.5)
    base_params.add('cen2', expr='cen1 + lamb_rb')
    base_params.add('wid_rb', value=1., vary=True, min=0.5, max=1.25)
    base_params.add('sig2', expr='sig1 * wid_rb')
    base_params.add('use_g2', value=False, vary=False)
    
    # Fe II --- negative (Gaussian 3)
    base_params.add('amp3', value=spec[34], min=0)
    base_params.add('cen3', value=lam[34], vary=True, min=lam[32], max=lam[34])
    base_params.add('sig3', value=0.03, min=0.02, max=0.032)
    base_params.add('use_g3', value=False, vary=False)
    
    # Fe II --- negative (Gaussian 4)
    base_params.add('amp4', value=spec[41], min=0)
    base_params.add('cen4', value=lam[41], vary=True, min=lam[39], max=lam[41])
    base_params.add('sig4', value=0.03, min=0.02, max=0.032)
    base_params.add('use_g4', value=False, vary=False)
    
    # Si IV --- negative (Gaussian 5)
    base_params.add('amp5', value=0.2*spec_max, min=0.12*spec_max, max=0.33*spec_max)
    base_params.add('cen5', value=1402.87, min=1402.824, max=1402.93)
    base_params.add('neg_sig', value=0.2, vary=True, min=0.15, max=0.26)
    base_params.add('sig5', expr='sig1 * neg_sig')
    base_params.add('use_g5', value=False, vary=False)
    
    # Si IV --- positive (Gaussian 6)
    base_params.add('amp6', value=0.3*spec_max, min=0.0*spec_max, max=0.5*spec_max)
    base_params.add('cen6', value=1402.77, min=1402.76, max=1402.85)
    base_params.add('sig6', value=0.03, min=0.005, max=0.036)
    base_params.add('use_g6', value=False, vary=False)
    
    # Generate all possible combinations of Gaussians 2-6
    gaussians = [2, 3, 4, 5, 6]
    
    best_fit = None
    best_aic = np.inf  # Using AIC (Akaike Information Criterion) to compare models
    best_combination = []
    
    # Try with just Gaussian 1
    model = Model(spectral_model)
    params = copy.deepcopy(base_params)
    result = model.fit(spec, params, x=lam, weights=1/spec_err)
    
    if result.aic < best_aic:
        best_aic = result.aic
        best_fit = result
        best_combination = [1]
    
    # Try all combinations of the other Gaussians with Gaussian 1
    for r in range(1, len(gaussians) + 1):
        for combo in combinations(gaussians, r):
            # Check for overlapping Gaussians 5 and 6 before fitting
            modified_combo = check_overlapping_gaussians(base_params, combo)
            
            # Create a copy of base parameters for this combination
            params = copy.deepcopy(base_params)
            
            # Set which Gaussians to use for this combination
            for g in modified_combo:
                params[f'use_g{g}'].value = True
            
            # Fit the model with this combination
            try:
                result = model.fit(spec, params, x=lam, weights=1/spec_err)
                
                # Check for post-fit overlapping issues
                valid_fit = post_fit_check_overlap(result, modified_combo)
                
                # Compare using AIC (lower is better) if valid
                if valid_fit and result.aic < best_aic:
                    best_aic = result.aic
                    best_fit = result
                    best_combination = [1] + list(modified_combo)  # Gaussian 1 is always included
            except Exception as e:
                print(f"Error fitting combination {[1] + list(modified_combo)}: {e}")
                continue
    
    # Return the best fit result
    print(f"Best combination: {best_combination}")
    return best_fit

# wrap the fit function to match the original function signature
def fit6(tm, px, lam, spec_cube, err_cube):
    return fit_best_combination(tm, px, lam, spec_cube, err_cube)


def plot_fit_results(tm, px, lam, spec_cube, time_iris, result):
    """
    Plot the best-fit spectral model along with individual Gaussian components.
    
    Parameters:
    -----------
    tm : int
        Time index.
    px : int
        Pixel index.
    lam : array
        Wavelength array.
    spec_cube : array
        Spectral data cube.
    time_iris : array
        Time array with datetime strings.
    result : lmfit.ModelResult
        Result from the best-fit model from fit6.
    
    Returns:
    --------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes objects for further customization if needed.
    """
    # Get the spectrum for this time and pixel
    spec = spec_cube[px, tm]
    
    # Get the fit parameters
    fit_res = result.params
    
    def gaussian(x, amplitude, center, sigma):
        """ Gaussian function """
        return amplitude * np.exp(-(x-center)**2 / (2*sigma**2))
    
    # Get which Gaussians were used in the best fit
    used_gaussians = [1] 
    for i in range(2, 7):
        param_name = f'use_g{i}'
        if param_name in fit_res and fit_res[param_name].value:
            used_gaussians.append(i)
    print(f"Gaussians used in best fit: {used_gaussians}")
    
    # colors and styles for plotting
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'teal']
    linestyles = ['dashed'] * 6
    
    # Calculate individual Gaussians only for those used in the fit
    gaussians = []
    for i in range(1, 7):
        if i in used_gaussians:
            amp = fit_res[f'amp{i}'].value
            cen = fit_res[f'cen{i}'].value
            sig = fit_res[f'sig{i}'].value
            
            if i in [3, 4, 5]:  # Negative Gaussians
                gaussians.append(-gaussian(lam, amp, cen, sig))
            else:  # Positive Gaussians
                gaussians.append(gaussian(lam, amp, cen, sig))
        else:
            # Add a zero array for consistency in indexing
            gaussians.append(np.zeros_like(lam))
    
    # --------------------- plot ----------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Format the time string for the plot
    try:
        time_str = datetime.strptime(time_iris[tm].decode("utf-8"), '%Y-%m-%dT%H:%M:%S.%f')
    except:
        # Handle case where the time format might be different
        time_str = f"Frame {tm}"
    
    # original spectrum
    ax.plot(lam, spec, label=f't={time_str}', color='brown', alpha=0.5,ds='steps-mid')
    
    
    # best fit
    ax.plot(lam, result.best_fit, color='black', linestyle='dashed', label='Total fit',ds='steps-mid')
    
    # plot individual Gaussians
    for i, g_idx in enumerate(used_gaussians):
        ax.plot(lam, gaussians[g_idx-1], color=colors[g_idx-1], linestyle=linestyles[g_idx-1], 
                label=f'G{g_idx}')
    
    # legend and labels
    ax.legend(loc='best', fontsize=10)
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Spectral Fit with Best Combination of Gaussians')
    
    # text box with fit statistics
    chisqr = result.chisqr
    n_data = len(spec)
    n_params = len([p for p in result.params.values() if p.vary])
    dof = max(1, n_data - n_params)
    red_chisqr = chisqr / dof
    aic = result.aic

    stats_text = (rf'Reduced $\chi^2$: {red_chisqr:.2f} | AIC: {aic:.2f}')
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5), fontsize=10)
    
    plt.tight_layout()
    return fig, ax


# Example usage:
# fig, ax = plot_fit_results(tm, px, lam, spec_cube, time_iris, result)
# plt.show()