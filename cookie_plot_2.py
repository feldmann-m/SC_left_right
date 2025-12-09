import os
import argparse
from glob import glob
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
import re
import xarray as xr

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import cookie_plot_utils as pu # Killians script that contains helping functions for plotting
import cookie_plotter as plotter # Killians plotting function for nice cookies

def set_colorbar_label(clb, stat, label, unit, delta):
    
    stats_mapping = {
    "mean": "mean",
    "std": "standard deviation",
    "q01": "1st percentile",
    "q05": "5th percentile",
    "q10": "10th percentile",
    "median": "median",
    "q90": "90th percentile",
    "q95": "95th percentile",
    "q99": "99th percentile"
    }
    
    if delta == True:
        clb.set_label(f"$\\Delta$ {label} [{unit}]\n({stats_mapping.get(stat, stat)})", fontsize = 1.25 * pu.font_size())
    else:
        clb.set_label(f"{label} [{unit}]\n({stats_mapping.get(stat, stat)})", fontsize = 1.25 * pu.font_size())
        
def round_nicely_arr(arr):
    i = 1
    while True:
        nice_rounded_arr = [round_nicely_number(element, i) for element in arr]
        if not contains_duplicates(nice_rounded_arr):
            break
        i += 1
    return nice_rounded_arr
        
def round_nicely_number(x, n):
    if x == 0:
        return 0  # If the number is zero, return zero directly
    
    # Determine the order of magnitude of the number
    order_of_magnitude = np.floor(np.log10(abs(x)))
    
    # Scale the number so that the first significant digit is in the units place
    scaled = x / (10 ** order_of_magnitude)
    
    # Round to n significant digit
    rounded = round(scaled, n)
    
    # Scale the number back to its original magnitude
    result = rounded * (10 ** order_of_magnitude)
    
    return result

def contains_duplicates(arr):
    return len(arr) != len(np.unique(arr))

def shrink_to_n(arr, n, lower_limit, upper_limit):
    # Step 1: Filter values within the lower and upper limits
    filtered = arr[(arr >= lower_limit) & (arr <= upper_limit)]
    
    if len(filtered) <= n:
        return arr
    else:
        # Calculate distances from the nearest limit
        distances_from_limits = np.minimum(np.abs(filtered - lower_limit), np.abs(filtered - upper_limit))
        
        # Sort the indices based on distances (farthest from the limits first)
        sorted_indices = np.argsort(distances_from_limits)
        
        # Remove the farthest values to retain only n closest to the limits
        filtered = np.delete(filtered, sorted_indices[:-n])  # Keep the closest n values
    
    # Step 3: Sort the final array in increasing order
    return np.sort(filtered)

def main(relative_time, subdomain, season, stat, variable, label, unit, cmap, direction, pressure_level = None, boundaries=None, boundaries_delta=None, cmap_delta=None):

    signature = 1 if direction == 'right' else -1
    nice_rounding = False
    
    present_composite = glob(f'/storage/workspaces/giub_meteo_impacts/ci01/supercell_climate/cookies/present_t-{relative_time}/comp_{direction}_movers/{subdomain}_{season}_signature{signature}_{stat}*')
    if not present_composite:
        return
    else:
        present_composite = present_composite[0]
    future_composite = glob(f'/storage/workspaces/giub_meteo_impacts/ci01/supercell_climate/cookies/future_t-{relative_time}/comp_{direction}_movers/{subdomain}_{season}_signature{signature}_{stat}*')
    if not future_composite:
        return
    else:
        future_composite = future_composite[0]
    
    # extract the number of cookies used for the composite
    pattern = r'_n(\d+)\.nc$'
    match_present = re.search(pattern, present_composite)
    match_future = re.search(pattern, future_composite)
    cookies_present = match_present.group(1)
    cookies_future = match_future.group(1)

    with xr.open_dataset(present_composite) as present:
        pass
    
    with xr.open_dataset(future_composite) as future:
        pass

    present = present.drop_vars('cookie_id')
    present = present.squeeze()
    
    future = future.drop_vars('cookie_id')
    future = future.squeeze()

    fig, axes = plt.subplots(1, 3, figsize = (pu.two_column() * 1.75, pu.two_column() * 1.75))
    
    vals_surf_present = present[variable] if pressure_level == None else present[variable].sel(pressure = pressure_level)
    vals_surf_future = future[variable] if pressure_level == None else future[variable].sel(pressure = pressure_level)
    
    if variable == 'FI':
        vals_surf_present = mpcalc.geopotential_to_height(vals_surf_present) / 10
        vals_surf_future = mpcalc.geopotential_to_height(vals_surf_future) / 10
    elif variable == 'QV_2M' or variable == 'QV':
        vals_surf_present = vals_surf_present * 1000
        vals_surf_future = vals_surf_future * 1000
    
    if boundaries is None:
        lower_boundary = np.nanmin([np.nanmin(vals_surf_present.values), np.nanmin(vals_surf_future.values)])
        upper_boundary = np.nanmax([np.nanmax(vals_surf_present.values), np.nanmax(vals_surf_future.values)])
        if lower_boundary != upper_boundary:
            boundaries = np.linspace(lower_boundary, upper_boundary, 15)
            if nice_rounding == True:
                boundaries = round_nicely_arr(boundaries)
        else:
            boundaries = np.linspace(lower_boundary - 0.5, upper_boundary + 0.5, 15)
    norm = mcolors.BoundaryNorm(boundaries, ncolors = 256, extend = 'both')
    
    t_contour = plotter.plt_composite(
        fig,
        axes[0],
        vals_surf = vals_surf_present,
        #vals_quiver = (present['U_10M'], present['V_10M']),
        #quiver_color = 'black',
        #quiver_scale = 20,
        #quiver_key_scale = 5,
        cmap = cmap,
        overlay_color = 'black',
        radius = 60,
        cbar_on = False,
        center_marker = True,
        levels = boundaries,
        norm = norm
    )
    plotter.text_on_circle_counterclock(axes[0], 68, 58.5, subdomain + ',' + f'ENV-{relative_time}h')
    plotter.text_on_circle(axes[0], 60, 24, str(cookies_present) + f' {direction} movers')
    axes[0].set_title('CTRL', pad = 15, fontsize = 2 * pu.font_size())

    plotter.plt_composite(
        fig,
        axes[1],
        vals_surf = vals_surf_future,
        #vals_quiver = (future['U_10M'], future['V_10M']),
        #quiver_color = 'black',
        #quiver_scale = 20,
        #quiver_key_scale = 5,
        cmap = cmap,
        overlay_color = 'black',
        radius = 60,
        cbar_on = False,
        center_marker = True,
        levels = boundaries,
        norm = norm
    )
    plotter.text_on_circle_counterclock(axes[1], 68, 58.5, subdomain + ',' + f'ENV-{relative_time}h')
    plotter.text_on_circle(axes[1], 60, 24, str(cookies_future) + f' {direction} movers')
    axes[1].set_title('PGW', pad = 15, fontsize = 2 * pu.font_size())
    
    
    vals_surf_delta = vals_surf_future - vals_surf_present
    lower_boundary = np.nanmin(vals_surf_delta.values)
    upper_boundary = np.nanmax(vals_surf_delta.values)
    if boundaries_delta is None:
        if lower_boundary < 0 and upper_boundary <= 0:
            if cmap == 'Reds':
                cmap_delta = plt.get_cmap('Blues_r')
            elif cmap == 'Reds_r':
                cmap_delta = plt.get_cmap('Reds_r')
            elif cmap == 'Greens':
                cmap_delta = plt.get_cmap('Oranges_r')
            if lower_boundary != upper_boundary:
                boundaries_delta = np.linspace(lower_boundary, upper_boundary, 16)
                if nice_rounding == True:
                    boundaries_delta = round_nicely_arr(boundaries_delta)
            else:
                boundaries_delta = np.linspace(lower_boundary - 0.5, upper_boundary + 0.5, 16)
        elif lower_boundary >= 0 and upper_boundary > 0:
            if cmap == 'Reds':
                cmap_delta = plt.get_cmap('Reds')
            elif cmap == 'Reds_r':
                cmap_delta = plt.get_cmap('Blues')
            elif cmap == 'Greens':
                cmap_delta = plt.get_cmap('Greens')
            if lower_boundary != upper_boundary:
                boundaries_delta = np.linspace(lower_boundary, upper_boundary, 16)
                if nice_rounding == True:
                    boundaries_delta = round_nicely_arr(boundaries_delta)
            else:
                boundaries_delta = np.linspace(lower_boundary - 0.5, upper_boundary + 0.5, 16)
        else:
            if cmap == 'Reds':
                cmap_delta = plt.get_cmap('RdBu_r')
            elif cmap == 'Reds_r':
                cmap_delta = plt.get_cmap('RdBu')
            elif cmap == 'Greens':
                cmap_delta = plt.get_cmap('BrBG')
            cols = cmap_delta(np.linspace(0, 1, 256))
            cols[127] = np.array([1, 1, 1, 1])
            cmap_delta = mcolors.ListedColormap(cols)
            boundary = np.nanmax([abs(lower_boundary), abs(upper_boundary)])
            if boundary != 0:
                boundaries_delta = np.linspace(-boundary, boundary, 16)
                if nice_rounding == True:
                    boundaries_delta = round_nicely_arr(boundaries_delta)
            else:
                boundaries_delta = np.linspace(-0.5, 0.5, 16)
       
    # load p-values to hatch the insignificant grid points
    variable_string = variable if pressure_level is None else variable + '_' + str(pressure_level)
    
    path_significances = f'/storage/workspaces/giub_meteo_impacts/ci01/supercell_climate/cookies/significances/significance_t-{relative_time}/{subdomain}/{direction}'
    p = np.loadtxt(f'{path_significances}/{variable_string}_significance_t-{relative_time}.csv', delimiter = ',')

    norm_delta = mcolors.BoundaryNorm(boundaries_delta, ncolors = 256, extend = 'both')
    plotter.plt_composite(
        fig,
        axes[2],
        vals_surf = vals_surf_delta,
        #vals_quiver = (future['U_10M'] - present['U_10M'], future['V_10M'] - present['V_10M']),
        #quiver_color = 'black',
        #quiver_scale = 5,
        #quiver_key_scale = 1,
        cmap = cmap_delta,
        overlay_color = 'black',
        radius = 60,
        cbar_on = False,
        center_marker = True,
        levels = boundaries_delta,
        norm = norm_delta,
        vals_hatch = p,
        hatch_level = 0.05
    )
    plotter.text_on_circle_counterclock(axes[2], 68, 58.5, subdomain + ',' + f'ENV-{relative_time}h')
    axes[2].set_title('$\Delta=$' + 'PGW' + '$-$' + 'CTRL', pad = 15, fontsize = 2 * pu.font_size())
    
    ax0 = axes[0].get_position().get_points().flatten()
    ax1 = axes[1].get_position().get_points().flatten()
    ax2 = axes[2].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([ax0[0], 0.33, ax1[2] - ax0[0], 0.015])
    ax_cbar_delta = fig.add_axes([ax2[0], 0.33, ax2[2] - ax2[0], 0.015])
    clb = plt.colorbar(cm.ScalarMappable(norm = norm, cmap = cmap), cax = ax_cbar, orientation = 'horizontal')
    clb.set_ticks(boundaries[::2])
    clb.minorticks_off()
    clb.ax.tick_params(rotation = 45)
    set_colorbar_label(clb, stat, label, unit, False)
    clb_delta = plt.colorbar(cm.ScalarMappable(norm = norm_delta, cmap = cmap_delta), cax = ax_cbar_delta, orientation = 'horizontal')
    clb_delta.set_ticks(boundaries_delta[::3])
    clb_delta.minorticks_off()
    clb_delta.ax.tick_params(rotation = 45)
    set_colorbar_label(clb_delta, stat, label, unit, True)
    
    if pressure_level == None:
        pressure_str = ''
    else:
        pressure_str = str(pressure_level)
    
    for ax in axes:
        ax.arrow(0, 0, 12, 0, head_width = 2, head_length = 2, fc = 'black', ec = 'black')
        
    labels = ['a)', 'b)', 'c)']
    for i, ax in enumerate(axes):
        ax.annotate(labels[i], xy = (0, 1), xycoords = 'axes fraction', xytext = (0, 0), textcoords = 'offset fontsize', fontsize = 12, verticalalignment = 'top', bbox = dict(facecolor = '1.0', edgecolor = 'none', boxstyle = 'square, pad = 0.2'), zorder = 10)
        
    base_path = f'composite_plots/'
    file_name = direction + '_' + subdomain + '_' + season + ' ' + variable + '_' + pressure_str + '_' + stat + '_t-' + str(relative_time) + '.png'
    full_path = os.path.join(base_path, file_name)

    os.makedirs(base_path, exist_ok = True)

    plt.savefig(fname = full_path, format = 'png', bbox_inches = 'tight', dpi = 400)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Visualize cookies from calculated composites")
    parser.add_argument("relative_time", type = int, help = "Relative time to mesocyclone detection time point")
    parser.add_argument("season", type = str, help = "Season")
    parser.add_argument("variable", type = str, help = "Variable to plot")
    parser.add_argument("label", type = str, help = "Label of the variable")
    parser.add_argument("unit", type = str, help = "Physical unit of the variable")
    parser.add_argument("cmap", type = str, help = "Colormap for CTRL and PGW plot")
    parser.add_argument("direction", type = str, help = "Left or right moving")
        
    parser.add_argument(
        "--pressure_level",
        type = int,
        default = None,
        help = "Pressure level when variable is 3D",
    )
    
    args = parser.parse_args()
    for subdomain in ['AL', 'NAL', 'SAL', 'MD', 'MDS', 'MDL', 'BI', 'EE', 'FR', 'IP', 'CE', 'NA', 'BA', 'REST', 'EUR']:
        for stat in ['mean', 'std', 'q01', 'q05', 'q10', 'median', 'q90', 'q95', 'q99']:
            main(args.relative_time, subdomain, args.season, stat, args.variable, args.label, args.unit, args.cmap, args.direction, args.pressure_level)