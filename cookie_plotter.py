import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.cm as cm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# import processor as processor

# sys.path.append("/home/kbrennan/phd/scripts")

import cookie_plot_utils as pu

pu.figure_setup()
plt.rcParams["hatch.linewidth"] = 0.5

def plot_comp_change(
    present,
    future,
    change,
    z_val,
    varname,
    label,
    map_params,
    sigma=2,
    cmap_abs="pink_r",
    cmap_change="PiYG_r",
    unit="",
    seasons=["MAM", "JJA", "SON"],
    conversion_factor=1,
    extend="both",
    mask_threshold=3,
    levels=None,
    levels_change=None,
):
    """
    gives 3xn grid of plots, first column is present (CNT), second column is future (PGW) and third column is the change.
    each column has it's own colorbar
    each row is one season, or there is just one row for combined seasons
    """

    projection = ccrs.RotatedPole(
        pole_longitude=map_params["rpole_lon"], pole_latitude=map_params["rpole_lat"]
    )
    n_rows = len(seasons)
    if n_rows == 1:
        height = pu.one_column()
    elif n_rows == 3:
        height = pu.one_column() * 0.7 * n_rows

    fig, axes = plt.subplots(
        n_rows,
        3,
        subplot_kw={"projection": projection},
        figsize=(pu.two_column(), height),
    )
    # make axes 2d if it's only 1d
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    letters = list(map(chr, range(97, 123)))
    for i, ax in enumerate(axes.flatten()):
        ax.add_feature(
            cfeature.COASTLINE, edgecolor=(0, 0, 0, 0.5), linewidth=0.5, alpha=0.7
        )
        ax.add_feature(
            cfeature.BORDERS, edgecolor=(0, 0, 0, 0.5), linewidth=0.5, alpha=0.7
        )

        ax.set_extent(map_params["map_extent"], crs=projection)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            linewidth=0.5,
            color="black",
            alpha=0.5,
            linestyle="--",
            draw_labels=True,
            x_inline=False,
            y_inline=False,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = False
        gl.bottom_labels = False
        pu.annotate_subplot(fig, ax, "(" + letters[i] + ")")

    axes[0, 0].set_title("CNT")
    axes[0, 1].set_title("PGW")
    axes[0, 2].set_title(r"$\Delta$")

    for i, season in enumerate(seasons):
        present_vals = present[season][varname].mean(dim="time") * conversion_factor
        future_vals = future[season][varname].mean(dim="time") * conversion_factor
        present_vals = processor.mask_inactive_areas(
            present_vals, future[season], present[season], threshold=mask_threshold
        )
        future_vals = processor.mask_inactive_areas(
            future_vals, future[season], present[season], threshold=mask_threshold
        )
        # add title on the side for season
        axes[i, 0].text(
            -0.04,
            0.5,
            season,
            transform=axes[i, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
        )

        pmesh = axes[i, 0].contourf(
            map_params["grid_rlon"],
            map_params["grid_rlat"],
            present_vals.T,
            cmap=cmap_abs,
            transform=projection,
            extend=extend,
            levels=levels,
        )
        axes[i, 0].set_xlabel("lon")
        axes[i, 0].set_ylabel("lat")

        pmesh = axes[i, 1].contourf(
            map_params["grid_rlon"],
            map_params["grid_rlat"],
            future_vals.T,
            cmap=cmap_abs,
            transform=projection,
            extend=extend,
            levels=levels,
        )

        axes[i, 1].set_xlabel("lon")
        axes[i, 1].set_ylabel("lat")

        pmesh_change = axes[i, 2].contourf(
            map_params["grid_rlon"],
            map_params["grid_rlat"],
            change[season][varname].T * conversion_factor,
            cmap=cmap_change,
            transform=projection,
            extend=extend,
            levels=levels_change,
        )

        # hatch insignificant values
        insignificant = z_val[season][varname] <= sigma
        insignificant = insignificant.astype(int)
        insignificant[np.isnan(z_val[season][varname])] = 0
        cf = axes[i, 2].contourf(
            map_params["grid_rlon"],
            map_params["grid_rlat"],
            insignificant.T,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["......"],
            transform=projection,
        )

    fig.subplots_adjust(wspace=0.05)
    fig.subplots_adjust(hspace=0.05)
    # colorbars
    # add cax by shifting all axes up
    fig.subplots_adjust(bottom=0.1)
    # get figure location from first axes
    figloc0 = axes[-1, 0].get_position()
    figloc1 = axes[-1, 1].get_position()
    figloc2 = axes[-1, 2].get_position()

    height = figloc0.y1 - figloc0.y0
    width = figloc1.x1 - figloc0.x0
    cax = fig.add_axes([figloc0.x0, figloc0.y0 - 0.13 * height, width, 0.08 * height])

    label_str = "{} {}".format(label, unit)
    cbar = plt.colorbar(
        pmesh,
        cax=cax,
        orientation="horizontal",
        # extend=extend,
        label=label_str,
        extendfrac=0.05,
    )

    width = figloc2.x1 - figloc2.x0
    cax = fig.add_axes([figloc2.x0, figloc2.y0 - 0.13 * height, width, 0.08 * height])

    label_str = "$\Delta${} {}".format(label, unit)
    cbar = plt.colorbar(
        pmesh_change,
        cax=cax,
        orientation="horizontal",
        # extend="both",
        label=label_str,
        extendfrac=0.1,
    )

    return fig, ax


def plot_change(
    change,
    z_val,
    varname,
    label,
    map_params,
    sigma=2,
    cmap="PiYG_r",
    unit="",
    extend="both",
):
    """
    the data variables can be either arrays or dict of arrays (e.g. with different seasons)
    """
    if "JJA" in change.keys():
        seasons = list(change.keys())
        n_seasons = len(seasons)
    else:
        n_seasons = 1
        seasons = [""]
        change = {"": change}
        z_val = {"": z_val}

    plt.rcParams["text.usetex"] = True
    projection = ccrs.RotatedPole(
        pole_longitude=map_params["rpole_lon"], pole_latitude=map_params["rpole_lat"]
    )
    if n_seasons == 1:
        fig, axes = plt.subplots(
            1,
            1,
            subplot_kw={"projection": projection},
            figsize=(pu.single_column(), pu.single_column() * 0.8),
            sharey=True,
        )
        axes = [axes]
    else:
        fig, axes = plt.subplots(
            1,
            n_seasons,
            subplot_kw={"projection": projection},
            figsize=(pu.two_column(), pu.one_column() * 0.7),
        )
        axes = axes.flatten()

    # ax.add_feature(
    #     cfeature.GSHHSFeature(scale="high", levels=[2]),
    #     linewidth=1,
    #     facecolor="None",
    #     alpha=0.6,
    # )

    change_all = np.concatenate([change[season][varname] for season in seasons])
    vmax = np.nanquantile(change_all, 0.99)
    vmin = np.nanquantile(change_all, 0.01)
    maxrange = np.max([np.abs(vmax), np.abs(vmin)])

    extend = "both"

    vmax = maxrange
    vmin = -maxrange

    levels = np.linspace(vmin, vmax, 12)

    for i, season in enumerate(seasons):
        ax = axes[i]
        ax.add_feature(
            cfeature.COASTLINE, edgecolor=(0, 0, 0, 0.5), linewidth=0.5, alpha=0.7
        )
        ax.add_feature(
            cfeature.BORDERS, edgecolor=(0, 0, 0, 0.5), linewidth=0.5, alpha=0.7
        )

        ax.set_extent(map_params["map_extent"], crs=projection)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            linewidth=0.5,
            color="black",
            alpha=0.5,
            linestyle="--",
            draw_labels=True,
            x_inline=False,
            y_inline=False,
        )
        gl.top_labels = False
        gl.right_labels = False

        if i > 0:
            gl.left_labels = False

        pmesh = ax.contourf(
            map_params["grid_rlon"],
            map_params["grid_rlat"],
            change[season][varname].T,
            cmap=cmap,
            transform=projection,
            levels=levels,
            extend=extend,
        )

        # hatch insignificant values
        insignificant = z_val[season][varname] <= sigma
        insignificant = insignificant.astype(int)
        insignificant[np.isnan(z_val[season][varname])] = 0
        cf = ax.contourf(
            map_params["grid_rlon"],
            map_params["grid_rlat"],
            insignificant.T,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["......"],
            transform=projection,
        )
        ax.set_title(season)

    patches = [
        mpatches.Patch(
            facecolor="none", hatch="......", label=r"$> " + str(int(sigma)) + "\sigma$"
        ),
    ]

    if len(seasons) == 1:
        cb_kwgs = {
            "ax": axes[0],
        }
    else:
        # make colorbar axis
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.83, 0.25, 0.02, 0.5])
        cb_kwgs = {
            "cax": cbar_ax,
        }

    # colorbar
    label = "{}\n within $r=\;${}$\,$km {}".format(label, map_params["radius_km"], unit)
    cticks = [20, 40, 60, 80, 100, 150, 200, 250, 300, 350]
    cticks = [x - 100 for x in cticks]
    cbar = plt.colorbar(
        pmesh,
        orientation="vertical",
        pad=0.02,
        shrink=0.78,
        extend=extend,
        label=label,
        **cb_kwgs,
        # ticks=cticks,
    )
    # ajust subplot hspace
    ax.legend(handles=patches, loc="center left", bbox_to_anchor=(1, 0))
    fig.subplots_adjust(wspace=0.05)
    return fig, ax


def plot_comparison(present, future, label, map_params, cmap="magma_r", extend="both"):
    """
    the data variables can be either arrays or dict of arrays (e.g. with different seasons)
    """

    projection = ccrs.RotatedPole(
        pole_longitude=map_params["rpole_lon"], pole_latitude=map_params["rpole_lat"]
    )
    fig, axes = plt.subplots(
        1, 2, subplot_kw={"projection": projection}, figsize=(8, 5)
    )

    vmax = np.nanquantile([present, future], 0.999)
    vmin = np.nanquantile([present, future], 0.001)

    levels = np.linspace(vmin, vmax, 12)
    for ax in axes:
        # ax.add_feature(
        #     cfeature.GSHHSFeature(scale="high", levels=[2]),
        #     linewidth=1,
        #     facecolor="None",
        #     alpha=0.6,
        # )
        ax.add_feature(
            cfeature.COASTLINE, edgecolor=(0, 0, 0, 0.5), linewidth=0.5, alpha=0.7
        )
        ax.add_feature(
            cfeature.BORDERS, edgecolor=(0, 0, 0, 0.5), linewidth=0.5, alpha=0.7
        )

        ax.set_extent(map_params["map_extent"], crs=projection)

    pmesh = axes[0].contourf(
        map_params["grid_rlon"],
        map_params["grid_rlat"],
        present.T,
        cmap=cmap,
        transform=projection,
        levels=levels,
        extend=extend,
    )
    axes[0].set_title("present")
    axes[0].set_xlabel("lon")
    axes[0].set_ylabel("lat")

    pmesh = axes[1].contourf(
        map_params["grid_rlon"],
        map_params["grid_rlat"],
        future.T,
        cmap=cmap,
        transform=projection,
        levels=levels,
        extend=extend,
    )
    axes[1].set_title("future")
    axes[1].set_xlabel("lon")
    axes[1].set_ylabel("lat")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.25, 0.02, 0.5])
    fig.colorbar(
        pmesh,
        cax=cbar_ax,
        label=label + "\nin " + str(map_params["radius_km"]) + " km radius",
        extend="max",
    )
    return fig, axes


def plt_composite(
    fig,
    ax,
    vals_surf=None,
    vals_cont=None,
    vals_quiver=None,
    cmap="cividis",
    cbar_label="",
    cont_label="",
    overlay_color="white",
    quiver_color=None,
    quiver_scale=80,
    quiver_key_scale=5,
    surface_levels=12,
    contour_levels=9,
    surface_ticks=None,
    clabel_pos="default",
    extend_surface="both",
    radius=40,
    cbar_on=True,
    negative_linestyles="dashed",
    linestyles=None,
    center_marker=False,
    norm=None,
    vals_hatch=None,
    hatch_level=2,
    flip_clabels=True,
    clabel_radius_factor=0.8,
    levels=None
):

    x = vals_surf.x
    y = vals_surf.y

    circle = mpatches.Circle(
        xy=(0, 0),
        radius=radius,
        edgecolor="k",
        facecolor="none",
        linewidth=pu.axis_lw(),
        transform=ax.transData,
        zorder=10,
    )

    if vals_surf is not None:
        # if surface_ticks is None:
        #     quantile = 0.005
        #     max_val = np.round(np.nanquantile(vals_surf,1-quantile)).astype(int)
        #     min_val = np.round(np.nanquantile(vals_surf,quantile)).astype(int)
        #     diff = max_val - min_val
        #     n_goal = 5
        #     step = diff / n_goal
        #     step = np.round(step).astype(int)
        #     step = np.max([1,step])
        #     surface_ticks = np.arange(min_val, max_val + step, step)
        #     if surface_ticks[-1] > max_val:
        #         surface_ticks = surface_ticks[:-1]
        #     if surface_ticks[0] < min_val:
        #         surface_ticks = surface_ticks[1:]

        t_contour = ax.contourf(
            x,
            y,
            vals_surf,
            levels=levels,
            cmap=cmap,
            #extend=extend_surface,
            #levels=surface_levels,
            norm=norm,
            extend='both'
        )
        ax.set_aspect("equal")

        if cbar_on:
            # cax = fig.add_axes(
            #     [
            #         ax.get_position().x0,
            #         ax.get_position().y0 - ax.get_position().height * 0.1,
            #         ax.get_position().width,
            #         ax.get_position().height * 0.05,
            #     ]
            # )
            if norm is not None:
                cbar = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = cmap), ax = ax, location="bottom", pad=0.065, shrink=0.9, label=cbar_label)
            else:
                cbar = fig.colorbar(
                    t_contour,
                    ax=ax,
                    location="bottom",
                    pad=0.065,
                    # fraction=0.3,  # 0.046,
                    shrink=0.9,
                    label=cbar_label,
                )
            cbar.ax.tick_params(rotation=45)
            if surface_ticks is not None:
                cbar.set_ticks(surface_ticks)
                cbar.set_ticklabels(surface_ticks)
                cbar.minorticks_off()

        for coll in t_contour.collections:
            coll.set_clip_path(circle)

    if vals_cont is not None:
        p_contour = ax.contour(
            x,
            y,
            vals_cont,
            colors=overlay_color,
            levels=contour_levels,
            linestyles=linestyles,
            negative_linestyles=negative_linestyles,
        )
        if clabel_pos == "default":
            clabels = ax.clabel(p_contour, p_contour.levels,
            inline=True,inline_spacing=-15,fontsize=8) #-40 for FI; -15 for TOT_PREC
        elif clabel_pos == "along_circle":
            positions = make_nice_contour_label_positions_radius(
                ax, p_contour, radius * clabel_radius_factor,flip=flip_clabels,
            )
            # print(positions)
            clabels = ax.clabel(p_contour, p_contour.levels, manual=positions)

        for coll in p_contour.collections:
            coll.set_clip_path(circle)
        for coll in clabels:
            coll.set_clip_path(circle)

        # label for contours if not empty
        if cont_label:
            ax.text(
                0,
                0.03,
                "cont.",
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.text(
                0,
                -0.045,
                cont_label,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

    if vals_quiver is not None:
        if quiver_color is None:
            quiver_color = overlay_color
        n_skip = 5
        q = ax.quiver(
            x[::n_skip],
            y[::n_skip],
            vals_quiver[0][::n_skip, ::n_skip],
            vals_quiver[1][::n_skip, ::n_skip],
            scale=quiver_scale,
            color=quiver_color,
        )
        qk = ax.quiverkey(
            q,
            0.7,
            -0.045,
            quiver_key_scale,
            str(quiver_key_scale) + r"$\,\mathrm{m\,s^{-1}}$",
            labelpos="E",
            coordinates="axes",
            color="black",
            # horizontalalignment="right"
        )
        q.set_clip_path(circle)

    if vals_hatch is not None:
        insignificant = vals_hatch > hatch_level
        insignificant = insignificant.astype(int)
        hatch = ax.contourf(
            x,
            y,
            insignificant,
            colors="none",
            hatches=["//////"],
            levels=[0.5, 1.5]
        )

        for coll in hatch.collections:
            coll.set_clip_path(circle)

    if center_marker:
        if center_marker is True:
            marker = "o"
        else:
            marker = center_marker
        ax.plot(0, 0, marker, markersize = 2, color=overlay_color)

    pad = 1 + 0.005
    ax.set_xlim(-radius * pad, radius * pad)
    ax.set_ylim(-radius * pad, radius * pad)

    attr_x = pu.assign_var("x")
    attr_y = pu.assign_var("y")
    ax.set_xlabel(attr_x["label"])
    ax.set_ylabel(attr_y["label"])

    # hide axes
    ax.axis("off")
    ax.add_patch(circle)
    ax.set_aspect("equal")

    return t_contour


def text_on_circle(ax, radius, angle, text, xy_0=(0, 0)):
    x = (
        np.cos(np.linspace(0, 2 * np.pi, 100) - np.pi / 2 + angle / 180 * np.pi)
        * radius
        * 1.02
        + xy_0[0]
    )
    y = (
        -np.sin(np.linspace(0, 2 * np.pi, 100) - np.pi / 2 + angle / 180 * np.pi)
        * radius
        * 1.02
        + xy_0[1]
    )

    text = pu.CurvedText(
        x=x,
        y=y,
        text=text,
        va="bottom",
        ha="right",
        axes=ax,
    )
    return

def text_on_circle_counterclock(ax, radius, angle, text, xy_0=(0, 0)):
    # Create the circle's coordinates
    angles = np.linspace(0, 2 * np.pi, 100) - np.pi / 2 - angle / 180 * np.pi
    x = np.cos(angles) * radius * 1.02 + xy_0[0]
    y = np.sin(angles) * radius * 1.02 + xy_0[1]

    # Display the text counterclockwise
    text = pu.CurvedText(
        x=x,
        y=y,
        text=text,
        va="bottom",
        ha="right",
        axes=ax,
    )
    return


def make_nice_contour_label_positions_radius(ax, p_contour, radius, flip=False):
    """generate locations for contour labels for coordinates where the paths from p_contour intersect a circle of radius radius centered at the origin"""
    clabel_pos = []
    lines = p_contour.collections
    for line in lines:
        paths = line.get_paths()
        for path in paths:
            x, y = path.vertices.T
            intersects = get_circle_intersection_points(x, y, radius)
            if len(intersects) == 0:
                continue
            if flip:
                clabel_pos.append(intersects[-1])
            else:
                clabel_pos.append(intersects[0])

    return clabel_pos


def get_circle_intersection_points(x, y, radius):
    intersects = []
    for i in range(len(x) - 1):
        # check wether the line segment from x[i] to x[i+1] intersects the circle
        x1, y1 = x[i], y[i]
        x2, y2 = x[i + 1], y[i + 1]
        r1 = np.sqrt((x1) ** 2 + (y1) ** 2) - radius
        r2 = np.sqrt((x2) ** 2 + (y2) ** 2) - radius
        # the line segment intersects the sign of r1 and r2 is different
        if r1 * r2 < 0:
            intersects.append((x1, y1))

    return intersects


def make_nice_contour_label_positions_line(ax, p_contour, x_pos=None, y_pos=None):
    """generate locations for contour labels for coordinates where the paths from p_contour intersect a line defined by x or y"""
    clabel_pos = []
    for line in p_contour.collections:
        for path in line.get_paths():
            x, y = path.vertices.T
            for i in range(len(x) - 1):
                # check wether the line segment from x[i] to x[i+1] intersects the line
                x1, y1 = x[i], y[i]
                x2, y2 = x[i + 1], y[i + 1]
                dx, dy = x2 - x1, y2 - y1
                if x_pos is not None:
                    x_int = x_pos
                    y_int = y1 + (x_pos - x1) * dy / dx
                elif y_pos is not None:
                    y_int = y_pos
                    x_int = x1 + (y_pos - y1) * dx / dy
                else:
                    raise ValueError("either x_pos or y_pos must be provided")
                # check if intersection points are on the line segment
                if (x_int - x1) * (x_int - x2) <= 0 and (y_int - y1) * (
                    y_int - y2
                ) <= 0:
                    clabel_pos.append((x_int, y_int))
                    break
            else:
                continue
            break
    return clabel_pos