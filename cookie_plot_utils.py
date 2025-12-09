import math
import matplotlib.pyplot as plt
import os
import subprocess
import tempfile
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredText
import matplotlib.transforms as mtransforms
import cartopy.io.shapereader as shpreader

from matplotlib import patches
from matplotlib import text as mtext
import numpy as np


def get_fig_size(fig_width=None, fig_height=None):
    """
    If no height is given, it is computed using the golden ratio.
    """
    if not fig_width:
        fig_width = textwidth()

    if not fig_height:
        golden_ratio = (1 + math.sqrt(5)) / 2
        fig_height = fig_width / golden_ratio

    size = (fig_width, fig_height)
    return size


"""
The following functions can be used by scripts to get the sizes of
the various elements of the figures.
"""


def fig_size():
    """Size of the figure in inches"""
    return 3.54


def label_size():
    """Size of axis labels"""
    return 10


def font_size():
    """Size of all texts shown in plots"""
    return 8


def ticks_size():
    """Size of axes' ticks"""
    return 8


def axis_lw():
    """Line width of the axes"""
    return 0.6


def plot_lw():
    """Line width of the plotted curves"""
    return 1


def one_column():
    """Return the size of a figure in one column (WCD)"""
    return 8.3 / 2.54


def single_column():
    return one_column()


def two_column():
    """Return the size of a figure in two columns (WCD)"""
    return 17.4 / 2.54


def textwidth():
    """Width of the text column in the document"""
    return 5.78


def textheight(factor=0.9):
    """Height of the text column in the document
    leave room for figure caption with factor"""
    return 8.17 * factor


def draw_fig(figsize=(3.54, 3.54)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    return fig, ax


def annotate_subplot(
    fig,
    ax,
    text,
    color="black",
    xpos=0,
    ypos=1,
    backgroundcolor=None,
    horizontalalignment="left",
):
    trans = mtransforms.ScaledTranslation(
        5 / 72 - 0.02, -10 / 72 + 0.01, fig.dpi_scale_trans
    )
    if backgroundcolor is None:
        ax.text(
            xpos,
            ypos,
            text,
            transform=ax.transAxes + trans,
            color=color,
            ha=horizontalalignment,
        )
    else:
        ax.text(
            xpos,
            ypos,
            text,
            transform=ax.transAxes + trans,
            color=color,
            bbox=dict(facecolor=backgroundcolor, edgecolor=[0, 0, 0, 0]),
            ha=horizontalalignment,
            zorder=2,
        )

    return ax


def boxplot(ax, data, positions, jitter=0.2, color="black", label=None, **kwargs):
    """
    boxplot with jittered fliers
    """
    bp = ax.boxplot(data, positions=positions, sym="", **kwargs)
    set_box_color(bp, color)
    if label is not None:
        # draw temporary lines and use them to create a legend
        ax.plot([], c=color, label=label)

    # add fliers separately with horizontal jitter
    for i, pos in enumerate(positions):
        iqr = np.percentile(data[i], 75) - np.percentile(data[i], 25)
        fliers = data[i][
            (data[i] > 1.5 * iqr + np.quantile(data[i], 0.75))
            | (data[i] < np.quantile(data[i], 0.25) - 1.5 * iqr)
        ]
        x = np.random.normal(pos, 0.08, size=len(fliers))
        ax.plot(
            x, fliers, marker=".", linestyle="", alpha=0.5, markersize=1, color=color
        )

    return ax


def set_box_color(bp, color):
    plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=color)
    plt.setp(bp["caps"], color=color)
    plt.setp(bp["medians"], color=color)


def figure_setup():
    """Set all the sizes to the correct values and use
    tex fonts for all texts.
    """
    params = {
        #"text.usetex": True,
        "figure.dpi": 400,
        "font.size": font_size(),
        #"font.serif": [],
        #"font.sans-serif": [],
        #"font.monospace": [],
        "axes.labelsize": label_size(),
        "axes.titlesize": font_size(),
        "axes.linewidth": axis_lw(),
        #'text.fontsize': font_size(),
        "legend.fontsize": font_size(),
        "legend.frameon": False,
        "xtick.labelsize": ticks_size(),
        "ytick.labelsize": ticks_size(),
        #"font.family": "serif",
        "lines.linewidth": plot_lw(),
    }
    plt.rcParams.update(params)
    return


def save_assortment(fig, file_name, tight=True, trim=False):
    """Save a Matplotlib figure as EPS/PNG/PDF to the given path and trim it."""

    save_fig(fig, file_name, fmt="png", tight=tight, trim=trim)
    save_fig(fig, file_name, fmt="pdf", tight=tight, trim=trim)
    # save_fig(fig, file_name, fmt='eps')
    # plt.style.use("dark_background")
    # save_fig(fig, file_name + "_dark", fmt="png", trim=False)
    # plt.style.use("default")


def save_fig(fig, file_name, fmt=None, dpi=600, tight=True, trim=True):
    """Save a Matplotlib figure as EPS/PNG/PDF to the given path and trim it."""

    if not fmt:
        fmt = file_name.strip().split(".")[-1]

    if fmt not in ["eps", "png", "pdf"]:
        raise ValueError("unsupported format: %s" % (fmt,))

    extension = ".%s" % (fmt,)
    if not file_name.endswith(extension):
        file_name += extension

    # trim it
    if trim:
        file_name = os.path.abspath(file_name)
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_name = tmp_file.name + extension

        # save figure
        if tight:
            fig.savefig(tmp_name, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(tmp_name, dpi=dpi)
        if fmt == "eps":
            subprocess.call(
                "epstool --bbox --copy %s %s" % (tmp_name, file_name), shell=True
            )
        elif fmt == "png":
            subprocess.call("convert %s -trim %s" % (tmp_name, file_name), shell=True)
        elif fmt == "pdf":
            subprocess.call("pdfcrop %s %s" % (tmp_name, file_name), shell=True)

    else:
        if tight:
            fig.savefig(file_name, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(file_name, dpi=dpi)


def assign_var(var):
    # standard values
    var = var.lower()

    norm = None
    cmap = "cividis"  # "viridis", 'plasma', 'inferno', 'magma', 'cividis','cubehelix'
    cmap_q = "Blues"
    flip = False

    #  if var starts with d_ it is a derivative
    derivative = False
    if var[0:2] == "d_":
        derivative = True
        var = var[2:]

    if var in ["the", "theta_e"]:
        label = r"$\Theta_e \; \mathrm{(K)}$"

    elif var in ["t"]:
        label = r"$T \; \mathrm{(^{\circ}C)}$"

    elif var in ["th", "theta"]:
        label = r"$\Theta \; \mathrm{(K)}$"

    elif var in ["head"]:
        label = r"$\mathit{arg} \; \mathrm{(rad)}$"
        cmap = "twilight"
        norm = mcolors.CenteredNorm()

    elif var in ["w"]:
        label = r"$w \; \mathrm{(m\,s^{-1})}$"
        cmap = "RdBu_r"
        norm = mcolors.CenteredNorm()

    elif var in ["w_rel"]:
        label = r"$\mathit{w_{rel}} \; \mathrm{(m\,s^{-1})}$"
        cmap = "RdBu_r"
        norm = mcolors.CenteredNorm()

    elif var in ["u"]:
        label = r"$u \; \mathrm{(m\,s^{-1})}$"
        cmap = "PiYG"
        norm = mcolors.CenteredNorm()

    elif var in ["v"]:
        label = r"$v \; \mathrm{(m\,s^{-1})}$"
        cmap = "PuOr"
        norm = mcolors.CenteredNorm()

    elif var in ["p"]:
        label = r"$p \; \mathrm{(hPa)}$"
        flip = True

    elif var in ["qr"]:
        label = r"$\mathit{q_{r}} \; \mathrm{(g\,kg^{-1})}$"
        cmap = "Blues"

    elif var in ["qi"]:
        label = r"$\mathit{q_{i}} \; \mathrm{(g\,kg^{-1})}$"
        cmap = "Purples"

    elif var in ["qv"]:
        label = r"$\mathit{q_{v}} \; \mathrm{(g\,kg^{-1})}$"
        cmap = "Greens"

    elif var in ["qg"]:
        label = r"$\mathit{q_{g}} \; \mathrm{(g\,kg^{-1})}$"
        cmap = "Purples"

    elif var in ["qg_ci", "tqg"]:
        label = r"$\int_{z}^{}\mathit{q_{g}} \; \mathrm{(g\,kg^{-1})}$"
        cmap = cmap_q

    elif var in ["tqi"]:
        label = r"$\int_{z}^{}\mathit{q_{i}} \; \mathrm{(g\,kg^{-1})}$"
        cmap = cmap_q

    elif var in ["tqr"]:
        label = r"$\int_{z}^{}\mathit{q_{r}} \; \mathrm{(g\,kg^{-1})}$"
        cmap = cmap_q

    elif var in ["tqs"]:
        label = r"$\int_{z}^{}\mathit{q_{s}} \; \mathrm{(g\,kg^{-1})}$"
        cmap = cmap_q

    elif var in ["tqc"]:
        label = r"$\int_{z}^{}\mathit{q_{c}} \; \mathrm{(g\,kg^{-1})}$"
        cmap = cmap_q

    elif var in ["qc"]:
        label = r"$\mathit{q_{c}} \; \mathrm{(g\,kg^{-1})}$"
        cmap = "Blues"

    elif var in ["qs"]:
        label = r"$\mathit{q_{s}} \; \mathrm{(g\,kg^{-1})}$"
        cmap = cmap_q

    elif var in ["qp"]:
        label = r"$q_{p} \; \mathrm{(g\,kg^{-1})}$"
        cmap = cmap_q

    elif var in ["qcc"]:
        label = r"$q_{c} \; \mathrm{(g\,kg^{-1})}$"
        cmap = cmap_q

    elif var in ["q_tot"]:
        label = r"$\mathit{q_{tot}} \; \mathrm{(g\,kg^{-1})}$"
        cmap = cmap_q

    elif var in ["cape_mu", "cape_ml", "cape_traj"]:
        # label = r"CAPE $\mathrm{(J\,kg^{-1})}$"
        label = r"CAPE $\text{J}\,\text{kg}^{-1}$"

    elif var in ["cin_ml", "cin_mu", "cin_traj"]:
        label = r"CIN $\mathrm{(J\,kg^{-1})}$"

    elif var == "z":
        label = r"$z \; \mathrm{(m)}$"

    elif var == "x":
        label = r"$x \; \mathrm{(km)}$"

    elif var == "y":
        label = r"$y \; \mathrm{(km)}$"

    elif var == "collimated":
        label = r"$\mathit{cf} \; \mathrm{(km)}$"

    elif var == "vort":
        label = r"$\zeta \; \mathrm{(s^{-1})}$"
        cmap = "PRGn"
        norm = mcolors.CenteredNorm()

    elif var == "div":
        label = r"$\nabla_H \cdot \mathbf{v} \; \mathrm{(s^{-1})}$"
        cmap = "BrBG"
        norm = mcolors.CenteredNorm()

    elif var == "pv":
        label = r"$\mathit{PV} \; \mathrm{(K\,m^2\,s^{-1})}$"
        cmap = "PiYG"
        norm = mcolors.CenteredNorm()

    elif var == "tot_prec":
        label = r"$R \; \mathrm{(mm\,h^{-1})}$"
        cmap = "Blues"

    elif var in ["dhail_mx", "dhail_av", "dhail"]:
        label = r"$\mathit{d_{hail}} \; \mathrm{(mm)}$"
        cmap = "plasma"

    elif var == "dhail_sd":
        label = r"$\sigma_{\mathit{d_{hail}}} \; \mathrm{(mm)}$"
        cmap = "Purples"

    elif var == "hsurf":
        label = r"$\mathit{h_{surf}} \; \mathrm{(m)}$"
        cmap = "Greys"  #'terrain'
        # norm = mcolors.Normalize(vmin=0, vmax=4000)

    elif var == "zagl":
        label = r"$\mathit{z_{agl}} \; \mathrm{(m)}$"
        cmap = "cividis"

    elif var == "hzerocl":
        label = r"$\mathit{h_{0^{\circ}}} \; \mathrm{(m)}$"
        cmap = "cividis"

    elif var in ["relhum", "rh"]:
        label = r"RH $\mathrm{(\%)}$"
        cmap = "Greens"
        norm = mcolors.Normalize(vmin=0, vmax=100)

    elif var == "dir":
        label = r"$\varphi \; \mathrm{(°)}$"
        cmap = "twilight"

    elif var == "spd":
        label = r"$\left| \mathbf{u} \right| \; \mathrm{(m\,s^{-1})}$"
        cmap = "OrRd"

    elif var == "dir_rel":
        label = r"$\varphi_{\mathit{rel}}$ ($\pi$)" r"$z \; \mathrm{(m)}$"
        cmap = "twilight"

    elif var == "spd_rel":
        label = r"$\left|\mathbf{u} \right|_{\mathit{rel}} \; \mathrm{(m\,s^{-1})}$"
        cmap = "OrRd"

    elif var == "shear_6km":
        label = r"$ \left| \frac{\delta\mathbf{u}}{\delta z} \right| ^{6\,\mathrm{km}}_{0\,\mathrm{km}} \; \mathrm{(m\,s^{-1})}$"

    elif var in ["shear_500hpa"]:
        label = r"$ \left| \frac{\delta\mathbf{u}}{\delta z} \right| ^{10\,\mathrm{m}}_{500\,\mathrm{hPa}} \; \mathrm{(m\,s^{-1})}$"

    elif var == "shear":
        label = r"$ \left| \frac{\delta\mathbf{u}}{\delta z} \right| \; \mathrm{(m\,s^{-1})}$"

    elif var == "srh_6km":
        label = r"$\mathit{SRH} \,_{0\,km}^{6\,km} \; \mathrm{(m^2 s^{-2})}$"

    elif var == "srh":
        label = r"$\mathit{SRH} \; \mathrm{(m^2 s^{-2})}$"

    elif var in ["dp", "td"]:
        label = r"$T_{\mathit{dp}} \; \mathrm{(°C)}$"

    elif var == "area_km":
        label = r"area $\mathrm{(km^2)}$"
        
    elif var == "wmaxshear_mu":
        label = r"$\text{m}**{2}\,\text{s}^{-2}$"

    else:
        print("variable not implemented yet: ", var)
        label = var

    if derivative:
        label = "$\Delta " + label[1:-3] + "\,s^{-1})}$"
        cmap = "PiYG"
        norm = mcolors.CenteredNorm()

    var_attributes = {"label": label, "norm": norm, "cmap": cmap, "flip": flip}
    # print(label)

    return var_attributes


def add_ch_border(ax, projection):
    shpfilename = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shpfilename)
    countries = reader.records()
    for country in countries:
        if country.attributes["ADM0_A3"] == "CHE":
            ax.add_geometries(
                [country.geometry],
                projection,
                edgecolor=(0, 0, 0, 1),
                linewidth=0.5,
                facecolor="none",
                label=country.attributes["ADM0_A3"],
            )
    return


def get_pv_colormap():
    pv_colors = [
        (79, 79, 238),
        (109, 134, 239),
        (147, 165, 234),
        (180, 198, 236),
        (210, 221, 229),
        (242, 215, 165),
        (232, 192, 136),
        (212, 161, 104),
        (227, 49, 34),
        (231, 127, 48),
        (239, 201, 68),
        (247, 247, 81),
        (145, 219, 77),
        (134, 182, 93),
    ]
    pv_colors = [tuple(v / 255 for v in c) for c in pv_colors]
    pv_borders = [-5, -2, 0, 0.2, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10]

    return pv_colors, pv_borders


class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    https://stackoverflow.com/questions/19353576/curved-text-rendering-in-matplotlib
    """

    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0], y[0], " ", **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == " ":
                ##make this an invisible 'a':
                t = mtext.Text(0, 0, "a")
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0, 0, c, **kwargs)

            # resetting unnecessary arguments
            t.set_ha("center")
            t.set_rotation(0)
            t.set_zorder(self.__zorder + 1)

            self.__Characters.append((c, t))
            axes.add_artist(t)

    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c, t in self.__Characters:
            t.set_zorder(self.__zorder + 1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self, renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        # preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w) / (figH * h)) * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])

        # points of the curve in figure coordinates:
        x_fig, y_fig = (
            np.array(l)
            for l in zip(
                *self.axes.transData.transform(
                    [(i, j) for i, j in zip(self.__x, self.__y)]
                )
            )
        )

        # point distances in figure coordinates
        x_fig_dist = x_fig[1:] - x_fig[:-1]
        y_fig_dist = y_fig[1:] - y_fig[:-1]
        r_fig_dist = np.sqrt(x_fig_dist**2 + y_fig_dist**2)

        # arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist), 0, 0)

        # angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]), (x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)

        rel_pos = 10
        for c, t in self.__Characters:
            # finding the width of c:
            t.set_rotation(0)
            t.set_va("center")
            bbox1 = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            # ignore all letters that don't fit:
            if rel_pos + w / 2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != " ":
                t.set_alpha(1.0)

            # finding the two data points between which the horizontal
            # center point of the character will be situated
            # left and right indices:
            il = np.where(rel_pos + w / 2 >= l_fig)[0][-1]
            ir = np.where(rel_pos + w / 2 <= l_fig)[0][0]

            # if we exactly hit a data point:
            if ir == il:
                ir += 1

            # how much of the letter width was needed to find il:
            used = l_fig[il] - rel_pos
            rel_pos = l_fig[il]

            # relative distance between il and ir where the center
            # of the character will be
            fraction = (w / 2 - used) / r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il] + fraction * (self.__x[ir] - self.__x[il])
            y = self.__y[il] + fraction * (self.__y[ir] - self.__y[il])

            # getting the offset when setting correct vertical alignment
            # in data coordinates
            t.set_va(self.get_va())
            bbox2 = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0] - bbox1d[0])

            # the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array(
                [
                    [math.cos(rad), math.sin(rad) * aspect],
                    [-math.sin(rad) / aspect, math.cos(rad)],
                ]
            )

            ##computing the offset vector of the rotated character
            drp = np.dot(dr, rot_mat)

            # setting final position and rotation:
            t.set_position(np.array([x, y]) + drp)
            t.set_rotation(degs[il])

            t.set_va("center")
            t.set_ha("center")

            # updating rel_pos to right edge of character
            rel_pos += w - used