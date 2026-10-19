#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import interpolate
from scipy.signal import fftconvolve
import matplotlib
import matplotlib.pyplot as plt
#!pip install svgpath2mpl matplotlib
from svgpath2mpl import parse_path
from astropy.convolution import Gaussian1DKernel, Box1DKernel, convolve
from tqdm import tqdm
from numba import njit


@njit(cache=True)
def _convol_gauss_kernel(x, y, sigma, half_width):
    """Discrete Gaussian convolution on an irregular but sorted wavelength grid."""
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        mu = x[i]
        # Match command_cut strict inequalities: lo < x < hi
        lo = np.searchsorted(x, mu - half_width, side="right")
        hi = np.searchsorted(x, mu + half_width, side="left")
        s = 0.0
        wsum = 0.0
        for j in range(lo, hi):
            u = (x[j] - mu) / sigma
            w = np.exp(-0.5 * u * u)
            s += y[j] * w
            wsum += w
        out[i] = s / wsum
    return out


@njit(cache=True)
def _convol_box_kernel(x, y, width):
    """Discrete box convolution on an irregular but sorted wavelength grid."""
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        mu = x[i]
        lo = np.searchsorted(x, mu - width, side="right")
        hi = np.searchsorted(x, mu + width, side="left")
        s = 0.0
        for j in range(lo, hi):
            s += y[j]
        out[i] = s / (hi - lo)
    return out


@njit(cache=True)
def _convol_rot_kernel(x, y, vsini_kms, beta):
    """
    Discrete rotational-broadening convolution on an irregular but sorted grid.

    Kernel width scales with wavelength (Δλ = vsini/c * λ), matching command_convol.
    """
    n = len(x)
    out = np.empty(n)
    vsini = vsini_kms / 299792.458
    for i in range(n):
        mu = x[i]
        half = vsini * mu
        lo = np.searchsorted(x, mu - half, side="right")
        hi = np.searchsorted(x, mu + half, side="left")
        c1 = 2.0 * (1.0 - beta) / (np.pi * half * (1.0 - beta / 3.0))
        c2 = beta / (2.0 * half * (1.0 - beta / 3.0))
        s = 0.0
        wsum = 0.0
        for j in range(lo, hi):
            z2 = ((mu - x[j]) / half) ** 2
            w = c1 * np.sqrt(1.0 - z2) + c2 * (1.0 - z2)
            s += y[j] * w
            wsum += w
        out[i] = s / wsum
    return out


# ---------------------------------------------------------------------------
# HLYMANA helpers (numba) for command_hlymana_fast_v2
# ---------------------------------------------------------------------------

@njit(cache=True)
def _voigth_njit(a, v):
    """Normalized Voigt H(a,v); Detlef Köster implementation (scalar)."""
    wpi = 1.77245385
    v = abs(v)
    vv = v * v

    if a < 0.2 and v > 5.0:
        h = ((2.12 / vv + 0.8463) / vv + 0.5642) * a / vv
        return h / wpi
    if a >= 0.2 and ((a > 1.4) or ((a + v) > 3.2)):
        aa = a * a
        u = (aa + vv) - 1.4142
        uu = u * u
        h = ((((aa - 10.0 * vv) * aa * 3.0 + 15.0 * vv * vv) / uu + 3.0 * vv - aa) / uu + 1.0) * a * 0.79788 / u
        return h / wpi
    h0 = np.exp(-vv)
    h2 = (1.0 - 2.0 * vv) * h0
    if v > 2.4:
        h1 = ((-0.0032783 * vv + 0.0429913 * v - 0.188326) * vv + 0.278712 * v + 0.55415) / (vv - 1.5)
        if a >= 0.2:
            hh1 = h1 + h0 * 1.12838
            hh2 = h2 + hh1 * 1.12838 - h0
            hh3 = (1.0 - h2) * 0.37613 - hh1 * 0.66667 * vv + hh2 * 1.12838
            hh4 = (3.0 * hh3 - hh1) * 0.37613 + h0 * 0.66667 * vv * vv
            h = ((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + h0) * (((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032)
            return h / wpi
        h = ((h2 * a + h1) * a + h0)
        return h / wpi
    if v <= 1.3:
        h1 = (0.42139 * vv - 2.34358 * v + 3.28868) * vv - 0.15517 * v - 1.1247
        if a >= 0.2:
            hh1 = h1 + h0 * 1.12838
            hh2 = h2 + hh1 * 1.12838 - h0
            hh3 = (1.0 - h2) * 0.37613 - hh1 * 0.66667 * vv + hh2 * 1.12838
            hh4 = (3.0 * hh3 - hh1) * 0.37613 + h0 * 0.66667 * vv * vv
            h = ((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + h0) * (((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032)
            return h / wpi
        h = ((h2 * a + h1) * a + h0)
        return h / wpi
    h1 = (-0.220416 * vv + 1.989196 * v - 6.61487) * vv + 9.39456 * v - 4.4848
    if a >= 0.2:
        hh1 = h1 + h0 * 1.12838
        hh2 = h2 + hh1 * 1.12838 - h0
        hh3 = (1.0 - h2) * 0.37613 - hh1 * 0.66667 * vv + hh2 * 1.12838
        hh4 = (3.0 * hh3 - hh1) * 0.37613 + h0 * 0.66667 * vv * vv
        h = ((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + h0) * (((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032)
        return h / wpi
    h = ((h2 * a + h1) * a + h0)
    return h / wpi


@njit(cache=True)
def _hlymana_tau_lorentz(lam, xlam, c0, c1, enh):
    """Optical depth for Lorentzian Lyman series (all lines, all wavelengths)."""
    n = len(lam)
    nlines = len(xlam)
    tau = np.zeros(n)
    for j in range(nlines):
        xj = xlam[j]
        c0j = c0[j]
        c1j = c1[j]
        for i in range(n):
            dlam = lam[i] - xj
            tau[i] += c0j / (1.0 + c1j * dlam * dlam)
    for i in range(n):
        tau[i] *= enh
    return tau


@njit(cache=True)
def _hlymana_tau_lorentz_points(lam_pts, xlam, c0, c1, enh):
    """Optical depth at a short list of wavelengths only."""
    n = len(lam_pts)
    nlines = len(xlam)
    tau = np.zeros(n)
    for i in range(n):
        t = 0.0
        x = lam_pts[i]
        for j in range(nlines):
            dlam = x - xlam[j]
            t += c0[j] / (1.0 + c1[j] * dlam * dlam)
        tau[i] = enh * t
    return tau


@njit(cache=True)
def _hlymana_tau_voigt(lam, xlam, a_arr, fak_arr, dndopp, enh, clight):
    n = len(lam)
    nlines = len(xlam)
    tau = np.zeros(n)
    for j in range(nlines):
        xj = xlam[j]
        aj = a_arr[j]
        fakj = fak_arr[j]
        dn = dndopp[j]
        xj2 = xj * xj
        for i in range(n):
            dlam = lam[i] - xj
            dnue = clight * 1.0e13 * dlam / xj2
            v = dnue / dn
            tau[i] += fakj * _voigth_njit(aj, v) * enh
    return tau


@njit(cache=True)
def _hlymana_tau_voigt_points(lam_pts, xlam, a_arr, fak_arr, dndopp, enh, clight):
    n = len(lam_pts)
    nlines = len(xlam)
    tau = np.zeros(n)
    for i in range(n):
        t = 0.0
        x = lam_pts[i]
        for j in range(nlines):
            dlam = x - xlam[j]
            dnue = clight * 1.0e13 * dlam / (xlam[j] * xlam[j])
            v = dnue / dndopp[j]
            t += fak_arr[j] * _voigth_njit(a_arr[j], v) * enh
        tau[i] = t
    return tau


@njit(cache=True)
def _hlymana_adaptive_refine(lam, flux, tau, xlam, c0, c1, a_arr, fak_arr, dndopp,
                             enh, clight, use_voigt, taucrit, max_iter):
    """
    Adaptive midpoint insertion where |Δτ/τ| > taucrit.
    Reuses existing τ and only evaluates newly inserted midpoints.
    Returns (lam, flux, tau, n_iters).
    """
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        n = len(lam)
        # count how many segments need a midpoint
        n_mid = 0
        for i in range(n - 1):
            tL = tau[i]
            denom = 1.0e-40 if tL == 0.0 else tL
            rel = abs((tau[i + 1] - tL) / denom)
            gap = lam[i + 1] - lam[i]
            if rel > taucrit and gap > 0.001:
                n_mid += 1

        if n_mid == 0:
            break

        midpoints = np.empty(n_mid)
        mid_left = np.empty(n_mid, dtype=np.int64)
        k = 0
        for i in range(n - 1):
            tL = tau[i]
            denom = 1.0e-40 if tL == 0.0 else tL
            rel = abs((tau[i + 1] - tL) / denom)
            gap = lam[i + 1] - lam[i]
            if rel > taucrit and gap > 0.001:
                midpoints[k] = 0.5 * (lam[i] + lam[i + 1])
                mid_left[k] = i
                k += 1

        if use_voigt:
            tau_mid = _hlymana_tau_voigt_points(
                midpoints, xlam, a_arr, fak_arr, dndopp, enh, clight
            )
        else:
            tau_mid = _hlymana_tau_lorentz_points(midpoints, xlam, c0, c1, enh)

        # Merge old grid + midpoints (already sorted by construction)
        n_out = n + n_mid
        lam_out = np.empty(n_out)
        flux_out = np.empty(n_out)
        tau_out = np.empty(n_out)
        j = 0
        mi = 0
        for i in range(n - 1):
            lam_out[j] = lam[i]
            flux_out[j] = flux[i]
            tau_out[j] = tau[i]
            j += 1
            if mi < n_mid and mid_left[mi] == i:
                lam_out[j] = midpoints[mi]
                flux_out[j] = 0.5 * (flux[i] + flux[i + 1])
                tau_out[j] = tau_mid[mi]
                j += 1
                mi += 1
        lam_out[j] = lam[n - 1]
        flux_out[j] = flux[n - 1]
        tau_out[j] = tau[n - 1]

        lam = lam_out
        flux = flux_out
        tau = tau_out

    return lam, flux, tau, iter_count


class wrplotter:
        
    def __init__(self):
        self.grid = None
        #add default colors of WRPLOT
        self.colors = ['white',       #0
                       'black',       #1
                       'red',         #2
                       'limegreen',   #3
                       'blue',        #4    
                       'yellow',      #5
                       'fuchsia',     #6
                       'cyan',        #7
                       'greenyellow', #8
                       'dodgerblue'   #9
                      ]

        #add symbols of WR
        bomb_symbol = parse_path("""m 256.762,183.29025 c -6.522,6.522 -17.087,6.522 -23.609,0 -14.733,-14.733 -34.294,-22.847 -55.066,-22.847 -20.772,0 -40.331,8.114 -55.066,22.847 -6.521,6.521 -17.087,6.521 -23.609,0 -6.521,-6.516 -6.521,-17.092 0,-23.609 21.038,-21.043 48.978,-32.63 78.674,-32.63 29.696,0 57.636,11.587 78.674,32.63 6.523,6.517 6.523,17.093 0.002,23.609 z m 250.34799,239.392 c -3.261,3.261 -7.533,4.892 -11.805,4.892 -4.272,0 -8.544,-1.631 -11.804,-4.892 l -11.13,-11.13 c -6.521,-6.521 -6.521,-17.092 0,-23.609 6.521,-6.521 17.097,-6.521 23.609,0 l 11.13,11.13 c 6.521,6.521 6.521,17.092 0,23.609 z m -66.784,-66.783 c -6.521,6.522 -17.097,6.522 -23.608,0 l -11.12999,-11.13 c -6.521,-6.521 -6.521,-17.092 0,-23.609 3.261,-3.261 7.53299,-4.892 11.80399,-4.892 4.271,0 8.544,1.631 11.804,4.892 l 11.13,11.13 c 6.521,6.522 6.521,17.092 0,23.609 z m 66.781,-11.13 -11.129,11.13 c -6.511,6.521 -17.087,6.521 -23.609,0 -6.521,-6.516 -6.521,-17.087 0,-23.609 l 11.13,-11.13 c 3.261,-3.261 7.533,-4.892 11.804,-4.892 4.271,0 8.544,1.631 11.804,4.892 6.522,6.516 6.522,17.087 0,23.609 z m -65.449,62.717 -23.609,23.609 c -19.52699,19.538 -51.31599,19.527 -70.83199,0 l -7.869,-7.869 -23.614,23.614 c -6.521,6.521 -17.087,6.521 -23.609,0 l -44.575,-44.575 c -21.699,9.182 -45.244,14.01 -69.55,14.01 -47.625,0 -92.343,-18.494 -125.924,-52.076 -69.435,-69.435 -69.435,-182.419 0,-251.853 33.582,-33.582 78.299,-52.076 125.924,-52.076 47.625,0 92.348,18.494 125.93,52.076 33.582,33.582 52.076,78.299 52.076,125.924 0,24.306 -4.828,47.851 -14.01,69.551 l 44.575,44.575 c 6.521,6.516 6.521,17.092 0,23.609 l -23.614,23.614 7.869,7.869 c 6.506,6.506 17.097,6.516 23.614,0 l 23.60899,-23.609 c 3.261,-3.261 7.533,-4.892 11.804,-4.892 4.271,0 8.544,1.631 11.804,4.892 6.522,6.515 6.522,17.092 10e-4,23.607 z M 280.321,135.95425 c -27.277,-27.272 -63.614,-42.293 -102.321,-42.293 -38.707,0 -75.038,15.022 -102.315,42.293 -56.414,56.418 -56.414,148.217 0,204.636 27.277,27.272 63.608,42.293 102.315,42.293 22.581,0 44.351,-5.121 63.991,-14.796 0.474,-0.305 0.957,-0.599 1.465,-0.856 11.984,-6.082 22.928,-13.665 32.612,-22.57 0.13,-0.12 0.259,-0.242 0.388,-0.362 1.31,-1.213 2.6,-2.448 3.863,-3.71 1.262,-1.263 2.495,-2.553 3.71,-3.863 0.12,-0.129 0.242,-0.259 0.362,-0.39 8.90499,-9.683 16.489,-20.628 22.57,-32.612 0.257,-0.509 0.551,-0.991 0.856,-1.465 9.675,-19.64 14.796,-41.411 14.796,-63.991 0.002,-38.706 -15.02,-75.037 -42.292,-102.314 z m 45.008,202.416 c -5.468,8.023 -11.62,15.652 -18.444,22.8 -0.973,1.021 -1.958,2.031 -2.956,3.029 -0.998,0.997 -2.008,1.982 -3.029,2.956 -7.148,6.823 -14.777,12.976 -22.8,18.444 l 25.828,25.828 47.22899,-47.229 z""")
        bomb_symbol.vertices -= bomb_symbol.vertices.mean(axis=0)
        self.symbols = [['',''],                                                             #0  nothing
                        ['','+'],                                                            #1  cross
                        ['',r'$*$'],                                                         #2  asterisk
                        ['','^'],                                                            #3  triangle
                        ['','s'],                                                            #4  square
                        ['-',''],                                                            #5  polygon
                        ['-',''],                                                            #6  cubic spline
                        ['--',''],                                                           #7  cubic spline, dashed
                        ['','o'],                                                            #8  circle
                        ['--',''],                                                           #9  polygon, dashed
                        ['-.',''],                                                           #10 polygon, dash-dotted
                        ['','v'],                                                            #11 triangle on its tip
                        ['','D'],                                                            #12 square on its corner
                        ['',bomb_symbol],                                                    #13 bomb (handle with care!)
                        ['','1'],                                                            #14 star with three spikes (Mercedes)
                        ['',matplotlib.path.Path([[-0.5,0],[0,1],[0.5,0],[0,-1]])],          #15 rhomb 
                        ['','|'],                                                            #16 vertical tick
                        ['',matplotlib.path.Path([[-1,0],[0,0.5],[1,0],[0,-0.5]])],          #17 horizontal rhomb
                        ['','*'],                                                            #18 star with five spikes,                                                            #20 polygon, dotted
                        ['',''],
                        [':',''],                                                            #20 polygon, dotted
                        ['-','+'],                                                           #21 polygon with crosses
                        ['-',r'$*$'],                                                        #22 polygon with asterisk
                        ['-','^'],                                                           #23 polygon with triangles
                        ['-','s'],                                                           #24 polygon with squares
                        ['-','v'],                                                           #25 polygon with triangles on their tips
                        ['','P'],                                                            #26 plus, filled
                        ['','X'],                                                            #27 St. Andrew's Cross, filled
                        ['-','o'],                                                           #28 polygon with circles
                        ['','p'],                                                            #29 pentagon
                        ['',(7,1,0)],                                                        #30 star with seven spikes
                        ['-','v'],                                                           #31 polygon with upside-down triangles (same as 25)
                        ['-','D'],                                                           #32 polygon with squares on their tips
                        ['',(14,1,0)],                                                       #33 star with 14 spikes
                        ['','']]
        
    ######################################################################################################################
    #
    #   
    #                                 PLOTTING PART OF WRPLOT
    #
    #
    ######################################################################################################################
        
    def figure(self, figsize=None, gridsize=None, fontsize=14, **kwargs):
        """
        Create a new figure, or activate an existing figure based on matplotlib.pyplot
        in the WRPLOT style. Adds an extra parameter defining a grid to easily create ago
        subplots.
        
        Parameters
        ----------
        num : int or str or `.Figure`, optional
            A unique identifier for the figure.
            If a figure with that identifier already exists, this figure is made
            active and returned. An integer refers to the ``Figure.number``
            attribute, a string refers to the figure label.
            If there is no figure with the identifier or *num* is not given, a new
            figure is created, made active and returned.  If *num* is an int, it
            will be used for the ``Figure.number`` attribute, otherwise, an
            auto-generated integer value is used (starting at 1 and incremented
            for each new figure). If *num* is a string, the figure label and the
            window title is set to this value.
        figsize : (float, float), default: :rc:`figure.figsize`
            Width, height in inches.
            A2 = 16.54, 23.39
            A3 = 11.69, 16.54
            A4 =  8.27, 11.69
        gridsize : (int, int), default (1,1)
            ncols, nrows
        dpi : float, default: :rc:`figure.dpi`
            The resolution of the figure in dots-per-inch.
        facecolor : color, default: :rc:`figure.facecolor`
            The background color.
        edgecolor : color, default: :rc:`figure.edgecolor`
            The border color.
        frameon : bool, default: True
            If False, suppress drawing the figure frame.
        FigureClass : subclass of `~matplotlib.figure.Figure`
            Optionally use a custom `.Figure` instance.
        clear : bool, default: False
            If True and the figure already exists, then it is cleared.
        tight_layout : bool or dict, default: :rc:`figure.autolayout`
            If ``False`` use *subplotpars*. If ``True`` adjust subplot
            parameters using `.tight_layout` with default padding.
            When providing a dict containing the keys ``pad``, ``w_pad``,
            ``h_pad``, and ``rect``, the default `.tight_layout` paddings
            will be overridden.
        constrained_layout : bool, default: :rc:`figure.constrained_layout.use`
            If ``True`` use constrained layout to adjust positioning of plot
            elements.  Like ``tight_layout``, but designed to be more
            flexible.  See
            :doc:`/tutorials/intermediate/constrainedlayout_guide`
            for examples.  (Note: does not work with `add_subplot` or
            `~.pyplot.subplot2grid`.)
        **kwargs : optional
            See `~.matplotlib.figure.Figure` for other possible arguments.
            
        Returns
        -------
        `~matplotlib.figure.Figure`
            The `.Figure` instance returned will also be passed to
            new_figure_manager in the backends, which allows to hook custom
            `.Figure` classes into the pyplot interface. Additional kwargs will be
            passed to the `.Figure` init function.
            
        Notes
        -----
        If you are creating many figures, make sure you explicitly call
        `.pyplot.close` on the figures you are not using, because this will
        enable pyplot to properly clean up the memory.
        `~matplotlib.rcParams` defines the default values, which can be modified
        in the matplotlibrc file.
        """
        fig = plt.figure(figsize=figsize,**kwargs)
        plt.rc('font', family='Helvetica')
        plt.rcParams["font.family"] = "sans-serif"#'serif'
        plt.rcParams["font.sans-serif"] = "Helvetica"
        plt.rcParams["font.size"]=fontsize
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = r'''
                                                    %\usepackage{mathtools}
                                                    \usepackage{amsmath,amssymb}
                                                    \usepackage{txfonts}
                                                    \usepackage{graphicx}
                                                    \renewcommand{\familydefault}{\rmdefault}
                                                    \usepackage{helvet}
                                                    %\renewcommand{\familydefault}{\sfdefault}
                                                    \fontfamily{phv}\selectfont
                                                    \usepackage{mathastext}
                                                    %\usepackage[frenchmath]{newtxsf} % for sans serif symbols
                                                    %\renewcommand{\familydefault}{\sfdefault}
                                                    % more packages here
                                                '''
        if gridsize==None:
            gridsize=(1,1)
        self.grid = fig.add_gridspec(gridsize[0],gridsize[1])
        return fig
    
    def get_grid(self):
        """    
        The grid of the figure and prints it, e.g. in case one forgot what the grid was.
        
        Parameters
        ----------
            
        Returns
        -------
        The grid of the figure.
        
        Notes
        -----
        """
        print("Info: the grid has shape: "+str(self.grid))
        return self.grid
    
    def get_subplot(self,xrange=None,yrange=None):
        """
        Returns/selects the subplot of interest. 
        
        Parameters
        ----------
        xrange : (int,int) Interval on the x-axis of the grid inbetween which the plot should be. 0 is left, n (number of grid points) is right.
            It is also possible to use -1 for n-1 as usual in python.
        yrange : (int,int) Interval on the y-axis of the grid inbetween which the plot should be. 0 is top, n (number of grid points) is bottom.
            It is also possible to use -1 for n-1 as usual in python.
            
        Returns
        -------
        The desired subplot, if possible. If no subplot is selected, it is assumed that the figure only contains 1 plot.
        
        Notes
        -----
        This function is not well tested and has to be handeled with care.
        """
        if xrange == None or yrange == None:
            return plt.subplot(111)
        #print(xrange.shape())
        if xrange[0] == xrange[1] or yrange[0] == yrange[1]:
            print("Error: one range needs distinct values! Please check.")
        return plt.subplot(self.grid[yrange[0]:yrange[1], xrange[0]:xrange[1]])
    
    def plot(self,x ,y , XRANGE=(None,None), YRANGE=(None,None), XTICKS=(None,None), YTICKS=(None,None), DARUNTER=(0.,0.), 
             LABELS=("",""), ALT=(False,False), NOLAB=(False,False), NOTICKS=(False,False), SYMBOL=1 ,
             SIZE=0.3 ,HATCHED=None, PEN=1 ,COLOR=1, spline_res=1e-3, **kwargs):
        """
        Creates a plot in the WRPLOT style, many commands are similar or inspired by WRPLOT commands.
        
        Parameters
        ----------
        x : int or array containing the data that should be plotted on the x-axis. It should have the same
            lenth as the y data.            
        y : int or array containing the data that should be plotted on the y-axis. It should have the same
            lenth as the x data.
        XRANGE : (float, float), default: (None,None)
            min x-value, max x-value, if None maplotlib will automatically decide
        YRANGE : (float, float), default: (None,None)
            min y-value, max y-value, if None maplotlib will automatically decide            
        XTICKS : (float, float), default: (None,None)
            major x-ticks (Note: in WRPLOT it is called LABEL), minor x-tick (Note: in WRPLOT it is called TICK). 
            if None maplotlib will automatically decide (be warned, it might look ugly!)
        YTICKS : (float, float), default: (None,None)
            major y-ticks (Note: in WRPLOT it is called LABEL), minor y-tick (Note: in WRPLOT it is called TICK). 
            If None maplotlib will automatically decide (be warned, it might look ugly!).
        LABELS=("","") : (str, str), default: ("","")
            xlabel, ylabel
        ALT : (boolean, boolean), default: (False,False)
            Place the labels and numbers of that axis at the alternate side (top or right, respectively) of the box.
            x-axis, y-axis
        NOLAB : (boolean, boolean), default: (False,False)
            Suppresses the labels on the respective axis. (Note, in WRPLOT major ticks are called LABELS!)
            x-axis, y-axis
        SYMBOL : int or (str,str), default: 1 (cross)
            If SYMBOL == int: Use the notation of WPLOT, e.g. 1 is a cross, 5 is a polygon and 21 is a polygon with crosses
                (see WRPLOT manual for list)
            IF SYMBOL == (str,str): Use custom matplotlib lines and markers. (linestyle, marker)
        SIZE : float, default: 0.3
             Set the marker size. If the size is negative, the area enclosed by the curve is filled (for symbols which draw
             a line), or the symbols are filled (in case of fillable symbols, see WRPLOT manual Appendix C).
        HATCHED : str, default: None
            Acts like negative size, but applies a hatched filling of areas. The WRPLOT options are diagonal "D" and finer 
            diagonal "d", vertical "V" and finer vertical "v", horizontal "H" and finer horizontal "h", checkerboard "K" and
            finer checkerboard "k", rotated checkerboard "C" and finer rotated checkerboard "c". Alternatively one can use
            all matplotlib hatches.
        PEN : float default: 1
            Set the line width.
        COLOR : int or str or (float,float,float) or (float,float,float,float), default: 1 (black)
            If SYMBOL == int: Use the notation of WPLOT, e.g. 0 is white and 1 is black
                (see WRPLOT manual for list)
            IF SYMBOL == str: Use the predefined colors from matplotlib. Or hex colors (staring with #)
            IF SYMBOL == (float,float,float): RGB colors between 0 and 1
            IF SYMBOL == (float,float,float,float): RGBA colors between 0 and 1
        spline_res : float, default: 1e-3
            When using a line with type cubic spline this value sets the resolution of the splines. 
        **kwargs : optional
            See `~.matplotlib.pyplot.plot` for other possible arguments.
            
        Returns
        -------
            None
        
        Notes
        -----
        This function is not well tested and has to be handeled with care.
        """
        # some standard definitions
        markeredgecolor = "black"
        markersize=14*abs(SIZE)
        markeredgewidth=abs(SIZE)
        linewidth=0.6*PEN
        
        XMIN = XRANGE[0]
        XMAX = XRANGE[1]
        YMIN = YRANGE[0]
        YMAX = YRANGE[1]
        XMAJOR = XTICKS[0]
        XMINOR = XTICKS[1]
        YMAJOR = YTICKS[0]
        YMINOR = YTICKS[1]
        
        # check if color format is okay
        if isinstance(COLOR, int):
            color = self.colors[COLOR]
        elif isinstance(COLOR, str) or (isinstance(COLOR, tuple) and 4 >= len(COLOR) >=3):
            color = COLOR
        else:
            print("ERROR: Color not in correct format.")
            print("Possible formats are:")
            print("int: 0-9")
            print("str: matplotlib color, or in hex (#000000 or #00000000)")
            print("tuple: 3 (RGB) or 4 (RGBA) values between 0-1")
            return
        
        # check if symbol format is okay
        if isinstance(SYMBOL, int):
            linestyle, symbol = self.symbols[SYMBOL]
            #if cubic spline
            if SYMBOL == 6 or SYMBOL == 7:
                tck, u = interpolate.splprep([x,y], s=0)
                unew = np.arange(0, 1.+spline_res, spline_res)
                out = interpolate.splev(unew, tck)
                x = out[0]
                y = out[1]
            # if + or * or |
            if SYMBOL == 1 or SYMBOL ==2 or SYMBOL == 14 or SYMBOL == 16 or SYMBOL == 21 or SYMBOL == 22:
                markeredgecolor=color
            # if BOMB
            if SYMBOL == 13:
                markersize=14*abs(SIZE)/0.4
                markeredgewidth=0.1*abs(SIZE)/0.4
            # if Mercedes or |
            if SYMBOL == 14 or SYMBOL == 16:
                markeredgewidth=3*abs(SIZE)
            # if rhombus
            if SYMBOL == 15 or SYMBOL == 17:
                markersize=14*abs(SIZE)/0.7
            # if line, make marker smaller
            if linestyle != '':
                markersize=markersize*0.5
        elif isinstance(SYMBOL, tuple) and len(SYMBOL) == 2:
            linestyle, symbol = SYMBOL
        else:
            print("ERROR: SYMBOL not in correct format.")
            print("Possible formats are:")
            print("int: 0-33")
            print("tuple(str, str): linestyle and marker")
                
        # check if user wants to change markersize by hand
        if "markersize" in kwargs or "ms" in kwargs:
            try:
                markersize=kwargs.pop("markersize")
            except:
                markersize=kwargs.pop("ms")
            
        # check if user wants to change markeredgecolor by hand
        if "markeredgecolor" in kwargs or "mec" in kwargs:
            try:
                markeredgecolor=kwargs.pop("markeredgecolor")
            except:
                markeredgecolor=kwargs.pop("mec")
                
        # check if user wants to change markeredgewidth by hand
        if "markeredgewidth" in kwargs or "mew" in kwargs:
            try:
                markeredgewidth=kwargs.pop("markeredgewidth")
            except:
                markeredgewidth=kwargs.pop("mew")
                
        # check if user wants to change linewidth by hand
        if "linewidth" in kwargs or "lw" in kwargs:
            try:
                linewidth=kwargs.pop("linewidth")
            except:
                linewidth=kwargs.pop("lw")
                
            
        # normal plotting (if SIZE is smaller than 1 the curve will be filled)
        if HATCHED == None and SIZE > 0:
            if isinstance(SYMBOL, int) and (SYMBOL == 3 or SYMBOL == 4 or 15 >= SYMBOL >= 11 or SYMBOL == 17 or SYMBOL == 18 or SYMBOL == 26 or SYMBOL == 27):
                markerfacecolor = "white"
            else: 
                markerfacecolor = color
            if isinstance(SYMBOL, int) and SYMBOL == 13:
                markerfacecolor = "black"

            # check if user wants to change markerfacecolor by hand
            if "markerfacecolor" in kwargs or "mfc" in kwargs:
                try:
                    markerfacecolor=kwargs.pop("markerfacecolor")
                except:
                    markerfacecolor=kwargs.pop("mfc")
                                    
            plt.plot(x,y,ls=linestyle,marker=symbol,color=color,markeredgecolor=markeredgecolor,
                     markersize=markersize,markeredgewidth=markeredgewidth,linewidth=linewidth,
                     markerfacecolor=markerfacecolor,**kwargs)
        # if SIZE < 0 the area enclosed by the curve is filled (for symbols which draw a line)
        # of the symbols are filled (in case of fillable symbols)
        elif HATCHED == None and SIZE < 0:
            alpha = 1
            # check if user wants to change alpha by hand (NOTE alpha only affects filled area)
            if "alpha" in kwargs:
                alpha=kwargs.pop("alpha")
                     
            y2 = 0
            # check if user wants to change the offset by hand
            if "y2" in kwargs:
                y2=kwargs.pop("y2")
            plt.plot(x,y,ls=linestyle,marker=symbol,color=color,markeredgecolor=markeredgecolor,
                     markersize=markersize,markeredgewidth=markeredgewidth,linewidth=linewidth,**kwargs)
            if linestyle != '':
                plt.fill_between(x,y,y2=y2,ls=linestyle,color=color,linewidth=linewidth,alpha=alpha,**kwargs)
        # if hatched is used it the area enclosed by the curve will be filled with the hatches
        elif isinstance(HATCHED, str):            
            alpha = 1
            # check if user wants to change alpha by hand (NOTE alpha only affects filled area)
            if "alpha" in kwargs:
                alpha=kwargs.pop("alpha")
                                     
            y2 = 0
            # check if user wants to change the offset by hand
            if "y2" in kwargs:
                y2=kwargs.pop("y2")
                
            plt.plot(x,y,ls=linestyle,marker=symbol,color=color,markeredgecolor=markeredgecolor,
                     markersize=markersize,markeredgewidth=markeredgewidth,linewidth=linewidth,**kwargs)
            
            # is it a WRPLOT hatch?
            hatch = HATCHED
            if HATCHED == "D":
                hatch = '//'
            elif HATCHED == "d":
                hatch = '///'
            elif HATCHED == "V":
                hatch = '||'
            elif HATCHED == "v":
                hatch = '|||'
            elif HATCHED == "H":
                hatch = '--'
            elif HATCHED == "h":
                hatch = '---'
            elif HATCHED == "K":
                hatch = '++'
            elif HATCHED == "k":
                hatch = '+++'
            elif HATCHED == "C":
                hatch = 'xx'
            elif HATCHED == "c":
                hatch = 'xxx'
            
            if linestyle != '':     
                facecolor = "white"
                # check if user wants to change alpha by hand (NOTE alpha only affects filled area)
                if "facecolor" in kwargs or "fc" in kwargs:
                    try:
                        facecolor=kwargs.pop("facecolor")
                    except:
                        facecolor=kwargs.pop("fc")
                        
                plt.fill_between(x,y,ls=linestyle,color=color,linewidth=linewidth,alpha=alpha,facecolor=facecolor, hatch=hatch,**kwargs)
        else:
            print("Error: Don't know what to plot.")
            print("It might be a problem with the hatches.")
        
        tmp = plt.gca()
        # some labeling stuff to make it look like WRPLOT
        if ALT[0] == False:
            plt.xlabel(LABELS[0])
        if ALT[0] == True or NOLAB[0] == True:
            plt.gca().xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        if ALT[1] == False:
            plt.ylabel(LABELS[1])
        if ALT[1] == True or NOLAB[1] == True:
            plt.gca().yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
            
        plt.axis(xmin = XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
        plt.tick_params(direction='in',which='both')
        plt.tick_params(which='major', length=6)
        plt.tick_params(which='minor', length=3)
        if XMAJOR != None:
            if DARUNTER[0] <= XMIN:
                custom_ticks = np.arange(DARUNTER[0],XMAX,XMAJOR)
                custom_ticks = custom_ticks[custom_ticks>=XMIN]
                plt.gca().set_xticks(custom_ticks)  
            elif DARUNTER[0] >= XMAX:
                custom_ticks = np.arange(DARUNTER[0],XMIN,-XMAJOR)
                custom_ticks = custom_ticks[custom_ticks<=XMAX]
                plt.gca().set_xticks(custom_ticks[::-1]) 
            else:
                lower_part = np.arange(DARUNTER[0],XMAX,XMAJOR)
                upper_part = np.arange(DARUNTER[0],XMIN,-XMAJOR)[::-1]
                custom_ticks = np.unique(np.concatenate((upper_part,lower_part),axis=None))
                plt.gca().set_xticks(custom_ticks)                 
            #plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=XMAJOR,offset=DARUNTER[0]))
        if XMINOR != None:
            if DARUNTER[0] <= XMIN:
                custom_ticks = np.arange(DARUNTER[0],XMAX,XMINOR)
                custom_ticks = custom_ticks[custom_ticks>=XMIN]
                plt.gca().set_xticks(custom_ticks,minor=True)  
            elif DARUNTER[0] >= XMAX:
                custom_ticks = np.arange(DARUNTER[0],XMIN,-XMINOR)
                custom_ticks = custom_ticks[custom_ticks<=XMAX]
                plt.gca().set_xticks(custom_ticks[::-1],minor=True) 
            else:
                lower_part = np.arange(DARUNTER[0],XMAX,XMINOR)
                upper_part = np.arange(DARUNTER[0],XMIN,-XMINOR)[::-1]
                custom_ticks = np.unique(np.concatenate((upper_part,lower_part),axis=None))
                plt.gca().set_xticks(custom_ticks,minor=True)     
#             plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(XMINOR))
        if YMAJOR != None:
            if DARUNTER[1] <= YMIN:
                custom_ticks = np.arange(DARUNTER[1],YMAX,YMAJOR)
                custom_ticks = custom_ticks[custom_ticks>=YMIN]
                plt.gca().set_yticks(custom_ticks)  
            elif DARUNTER[1] >= YMAX:
                custom_ticks = np.arange(DARUNTER[1],YMIN,-YMAJOR)
                custom_ticks = custom_ticks[custom_ticks<=YMAX]
                plt.gca().set_yticks(custom_ticks[::-1]) 
            else:
                lower_part = np.arange(DARUNTER[1],YMAX,YMAJOR)
                upper_part = np.arange(DARUNTER[1],YMIN,-YMAJOR)[::-1]
                custom_ticks = np.unique(np.concatenate((upper_part,lower_part),axis=None))
                plt.gca().set_yticks(custom_ticks)   
#             plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(YMAJOR))
        if YMINOR != None:
            if DARUNTER[1] <= YMIN:
                custom_ticks = np.arange(DARUNTER[1],YMAX,YMINOR)
                custom_ticks = custom_ticks[custom_ticks>=YMIN]
                plt.gca().set_yticks(custom_ticks,minor=True)  
            elif DARUNTER[1] >= YMAX:
                custom_ticks = np.arange(DARUNTER[1],YMIN,-YMINOR)
                custom_ticks = custom_ticks[custom_ticks<=YMAX]
                plt.gca().set_yticks(custom_ticks[::-1],minor=True) 
            else:
                lower_part = np.arange(DARUNTER[1],YMAX,YMINOR)
                upper_part = np.arange(DARUNTER[1],YMIN,-YMINOR)[::-1]
                custom_ticks = np.unique(np.concatenate((upper_part,lower_part),axis=None))
                plt.gca().set_yticks(custom_ticks,minor=True)   
#             plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(YMINOR))
        if XMIN == None:
            XMIN = plt.gca().get_xlim()[0]
        if XMAX == None:
            XMAX = plt.gca().get_xlim()[1]
            
        ax2 = plt.gca().twiny()
        ax2.set_xlim(xmin=XMIN, xmax=XMAX)
        if XMAJOR != None:
            if DARUNTER[0] <= XMIN:
                custom_ticks = np.arange(DARUNTER[0],XMAX,XMAJOR)
                custom_ticks = custom_ticks[custom_ticks>=XMIN]
                ax2.set_xticks(custom_ticks)  
            elif DARUNTER[0] >= XMAX:
                custom_ticks = np.arange(DARUNTER[0],XMIN,-XMAJOR)
                custom_ticks = custom_ticks[custom_ticks<=XMAX]
                ax2.set_xticks(custom_ticks[::-1]) 
            else:
                lower_part = np.arange(DARUNTER[0],XMAX,XMAJOR)
                upper_part = np.arange(DARUNTER[0],XMIN,-XMAJOR)[::-1]
                custom_ticks = np.unique(np.concatenate((upper_part,lower_part),axis=None))
                ax2.set_xticks(custom_ticks)   
#             ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(XMAJOR))
        if XMINOR != None:
            if DARUNTER[0] <= XMIN:
                custom_ticks = np.arange(DARUNTER[0],XMAX,XMINOR)
                custom_ticks = custom_ticks[custom_ticks>=XMIN]
                ax2.set_xticks(custom_ticks,minor=True)  
            elif DARUNTER[0] >= XMAX:
                custom_ticks = np.arange(DARUNTER[0],XMIN,-XMINOR)
                custom_ticks = custom_ticks[custom_ticks<=XMAX]
                ax2.set_xticks(custom_ticks[::-1],minor=True) 
            else:
                lower_part = np.arange(DARUNTER[0],XMAX,XMINOR)
                upper_part = np.arange(DARUNTER[0],XMIN,-XMINOR)[::-1]
                custom_ticks = np.unique(np.concatenate((upper_part,lower_part),axis=None))
                ax2.set_xticks(custom_ticks,minor=True)     
#             ax2.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(XMINOR))
        if ALT[0] == False or NOLAB[0] == True:
            ax2.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax2.set_xlabel(LABELS[0])
        ax2.tick_params(direction='in',which='both')
        ax2.tick_params(which='major', length=6)
        ax2.tick_params(which='minor', length=3)
        
        if YMIN == None:
            YMIN = plt.gca().get_ylim()[0]
        if YMAX == None:
            YMAX = plt.gca().get_ylim()[1]
        ax2 = plt.gca().twinx()
        ax2.set_ylim(ymin=YMIN, ymax=YMAX)
        if YMAJOR != None:
            if DARUNTER[1] <= YMIN:
                custom_ticks = np.arange(DARUNTER[1],YMAX,YMAJOR)
                custom_ticks = custom_ticks[custom_ticks>=YMIN]
                ax2.set_yticks(custom_ticks)  
            elif DARUNTER[1] >= YMAX:
                custom_ticks = np.arange(DARUNTER[1],YMIN,-YMAJOR)
                custom_ticks = custom_ticks[custom_ticks<=YMAX]
                ax2.set_yticks(custom_ticks[::-1]) 
            else:
                lower_part = np.arange(DARUNTER[1],YMAX,YMAJOR)
                upper_part = np.arange(DARUNTER[1],YMIN,-YMAJOR)[::-1]
                custom_ticks = np.unique(np.concatenate((upper_part,lower_part),axis=None))
                ax2.set_yticks(custom_ticks)   
#             ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(YMAJOR))
        if YMINOR != None:
            if DARUNTER[1] <= YMIN:
                custom_ticks = np.arange(DARUNTER[1],YMAX,YMINOR)
                custom_ticks = custom_ticks[custom_ticks>=YMIN]
                ax2.set_yticks(custom_ticks,minor=True)  
            elif DARUNTER[1] >= YMAX:
                custom_ticks = np.arange(DARUNTER[1],YMIN,-YMINOR)
                custom_ticks = custom_ticks[custom_ticks<=YMAX]
                ax2.set_yticks(custom_ticks[::-1],minor=True) 
            else:
                lower_part = np.arange(DARUNTER[1],YMAX,YMINOR)
                upper_part = np.arange(DARUNTER[1],YMIN,-YMINOR)[::-1]
                custom_ticks = np.unique(np.concatenate((upper_part,lower_part),axis=None))
                ax2.set_yticks(custom_ticks,minor=True)   
#             ax2.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(YMINOR))
        if ALT[1] == False or NOLAB[1] == True:
            ax2.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax2.set_ylabel(LABELS[1])        #
        ax2.tick_params(direction='in',which='both')
        ax2.tick_params(which='major', length=6)
        ax2.tick_params(which='minor', length=3)
        plt.sca(tmp)
            
    def show(self):
        plt.show()
        return
    
    ######################################################################################################################
    #
    #   
    #                                 DATASET PART OF WRPLOT
    #
    #
    ######################################################################################################################
    
    def is_float(self,element):
        try:
            float(element)
            return True
        except ValueError:
            return False

    
    def command_include(self, path, inckey=None, dataset=1, **kwargs):
        """    
        !! this routine is stolen from Andreas Sander !!
        Loads a datafile which contains a table. It is possible to load also tables contained in 
        files from PoWR (e.g. formal.plot) which have the format N=n. (see WRPLOT manual)
        
        Parameters
        ----------
        path: str
            Path to the dataset that should be loaded.
        inckey: str default: None
            If set to "None" it assumes that the file is an XYTABLE,
            else it will search for an incremental key which within the file that indicates the start of a
            datatable.
        dataset: int default: 1
            Number of the dataset inside a datatable, that should be read (only important when reading PoWR files)
        
        Returns
        -------
        The desired dataset.
        
        Notes
        -----
        This function is not well tested.
        """
        comments=['FIN', 'N=', '#', "*", "!", "/", "\\", "PLOT","KAS", "HEADER", "X", "Y","MASS"]
        # check if user wants to set comments by hand 
        if "comments" in kwargs:
            comments=kwargs.pop("comments")
            
        # if not inckey, just load file
        if inckey == None or inckey == "":
            return np.loadtxt(path,comments=comments,**kwargs).T
        # if inckey set find inckey in file and the end of the tabel
        else:
            file = open(path, 'r')
            startkey = "N=" 
            endkeys = ["N=", "FINISH", "END"]

            xdata = []
            ydata = []    
            readindex = -1
            nskip = dataset - 1
            foundkey = (inckey == "")
            for curline in file:  
                if (not foundkey): 
                    keypos = curline.find(inckey) 
                    if (keypos == -1):
                        continue
                    else:
                        foundkey = True

                if ((readindex == -1) and foundkey):
                    readindex = 0

                if (readindex == 0):
                    if (curline.strip().startswith(startkey)):
                        if (nskip > 0):
                            nskip = nskip - 1
                        else:
                            readindex = 1
                    continue
                elif (readindex == 1 or readindex == 2):
        #           now we need to read out pairs of lines

                    rawline = curline.rstrip()
                    for endkey in endkeys:
                        if (rawline.strip().startswith(endkey)):
                            if (readindex == 2):
                                print ("FATAL ERROR: odd number of xy-lines")
                                return False
                            readindex = 99
                            break
                    xynewset = rawline.split()
                    if (readindex == 1):
                        for xynew in xynewset:
                            xdata.append(float(xynew))
                        readindex = 2
                    elif (readindex == 2):
                        for xynew in xynewset:
                            ydata.append(float(xynew))
                        readindex = 1
                elif (readindex > 9):
                    break

            file.close()
            
            if (not foundkey):
                print ('Could not find keyword ', inckey)
                return 0

            if (readindex == 0):
                print ('Could not find dataset ', dataset)
                
            xdata = np.float64(xdata)            
            ydata = np.float64(ydata)
            return np.vstack((xdata,ydata))
    
    def command_log(self, dataset, axis=None):
        """    
        Takes the logarithm with basis 10 of a dataset. It has the possibity to only take the logarithm of one axis.
        
        Parameters
        ----------
        dataset: np.array or list
            Dataset that should be modified.
        axis: str, int, default: None
            If axis = None, logarithm of all values is taken.
            If axis = "x" or axis = 0, logarithm of x-axis is taken
            If axis = "y" or axis = 1, logarithm of y-axis is taken
            
        Returns
        -------
        The modified dataset.
        
        Notes
        -----
        If one axis is specified the dataset will become a 2D array.
        """
        if axis == None or axis == "both":
            return np.log10(dataset)
        elif axis == "x" or axis == 0:
            return np.array([np.log10(dataset[0]),dataset[1]])
        elif axis == "y" or axis == 1:
            return np.array([dataset[0],np.log10(dataset[1])])
        else:
            print("Instructions not clear, choose x or y axis.")
            print("Warning: Dataset not modified.")
            return dataset
        
    def command_dex(self, dataset, axis=None):
        """    
        De-logarithmizes a dataset. It has the possibity to only take the logarithm of one axis.
        
        Parameters
        ----------
        dataset: np.array or list
            Dataset that should be modified.
        axis: str, int, default: None
            If axis = None, De-logarithmizes all values.
            If axis = "x" or axis = 0, De-logarithmizes all values of the x-axis
            If axis = "y" or axis = 1, De-logarithmizes all values of the y-axis
            
        Returns
        -------
        The modified dataset.
        
        Notes
        -----
        If one axis is specified the dataset will become a 2D array.
        """
        if axis == None or axis == "both":
            return np.array([10**dataset[0],10**dataset[1]])
        elif axis == "x" or axis == 0:
            return np.array([10**dataset[0],dataset[1]])
        elif axis == "y" or axis == 1:
            return np.array([dataset[0],10**dataset[1]])
        else:
            print("Instructions not clear, choose x or y axis.")
            print("Warning: Dataset not modified.")
            return dataset
        
    def command_cut(self, dataset, cuttingrange, axis=None):
        """    
        Cuts off the edges of the cuttingrange of an axis.
        
        Parameters
        ----------
        dataset: np.array or list
            Dataset that should be modified.
        cuttingrange: list or tuple
            If the axis is defined (x or y) it must contain two elements, the values at which to cut.
            If the axis is not defined (None) it must be a list/tuple of two lists/tuples that contain 
            two values that define the cutting ranges.
        axis: str, int, default: None
            If axis = None, cuttingrange must contain 4 values.
            If axis = "x" or axis = 0, logarithm of x-axis is taken
            If axis = "y" or axis = 1, logarithm of y-axis is taken
            
        Returns
        -------
        The modified dataset.
        
        Notes
        -----
        If the dataset will become a 2D array.
        """
        if axis == "x" or axis == 0:
            arrange = np.where((cuttingrange[0] < dataset[0]) & (dataset[0] < cuttingrange[1]))
            return np.array([dataset[0][arrange],dataset[1][arrange]])
        elif axis == "y" or axis == 1:
            arrange = np.where((cuttingrange[0] < dataset[1]) & (dataset[1] < cuttingrange[1]))
            return np.array([dataset[0][arrange],dataset[1][arrange]])
        else:
            try:
                arrange = np.where((cuttingrange[0][0] < dataset[0]) & (dataset[0] < cuttingrange[0][1]) & 
                                   (cuttingrange[1][0] < dataset[1]) & (dataset[1] < cuttingrange[1][1]))
                return np.array([dataset[0][arrange],dataset[1][arrange]])
            except:
                print("Error: could not cut dataset. You probably forgot to set the axis (x or y).")
                print("Returning unmodified dataset.")
                return dataset
            
    def gauss(self, x, mu=0, sigma=1):
        """    
        Gaussian function for given range x.
        
        Parameters
        ----------
        x: np.array or list or int
            x-values over which the Gaussian should be calculated.
        mu: float default: 0
            Offset from the center. 
        sigma: float default: 1
            Standard deviation of the Gaussian curve. 
            
        Returns
        -------
        The Gaussian curve or the value of the curve at position x.
        """
        gauss = np.exp(-0.5*((x-mu)/sigma)**2)
        return gauss/np.sum(gauss)
    
    def box(self, x, mu=0, width=1):
        """    
        Box or hat function for given range x.
        
        Parameters
        ----------
        x: np.array or list or int
            x-values over which the Box function should be calculated.
        mu: float default: 0
            Offset from the center. 
        width: float default: 1
            Half width of the Box (like in WRPLOT?). 
            
        Returns
        -------
        The Box function or the value of the curve at position x.
        
        Notes
        -----
        The Box function is NOT normalized to 1.
        """
        return np.array([1 if  (a >= mu-width) & (a <= mu+width) else 0 for a in x])
    
    def rot(self, x, mu=0, vsini=0, epsilon=0):
        """    
        Ellipsoidal function to mimic the rotational broadening effect for given range x.
        Calculated using equation 18.14 from David F. Gray "The Observation And Analysis Of Stellar Photospheres".
        
        Parameters
        ----------
        x: np.array or list or int
            x-values over which the rotational broadening should be calculated.
        mu: float default: 0
            Central wavelenght. 
        vsini: float default: 0
            Projected rotational velocity of the object in km/s (this is different to WRPLOT!). 
        epsilon: float default: 0
            Limb darkening coefficient. By default this effect is ignored.
            
        Returns
        -------
        The Rotation ellipse curve or the value of the curve at position x.
        """
        vsini = vsini / 299792.458
        del_lambda = mu-x
        del_lambda_L = vsini*mu
        c1 = 2 * (1 - epsilon) / (np.pi * del_lambda_L * (1 - epsilon / 3.))
        c2 = epsilon / (2.* del_lambda_L * (1. - epsilon/3.))
        return c1 * (1-(del_lambda/del_lambda_L)**2)**0.5 + c2 * (1-(del_lambda/del_lambda_L)**2)
            
    def command_convol(self, dataset, shape="gaussian", value=0, beta=0):
        """    
        Convolution of a dataset with a given function/shape.
        
        Parameters
        ----------
        dataset: np.array or list or int
            Dataset that should be modified.
        shape: str default: gaussian
            Possible options: "gauss" or gaussian", "box" or "kasten", "rot" or "rotation"
        value: float default: 0
            A value of with of an corresponding curve:
            If shape == "gauss", the value corresponds to the FWHM.
            If shape == "box", the value corresponds to the half width of the box.
            If shape == "rot", the value corresponds to the vsini of the rotation (note this is somewhat different to WRPLOT!)
        beta: float default: 0
            This value is only needed for the rotational broadening and is the Limb darkening coefficient.
            By default limb-darkening is ignored.
            
        Returns
        -------
        The modified dataset. 
        """
        
        if value == 0:
            return dataset
        
        shape = shape.lower()
        if shape == "gauss" or shape == "gaussian":
            sigma = value/2.354820045 #FWHM = 2 sqrt(2 ln(2)) stddev
            new_y = []
            for mu in dataset[0]:
                temp_data = self.command_cut(dataset,(mu-4.5*sigma,mu+4.5*sigma),axis="x")
                gaussian = self.gauss(temp_data[0],mu,sigma)
                new_y.append(np.dot(temp_data[1],gaussian/sum(gaussian)))
            return np.array([dataset[0],new_y])
        elif shape == "box" or shape == "kasten":
            new_y = []
            for mu in dataset[0]:
                temp_data = self.command_cut(dataset,(mu-value,mu+value),axis="x")
                boxprof = self.box(temp_data[0],mu,value)
                new_y.append(np.dot(temp_data[1],boxprof/sum(boxprof)))
            return np.array([dataset[0],new_y])
        elif shape == "rot" or shape == "rotation":
            new_y = []
            for mu in dataset[0]:
                temp_data = self.command_cut(dataset,(mu-value/299792.458*mu,mu+value/299792.458*mu),axis="x")
                rot_prof = self.rot(temp_data[0],mu,value,beta)
                new_y.append(np.dot(temp_data[1],rot_prof/sum(rot_prof)))
            return np.array([dataset[0],new_y])
            
    def command_convol_fast(self, dataset, shape="gaussian", value=0, beta=0,
                            precision=1e-4, allow_fft=False):
        """    
        Fast convolution agreeing with ``command_convol`` to within ``precision``.

        By default this always uses the exact discrete, per-point weighted sums
        as ``command_convol`` (numba-accelerated kernel path). An optional FFT
        path exists for speed on uniform grids, but it is **opt-in** via
        ``allow_fft=True`` because FFT assumes uniform Δλ and a single fixed
        kernel — conditions that real spectra and rotational broadening often
        violate.

        Parameters
        ----------
        dataset: np.array or list or int
            Dataset that should be modified. If the wavelength axis is not
            sorted ascending, it is sorted (with flux) before convolution.
        shape: str default: gaussian
            Possible options: "gauss" or gaussian", "box" or "kasten", "rot" or "rotation"
        value: float default: 0
            A value of with of an corresponding curve:
            If shape == "gauss", the value corresponds to the FWHM.
            If shape == "box", the value corresponds to the half width of the box.
            If shape == "rot", the value corresponds to the vsini of the rotation (note this is somewhat different to WRPLOT!)
        beta: float default: 0
            This value is only needed for the rotational broadening and is the Limb darkening coefficient.
            By default limb-darkening is ignored.
        precision: float or None default: 1e-4
            Maximum absolute flux difference allowed relative to ``command_convol``.
            Default ``1e-4`` is a typical “scientifically indistinguishable”
            tolerance. Pass ``None`` (or ``<= 0``) to request a fully exact
            discrete match (machine precision). Values tighter than ~1e-10 also
            keep the kernel path even if ``allow_fft=True``.
        allow_fft: bool default: False
            If False (default), always use the exact kernel path.
            If True, FFT may be used **only when all** of the following hold:
              - ``precision`` is set and ``>= 1e-10`` (not forcing exact),
              - ``shape`` is Gaussian or box (not rotation; rot width ∝ λ),
              - the wavelength grid is mathematically uniform.
            FFT is faster but can disagree with ``command_convol`` on irregular
            sampling; leave this False unless you know the grid is uniform and
            accept the ``precision`` tolerance.
            
        Returns
        -------
        The modified dataset. 
        """
        if value == 0:
            return dataset

        x = np.asarray(dataset[0], dtype=float).copy()
        y = np.asarray(dataset[1], dtype=float).copy()
        if x.size > 1 and np.any(x[1:] < x[:-1]):
            order = np.argsort(x, kind="mergesort")
            x = x[order]
            y = y[order]

        shape = shape.lower()
        force_exact = precision is None or precision <= 0

        # FFT is opt-in (allow_fft=False by default). Even when enabled it is
        # only safe for constant-width kernels on a uniform grid, and only when
        # the caller does not demand a fully exact match via precision.
        use_fft = (
            allow_fft
            and not force_exact
            and shape in ("gauss", "gaussian", "box", "kasten")
            and x.size > 1
            and np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-9, atol=0.0)
            and precision >= 1e-10
        )

        if use_fft:
            y_new = self._convol_fft_uniform(x, y, shape, value, beta)
        elif shape in ("gauss", "gaussian"):
            sigma = value / 2.354820045
            y_new = _convol_gauss_kernel(x, y, sigma, 4.5 * sigma)
        elif shape in ("box", "kasten"):
            y_new = _convol_box_kernel(x, y, value)
        elif shape in ("rot", "rotation"):
            y_new = _convol_rot_kernel(x, y, value, beta)
        else:
            raise ValueError(f"Unknown convolution shape: {shape}")

        return np.array([x, y_new])

    def _convol_fft_uniform(self, x, y, shape, value, beta=0):
        """FFT convolution for Gaussian/box on a verified uniform wavelength grid."""
        dx = float(x[1] - x[0])
        if shape in ("gauss", "gaussian"):
            sigma = value / 2.354820045
            half_width = max(int(np.ceil(4.5 * sigma / dx)), 1)
            xk = dx * np.arange(-half_width, half_width + 1)
            kernel = np.exp(-0.5 * ((xk / sigma) ** 2))
        else:  # box / kasten
            half_width = max(int(np.floor(value / dx)), 0)
            xk = dx * np.arange(-half_width, half_width + 1)
            kernel = ((xk > -value) & (xk < value)).astype(float)
            if kernel.sum() == 0:
                kernel = np.array([1.0])
        kernel /= kernel.sum()
        pad = len(kernel) // 2
        y_padded = np.pad(y, pad, mode="edge")
        return fftconvolve(y_padded, kernel, mode="valid")

    def command_write(self,file,dataset,**kwargs):
        """    
        Saves a dataset to a textfile.
        
        Parameters
        ----------
        file: str
            Name of the final file. Do not forget the file extension.
        dataset: np.array or list or int
            Dataset that should be saved.
        kwargs: 
            Everything that can be used in the numpy.savetxt function.
        """
        np.savetxt(file,dataset.T,**kwargs)
    
    def command_plus(self,dataset,value,axis="x"):
        """    
        Adds a value to a given axis. It is possible to shift both axes simultaneously.
        
        Parameters
        ----------
        dataset: np.array or list
            Dataset that should be modified.
        value: int, float, list or tuple
            If the axis is defined (x or y) it must be a int or float.
            If the axis is not defined (e.g., None) it can be a list/tuple that contain two values that define 
            the shift.
        axis: str, int, default: None
            If axis = None, value added to both axes.
            If axis = "x" or axis = 0, value added to x-axis
            If axis = "y" or axis = 1, value added to y-axis
            
        Returns
        -------
        The modified dataset.
        """
        if axis == "x" or axis == 0:
            return np.array([dataset[0]+value,dataset[1]])
        elif axis == "y" or axis == 1:
            return np.array([dataset[0],dataset[1]+value])
        else:
            if isinstance(value, int) or isinstance(value, float):
                return np.array([dataset[0]+value,dataset[1]+value])
            else:
                return np.array([dataset[0]+value[0],dataset[1]+value[1]])
            
            
    def command_minus(self,dataset,value,axis="x"):
        """    
        Subtracts a value to a given axis. It is possible to shift both axes simultaneously.
        
        Parameters
        ----------
        dataset: np.array or list
            Dataset that should be modified.
        value: int, float, list or tuple
            If the axis is defined (x or y) it must be a int or float.
            If the axis is not defined (e.g., None) it can be a list/tuple that contain two values that define 
            the shift.
        axis: str, int, default: None
            If axis = None, value subtracted from both axes.
            If axis = "x" or axis = 0, value subtracted from x-axis
            If axis = "y" or axis = 1, value subtracted from y-axis
            
        Returns
        -------
        The modified dataset.
        """
        if axis == "x" or axis == 0:
            return np.array([dataset[0]-value,dataset[1]])
        elif axis == "y" or axis == 1:
            return np.array([dataset[0],dataset[1]-value])
        else:
            if isinstance(value, int) or isinstance(value, float):
                return np.array([dataset[0]-value,dataset[1]-value])
            else:
                return np.array([dataset[0]-value[0],dataset[1]-value[1]])
            
    
    def command_multiply(self,dataset,value,axis="x"):
        """    
        Multiplys all values of a given axis. It is possible to multiply both axes simultaneously.
        
        Parameters
        ----------
        dataset: np.array or list
            Dataset that should be modified.
        value: int, float, list or tuple, or 2D array
            If the axis is defined (x or y) it must be a int or float or a 2D array. 
            If it is a 2D array, the other axis must be exactly the same!
            If the axis is not defined (e.g., None) it can also be a list/tuple that contain two values that define 
            the shift.
        axis: str, int, default: None
            If axis = None, both axes multiplied by value.
            If axis = "x" or axis = 0, x-axis multiplied by value.
            If axis = "y" or axis = 1, y-axis multiplied by value.
            
        Returns
        -------
        The modified dataset.
        """
        if axis == "x" or axis == 0:
            if isinstance(value, int) or isinstance(value, float):
                return np.array([dataset[0]*value,dataset[1]])
            elif (dataset[1] == value[1]).all():
                return np.array([dataset[0]*value[0],dataset[1]])
            else:
                print("Warning unknown type of value given.")
                return             
        elif axis == "y" or axis == 1:
            if isinstance(value, int) or isinstance(value, float):
                return np.array([dataset[0],dataset[1]*value])
            elif (dataset[0] == value[0]).all():
                return np.array([dataset[0],dataset[1]*value[1]])
            else:
                print("Warning unknown type of value given.")
                return 
        else:
            if isinstance(value, int) or isinstance(value, float):
                return np.array([dataset[0]*value,dataset[1]*value])
            else:
                return np.array([dataset[0]*value[0],dataset[1]*value[1]])
            
    
    def command_divide(self,dataset,value,axis="x"):
        """    
        Divides all values of a given axis. It is possible to divide both axes simultaneously.
        
        Parameters
        ----------
        dataset: np.array or list
            Dataset that should be modified.
        value: int, float, list or tuple
            If the axis is defined (x or y) it must be a int or float.
            If the axis is not defined (e.g., None) it can also be a list/tuple that contain two values that define 
            the shift.
        axis: str, int, default: None
            If axis = None, both axes divided by value.
            If axis = "x" or axis = 0, x-axis divided by value.
            If axis = "y" or axis = 1, y-axis divided by value.
            
        Returns
        -------
        The modified dataset.
        """
        if axis == "x" or axis == 0:
            return np.array([dataset[0]/value,dataset[1]])
        elif axis == "y" or axis == 1:
            return np.array([dataset[0],dataset[1]/value])
        else:
            if isinstance(value, int) or isinstance(value, float):
                return np.array([dataset[0]/value,dataset[1]/value])
            else:
                return np.array([dataset[0]/value[0],dataset[1]/value[1]])

    def command_filter(self, magnitude, filter_name):
        zero_point = 0
        central_wavelength = 0
        if filter_name == "U":
            central_wavelength = 3600
            zero_point = 1790.*2.99792458E-05/(central_wavelength**2)
        elif filter_name == "B":
            central_wavelength = 4400
            zero_point = 4063.*2.99792458E-05/(central_wavelength**2)
        elif filter_name == "V":
            central_wavelength = 5550
            zero_point = 3636.*2.99792458E-05/(central_wavelength**2)
        elif filter_name == "I":
            central_wavelength = 7900
            zero_point = 2416.*2.99792458E-05/(central_wavelength**2)
        elif filter_name == "J":
            central_wavelength = 12350
            zero_point = 1594.*2.99792458E-05/(central_wavelength**2)
        elif filter_name == "H":
            central_wavelength = 16620
            zero_point = 1024.*2.99792458E-05/(central_wavelength**2)
        elif filter_name == "K":
            central_wavelength = 21590
            zero_point = 666.7*2.99792458E-05/(central_wavelength**2)
        elif filter_name == "MCPS:U":
            zero_point = 4.08739e-9
            central_wavelength = 3706.29
        elif filter_name == "MCPS:B":
            zero_point = 6.21086e-9
            central_wavelength = 4394.48
        elif filter_name == "MCPS:V":
            zero_point = 3.64047e-9
            central_wavelength = 5438.23
        elif filter_name == "MCPS:I":
            zero_point = 9.23651e-10
            central_wavelength = 8568.89
        elif filter_name == "2MASS:J":
            zero_point = 3.0596e-10
            central_wavelength = 12321.15
        elif filter_name == "2MASS:H":
            zero_point = 1.11064e-10
            central_wavelength = 16422.96
        elif filter_name == "2MASS:K":
            zero_point = 4.17999e-11
            central_wavelength = 21557.65
        elif filter_name == "Gaia:G":
            zero_point = 2.81581e-9
            central_wavelength = 5822.39  
        elif filter_name == "SDSS:g":
            zero_point = 4.90883e-9
            central_wavelength = 4669.48       
        elif filter_name == "SMASH:u":
            zero_point = 8.62456e-9
            central_wavelength = 3648.48  
        elif filter_name == "SMASH:g":
            zero_point = 4.70792e-9
            central_wavelength = 4769.90
        elif filter_name == "SMASH:r":
            zero_point = 2.64299e-9
            central_wavelength = 6370.44
        elif filter_name == "SMASH:i":
            zero_point = 1.78253e-9
            central_wavelength = 7774.30
        elif filter_name == "SMASH:z":
            zero_point = 1.29484e-9
            central_wavelength = 9154.88 	
        elif filter_name == "UVIT:172":
            zero_point = 3.6869e-8
            central_wavelength = 1719.32
        else: 
            print("Warning: Filter not found...")
            return 
            
        flux = 10**( magnitude/(-2.5) ) * zero_point # erg/cm^2/s/Ang
        return central_wavelength,flux           
        
            
    def command_ari(self,dataset1,dataset2,axis="y"):
        """    
        Adds one datasets to another, using the dataset with the finer resolution as basis.
        The combined dataset only covers the range covered by both datasets.
        One can add first or second column of the datasets (x or y).
        
        Parameters
        ----------
        dataset1: np.array or list
            First dataset.
        dataset2: np.array or list
            Second dataset that should be added.
        axis: str, int, default: None
            If axis = "x" or axis = 0, y-axis values are unchanged, x-axis are added.
            If axis = "y" or axis = 1, x-axis values are unchanged, y-axis are added.
            
        Returns
        -------
        The combined dataset.
        
        Notes
        -----
        The "x" and "y" notation for the axes might be opposite to WRPLOT, but I found 
        it more logical to say which axis should be merged.
        """

        index_1 = int(0)
        index_2 = int(1)
        cut_axis = "x"
        if axis == "x":
            index_1 = int(1)
            index_2 = int(0)
            cut_axis="y"

        wavemin = max(min(dataset1[index_1]),min(dataset2[index_1]))
        wavemax = min(max(dataset1[index_1]),max(dataset2[index_1]))

        dataset1 = self.command_cut(dataset1, [wavemin,wavemax],axis=cut_axis)
        dataset2 = self.command_cut(dataset2, [wavemin,wavemax],axis=cut_axis)
        
        if (len(dataset1[index_1])>len(dataset2[index_1])):
            wavegrid = dataset1[index_1]
            tmp = dataset2
            flux = dataset1[index_2]
        else:
            wavegrid = dataset2[index_1]
            tmp = dataset1
            flux = dataset2[index_2]

        flux_to_add = interpolate.interp1d(tmp[index_1],tmp[index_2], bounds_error=False, fill_value=0,kind='linear')(wavegrid)
        flux = flux + flux_to_add

        # keep the same format (x,y) as the original arrays
        if axis == "x":
            return np.array([flux,wavegrid])
        return np.array([wavegrid,flux])
            
    def command_insert(self,dataset,x):
        """    
        Insert a point a x by linear interpolation between two existing points.
        
        Parameters
        ----------
        dataset: np.array or list
            Dataset that should be modified.
        x: int, float
            Datapoint that should be inserted.
            
        Returns
        -------
        The modified dataset.
        
        Notes
        -----
        The input array has to be monotonic!
        If the dataset will become a 2D array. 
        """
        xdata = np.float64(dataset[0])
        # if not monotinic we cannot interpolate
        if not (np.sort(xdata) == xdata).all():
            print("Error: Dataset not monotonic.")
            print("Returining unmodified dataset.")
            return dataset
        ydata = np.float64(dataset[1])
        
        if x > max(dataset[0]) or x < min(dataset[0]):
            print("Error: Datapoint is outside the dataset.")
            print("Returining unmodified dataset.")
            return dataset
        
        indices = -1
        for L in range(len(xdata)-1):
            DR1 = (xdata[L]-x)
            DR2 = (xdata[L+1]-x)
            XM = DR1*DR2
            if XM == 0.:
                print("Warning: Datapoint already exits!")
                print("Returining unmodified dataset.")
                return dataset
            elif XM < 0.:
                ### hochschieben der elemente ab L1+1
                # first add dummy element at the end
                xdata = np.append(xdata,0)
                ydata = np.append(ydata,0)
                # and then shift everything by one
                for LL in range(len(xdata)-2,L,-1):
                    ydata[LL+1] = ydata[LL]
                    xdata[LL+1] = xdata[LL]
                
                P = (x - xdata[L]) / (xdata[L+2] - xdata[L])
                Q = 1. - P   
                #now insert new datapoint
                xdata[L+1] = x
                ydata[L+1] = Q*ydata[L]+P*ydata[L+2]
                return np.array([xdata,ydata])
        
        print("Error: Dataset out of range. Cannot interpolate.")
        print("Returining unmodified dataset.")
        return dataset   

    def command_sort(self,dataset):        
        indices = dataset[0].argsort()
        return np.array([dataset[0][indices],dataset[1][indices]])
    
    def voigth(self,a,v):
        """
        normalized (!!!) VOIGT function H(a,v)
        according to the implementation of Detlef Köster
        """
        wpi = 1.77245385
        v = abs(v)
        vv = v*v
        
        if a < 0.2 and v > 5.0:
            h = ((2.12/vv+0.8463)/vv+0.5642)*a/vv
            return h/wpi
        if a >= 0.2 and ((a > 1.4) or ((a + v) > 3.2)):
            aa = a*a
            u = (aa+vv)-1.4142
            uu = u*u
            h = ((((aa-10.*vv)*aa*3.+15.*vv*vv)/uu+3.*vv-aa)/uu+1.)*A*0.79788/u
            return h/wpi
        h0 = np.exp(-vv)
        h2 = (1.-2.*vv)*h0
        if v > 2.4:
            h1 = ((-0.0032783*vv+0.0429913*v-0.188326)*vv+0.278712*v+0.55415)/(vv-1.5)
            if a >= 0.2:
                hh1 = h1+h0*1.12838
                hh2 = h2+hh1*1.12838-h0
                hh3 = (1.-h2)*0.37613-hh1*0.66667*vv+hh2*1.12838
                hh4 = (3.*hh3-hh1)*0.37613+h0*0.66667*vv*vv
                h = ((((hh4*a+hh3)*a+hh2)*a+hh1)*a+h0)*(((-0.122727278*a+0.532770573)*a-0.96284325)*a+0.979895032)
                return h/wpi
            else:
                h=((h2*a+h1)*a+h0)
                return h/wpi
        if v <= 1.3:
            h1 = (0.42139*vv-2.34358*v+3.28868)*vv-0.15517*v-1.1247
            if a >= 0.2:
                hh1 = h1+h0*1.12838
                hh2 = h2+hh1*1.12838-h0
                hh3 = (1.-h2)*0.37613-hh1*0.66667*vv+hh2*1.12838
                hh4 = (3.*hh3-hh1)*0.37613+h0*0.66667*vv*vv
                h = ((((hh4*a+hh3)*a+hh2)*a+hh1)*a+h0)*(((-0.122727278*a+0.532770573)*a-0.96284325)*a+0.979895032)
                return h/wpi
            else:
                h=((h2*a+h1)*a+h0)
                return h/wpi
        h1 = (-0.220416*vv+1.989196*v-6.61487)*vv+9.39456*v-4.4848
        if a >= 0.2:
            hh1 = h1+h0*1.12838
            hh2 = h2+hh1*1.12838-h0
            hh3 = (1.-h2)*0.37613-hh1*0.66667*vv+hh2*1.12838
            hh4 = (3.*hh3-hh1)*0.37613+h0*0.66667*vv*vv
            h = ((((hh4*a+hh3)*a+hh2)*a+hh1)*a+h0)*(((-0.122727278*a+0.532770573)*a-0.96284325)*a+0.979895032)
            return h/wpi
        else:
            h=((h2*a+h1)*a+h0)
            return h/wpi
        
    def command_hlymana(self,dataset,key="coldens",value=-1,vrad=0,temp=-1,vturb=0,info=False):
        """
        Apply corrections for interstellar absorption by hydrogen Lyman lines (from Lyα to Ly20).
        By default, HLYMAN uses the formalism from Groenewegen & Lamers (1989) which accounts only 
        for the damping (i.e. Lorentzian) profiles but no Doppler cores. However, if one of the 
        parameters T and/or VTURB is specified, HLYMAN applies Voigt profiles with the according 
        Doppler velocity.
        
        Parameters
        ----------
        dataset: np.array or list
            Dataset that should be modified.
            First column should have wavelenghts in Angstrom 
            Second column should be linear flux (i.e. in erg/s/cm^2/Ang)
        keyword: string
            Possible options: "EBV" or "ebv" or "COLDENS" or "coldens"
            If EBV is used, the column density is calculated using n_H = 3.7e21*EBV (Groenewegen & Lamers 1989)
        value: int, float
            The value of EBV or COLDENS.
        vrad: int, float
            The radial velocity in km/s
        temp: int, float
            Temperature in Kelvin; 
            if T > -1:
                Account for Doppler broadening with the thermal velocity
        vturb: int, float
            Turbulence velocity in km/s
            if vturb > -1:
                additional Doppler broadening
            
        Returns
        -------
        The modified dataset.

        Notes
        -------
        The dataset must be monotonic.
        """
        # wavelenghts of H Ly lines
        XLAM0 = np.float64([1215.671, 1025.722, 972.5366, 949.74297, 
                 937.80344, 930.74824, 926.22571, 923.15037, 
                 920.96313, 919.35146, 918.12939, 917.18061,
                 916.42917, 915.82388, 915.32904, 914.91936,
                 914.57629, 914.28622, 914.0386, 913.82569,   
                 913.64126, 913.48036, 913.33922, 913.2147,   
                 913.10431, 913.00597, 912.918, 912.83898,   
                 912.76775, 912.70331, 912.64482])

        # constants of formulae 16 and 22 from GROENEWEGEN & LAMERS, 1989
        # note: extrapolated for lines higher than Ly 1-11!
        C0 = np.float64([7.056E-11, 4.428E-11, 3.788E-11, 3.522E-11, 
              3.381E-11, 3.298E-11, 3.243E-11, 3.204E-11, 
              3.177E-11, 3.156E-11, 3.141E-11, 3.129E-11, 
              3.12E-11, 3.112E-11, 3.106E-11, 3.101E-11, 
              3.097E-11, 3.093E-11, 3.09E-11, 3.087E-11, 
              3.085E-11, 3.083E-11, 3.081E-11, 3.079E-11, 
              3.078E-11, 3.077E-11, 3.076E-11, 3.075E-11,  
              3.074E-11, 3.073E-11, 3.072E-11])

        C1 = np.float64([1.657E+09, 3.575E+10, 2.402E+11, 9.867E+11, 
              3.059E+12, 7.871E+12, 1.773E+13, 3.628E+13, 
              6.036E+13, 9.76E+13, 1.427E+14, 1.941E+14, 
              2.505E+14, 3.106E+14, 3.734E+14, 4.381E+14, 
              5.042E+14, 5.712E+14, 6.387E+14, 7.064E+14, 
              7.743E+14, 8.421E+14, 9.098E+14, 9.773E+14, 
              1.044E+15, 1.111E+15, 1.178E+15, 1.244E+15, 
              1.31E+15, 1.376E+15, 1.441E+15])

        # f values (oscillator strengths) of the Lyman lines
        FVALUE = np.float64([4.1620E-01, 7.9100E-02, 2.8990E-02, 1.3940E-02,
                  7.7990E-03, 4.8140E-03, 3.1830E-03, 2.2160E-03, 
                  1.6050E-03, 1.2010E-03, 9.2140E-04, 7.2270E-04,
                  5.7740E-04, 4.6860E-04, 3.8560E-04, 3.2110E-04, 
                  2.7030E-04, 2.2960E-04, 1.9670E-04, 1.6980E-04, 
                  1.4760E-04, 1.2910E-04, 1.1360E-04, 1.0050E-04, 
                  8.9290E-05, 7.9700E-05, 7.1450E-05, 6.4300E-05, 
                  5.8070E-05, 5.2610E-05, 4.7830E-05])

        # Gamma values (damping constants) of the Lyman lines
        # The first 5 Gammas are from Morton (2003), ApJS 149: 205-238;
        # this exactly reproduces the Groenewegen & Lamers profiles.
        # Further values  were computed by Thomas Rauch (08.09.2012), 
        # according to Groenewegen & Lamers (1989); 
        GAMM = np.float64([6.26500E+08, 1.89700E+08, 8.12700E+07, 4.20400E+07,
                 2.45000E+07, 1.4043E+07, 9.3788E+06, 6.5742E+06, 
                 4.7849E+06, 3.5931E+06, 2.7643E+06, 2.1728E+06, 
                 1.7388E+06, 1.4131E+06, 1.1641E+06, 9.7030E+05, 
                 8.1739E+05, 6.9477E+05, 5.9556E+05, 5.1437E+05,
                 4.4730E+05, 3.9138E+05, 3.4449E+05, 3.0485E+05,
                 2.7091E+05, 2.4188E+05, 2.1689E+05, 1.9522E+05,
                 1.7632E+05, 1.5978E+05, 1.4527E+05])

        # hydrogen column density for E(B-V) = 1mag
        C2 = 3.8e21
        CLIGHT = 299792.458
                
        #Constants for Voigt profile in cgs units
        ATMASSH = 1.00794
        XKBOLTZ = 1.38065000e-16
        XMP = 1.67262160e-24
        XLYEDGE0 = 911.2670504
        PI = 3.1415927

        TEMP = temp
        VTURB = vturb
        VRAD = vrad
        
        # Insert lambda point when relative change of TAU exceeds TAUCRIT
        TAUCRIT = 0.1
        # Insert MAXNPOINTS extra points per line on top of TAU criterion
        MAXNPOINTS = 0
        
        XLAM = XLAM0 * (1+VRAD/CLIGHT)
        XLYEDGE = XLYEDGE0 * (1+VRAD/CLIGHT)
        
        # convert vturb from km/s to cm/s
        VTURB = VTURB * 1e5
        
        ENH = value
        key = key.lower()
        if key == "ebv":
            ENH = C2 * value
        elif key != "coldens":
            print("Error: Unknown keyword in HLYMANA.")
            print("Please use 'coldens' or 'ebv'.")
            print("Returning unmodified dataset.")
            return dataset
        
        if ENH == 0.:
            return dataset
        
        # expand data to insert Lyman lines
        # dataset must be monotonic
        if not (np.sort(dataset[0]) == dataset[0]).all():
            print("Warning: Dataset not monotonic.")
            print("I will sort it for you.")
            dataset = self.command_sort(dataset)
            if not (np.sort(dataset[0]) == dataset[0]).all():
                print("Error: something when wrong when sorting the array.")
                return 
        
        XLAMMIN = min(dataset[0])
        XLAMMAX = max(dataset[0])
        
        #insert also datapoitns between line centers
        XINSOLD = dataset[0][0]
        
        for I in range(len(XLAM)-1,-1,-1):
            DLAMBDA = 1.0 / max(1, MAXNPOINTS) / (float(I+1) * float(I+1))
            for J in range(-MAXNPOINTS//2,MAXNPOINTS//2+1):
                XINS = XLAM[I] + np.sign(J) * J * J * DLAMBDA
                XDL1 = XLAMMIN - XINS
                XDR1 = XLAMMAX - XINS
                if XDL1 * XDR1 > 0:
                    continue
                dataset = self.command_insert(dataset, XINS)
                # At least one data point per line inserted, add another one for taucrit in between:
                if MAXNPOINTS == 0:
                    XMID = (XINSOLD + XINS) / 2.
                    dataset = self.command_insert(dataset, XMID)
                    XINSOLD = XINS
                    
        # insert two points around the edge
        if ((XLYEDGE - dataset[0][0])*(XLYEDGE-dataset[0][-1]) < 0):
            dataset = self.command_insert(dataset, XLYEDGE-0.001)
            dataset = self.command_insert(dataset, XLYEDGE+0.001)
            
        ###### multiplication with absorption profile        
        # initialize tau criterion
        TAUOLD = 1e-40
        
        tau = []
        xdata = []
        ydata = []
        
        I = 0
        while I < len(dataset[0]):
            TAU = 0.0
            
            if(TEMP > 0. or VTURB > 0.):
                # Temperature given --> voigt profile
                for LY in range(len(XLAM)):
                    VDOPP = np.sqrt(2.*XKBOLTZ*TEMP/(XMP*ATMASSH)+VTURB*VTURB)
                    DNDOPP = VDOPP /(XLAM[LY]*1e-8)
                    # Constant A for normalized voight function
                    A = GAMM[LY] / (4.0*np.pi*DNDOPP)
                    # variable V for normalized voigt
                    DLAM = dataset[0][I] - XLAM[LY]
                    DNUE = CLIGHT*1e13*DLAM/(XLAM[LY]*XLAM[LY])
                    V = DNUE / DNDOPP
                    HH = self.voigth(a=A,v=V)
                    # cross section = (pi e^2)/mc * f * voigth / DNDOPP
                    SIGMACL = 0.026537
                    FAK = SIGMACL * FVALUE[LY] / DNDOPP
                    TAU = TAU + FAK * HH * ENH
                
                V = 0.0
                TAULYMEDGE = FAK * self.voigth(a=A,v=V) * ENH
            else:
                # no or neg. Temperature given --> Lorentz profile
                for LY in range(len(XLAM)):
                    DLAM = dataset[0][I]-XLAM[LY]
                    TAU = TAU + ENH * C0[LY] / (1.0+C1[LY]*DLAM*DLAM)
                TAULYMEDGE = ENH * C0[LY]
            
            if TAUOLD == 0:
                print("Tau = 0")
                
            if I > 0 and TAUCRIT > 0 and abs((TAU - TAUOLD)/TAUOLD) > TAUCRIT and   abs(dataset[0][I]-dataset[0][I-1])>0.001:
                XINS = (dataset[0][I]+dataset[0][I-1])/2.
                dataset = self.command_insert(dataset,XINS)
                XINS = (dataset[0][I]+dataset[0][I+1])/2.
                dataset = self.command_insert(dataset,XINS)
                I = I-1
                continue
                
            TAUOLD = TAU
            tau.append(TAU)
            xdata.append(dataset[0][I])
            ydata.append(dataset[1][I])
            I = I +1
            
        for i,tt in enumerate(tau):
            if (tt < 10.):
                # Prevent floating overflow for positive tau (dereddening)
                ydata[i] = ydata[i]*np.exp(tt)
            else:
                ydata[i] = 0.0
                print("Warning: In Lyman alpha correction tau is ", TAU, ">10. Hence, the flux is set to 0.0")
            
        dataset = np.array([xdata,ydata])
                
        if TAUCRIT > 0 and MAXNPOINTS != 0:
            print("HLYMANA: Dataset expanded to ",len(dataset[0]), " datapoints because of TAUCRIT and MAXNPOINTS.")
        elif MAXNPOINTS == 0 and TAUCRIT > 0:
            print("HLYMANA: Dataset expanded to ",len(dataset[0]), " datapoints because of TAUCRIT.")
        elif MAXNPOINTS != 0 and TAUCRIT < 0:
            print("HLYMANA: Dataset expanded to ",len(dataset[0]), " datapoints because of MAXNPOINTS.")
        else:
            print("HLYMANA: Dataset expanded to ",len(dataset[0]), " Datapoints only for HLYMAN line center.")
            
        TAULYMEDGE = ENH * C0[0]
        EXPTAULYMEDGE = 0.0
        
        if TAULYMEDGE < 10.:
            EXPTAULYMEDGE = np.exp(TAULYMEDGE)
            
        for I in range(len(dataset[0])):
            if dataset[0][I] < XLAM[-1] and dataset[0][I] > XLYEDGE:
                dataset[1][I] = dataset[1][I]*EXPTAULYMEDGE
            
        return dataset

    def command_hlymana_fast(self, dataset, key="coldens", value=-1, vrad=0, temp=-1, vturb=0, info=False):
        """
        Apply corrections for interstellar absorption by hydrogen Lyman lines (from Lyα to Ly20).
        By default, HLYMAN uses the formalism from Groenewegen & Lamers (1989) which accounts only 
        for the damping (i.e. Lorentzian) profiles but no Doppler cores. However, if one of the 
        parameters T and/or VTURB is specified, HLYMAN applies Voigt profiles with the according 
        Doppler velocity.
        
        Parameters
        ----------
        dataset: np.array or list
            Dataset that should be modified.
            First column should have wavelenghts in Angstrom 
            Second column should be linear flux (i.e. in erg/s/cm^2/Ang)
        keyword: string
            Possible options: "EBV" or "ebv" or "COLDENS" or "coldens"
            If EBV is used, the column density is calculated using n_H = 3.7e21*EBV (Groenewegen & Lamers 1989)
        value: int, float
            The value of EBV or COLDENS.
        vrad: int, float
            The radial velocity in km/s
        temp: int, float
            Temperature in Kelvin; 
            if T > -1:
                Account for Doppler broadening with the thermal velocity
        vturb: int, float
            Turbulence velocity in km/s
            if vturb > -1:
                additional Doppler broadening
            
        Returns
        -------
        The modified dataset.

        Notes
        -------
        The dataset must be monotonic.
        """
        import numpy as np
    
        # --- constants and line data (same as your original) ---
        XLAM0 = np.float64([1215.671, 1025.722, 972.5366, 949.74297,
                 937.80344, 930.74824, 926.22571, 923.15037,
                 920.96313, 919.35146, 918.12939, 917.18061,
                 916.42917, 915.82388, 915.32904, 914.91936,
                 914.57629, 914.28622, 914.0386, 913.82569,
                 913.64126, 913.48036, 913.33922, 913.2147,
                 913.10431, 913.00597, 912.918, 912.83898,
                 912.76775, 912.70331, 912.64482])
    
        C0 = np.float64([7.056E-11, 4.428E-11, 3.788E-11, 3.522E-11,
              3.381E-11, 3.298E-11, 3.243E-11, 3.204E-11,
              3.177E-11, 3.156E-11, 3.141E-11, 3.129E-11,
              3.12E-11, 3.112E-11, 3.106E-11, 3.101E-11,
              3.097E-11, 3.093E-11, 3.09E-11, 3.087E-11,
              3.085E-11, 3.083E-11, 3.081E-11, 3.079E-11,
              3.078E-11, 3.077E-11, 3.076E-11, 3.075E-11,
              3.074E-11, 3.073E-11, 3.072E-11])
    
        C1 = np.float64([1.657E+09, 3.575E+10, 2.402E+11, 9.867E+11,
              3.059E+12, 7.871E+12, 1.773E+13, 3.628E+13,
              6.036E+13, 9.76E+13, 1.427E+14, 1.941E+14,
              2.505E+14, 3.106E+14, 3.734E+14, 4.381E+14,
              5.042E+14, 5.712E+14, 6.387E+14, 7.064E+14,
              7.743E+14, 8.421E+14, 9.098E+14, 9.773E+14,
              1.044E+15, 1.111E+15, 1.178E+15, 1.244E+15,
              1.31E+15, 1.376E+15, 1.441E+15])
    
        FVALUE = np.float64([4.1620E-01, 7.9100E-02, 2.8990E-02, 1.3940E-02,
                  7.7990E-03, 4.8140E-03, 3.1830E-03, 2.2160E-03,
                  1.6050E-03, 1.2010E-03, 9.2140E-04, 7.2270E-04,
                  5.7740E-04, 4.6860E-04, 3.8560E-04, 3.2110E-04,
                  2.7030E-04, 2.2960E-04, 1.9670E-04, 1.6980E-04,
                  1.4760E-04, 1.2910E-04, 1.1360E-04, 1.0050E-04,
                  8.9290E-05, 7.9700E-05, 7.1450E-05, 6.4300E-05,
                  5.8070E-05, 5.2610E-05, 4.7830E-05])
    
        GAMM = np.float64([6.26500E+08, 1.89700E+08, 8.12700E+07, 4.20400E+07,
                     2.45000E+07, 1.4043E+07, 9.3788E+06, 6.5742E+06,
                     4.7849E+06, 3.5931E+06, 2.7643E+06, 2.1728E+06,
                     1.7388E+06, 1.4131E+06, 1.1641E+06, 9.7030E+05,
                     8.1739E+05, 6.9477E+05, 5.9556E+05, 5.1437E+05,
                     4.4730E+05, 3.9138E+05, 3.4449E+05, 3.0485E+05,
                     2.7091E+05, 2.4188E+05, 2.1689E+05, 1.9522E+05,
                     1.7632E+05, 1.5978E+05, 1.4527E+05])
    
        C2 = 3.8e21
        CLIGHT = 299792.458  # km/s
    
        ATMASSH = 1.00794
        XKBOLTZ = 1.38065000e-16
        XMP = 1.67262160e-24
        XLYEDGE0 = 911.2670504
        SIGMACL = 0.026537
    
        # adaptive insertion params (same names as your code)
        TAUCRIT = 0.1
        MAXNPOINTS = 0
    
        # --- prepare dataset (ensure numpy arrays) ---
        lam = np.asarray(dataset[0], dtype=np.float64)
        flux = np.asarray(dataset[1], dtype=np.float64)
    
        # sort if not monotonic
        if not np.all(np.diff(lam) > 0):
            # minimal safe sort (faster than calling external command_sort)
            order = np.argsort(lam)
            lam = lam[order]
            flux = flux[order]
    
        lam_min = lam[0]
        lam_max = lam[-1]
    
        # shift lines and Lyman edge by radial velocity
        XLAM = XLAM0 * (1 + vrad / CLIGHT)
        XLYEDGE = XLYEDGE0 * (1 + vrad / CLIGHT)
    
        # compute ENH (column density)
        key_lower = key.lower()
        if key_lower == "ebv":
            ENH = C2 * value
        elif key_lower == "coldens":
            ENH = value
        else:
            print("Error: Unknown keyword in HLYMANA.")
            print("Please use 'coldens' or 'ebv'.")
            print("Returning unmodified dataset.")
            return dataset
    
        if ENH == 0.:
            return dataset
    
        # --- ensure line centers and Lyman edge points are present in grid ---
        # Add line centers that fall inside the dataset range
        centers_to_add = XLAM[(XLAM >= lam_min) & (XLAM <= lam_max)]
        if centers_to_add.size:
            lam = np.concatenate([lam, centers_to_add])
            flux = np.concatenate([flux, np.interp(centers_to_add, dataset[0], dataset[1])])
        # Add small points around Lyman edge if within range (like your original)
        if (XLYEDGE > lam_min) and (XLYEDGE < lam_max):
            edgepts = np.array([XLYEDGE - 0.001, XLYEDGE + 0.001])
            lam = np.concatenate([lam, edgepts])
            flux = np.concatenate([flux, np.interp(edgepts, dataset[0], dataset[1])])
    
        # sort after adding
        order = np.argsort(lam)
        lam = lam[order]
        flux = flux[order]
    
        # Precompute Voigt/Lorentz parameters for lines once
        VTURB_CGS = vturb * 1e5
        use_voigt = (temp > 0.) or (vturb > 0.)
        if use_voigt:
            VDOPP = np.sqrt(2.0 * XKBOLTZ * temp / (XMP * ATMASSH) + VTURB_CGS**2)  # cm/s
            DNDOPP = VDOPP / (XLAM * 1e-8)  # Hz
            A_arr = GAMM / (4.0 * np.pi * DNDOPP)
            FAK_arr = SIGMACL * FVALUE / DNDOPP
        else:
            # Lorentz: nothing to precompute beyond C0,C1 arrays above
            pass
    
        # --- function to compute tau on a wavelength array (vectorized over lam) ---
        def compute_tau(lam_array):
            # lam_array: 1D numpy array
            if use_voigt:
                tau_local = np.zeros_like(lam_array, dtype=np.float64)
                # loop over lines (31 iterations) -- acceptable
                for j in range(len(XLAM)):
                    DLAM = lam_array - XLAM[j]                       # angstrom
                    DNUE = CLIGHT * 1e13 * DLAM / (XLAM[j]**2)      # normalized freq diff
                    V = DNUE / DNDOPP[j]                            # dimensionless v
                    # call your voigt for whole vector V -> produce H(v)
                    # we build H vector by calling voigt per V (self.voigth expects scalars)
                    # but do it once per line (vector size = len(lam_array))
                    H = np.empty_like(V)
                    # using list comprehension is faster than Python loop for large V
                    # but to keep compatibility we explicitly fill via numpy-ish loop:
                    for k, vk in enumerate(V):
                        H[k] = self.voigth(a=A_arr[j], v=vk)
                    tau_local += FAK_arr[j] * H * ENH
                return tau_local
            else:
                # Lorentzian sum (vectorized)
                DLAM = lam_array[:, None] - XLAM[None, :]
                denom = 1.0 + C1[None, :] * (DLAM**2)
                sumterm = np.sum(C0[None, :] / denom, axis=1)
                return ENH * sumterm
    
        # --- Adaptive refinement loop (batch midpoint insertion) ---
        # We'll iteratively compute tau on the current lam grid and insert midpoints
        # where relative change in tau between adjacent points exceeds TAUCRIT.
        max_iter = 20  # safety cap to avoid infinite loops
        iter_count = 0
        lam_curr = lam
        flux_curr = flux
    
        while True:
            iter_count += 1
            tau_curr = compute_tau(lam_curr)
            # relative change between adjacent tau entries (use tiny floor to avoid div by zero)
            tau_left = tau_curr[:-1]
            tau_right = tau_curr[1:]
            denom = np.where(tau_left == 0.0, 1e-40, tau_left)
            rel_change = np.abs((tau_right - tau_left) / denom)
    
            # find indices where criterion triggers and spacing is large enough
            gaps = lam_curr[1:] - lam_curr[:-1]
            need_refine_mask = (rel_change > TAUCRIT) & (gaps > 0.001)
    
            if not np.any(need_refine_mask):
                break  # grid is sufficiently refined
    
            # Create midpoints for all segments that need refining (batch)
            indices = np.nonzero(need_refine_mask)[0]
            midpoints = (lam_curr[indices] + lam_curr[indices + 1]) * 0.5
    
            # Insert midpoints into lam_curr and interpolate flux_curr at those midpoints
            lam_new = np.concatenate([lam_curr, midpoints])
            flux_new = np.concatenate([flux_curr, np.interp(midpoints, lam_curr, flux_curr)])
            order = np.argsort(lam_new)
            lam_curr = lam_new[order]
            flux_curr = flux_new[order]
    
            # safety: avoid too many iterations
            if iter_count >= max_iter:
                if info:
                    print("HLYMANA: reached max refinement iterations.")
                break
    
        # After refinement, compute final tau & apply to flux
        tau_final = compute_tau(lam_curr)
    
        # Prevent floating overflow for positive tau (dereddening) as original code:
        flux_out = np.where(tau_final < 10.0, flux_curr * np.exp(tau_final), 0.0)
        for idx, tval in enumerate(tau_final):
            if tval >= 10.0:
                # Optionally warn once, matching original behavior (original warns per time >10)
                # But to avoid excessive printing, warn only if info True
                if info:
                    print("Warning: In Lyman alpha correction tau >= 10. Flux set to 0.0 at some points.")
                # zeroed already by np.where
    
        # Apply Lyman edge multiplicative factor for wavelengths between XLYEDGE and smallest XLAM
        TAULYMEDGE = ENH * C0[0]
        EXPTAULYMEDGE = np.exp(TAULYMEDGE) if TAULYMEDGE < 10.0 else 0.0
        # find indices that are between XLYEDGE and last XLAM (smallest XLAM value)
        lam_min_line = XLAM[-1]  # note: XLAM is sorted descending? but original used XLAM[-1], keep that semantics
        # apply factor for lam where lam < XLAM[-1] and lam > XLYEDGE
        mask = (lam_curr < XLAM[-1]) & (lam_curr > XLYEDGE)
        lam_curr[mask]  # no-op to emphasize use
        flux_out[mask] = flux_out[mask] * EXPTAULYMEDGE
    
        if info:
            print(f"HLYMANA: Dataset expanded to {len(lam_curr)} Datapoints after adaptive refinement (iters={iter_count}).")
    
        return np.array([lam_curr, flux_out])

    def command_hlymana_fast_v2(
        self,
        dataset,
        key="coldens",
        value=-1,
        vrad=0,
        temp=-1,
        vturb=0,
        info=False,
        cut_lyman=True,
        lyman_window=(900.0, 1300.0),
    ):
        """
        Faster rewrite of ``command_hlymana_fast``.

        Same physics and adaptive τ-refinement criterion, but:
          * Lorentzian / Voigt optical-depth sums are numba-compiled
          * refinement only evaluates τ at newly inserted midpoints
          * midpoints are merged in order (no full re-sort each iteration)
          * final τ is reused from the last refinement pass
          * by default only the Lyman window is processed, then stitched
            back onto the unmodified spectrum (see ``cut_lyman``)

        Parameters
        ----------
        cut_lyman : bool, default True
            If True, apply the correction only inside ``lyman_window`` and
            leave wavelengths outside that range unchanged. Set False to
            process the full input spectrum.
        lyman_window : tuple (lo, hi), default (900, 1300)
            Observed-frame wavelength window in Angstrom used when
            ``cut_lyman`` is True. Shifted with ``vrad`` like the lines.
        """
        XLAM0 = np.float64([1215.671, 1025.722, 972.5366, 949.74297,
                 937.80344, 930.74824, 926.22571, 923.15037,
                 920.96313, 919.35146, 918.12939, 917.18061,
                 916.42917, 915.82388, 915.32904, 914.91936,
                 914.57629, 914.28622, 914.0386, 913.82569,
                 913.64126, 913.48036, 913.33922, 913.2147,
                 913.10431, 913.00597, 912.918, 912.83898,
                 912.76775, 912.70331, 912.64482])

        C0 = np.float64([7.056E-11, 4.428E-11, 3.788E-11, 3.522E-11,
              3.381E-11, 3.298E-11, 3.243E-11, 3.204E-11,
              3.177E-11, 3.156E-11, 3.141E-11, 3.129E-11,
              3.12E-11, 3.112E-11, 3.106E-11, 3.101E-11,
              3.097E-11, 3.093E-11, 3.09E-11, 3.087E-11,
              3.085E-11, 3.083E-11, 3.081E-11, 3.079E-11,
              3.078E-11, 3.077E-11, 3.076E-11, 3.075E-11,
              3.074E-11, 3.073E-11, 3.072E-11])

        C1 = np.float64([1.657E+09, 3.575E+10, 2.402E+11, 9.867E+11,
              3.059E+12, 7.871E+12, 1.773E+13, 3.628E+13,
              6.036E+13, 9.76E+13, 1.427E+14, 1.941E+14,
              2.505E+14, 3.106E+14, 3.734E+14, 4.381E+14,
              5.042E+14, 5.712E+14, 6.387E+14, 7.064E+14,
              7.743E+14, 8.421E+14, 9.098E+14, 9.773E+14,
              1.044E+15, 1.111E+15, 1.178E+15, 1.244E+15,
              1.31E+15, 1.376E+15, 1.441E+15])

        FVALUE = np.float64([4.1620E-01, 7.9100E-02, 2.8990E-02, 1.3940E-02,
                  7.7990E-03, 4.8140E-03, 3.1830E-03, 2.2160E-03,
                  1.6050E-03, 1.2010E-03, 9.2140E-04, 7.2270E-04,
                  5.7740E-04, 4.6860E-04, 3.8560E-04, 3.2110E-04,
                  2.7030E-04, 2.2960E-04, 1.9670E-04, 1.6980E-04,
                  1.4760E-04, 1.2910E-04, 1.1360E-04, 1.0050E-04,
                  8.9290E-05, 7.9700E-05, 7.1450E-05, 6.4300E-05,
                  5.8070E-05, 5.2610E-05, 4.7830E-05])

        GAMM = np.float64([6.26500E+08, 1.89700E+08, 8.12700E+07, 4.20400E+07,
                     2.45000E+07, 1.4043E+07, 9.3788E+06, 6.5742E+06,
                     4.7849E+06, 3.5931E+06, 2.7643E+06, 2.1728E+06,
                     1.7388E+06, 1.4131E+06, 1.1641E+06, 9.7030E+05,
                     8.1739E+05, 6.9477E+05, 5.9556E+05, 5.1437E+05,
                     4.4730E+05, 3.9138E+05, 3.4449E+05, 3.0485E+05,
                     2.7091E+05, 2.4188E+05, 2.1689E+05, 1.9522E+05,
                     1.7632E+05, 1.5978E+05, 1.4527E+05])

        C2 = 3.8e21
        CLIGHT = 299792.458
        ATMASSH = 1.00794
        XKBOLTZ = 1.38065000e-16
        XMP = 1.67262160e-24
        XLYEDGE0 = 911.2670504
        SIGMACL = 0.026537
        TAUCRIT = 0.1
        MAX_ITER = 20

        lam_full = np.asarray(dataset[0], dtype=np.float64).copy()
        flux_full = np.asarray(dataset[1], dtype=np.float64).copy()

        if not np.all(np.diff(lam_full) > 0):
            order = np.argsort(lam_full)
            lam_full = lam_full[order]
            flux_full = flux_full[order]

        XLAM = XLAM0 * (1.0 + vrad / CLIGHT)
        XLYEDGE = XLYEDGE0 * (1.0 + vrad / CLIGHT)

        key_lower = key.lower()
        if key_lower == "ebv":
            ENH = C2 * value
        elif key_lower == "coldens":
            ENH = float(value)
        else:
            print("Error: Unknown keyword in HLYMANA.")
            print("Please use 'coldens' or 'ebv'.")
            print("Returning unmodified dataset.")
            return dataset

        if ENH == 0.0:
            return np.array([lam_full, flux_full])

        # Optional cut–apply–stitch: only process the Lyman window
        win_lo = float(lyman_window[0]) * (1.0 + vrad / CLIGHT)
        win_hi = float(lyman_window[1]) * (1.0 + vrad / CLIGHT)
        if cut_lyman:
            mid_mask = (lam_full >= win_lo) & (lam_full <= win_hi)
            if not np.any(mid_mask):
                if info:
                    print(
                        f"HLYMANA: cut_lyman window [{win_lo:.1f}, {win_hi:.1f}] "
                        "does not overlap spectrum; returning unmodified."
                    )
                return np.array([lam_full, flux_full])
            left_mask = lam_full < win_lo
            right_mask = lam_full > win_hi
            lam_left, flux_left = lam_full[left_mask], flux_full[left_mask]
            lam_right, flux_right = lam_full[right_mask], flux_full[right_mask]
            lam = lam_full[mid_mask].copy()
            flux = flux_full[mid_mask].copy()
        else:
            lam_left = flux_left = lam_right = flux_right = None
            lam = lam_full
            flux = flux_full

        lam_min = lam[0]
        lam_max = lam[-1]

        # Seed line centers / Lyman edge inside the processed segment
        centers_to_add = XLAM[(XLAM >= lam_min) & (XLAM <= lam_max)]
        extra_lam = []
        if centers_to_add.size:
            extra_lam.append(centers_to_add)
        if (XLYEDGE > lam_min) and (XLYEDGE < lam_max):
            extra_lam.append(np.array([XLYEDGE - 0.001, XLYEDGE + 0.001]))
        if extra_lam:
            extra = np.concatenate(extra_lam)
            flux_extra = np.interp(extra, lam, flux)
            lam = np.concatenate([lam, extra])
            flux = np.concatenate([flux, flux_extra])
            order = np.argsort(lam)
            lam = lam[order]
            flux = flux[order]

        use_voigt = (temp > 0.0) or (vturb > 0.0)
        if use_voigt:
            VTURB_CGS = vturb * 1e5
            VDOPP = np.sqrt(2.0 * XKBOLTZ * temp / (XMP * ATMASSH) + VTURB_CGS ** 2)
            DNDOPP = VDOPP / (XLAM * 1e-8)
            A_arr = GAMM / (4.0 * np.pi * DNDOPP)
            FAK_arr = SIGMACL * FVALUE / DNDOPP
            tau0 = _hlymana_tau_voigt(lam, XLAM, A_arr, FAK_arr, DNDOPP, ENH, CLIGHT)
        else:
            A_arr = np.empty(0)
            FAK_arr = np.empty(0)
            DNDOPP = np.empty(0)
            tau0 = _hlymana_tau_lorentz(lam, XLAM, C0, C1, ENH)

        lam_curr, flux_curr, tau_final, iter_count = _hlymana_adaptive_refine(
            lam, flux, tau0, XLAM, C0, C1, A_arr, FAK_arr, DNDOPP,
            ENH, CLIGHT, use_voigt, TAUCRIT, MAX_ITER,
        )

        if info and iter_count >= MAX_ITER:
            print("HLYMANA: reached max refinement iterations.")

        flux_out = np.where(tau_final < 10.0, flux_curr * np.exp(tau_final), 0.0)
        if info and np.any(tau_final >= 10.0):
            print("Warning: In Lyman alpha correction tau >= 10. Flux set to 0.0 at some points.")

        TAULYMEDGE = ENH * C0[0]
        EXPTAULYMEDGE = np.exp(TAULYMEDGE) if TAULYMEDGE < 10.0 else 0.0
        mask = (lam_curr < XLAM[-1]) & (lam_curr > XLYEDGE)
        flux_out[mask] *= EXPTAULYMEDGE

        if cut_lyman:
            parts_x = []
            parts_y = []
            if lam_left.size:
                parts_x.append(lam_left)
                parts_y.append(flux_left)
            parts_x.append(lam_curr)
            parts_y.append(flux_out)
            if lam_right.size:
                parts_x.append(lam_right)
                parts_y.append(flux_right)
            lam_curr = np.concatenate(parts_x)
            flux_out = np.concatenate(parts_y)

        if info:
            n_proc = int(np.sum((lam_curr >= win_lo) & (lam_curr <= win_hi))) if cut_lyman else len(lam_curr)
            print(
                f"HLYMANA: Dataset expanded to {len(lam_curr)} Datapoints "
                f"after adaptive refinement (iters={iter_count}"
                + (f", cut_lyman=[{win_lo:.1f},{win_hi:.1f}], processed≈{n_proc}" if cut_lyman else "")
                + ")."
            )

        return np.array([lam_curr, flux_out])

    def __polynom(self,X,COEFF):
        Y = COEFF[-1]
        for i in range(len(COEFF)-2,-1,-1):
            Y = Y*X+COEFF[i]
        return Y

    def __spline(self,XX, Y, X):
        """
        CUBIC INTERPOLATION WITH FIXED DERIVATIVES AT THE MESH POINTS
        THESE DERIVATIVES ARE CALCULATED BY LINEAR I5TERPOLATION BETWEEN
        NEIGHBOURING POINTS
            
            X(L) = GRID OF MESH POINTS
            Y(L) = FUNCTION TO BE INTERPOLATED
            P1,P2,P3,P4 = RESULTING COEFFICIENTS
            
        EVALUATION OF THE INTERPOLATED VALUE Z(X) : 
            
            FX = P1*(XX-X(L-1))**3 + P2*(XX-X(L-1)) + P3*(X(L)-XX)**3 + P4*(X(L)-XX)
                 
        THE COEFFICIENT MATRIX H HAS BEEN INVERTED ANALYTICALLY
        """
        # FIND INDEX INTERVAL (L, L-1) CONTAINING XX
        L = 1
        while (L < len(X)):
            if (XX-X[L-1])*(XX-X[L]) <= 0:
                break;
            L += 1

        # SET UP THE COEFFICIENT MATRIX
        D1 = 1. / (X[L] - X[L-1])
        D2 = D1 * D1
        D3 = D1 * D2
        D23 = D2 / 3.
        H11 = D3
        H12 = -D3
        H13 = D23
        H14 = 2. * D23
        H21 = -D1
        H22 = 2. * D1
        H23 = -0.333333333333333
        H24 = -0.666666666666666
        H31 = -D3
        H32 = D3
        H33 = -2. * D23
        H34 = -D23
        H41 = 2. * D1
        H42 = -D1
        H43 = 0.666666666666666
        H44 = 0.333333333333333
        
        # FOR THE BOUNDARY INTERVALS THE DERIVATIVE CANNOT EXTEND OVER THE BOUNDARY
        LA = max(L-2,0)
        LB = min(L+1,len(X)-1)

        # FUNCTION TO BE INTERPOLATED: ZRAY
        F1 = Y[L-1]
        F2 = Y[L]

        F3 = 0
        F4 = 0
        if True:
            F3 = (Y[L]-Y[LA]) / (X[L]-X[LA])
            F4 = (Y[LB]-Y[L-1]) / (X[LB]-X[L-1])
        else:
            # Neue Version wrh 16-May-2002 13:39:29
            # Statt der zentrierten Ableitung werden gewichtete Mittel der
            # Ableitungen der angrenzenden Intervalle genommen. Das entspricht
            # der Ableitung eines Parabel-Fits durch drei Punkte.
            FS0 = (Y[L] - Y[L-1]) / (X[L] - X[L-1])
            if (L == 1):
                F3 = FS0
            else:
                P = (X[L] - X[L-1]) / (X[L] - X[L-2])
                FSM = (Y[L-1] - Y[L-2]) / (X[L-1] - X[L-2])
                F3 = P * FSM + (1.-P) * FS0
            if (L == len(X)-1):
                F4 = FS0
            else:
                P = (X[L] - X[L-1]) / (X[L+1] - X[L-1])
                FSP = (Y[L+1] - Y[L]) / (X[L+1] - X[L])
                F4 = P * FSP + (1.-P) * FS0
        
        # SET TRUE FOR MONO OPTION
        # if False:
        #     S4 = ( Y[L] - Y[L-1] ) / ( X[L] - X[L-1] )
        #     # We are not in the first interval:
        #     if (LA != L-2):
        #         S3 = S4
        #     else
        #         S3 = ( Y[L-1] - Y[L-2] ) / ( X[L-1] - X[L-2] )
       
        #     # We are not in the last interval:
        #     if (LB != L+1):
        #         S5 = S4
        #     else:
        #         S5 = ( Y[L+1] - Y[L] ) / ( X[L+1] - X[L] )

        #     F3 = np.sign
        #     F3 = (SIGN(1.0,S3)+SIGN(1.0,S4))*MIN(ABS(S3),ABS(S4),0.5*ABS(F3))
        #     F4 = (SIGN(1.0,S4)+SIGN(1.0,S5))*MIN(ABS(S4),ABS(S5),0.5*ABS(F4))

        #  CALCULATE POLYNOMIAL COEFFICIENTS: P(VECTOR) = H(MATRIX) * F(VECTOR)
        P1 = H11*F1 + H12*F2 + H13*F3 + H14*F4
        P2 = H21*F1 + H22*F2 + H23*F3 + H24*F4
        P3 = H31*F1 + H32*F2 + H33*F3 + H34*F4 
        P4 = H41*F1 + H42*F2 + H43*F3 + H44*F4

        # EVALUATE THE INTERPOLATION POLYNOMIAL AT XX
        DLM = XX - X[L-1]
        DLM3 = DLM * DLM * DLM
        DL = X[L] - XX
        DL3 = DL * DL * DL
        return P1 * DLM3 + P2 * DLM + P3 * DL3 + P4 * DL
        
    def command_reddening(self,dataset,ebv,law="seaton",Rv=-1,info=False):
        """
        Apply corrections for interstellar reddening. 
        By default, HLYMAN uses the formalism from Groenewegen & Lamers (1989) which accounts only 
        for the damping (i.e. Lorentzian) profiles but no Doppler cores. However, if one of the 
        parameters T and/or VTURB is specified, HLYMAN applies Voigt profiles with the according 
        Doppler velocity.
        
        Parameters
        ----------
        dataset: np.array or list
            Dataset that should be modified.
            First column should be log wavelenghts in Angstrom 
            Second column should be log flux (i.e. in erg/s/cm^2/Ang)
        ebv: int, float
            Possible options: "EBV" or "ebv" or "COLDENS" or "coldens"
            If EBV is used, the column density is calculated using n_H = 3.7e21*EBV (Groenewegen & Lamers 1989)
        law: string
            Reddening law that should be applied. 
            Possible inputs: "SEATON", "seaton",            | Seaton (1979)
                             "CARDELLI", "cardelli",        | Cardelli et al. (1989)
                             "FITZPATRICK", "fitzpatrick"   | Fitzpatrick (1999)
                             "LMC", "lmc",                  | Howarth (1983)
                             "SMC", "smc",                  | Howarth (1983)
                             "SMC_BAR", "smc_bar"           | Gordon et al. (2003)
                             
        Rv: int, float
            RV = AV / EBV , where AV is the extinction in the V band.
            Typically set by the law. 

             "SEATON",            | Rv = 3.1     (Nandy et al. (1975))
             "CARDELLI",          | Rv = 3.1
             "FITZPATRICK",       | Rv = 3.1
             "LMC",               | Rv = 3.2
             "SMC",               | Rv = 2.7     (Bouchet et al. (1985))
             "SMC BAR",           | Rv = 2.74
            
            if Rv > -1:
                use user input.

                
        temp: int, float
            Temperature in Kelvin; 
            if T > -1:
                Account for Doppler broadening with the thermal velocity
        vturb: int, float
            Turbulence velocity in km/s
            if vturb > -1:
                additional Doppler broadening
            
        Returns
        -------
        The modified dataset.

        Notes:
        -------
        The dataset must be monotonic. 
        The dataset is automatically expanded to 1000 points if there are less.
        All reddening laws are extended towards the red end (outside their “range”) according to a table from Moneti et al. (2001), 
        above λ > 24 μm an extrapolation propto λ^(−2) is extended.
        In the X-ray regime (λ = 1.24 − 413 Ang) the absorption is calculated according to Table 2 of Morrison & McCammon (1983). 
        The edges in this wavelength range arise due to the K-shell absorption of different elements (a solar composition is assumed). 
        In the range λ = 413 − 504 Ang (the He I edge) these tabulated polynoms are extrapolated.
        In the range of λ = 504 − 911 Ang, i.e. between the Lyman edge of H and the He I edge, only the opacity of H I is considered 
        in the same way as in the PoWR code (including the Gaunt factor).
        """
             
        law = law.lower()
        if not (law == "cardelli" or law == "fitzpatrick" or law == "seaton" or law == "lmc" or law == "smc" or law == "smc_bar"):
            print("Error: Reddedning law not found.")
            print("Possible laws are: seaton, cardelli, fitzpatrick, lmc, smc, smc_bar")
            return

        # setting up default values for the different laws
        # since they are used sometimes for different ranges,
        # they all need to be defined here
        
        # LMC
        RLMC = 3.2
        # SMC
        RSMC = 2.7
        #FITZPATRICK
        Rfitz = 3.1        
        # for SMC_BAR: special law from Gordon et al. (2003) for < 3000 Ang 
        # for longer wavelength, the FITZPATRICK law is taken with RV=2.74
        RSMCBAR = 2.74
        
        if law == "cardelli":
            if Rv < 0:
                Rv = 3.1
        elif law == "fitzpatrick":
            if Rv > 0:
                Rfitz = Rv
        elif law == "smc":
            if Rv > 0:
                RSMC = Rv

        if (law == "seaton") and Rv > 0:
            print("Warning: Law 'seaton' is implemented with a fixed Rv = 3.1")
            print("Warning: Continuing with default value...")
        elif (law == "smc_bar") and Rv > 0:
            print("Warning: Law 'seaton' is implemented with a fixed Rv = 2.74")
            print("Warning: Continuing with default value...")
        elif (law == "lmc") and Rv > 0:
            print("Warning: Law 'lmc' is implemented with a fixed Rv = 3.2")
            print("If you want to change the Rv value here, use the law 'smc'.")
            print("Warning: Continuing with default value...")

        # dataset must be monotonic
        if not (np.sort(dataset[0]) == dataset[0]).all():
            print("Warning: Dataset not monotonic.")
            print("I will sort it for you.")
            dataset = self.command_sort(dataset)
            if not (np.sort(dataset[0]) == dataset[0]).all():
                print("Error: something when wrong when sorting the array.")
                return 
        
        # X-ray absorption for solar abundances
        # Morrison R. & McCammon D., 1983, ApJ 270, 119 Table 2
        EX = [0.030, 0.100, 0.284, 0.400, 0.532, 0.707, 0.867, 1.303, 
              1.840, 2.471, 3.210, 4.038, 7.111, 8.331, 10.00]
        CX0 = [17.3,  34.6,  78.1,  71.4,  95.5, 308.9, 120.6,
               141.3, 202.7, 342.7, 352.2, 433.9, 629.0, 701.2]
        CX1 = [608.1, 267.9,  18.8,  66.8, 145.8,-380.6, 169.3,
               146.8, 104.7,  18.7,  18.7,  -2.4,  30.9,  25.2]
        CX2 = [-2150.,-476.1,   4.3, -51.4, -61.1, 294.0, -47.7,
               -31.5, -17.0,   0.0,   0.0,  0.75,   0.0,   0.0]
 

        # THE DATA SET IS RESTRICTED TO THAT WAVELENGTH RANGE
        # FOR WHICH THE REDENING LAW IS IMPLEMENTED, 
        # AND TO THE LIMITS OF THE BOX
        # NOTE: JUST ONE POINT OUTSIDE EACH END OF THE INTERVAL MAY BE KEPT
        N = 1000
        while(len(dataset[0]) < N):
            # FIND LARGEST DISTANCE
            D1 = 0.
            XMID = 0.
            for i in range(len(dataset[0])-1):
                if abs(dataset[0][i]-dataset[0][i+1]) > D1:
                    D1 = abs(dataset[0][i]-dataset[0][i+1])
                    XMID = (dataset[0][i]+dataset[0][i+1])/2
            # ADD DATAPOINT USING LINEAR INTERPOLATION
            dataset = self.command_insert(dataset,XMID)

        # REDDENING CORRECTION

        for i in range(len(dataset[0])):
            # skip X-ray branch if lambda > 911.7634 Angstrom
            if dataset[0][i] <= 2.959882:
                # X-ray branch (same for all reddening keywords)
                # 1.0933705 = log lambda in Ang corresponding 1 keV (lambda = 12.39854 Ang)
                XLOGE =  1.0933705 - dataset[0][i]
                E = 10**XLOGE

                # no Reddening for wavelenghts shorter than tabulated range
                # --> leave flux unreddened
                if ( E <= EX[-1]):
                    
                    SIGMA = 0
                    # between the LYMAN edge and the HeI edge (911 - 504 Ang) we calculate
                    # the cross section like in the PoWR code
                    # i.e. hydrogenic * gaunt-bf
                    if ( E < EX[0] and dataset[0][i] > 2.70265):
                        XX = 10**dataset[0][i]/911.76
                        XINV = 1./XX

                        
                        # THE FOLLOWING DATA ARE FOR MIHALAS' FIT OF GAUNT-BF H I (Lyman-cont)
                        GAUNT = 1.2302628 + XX*(-0.50812150+XX*0.10631895) + (-3.1927214e-2+(8.9105122e-4+-1.1544111e-5*XINV)*XINV)*XINV
                        # Hydrogen cross section (classical, without gaunt-bf) = 7.92e-18
                        SIGMA = 7.92e-18 * XX * XX * XX * GAUNT
                    # between HeI edge and the tabulated X-ray law (504 - 413 Ang) the 
                    # tabulated law is extrapolated
                    elif ( E < EX[0] and dataset[0][i] <= 2.70265):
                        SIGMA = ((CX0[0] / E + CX1[0]) / E + CX2[0]) / E *1e-24
                    # for all other ranges, apply the tabulated absorption
                    else:
                        # find nearest neighbour in EX (it is sorted)
                        j = 0
                        dist = abs(E-EX[0])
                        while (True):
                            if abs(E-EX[j+1]) < dist:
                                dist = abs(E-EX[j+1])
                                j += 1
                            else:
                                break;
                        SIGMA = ((CX0[j] / E + CX1[j]) / E + CX2[j]) / E *1e-24

                    # opt. depth TAU = SIGMA * COLDENS (Hydrogen column density)
                    # COLDENS = ebv * 3.8e21
                    # 3.8e21 = 10. ** 21.58: = HYDROGEN COLUMN DENSITY FOR E(B-V)= 1 MAG (GROENEWEGEN & LAMERS, 1989)
                    # Warning: Bohlin et al. (1978: ApJ 224, 132) recommend 5.8E21
                    dataset[1][i] = dataset[1][i] - SIGMA * ebv * 3.8e21 * 0.4342944
            else:
                # non-X-ray branch
                # Z is in 1/mu-m
                Z = 10**(4.-dataset[0][i])
                
                F = 0

                # CARDELLI law  (CARDELLI ET AL. (APJ 345, 245))
                # Note: originally defined for lambda > 3.33mu
                # for longer lambda, the IR law is used
                # this limit was changed to 2.44mu (Z > 0.41) for smoother
                # transition to Moneti's law   
                if law == "cardelli" and Z >= 0.41:                        
                    ACAR = 0
                    BCAR = 0
                    XCAR = Z
                    
                    if XCAR < 3.3:
                        if XCAR < 1.1:
                            ACAR = 0.574 * XCAR**1.61
                            BCAR = -0.527 * XCAR**1.61
                        else:
                            YCAR = XCAR - 1.82
                            ACAR = self.__polynom(YCAR,[1., 0.17699, -0.50447, -0.02427, 0.72085, 0.01979, -0.77530, 0.32999])
                            BCAR = self.__polynom(YCAR,[0., 1.41338, 2.28305, 1.07233, -5.38434, -0.62251, 5.30260, -2.09002])
                    elif XCAR < 8.0:
                        FACAR = 0.
                        FBCAR = 0.
                        if XCAR >= 5.9:
                            FACAR = -0.04473 * (XCAR - 5.9)**2 - 0.009779 * (XCAR - 5.9)**3
                            FBCAR =  0.2130  * (XCAR - 5.9)**2 + 0.1207 * (XCAR - 5.9)**3
                            
                        ACAR = FACAR + 1.752 - 0.316*XCAR - 0.104/((XCAR-4.67)**2 + 0.341)
                        BCAR = FBCAR - 3.090 + 1.825*XCAR + 1.206/((XCAR-4.62)**2 + 0.263)
                    elif XCAR <= 10:
                        XXCAR = XCAR - 8
                        ACAR = self.__polynom(XXCAR,[-1.073, -0.628, 0.137, -0.070])
                        BCAR = self.__polynom(XXCAR,[13.670, 4.257, -0.420, 0.374])
                    else:
                        # bluemost formula is extrapolated
                        XXCAR = XCAR - 8
                        ACAR = self.__polynom(XXCAR,[-1.073, -0.628, 0.137, -0.070])
                        BCAR = self.__polynom(XXCAR,[13.670, 4.257, -0.420, 0.374])
                        
                    # NOTE : The Cardelli Function is 1. at V-Band and NOT 3.1 as the other formulae
                    F = Rv * ACAR + BCAR
                # Fitzpatrick-Law, adopted after his IDL-Routine... (FITZPATRICK AND MASSA (PASP 1999))
                # long-wave limit adopted at 5mu (Z=0.2)
                # -- this limit changed to 4mu (Z=0.25) for better transition to Moneti    
                elif (( law == "fitzpatrick" and Z > 0.25) or (law == "smc" and (Z < 1.83 and Z > 0.2)) or (law == 'smc_bar' and(Z <= 3.375 and Z > 0.2))):
                    if (law == "smc"):
                        Rfitz = RSMC
                    elif law == "smc_bar":
                        Rfitz = RSMCBAR
                    fitz_c2    = -0.824 + 4.717 / Rfitz
                    fitz_c1    =  2.030 - 3.007 * fitz_c2
                    if Z >= 3.703704:
                        yuv = fitz_c1 +  fitz_c2*Z + 3.23*Z**2/((Z**2-4.596**2)**2 + (Z*0.99)**2)
                        if Z > 5.9:
                            yuv=yuv+ 0.41*(0.5392*(Z-5.9)**2+0.05644*(Z-5.9)**3)
                        yuv = yuv + Rfitz
                        # Da Fitzpatrick und Massa in E(lambda-V)\E(B-V) rechnen, Cardelli jedoch in 
                        # A(lambda)\A(V) ist nach Cardelli et al die Umrechnung gegeben durch 
                        # A(lambda)\A(V) = FM(x)/R_V + 1 aber die wird hier nicht gebraucht.
                        F = yuv
                    elif Z < 3.703704:
                        # prepare the coefficients for the spline interpolation
                        yopir = np.zeros(9)
                        yopir[0] = 0.0
                        yopir[1] = 0.26469*Rfitz/3.1 # normed at R=3.1 in IR
                        yopir[2] = 0.82925*Rfitz/3.1 # normed at R=3.1 in IR
                        yopir[3] = -4.22809e-01 + 1.00270*Rfitz + 2.13572e-04*Rfitz**2
                        yopir[4] = -5.13540e-02 + 1.00216*Rfitz - 7.35778e-05*Rfitz**2
                        yopir[5] = 7.00127e-01  + 1.00184*Rfitz - 3.32598e-05*Rfitz**2
                        yopir[6] = 1.19456      + 1.01707*Rfitz - 5.46959e-03*Rfitz**2 + 7.97809e-04*Rfitz**3 - 4.45636e-05*Rfitz**4
                        yopir[7] = fitz_c1 + fitz_c2*3.703704  + (3.23*3.703704 **2/((3.703704 **2-4.596**2)**2 + (3.703704 *0.99)**2)) + Rfitz
                        yopir[8] = fitz_c1 + fitz_c2*3.846154 + (3.23*3.846154**2/((3.846154**2-4.596**2)**2 + (3.846154*0.99)**2)) + Rfitz
                        F = self.__spline(Z,yopir,[0.0, 0.377358, 0.819672, 1.666667, 1.828153, 2.141328, 2.433090, 3.703704, 3.846154])
                # LMC law + SMC law
                # Note: for lambda > 5464 Ang, the default (SEATON) law is used
                elif law == "lmc" and Z >= 1.83:
                    # Note: Ian's formula is bad for Lambda > 5464 Ang --> galactic law taken
                    # F = ((1.86 - 0.48 * Z) - 0.1) * Z
                    
                    # Lambda > 3636 Ang
                    if (Z <= 2.75):
                        ZZ = Z - 1.83
                        F = RLMC + ZZ * (2.04 + 0.094*ZZ)
                    # Lambda < 3636 Ang
                    else:
                        ZZ = Z -4.557
                        F = RLMC - 0.236 + Z * (0.462 + 0.105*Z) + 0.454/(ZZ*ZZ+0.293)
                elif law == "smc" and Z >= 1.83:
                    # Note: Ian's formula is bad for Lambda > 5464 Ang --> galactic law taken
                    # F = ((1.86 - 0.48 * Z) - 0.1) * Z
                    
                    # Lambda > 3636 Ang
                    if (Z <= 2.75):
                        ZZ = Z - 1.83
                        F = RSMC + ZZ * (2.04 + 0.094*ZZ)
                    # Lambda < 3636 Ang
                    else:
                        ZZ = Z -4.557
                        F = RSMC - 0.236 + Z * (0.462 + 0.105*Z) + 0.454/(ZZ*ZZ+0.293)
                        
                # new law for the SMC bar by Gordon et al. (2003)
                # Note: for lambda > 21980 Ang, the default (SEATON) law is used
                elif law == "smc_bar" and Z > 3.375:
                    FZ = 0
                    if (Z >= 5.9):
                        FZ = 0.5392 * (Z - 5.9)**2 + 0.05644 * (Z - 5.9)**3
                    F = RSMCBAR - 4.959 + 2.264*Z + 0.389*Z*Z/((Z*Z - 21.16)**2 + Z*Z) + 0.461*FZ
                
                # SEATON law (default)
                # Note: for lambda > 33333 Ang, the default law is used
                else:
                    # RANGE 1000 - 10000 ANGSTROEM 
                    if dataset[0][i] < 4:
                        if Z < 2.7:
                            # INTERPOLATION TABLE FOR PART OF SEATON'S RANGE
                            F = self.__spline(Z,[1.36,1.44,1.84,2.04,2.24,2.44,2.66,2.88,3.14,3.36,3.56,3.77,3.96,4.15,4.26,4.40,4.52,4.64], 
                                              [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7])
                            # F = -1000
                        elif Z <= 3.65:
                            F = 1.56+1.048*Z+1.01/((Z-4.6)**2+0.28)
                        elif Z <= 7.14:
                            F = 2.29+0.848*Z+1.01/((Z-4.6)**2+0.28)
                        else:
                            F = 16.17+Z*(0.2975*Z-3.2)
                    # redmost reddening law (Moneti or extrapolation)
                    # Since the Moneti Table refers A_V, we scale with R_V
                    else:
                        RVMONETI = 3.1
                        if law == "cardelli":
                            # the following enhancement of the deviation from RV=3.1
                            # is arbiitrary, just in order to make a smooth transition 
                            # of the Monati law at 2.44mu 
                            RVMONETI = 3.1 + (Rv - 3.1) * 1.4
                        elif law == "lmc":
                            RVMONETI = RLMC
                        elif law == "smc":
                            RVMONETI = RSMC
                        elif law == "smc_bar":
                            RVMONETI = RSMCBAR
                        # Range 10 000 - 240 000 ANG after Moneti
                        if dataset[0][i] < 5.380:
                            # Interpolation for IR after Moneti et al 2001: 10000-240000 ANG
                            F = RVMONETI*self.__spline(dataset[0][i],
                                                       [0.4097, 0.2804, 0.1736, 0.1108, 0.0964, 0.0923, 0.0715,                 
                                                        0.0530, 0.0510, 0.0492, 0.0477, 0.0464, 0.0452, 0.0441,
                                                        0.0431, 0.0463, 0.0628, 0.0989, 0.1454, 0.1431, 0.1250,
                                                        0.1023, 0.0835, 0.0721, 0.0634, 0.0569, 0.0545, 0.0553,
                                                        0.0561, 0.0573, 0.0589, 0.0612, 0.0643, 0.0677, 0.0707, 
                                                        0.0726, 0.0730, 0.0721, 0.0704, 0.0683, 0.0663, 0.0644,
                                                        0.0626, 0.0610, 0.0586, 0.0561, 0.0537, 0.0515],
                                                       [4.000, 4.097, 4.217, 4.342, 4.398, 4.477, 4.544, 4.602, 
                                                        4.653, 4.699, 4.740, 4.778, 4.813, 4.845, 4.875, 4.903, 
                                                        4.929, 4.954, 4.978, 5.000, 5.021, 5.041, 5.061, 5.079, 
                                                        5.097, 5.114, 5.130, 5.146, 5.161, 5.176, 5.190, 5.204,
                                                        5.217, 5.230, 5.243, 5.255, 5.267, 5.279, 5.290, 5.301,
                                                        5.312, 5.322, 5.332, 5.342, 5.352, 5.362, 5.371, 5.380])
                        # > 240 000 ANG: extrapolation proportional 1/Lambda^2
                        else:
                            F = RVMONETI * 0.0515 * 10**(2 * (5.380-dataset[0][i]))
                            
                dataset[1][i] -= 0.4 * F * ebv     
        
        return dataset

    def __plot_ident(self,text,id_x,id_xoff=0,id_y=1.05,id_len=0.05,id_fontsize=10,id_linesize=0.75,id_horizontal=False,color="black",bbox_alpha=0):
        """
        Plot line identifications in a pretty and convenient way.
        
        Parameters
        ----------
        text: str
            Text that should be displayed above the line.
        id_x: int, float
            Wavelenght of the line where the text should point to.
        id_xoff: int, float
            Offset of the text to the line identification. Often used to avoid that text overlaps.                 
        id_y: int, float
            Y value, where the line should start.
        id_len: int, float
            The lenght of the line.
        id_fontsize: int, float
            Fontsize of the text that should be displayed.
        id_linesize: int, float
            Width of the line.
        id_horitontal: bool
            Normally the text is shown vertically over the spectrum to save space.
            If True: Shows the text horizontally.
        bbox_alpha: float
            Decide if you want to have a white bounding box in the background for better readability.
            0 means transparent, 1 is fully opaque.

        Returns:
        -------
        Array containing: left and right boundary of the text box

        Notes:
        -------
        This is inspired by WRPLOT, but not the actual implementation.
        """

        # The line consists of 3 elements:
        # a straight line at the bottom, 
        # a potentially tilted line, if there is an offset 
        # and another straight line to the text 
        # I split them (2/10), (7/10), (3/10)
        plt.plot([id_x,id_x],[id_y,id_y+id_len*0.4],"-",color=color,lw=id_linesize)
        plt.plot([id_x,id_x+id_xoff],[id_y+id_len*0.4,id_y+id_len*0.5],"-",color=color,lw=id_linesize)
        plt.plot([id_x+id_xoff,id_x+id_xoff],[id_y+id_len*0.5,id_y+id_len],"-",color=color,lw=id_linesize)

        rot = 90
        if id_horizontal:
            rot = 0
        else:
            text_box = plt.text(id_x+id_xoff,id_y+id_len*1.4,text, rotation=rot,fontsize=id_fontsize,ha="center",color=color,
                                bbox=dict(facecolor='white', alpha = bbox_alpha,edgecolor='white',pad=0)
                               )

        # get text coordinates on figure
        r = plt.gca().get_figure().canvas.get_renderer()
        text_box = text_box.get_window_extent(renderer=r)

        # convert it to plot coordinates
        inv = plt.gca().transData.inverted()
        arr = inv.transform(text_box)

        # I only check if they collide in x direction 
        # returns 
        return [arr[0][0],arr[1][0]]

    def __plot_idmult(self,text,id_x,id_xoff=0,id_y=1.05,id_len=0.05,id_fontsize=10,id_linesize=0.75,id_horizontal=False,color="black",bbox_alpha=0):
        """
        Plot multiple line identifications in a pretty and convenient way.
        
        Parameters
        ----------
        text: str
            Text that should be displayed above the lines.
        id_x: np.array or list
            Array containing the wavelenghts of the lines where the text should point to.
        id_xoff: int, float
            Offset of the text to the line identification. Often used to avoid that text overlaps.                 
        id_y: int, float
            Y value, where the line should start.
        id_len: int, float
            The lenght of the line.
        id_fontsize: int, float
            Fontsize of the text that should be displayed.
        id_linesize: int, float
            Width of the line.
        id_horitontal: bool
            Normally the text is shown vertically over the spectrum to save space.
            If True: Shows the text horizontally.
        bbox_alpha: float
            Decide if you want to have a white bounding box in the background for better readability.
            0 means transparent, 1 is fully opaque.

        Returns:
        -------
        Array containing: left and right boundary of the text box

        Notes:
        -------
        This is inspired by WRPLOT, but not the actual implementation.
        """

        # The line consists of 3 elements:
        # a straight lines at the bottom, with one horizontal line connecting them
        # a potentially tilted (if there is an offset) line from the center of the dataset 
        # and another straight line to the text 
        # I split the heights in (2/10), (7/10), (3/10) ratios
        for x in id_x:
            plt.plot([x,x],[id_y,id_y+id_len*0.4],"-",color=color,lw=id_linesize)
        plt.plot([id_x[0],id_x[-1]],[id_y+id_len*0.4,id_y+id_len*0.4],"-",color=color,lw=id_linesize)
        center = (id_x[0]+id_x[-1])/2
        plt.plot([center,center+id_xoff],[id_y+id_len*0.4,id_y+id_len*0.5],"-",color=color,lw=id_linesize)
        plt.plot([center+id_xoff,center+id_xoff],[id_y+id_len*0.5,id_y+id_len],"-",color=color,lw=id_linesize)

        rot = 90
        if id_horizontal:
            rot = 0
        else:
            text_box = plt.text(center+id_xoff,id_y+id_len*1.4,text, rotation=rot,fontsize=id_fontsize,ha="center",color=color,
                                bbox=dict(facecolor='white', alpha = bbox_alpha,edgecolor='white',pad=0)
                               )

        # get text coordinates on figure
        r = plt.gca().get_figure().canvas.get_renderer()
        text_box = text_box.get_window_extent(renderer=r)

        # convert it to plot coordinates
        inv = plt.gca().transData.inverted()
        arr = inv.transform(text_box)

        # I only check if they collide in x direction 
        # returns 
        return [arr[0][0],arr[1][0]]

    def __remove_WRPLOT_special_characters(self, par):
        """
        Function that removes WRPLOT specific commands and characters from a string
        
        Parameters
        ----------
        par: str
            String that should be formatted.

        Notes:
        -------
        Many of the WRPLOT functions are lost here and the list might not be complete
        """
        # WRPLOT has many options to change the text, for now, I am ignoring them
        par = par.replace("&T","") #tiefstellen
        par = par.replace("&H","") #hochstellen
        par = par.replace("&M","") #zurück zu normal, middle vertical alignment
        par = par.replace("&E","") #compact letter spacing
        par = par.replace("&B","") #broad letter spacing
        par = par.replace("&I","") #italics
        par = par.replace("&N","") #normal font style
        par = par.replace("&R","") #inclination to the right
        par = par.replace("&L","") #inclination to the left
        par = par.replace("&G","") #uninclined font (gerade)
        par = par.replace("&F","") #boldface
        par = par.replace("&f","") #quit boldface
        par = par.replace("&W","") #quasi-bold-face
        par = par.replace("&0","") #change color
        par = par.replace("&1","") #change color
        par = par.replace("&2","") #change color
        par = par.replace("&3","") #change color
        par = par.replace("&4","") #change color
        par = par.replace("&5","") #change color
        par = par.replace("&6","") #change color
        par = par.replace("&7","") #change color
        par = par.replace("&8","") #change color
        par = par.replace("&9","") #change color
        par = par.replace("&PW","") #locally change the font to WRplot (only at the beginning of the string)
        par = par.replace("&PT","") #locally change the font to Times (only at the beginning of the string)
        par = par.replace("&PH","") #locally change the font to Times (only at the beginning of the string)
        par = par.replace('"',"") #locally change the font to Helvetica (only at the beginning of the string)

        #decode WRPLOT specific symbols to latex (list might be incomplete)
        par = par.replace(r"\o",r"\cdot") #filled multiplyer
        par = par.replace(r"\S",r"_\odot") #solar units
        par = par.replace(r"\A",r"\AA") #Angstrom
        par = par.replace(r"\M",r"\dot{M}") #Mdot
        par = par.replace(r"\8",r"_\infinity") #infinity subscript
        par = par.replace(r"\*",r"\ast") #asterix
        par = par.replace(r"\>",r"\rightarrow") #right arrow
        par = par.replace(r"\<",r"\leftarrow") #left arrow
        par = par.replace(r"\+",r"\pm") #plus-minus
        par = par.replace(r"&&",r"\&") #and
        par = par.replace(r"&a",r"ä") #ä
        par = par.replace(r"&o",r"ö") #ö
        par = par.replace(r"&u",r"ü") #ü
        par = par.replace(r"&A",r"Ä") #Ä
        par = par.replace(r"&O",r"Ö") #Ö
        par = par.replace(r"&U",r"Ü") #Ü
        par = par.replace(r"&s",r"ß") #ß

        #greek letters (not sure if all are included and how, only using the most common ones here=
        par = par.replace(r"#a#",r"\alpha") #alpha
        par = par.replace(r"#A#",r"\Alpha") #alpha
        par = par.replace(r"#b#",r"\beta") #beta
        par = par.replace(r"#B#",r"\Beta") #beta
        par = par.replace(r"#g#",r"\gamma") #gamma
        par = par.replace(r"#G#",r"\Gamma") #gamma
        par = par.replace(r"#e#",r"\epsilon") #epsilon
        par = par.replace(r"#E#",r"\Epsilon") #epsilon
        par = par.replace(r"#d#",r"\delta") #delta
        par = par.replace(r"#D#",r"\Delta") #delta
        par = par.replace(r"#z#",r"\zeta") #zeta
        par = par.replace(r"#Z#",r"\zeta") #zeta
        
        par = par.replace('\n','')

        return par

    def plot_idents(self, file, id_xoff=0,id_y=1.05,id_len=0.05,id_fontsize=12,id_linesize=0.75,id_horizontal=False,color="black",bbox_alpha=0):
        """
        Read and plot line identifications from a file.
        
        Parameters
        ----------
        file: str
            File containing the line identifications
        id_x: np.array or list
            Array containing the wavelenghts of the lines where the text should point to.
        id_xoff: int, float
            Offset of the text to the line identification. Often used to avoid that text overlaps.                 
        id_y: int, float
            Y value, where the line should start.
        id_len: int, float
            The lenght of the line.
        id_fontsize: int, float
            Fontsize of the text that should be displayed.
        id_linesize: int, float
            Width of the line.
        id_horitontal: bool
            Normally the text is shown vertically over the spectrum to save space.
            If True: Shows the text horizontally.
        bbox_alpha: float
            Decide if you want to have a white bounding box in the background for better readability.
            0 means transparent, 1 is fully opaque.

        Notes:
        -------
        This is inspired by WRPLOT, but not the actual implementation.
        """
        
        # get limits of current plot
        xmin, xmax = plt.gca().get_xlim()
        wavelenghts = []
        text = []
        id_ys = []
        default_id_y = id_y
        
        # load file containing line identifications
        with open(file, 'r') as f:
            # Read each line in the file
            for line in f:
                # ignore comments
                if ("*" in line):
                    continue
                # add line ident
                elif (r"\IDENT" in line):
                    # IDENT is always followed by a number and then the text
                    # spaces can be ignored
                    line = line.replace(r"\IDENT","")
                    # remove empty spaces at front
                    while line[0] == " ":
                        line = line[1:]
                    # now get the number
                    number = ""
                    while line[0] == "1" or line[0] == "2" or line[0] == "3" or line[0] == "4" or line[0] == "5" or line[0] == "6" or line[0] == "7" or line[0] == "8" or line[0] == "9" or line[0] == "0" or line[0] == ".":
                        number += line[0]
                        line = line[1:]
                    # remove idents that are not in plotting range
                    number = float(number)
                    if number > xmin and number < xmax:
                        wavelenghts.append(number)
                        # remove empty spaces at front
                        while line[0] == " ":
                            line = line[1:]
                        line = self.__remove_WRPLOT_special_characters(line)
                        text.append("$"+line+"$")
                        id_ys.append(id_y)
                # add idmult
                elif (r"\IDMULT" in line):
                    # IDMULT is always followed by a set of number and then the text
                    # spaces can be ignored
                    line = line.replace(r"\IDMULT","")
                    # remove empty spaces at front
                    while line[0] == " ":
                        line = line[1:]
                    numbers = []
                    number = ""
                    in_range = True
                    # continue adding numbers as long as no text is there
                    while (not (line[0].isalpha() or line[0] == '&' or line[0] == '\\' or line[0] == '"')):
                        # now get the number
                        number = ""
                        while line[0] == "1" or line[0] == "2" or line[0] == "3" or line[0] == "4" or line[0] == "5" or line[0] == "6" or line[0] == "7" or line[0] == "8" or line[0] == "9" or line[0] == "0" or line[0] == ".":
                            number += line[0]
                            line = line[1:]
                        number = float(number)
                        if number > xmin and number < xmax:
                            numbers.append(float(number))
                        # do not add, if idents are outside of plot
                        else:
                            in_range = False
                            
                        # remove empty spaces to next number or word
                        while line[0] == " ":
                            line = line[1:]
                    #check if numbers and text should be added
                    if in_range:
                        wavelenghts.append(numbers)
                        line = self.__remove_WRPLOT_special_characters(line)
                        text.append("$"+line+"$")
                        id_ys.append(id_y)
                elif (r"\IDY" in line):
                    # sometimes users define IDY externally with IDYNORM I replace it with my default
                    if (r"IDYNORM" in line):
                        id_y = default_id_y
                        continue
                    line = line.replace(r"\IDY","")
                    # for now I do not care if it is in cm or real units. Most of them are in "U"
                    line = line.replace(r"U","")
                    line = line.replace(r"cm","")
                    line = line.replace(r"=","")
                    line = line.replace('\n',"")
                    id_y = float(line)

        #no idents in plotting range
        if len(wavelenghts) == 0:
            return

        # get sizes of the textboxes such taht text is not overlapping
        text_boxes = []
        if isinstance(wavelenghts[0], float) or isinstance(wavelenghts[0], int):
            text_boxes.append(self.__plot_ident(text[0],wavelenghts[0],id_xoff=0,id_y=id_ys[0],id_len=id_len,id_fontsize=id_fontsize,id_linesize=id_linesize,id_horizontal=id_horizontal,color=color,bbox_alpha=bbox_alpha))
        else:
            text_boxes.append(self.__plot_idmult(text[0],wavelenghts[0],id_xoff=0,id_y=id_ys[0],id_len=id_len,id_fontsize=id_fontsize,id_linesize=id_linesize,id_horizontal=id_horizontal,color=color,bbox_alpha=bbox_alpha))
        
        # assume that we only use single lined idents
        width = abs(text_boxes[0][0]-text_boxes[0][1])
        for i in range(1,len(wavelenghts)):
            #check if the line is left of the right boundary of the latest (most right) text 
            # if True --> need to shift!
            offs = 0
                
            if isinstance(wavelenghts[i], float) or isinstance(wavelenghts[i], int):
                # print(wavelenghts[i], text_boxes[-1][1]+width/2)
                if wavelenghts[i] < text_boxes[-1][1]+width/2:
                    # print(text[i])
                    offs = text_boxes[-1][1]-wavelenghts[i] + width
                # avoid overflow of plotting box
                if wavelenghts[i]+offs < xmax:
                    text_boxes.append(self.__plot_ident(text[i],wavelenghts[i],id_xoff=offs,id_y=id_ys[i],id_len=id_len,id_fontsize=id_fontsize,id_linesize=id_linesize,id_horizontal=id_horizontal,color=color,bbox_alpha=bbox_alpha))    
            else:
                center = (wavelenghts[i][0]+wavelenghts[i][-1])/2
                # print(center, text_boxes[-1][1]+width/2)
                if center < text_boxes[-1][1]+width/2:
                    # print(text[i])
                    offs = text_boxes[-1][1]-center + width
                # avoid overflow of plotting box
                if center+offs < xmax:
                    text_boxes.append(self.__plot_idmult(text[i],wavelenghts[i],id_xoff=offs,id_y=id_ys[i],id_len=id_len,id_fontsize=id_fontsize,id_linesize=id_linesize,id_horizontal=id_horizontal,color=color,bbox_alpha=bbox_alpha))
    
    def command_ismline(self,dataset,coldens,l0,fvalue,gamm,atmass=None,vrad=0.0,temp=-1.0,vturb=0.0,expand=True,window=5.0,info=False):
        """
        Apply a single ISM absorption-line correction.
    
        Parameters
        ----------
        dataset : [wave, flux]
        coldens : float
            Column density [cm^-2].
    
            WRPLOT convention:
            coldens < 0 => absorption
            coldens > 0 => emission-like correction
    
        l0 : float
            Rest wavelength [Angstrom]
    
        fvalue : float
            Oscillator strength
    
        gamm : float
            Damping constant [s^-1]
    
        atmass : float
            Atomic mass [amu]
    
        vrad : float
            Radial velocity [km/s]
    
        temp : float
            Temperature [K]
    
        vturb : float
            Turbulent velocity [km/s]
    
        window : float
            Region around line center where the profile is evaluated.
        """
    
        # ------------------------------------------------------------------
        # constants
        # ------------------------------------------------------------------
    
        CLIGHT = 299792.458
    
        XKBOLTZ = 1.38065000e-16
        CLCM    = 2.99792458e10
        ECHAR   = 4.80325000e-10
        XME     = 9.10938190e-28
        XMP     = 1.67262160e-24
    
        PI = np.pi
        QPI = 1.0 / PI
        QC = 1.0 / CLCM
    
        FRBB = PI * ECHAR * ECHAR / (XME * CLCM)
    
        # ------------------------------------------------------------------
        # sanity checks
        # ------------------------------------------------------------------
    
        if fvalue <= 0:
            raise ValueError("FVALUE must be > 0")
    
        if gamm <= 0:
            raise ValueError("GAMM must be > 0")
    
        if l0 <= 0:
            raise ValueError("L0 must be > 0")
    
        if coldens == 0:
            raise ValueError("COLDENS cannot be zero")
    
        if temp > 0 and (atmass is None or atmass <= 0):
            raise ValueError("ATMASS required when TEMP > 0")
    
        # ------------------------------------------------------------------
        # sort spectrum if needed
        # ------------------------------------------------------------------
    
        if not np.all(np.diff(dataset[0]) > 0):
            dataset = self.command_sort(dataset)
    
        x = np.asarray(dataset[0], dtype=float)
        y = np.asarray(dataset[1], dtype=float)
    
        # ------------------------------------------------------------------
        # radial velocity shift
        # ------------------------------------------------------------------
    
        xlam = l0 * (1.0 + vrad / CLIGHT)
    
        # km/s -> cm/s
        vturb_cgs = vturb * 1.0e5
    
        # ------------------------------------------------------------------
        # expand wavelength grid
        # ------------------------------------------------------------------
    
        if expand:
    
            dlambda = 1.0e-4
    
            for j in range(-500, 501):
    
                xins = xlam + np.sign(j) * j * j * dlambda
    
                if abs(xins - xlam) <= window:
                    dataset = self.command_insert(dataset, xins)
    
            x = np.asarray(dataset[0], dtype=float)
            y = np.asarray(dataset[1], dtype=float)
    
            if info:
                print(f"ISMLINE: dataset expanded to {len(x)} points")
    
        # ------------------------------------------------------------------
        # local window
        # ------------------------------------------------------------------
    
        loc_mask = np.abs(x - xlam) <= window
        idx = np.where(loc_mask)[0]
    
        if len(idx) == 0:
            return np.array([x, y])
    
        # ------------------------------------------------------------------
        # initialize output
        # ------------------------------------------------------------------
    
        yout = y.copy()
    
        # ------------------------------------------------------------------
        # Voigt profile
        # ------------------------------------------------------------------
    
        if temp > 0 or vturb > 0:
    
            if temp > 0:
    
                vdopp = np.sqrt(
                    2.0 * XKBOLTZ * temp /
                    (XMP * atmass)
                    + vturb_cgs * vturb_cgs
                )
    
                dndopp = xlam * 1.0e-8 * QC * vdopp
    
            else:
    
                dndopp = xlam * 1.0e-8 * QC * vturb_cgs
    
            qdndopp = 1.0 / dndopp
    
            a = (
                0.25 * gamm * QPI *
                qdndopp *
                (xlam * 1.0e-8)**2 *
                QC
            )
    
            fak = (
                FRBB *
                fvalue *
                qdndopp *
                (xlam * 1.0e-8)**2 *
                QC
            )
    
            tau_min = 0.0
    
            for i in idx:
    
                dlam = x[i] - xlam
    
                v = abs(dlam) * 1.0e-8 * qdndopp
    
                hh = self.voigth(a, v)
    
                tau = fak * hh * coldens
    
                tau_min = min(tau_min, tau)
    
                if tau < 10.0:
                    yout[i] *= np.exp(tau)
                else:
                    yout[i] = 0.0
    
            if info:
                print("ISMLINE: tau_min =", tau_min)
    
        # ------------------------------------------------------------------
        # Lorentz profile
        # ------------------------------------------------------------------
    
        else:
    
            c0 = (
                4.0 * PI * ECHAR * ECHAR /
                (XME * CLCM * gamm)
            ) * fvalue
    
            c1 = (
                4.0 * PI * CLCM /
                ((xlam * 1.0e-8)**2 * gamm)
            )**2
    
            dlam = (x[idx] - xlam) * 1.0e-8
    
            tau = c0 * coldens / (1.0 + c1 * dlam * dlam)
    
            if info:
                print("ISMLINE: tau_min =", np.min(tau))
    
            good = tau < 10.0
    
            yout[idx[good]] *= np.exp(tau[good])
            yout[idx[~good]] = 0.0
    
        return np.array([x, yout])
        
    def command_ismline_fast(self,dataset,coldens,l0,fvalue,gamm,atmass=None,vrad=0.0,temp=-1.0,vturb=0.0,window=5.0,info=False):
        """
        Apply corrections for interstellar absorption by hydrogen Lyman lines (from Lyα to Ly20).
        By default, HLYMAN uses the formalism from Groenewegen & Lamers (1989) which accounts only 
        for the damping (i.e. Lorentzian) profiles but no Doppler cores. However, if one of the 
        parameters T and/or VTURB is specified, HLYMAN applies Voigt profiles with the according 
        Doppler velocity.
        
        Parameters
        ----------
        dataset: np.array or list
            Dataset that should be modified.
            First column should have wavelenghts in Angstrom 
            Second column should be linear flux (i.e. in erg/s/cm^2/Ang)
        coldens: int, float
            The value of the coloumn density.
        l0: float
            wavelenth of ISM line
        fvalue: float
            Oscillator strength
        gamm: float
            Damping factor
        atmass: float
            Atomic mass
        vrad: int, float
            The radial velocity in km/s
        temp: int, float
            Temperature in Kelvin; 
            if T > -1:
                Account for Doppler broadening with the thermal velocity
        vturb: int, float
            Turbulence velocity in km/s
            if vturb > -1:
                additional Doppler broadening
            
        Returns
        -------
        The modified dataset.

        Notes
        -------
        The dataset must be monotonic.
        """
        import numpy as np
    
        # --- constants and line data (same as your original) ---
    
        CLIGHT = 299792.458
    
        XKBOLTZ = 1.38065000e-16
        CLCM    = 2.99792458e10
        ECHAR   = 4.80325000e-10
        XME     = 9.10938190e-28
        XMP     = 1.67262160e-24
        SIGMACL = 0.026537

        XLAM = l0 * (1.0 + vrad / CLIGHT)
        XLAM0 = np.float64([XLAM])
    
        C0 = np.float64([4 * np.pi * ECHAR * ECHAR / ( XME * CLCM * gamm ) * fvalue])
    
        C1 = np.float64([(4 * np.pi * CLCM / (XLAM * 1e-8 * XLAM * 1e-8 * gamm))**2])
    
        FVALUE = np.float64([fvalue])
    
        GAMM = np.float64([gamm])
    
        ATMASSH = atmass
    
        # adaptive insertion params (same names as your code)
        TAUCRIT = 0.1
        MAXNPOINTS = 0
    
        # --- prepare dataset (ensure numpy arrays) ---
        lam = np.asarray(dataset[0], dtype=np.float64)
        flux = np.asarray(dataset[1], dtype=np.float64)
    
        # sort if not monotonic
        if not np.all(np.diff(lam) > 0):
            # minimal safe sort (faster than calling external command_sort)
            order = np.argsort(lam)
            lam = lam[order]
            flux = flux[order]
    
        lam_min = lam[0]
        lam_max = lam[-1]
    
        # compute ENH (column density)
        ENH = coldens
    
        # --- ensure line centers and Lyman edge points are present in grid ---
        # Add line centers that fall inside the dataset range
        # centers_to_add = XLAM[(XLAM >= lam_min) & (XLAM <= lam_max)]
        # if centers_to_add.size:
        if lam_min <= XLAM <= lam_max:
            lam = np.concatenate([lam, [XLAM]])
            flux = np.concatenate([flux, np.interp([XLAM], dataset[0], dataset[1])])
    
        # sort after adding
        order = np.argsort(lam)
        lam = lam[order]
        flux = flux[order]
    
        # Precompute Voigt/Lorentz parameters for lines once
        VTURB_CGS = vturb * 1e5
        use_voigt = (temp > 0.) or (vturb > 0.)
        if use_voigt:
            VDOPP = np.sqrt(2.0 * XKBOLTZ * temp / (XMP * ATMASSH) + VTURB_CGS**2)  # cm/s
            DNDOPP = VDOPP / (np.float64([XLAM]) * 1e-8)  # Hz
            A_arr = GAMM / (4.0 * np.pi * DNDOPP)
            FAK_arr = SIGMACL * FVALUE / DNDOPP
        else:
            # Lorentz: nothing to precompute beyond C0,C1 arrays above
            pass
    
        # --- function to compute tau on a wavelength array (vectorized over lam) ---
        def compute_tau(lam_array):
            # lam_array: 1D numpy array
            if use_voigt:
                tau_local = np.zeros_like(lam_array, dtype=np.float64)
                # loop over lines (31 iterations) -- acceptable
                for j in range(len([XLAM])):
                    DLAM = lam_array - [XLAM][j]                       # angstrom
                    DNUE = CLIGHT * 1e13 * DLAM / ([XLAM][j]**2)      # normalized freq diff
                    V = DNUE / DNDOPP[j]                            # dimensionless v
                    # call your voigt for whole vector V -> produce H(v)
                    # we build H vector by calling voigt per V (self.voigth expects scalars)
                    # but do it once per line (vector size = len(lam_array))
                    H = np.empty_like(V)
                    # using list comprehension is faster than Python loop for large V
                    # but to keep compatibility we explicitly fill via numpy-ish loop:
                    for k, vk in enumerate(V):
                        H[k] = self.voigth(a=A_arr[j], v=vk)
                    tau_local += FAK_arr[j] * H * ENH
                return tau_local
            else:
                # Lorentzian sum (vectorized)
                DLAM = lam_array[:, None] - XLAM[None, :]
                denom = 1.0 + C1[None, :] * (DLAM**2)
                sumterm = np.sum(C0[None, :] / denom, axis=1)
                return ENH * sumterm
    
        # --- Adaptive refinement loop (batch midpoint insertion) ---
        # We'll iteratively compute tau on the current lam grid and insert midpoints
        # where relative change in tau between adjacent points exceeds TAUCRIT.
        max_iter = 20  # safety cap to avoid infinite loops
        iter_count = 0
        lam_curr = lam
        flux_curr = flux
    
        while True:
            iter_count += 1
            tau_curr = compute_tau(lam_curr)
            # relative change between adjacent tau entries (use tiny floor to avoid div by zero)
            tau_left = tau_curr[:-1]
            tau_right = tau_curr[1:]
            denom = np.where(tau_left == 0.0, 1e-40, tau_left)
            rel_change = np.abs((tau_right - tau_left) / denom)
    
            # find indices where criterion triggers and spacing is large enough
            gaps = lam_curr[1:] - lam_curr[:-1]
            need_refine_mask = (rel_change > TAUCRIT) & (gaps > 0.001)
    
            if not np.any(need_refine_mask):
                break  # grid is sufficiently refined
    
            # Create midpoints for all segments that need refining (batch)
            indices = np.nonzero(need_refine_mask)[0]
            midpoints = (lam_curr[indices] + lam_curr[indices + 1]) * 0.5
    
            # Insert midpoints into lam_curr and interpolate flux_curr at those midpoints
            lam_new = np.concatenate([lam_curr, midpoints])
            flux_new = np.concatenate([flux_curr, np.interp(midpoints, lam_curr, flux_curr)])
            order = np.argsort(lam_new)
            lam_curr = lam_new[order]
            flux_curr = flux_new[order]
    
            # safety: avoid too many iterations
            if iter_count >= max_iter:
                if info:
                    print("HLYMANA: reached max refinement iterations.")
                break
    
        # After refinement, compute final tau & apply to flux
        tau_final = compute_tau(lam_curr)
    
        # Prevent floating overflow for positive tau (dereddening) as original code:
        flux_out = np.where(tau_final < 10.0, flux_curr * np.exp(tau_final), 0.0)
        for idx, tval in enumerate(tau_final):
            if tval >= 10.0:
                # Optionally warn once, matching original behavior (original warns per time >10)
                # But to avoid excessive printing, warn only if info True
                if info:
                    print("Warning: In Lyman alpha correction tau >= 10. Flux set to 0.0 at some points.")
                # zeroed already by np.where
    
        # # Apply Lyman edge multiplicative factor for wavelengths between XLYEDGE and smallest XLAM
        # TAULYMEDGE = ENH * C0[0]
        # EXPTAULYMEDGE = np.exp(TAULYMEDGE) if TAULYMEDGE < 10.0 else 0.0
        # # find indices that are between XLYEDGE and last XLAM (smallest XLAM value)
        # lam_min_line = XLAM[-1]  # note: XLAM is sorted descending? but original used XLAM[-1], keep that semantics
        # # apply factor for lam where lam < XLAM[-1] and lam > XLYEDGE
        # mask = (lam_curr < XLAM[-1]) & (lam_curr > XLYEDGE)
        # lam_curr[mask]  # no-op to emphasize use
        # flux_out[mask] = flux_out[mask] * EXPTAULYMEDGE
    
        if info:
            print(f"HLYMANA: Dataset expanded to {len(lam_curr)} Datapoints after adaptive refinement (iters={iter_count}).")
    
        return np.array([lam_curr, flux_out])