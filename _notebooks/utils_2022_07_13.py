import numpy as np
import pickle
from scipy import optimize as opt
from matplotlib import pyplot as plt
import proplot as pplt
from ipywidgets import interactive
from ipywidgets import widgets


def max_indices(array):
    """Return indices of maximum element in array."""
    return np.unravel_index(np.argmax(array), array.shape)    


def copy_into_new_dim(array, shape, axis=-1, method='broadcast', copy=False):
    """Copy an array into one or more new dimensions.
    
    The 'broadcast' method is much faster since it works with views instead of copies. 
    See 'https://stackoverflow.com/questions/32171917/how-to-copy-a-2d-array-into-a-3rd-dimension-n-times'
    """
    if type(shape) in [int, np.int32, np.int64]:
        shape = (shape,)
    if method == 'repeat':
        for i in range(len(shape)):
            array = np.repeat(np.expand_dims(array, axis), shape[i], axis=axis)
        return array
    elif method == 'broadcast':
        if axis == 0:
            new_shape = shape + array.shape
        elif axis == -1:
            new_shape = array.shape + shape
        for _ in range(len(shape)):
            array = np.expand_dims(array, axis)
        if copy:
            return np.broadcast_to(array, new_shape).copy()
        else:
            return np.broadcast_to(array, new_shape)
    return None


def make_slice(n, axis=0, ind=0):
    """Return a slice index."""
    if type(axis) is int:
        axis = [axis]
    if type(ind) is int:
        ind = [ind]
    idx = n * [slice(None)]
    for k, i in zip(axis, ind):
        if i is None:
            continue
        idx[k] = slice(i[0], i[1]) if type(i) in [tuple, list, np.ndarray] else i
    return tuple(idx)


def project(array, axis=0):
    """Project array onto one or more axes."""
    if type(axis) is int:
        axis = [axis]
    axis_sum = tuple([i for i in range(array.ndim) if i not in axis])
    proj = np.sum(array, axis=axis_sum)
    # Handle out of order projection. Right now it just handles 2D, but
    # it should be extended to higher dimensions.
    if proj.ndim == 2 and axis[0] > axis[1]:
        proj = np.moveaxis(proj, 0, 1)
    return proj


def get_grid_coords(*xi, indexing='ij'):
    """Return array of shape (N, D), where N is the number of points on 
    the grid and D is the number of dimensions."""
    return np.vstack([X.ravel() for X in np.meshgrid(*xi, indexing=indexing)]).T


def dist_cov(f, coords):
    """Compute the distribution covariance matrix.
    
    Parameters
    ----------
    f : ndarray
        The distribution function (weights).
    coords : list[ndarray]
        List of coordinates along each axis of `H`. Can also
        provide meshgrid coordinates.
        
    Returns
    -------
    Sigma : ndarray, shape (n, n)
    means : ndarray, shape (n,)
    """
    if coords[0].ndim == 1:
        COORDS = np.meshgrid(*coords, indexing='ij')
    n = f.ndim
    f_sum = np.sum(f)
    if f_sum == 0:
        return np.zeros((n, n)), np.zeros((n,))
    means = np.array([np.average(C, weights=f) for C in COORDS])
    Sigma = np.zeros((n, n))
    for i in range(Sigma.shape[0]):
        for j in range(i + 1):
            X = COORDS[i] - means[i]
            Y = COORDS[j] - means[j]
            EX = np.sum(X * f) / f_sum
            EY = np.sum(Y * f) / f_sum
            EXY = np.sum(X * Y * f) / f_sum
            Sigma[i, j] = EXY - EX * EY
    Sigma = symmetrize(Sigma)
    return Sigma, means


def rms_ellipse_dims(sig_xx, sig_yy, sig_xy):
    """Return semi-axes and tilt angle of the RMS ellipse in the x-y plane.
    
    Parameters
    ----------
    sig_xx, sig_yy, sig_xy : float
        Covariance between x-x, y-y, x-y.
    
    Returns
    -------
    angle : float
        Tilt angle of the ellipse below the x axis (radians).
    cx, cy : float
        Semi-axes of the ellipse.
    """
    angle = -0.5 * np.arctan2(2 * sig_xy, sig_xx - sig_yy)
    sn, cs = np.sin(angle), np.cos(angle)
    cx = np.sqrt(abs(sig_xx*cs**2 + sig_yy*sn**2 - 2*sig_xy*sn*cs))
    cy = np.sqrt(abs(sig_xx*sn**2 + sig_yy*cs**2 + 2*sig_xy*sn*cs))
    return angle, cx, cy



def plot_image(image, x=None, y=None, ax=None, profx=False, profy=False, 
               prof_kws=None, frac_thresh=None, contour=False, contour_kws=None,
               return_mesh=False,
               **plot_kws):
    """Plot image with profiles overlayed.
    
    To do: clean up, add documentation.
    """
    log = 'norm' in plot_kws and plot_kws['norm'] == 'log'
    if log:
        if 'colorbar' in plot_kws and plot_kws['colorbar']:
            if 'colorbar_kw' not in plot_kws:
                plot_kws['colorbar_kw'] = dict()
            plot_kws['colorbar_kw']['formatter'] = 'log'
    if contour and contour_kws is None:
        contour_kws = dict()
        contour_kws.setdefault('color', 'white')
        contour_kws.setdefault('lw', 1.0)
        contour_kws.setdefault('alpha', 0.5)
    if x is None:
        x = np.arange(image.shape[0])
    if y is None:
        y = np.arange(image.shape[1])
    if x.ndim == 2:
        x = x.T
    if y.ndim == 2:
        y = y.T
    image_max = np.max(image)
    if frac_thresh is not None:
        floor = max(1e-12, frac_thresh * image_max)
        image[image < floor] = 0
    if log:
        # floor = 1e-12
        # if image_max > 0:
        #     floor = np.min(image[image > 0])
        # image = image + floor
        image = np.ma.masked_less_equal(image, 0)
    mesh = ax.pcolormesh(x, y, image.T, **plot_kws)
    if contour:
        ax.contour(x, y, image.T, **contour_kws)
    if profx or profy:
        if prof_kws is None:
            prof_kws = dict()
        prof_kws.setdefault('color', 'white')
        prof_kws.setdefault('lw', 1.0)
        prof_kws.setdefault('scale', 0.15)
        prof_kws.setdefault('kind', 'line')
        _prof_kws = prof_kws.copy()
        scale = _prof_kws.pop('scale')
        kind = _prof_kws.pop('kind')
        fx = np.sum(image, axis=1)
        fy = np.sum(image, axis=0)
        fx_max = np.max(fx)
        fy_max = np.max(fy)
        if fx_max > 0:
            fx = fx / fx_max
        if fy_max > 0:
            fy = fy / fy.max()
        x1 = x
        y1 = scale * np.abs(y[-1] - y[0]) * fx
        x2 = np.abs(x[-1] - x[0]) * scale * fy
        y1 = y1 + y[0]
        x2 = x2 + x[0]
        y2 = y
        for i, (x, y) in enumerate(zip([x1, x2], [y1, y2])):
            if i == 0 and not profx:
                continue
            if i == 1 and not profy:
                continue
            if kind == 'line':
                ax.plot(x, y, **_prof_kws)
            elif kind == 'bar':
                if i == 0:
                    ax.bar(x, y, **_prof_kws)
                else:
                    ax.barh(y, x, **_prof_kws)
            elif kind == 'step':
                ax.step(x, y, where='mid', **_prof_kws)
    if return_mesh:
        return ax, mesh
    else:
        return ax


def corner(
    image, 
    coords=None,
    labels=None, 
    diag_kind='line',
    frac_thresh=None,
    fig_kws=None, 
    diag_kws=None, 
    prof=False,
    prof_kws=None,
    return_fig=False,
    **plot_kws
):
    """Plot all 1D/2D projections in a matrix of subplots.
    
    To do: 
    
    Clean this up and merge with `scdist.tools.plotting.corner`, 
    which performs binning first. I believe in scdist I also found
    a nicer way to handle diag on/off. (One function that plots
    the off-diagonals, recieving axes as an argument.
    """
    n = image.ndim
    if labels is None:
        labels = n * ['']
    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault('figwidth', 1.5 * n)
    fig_kws.setdefault('aligny', True)
    if diag_kws is None:
        diag_kws = dict()
    diag_kws.setdefault('color', 'black')
    plot_kws.setdefault('ec', 'None')
    
    if coords is None:
        coords = [np.arange(s) for s in image.shape]
    
    if diag_kind is None or diag_kind.lower() == 'none':
        axes = _corner_nodiag(
            image, 
            coords=coords,
            labels=labels, 
            frac_thresh=frac_thresh,
            fig_kws=fig_kws, 
            prof=prof,
            prof_kws=prof_kws,
            return_fig=return_fig,
            **plot_kws
        )
        return axes
    
    fig, axes = pplt.subplots(
        nrows=n, ncols=n, sharex=1, sharey=1, 
        spanx=False, spany=False, **fig_kws
    )
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
            elif i == j:
                h = project(image, j)
                if diag_kind == 'line':
                    ax.plot(h, **diag_kws)
                elif diag_kind == 'bar':
                    ax.bar(h, **diag_kws)
                elif diag_kind == 'step':
                    ax.step(h, where='mid', **diag_kws)
            else:
                if prof == 'edges':
                    profx = i == n - 1
                    profy = j == 0
                else:
                    profx = profy = prof
                H = project(image, (j, i))
                plot_image(H, ax=ax, x=coords[j], y=coords[i],
                           profx=profx, profy=profy, prof_kws=prof_kws, 
                           frac_thresh=frac_thresh, **plot_kws)
    for ax, label in zip(axes[-1, :], labels):
        ax.format(xlabel=label)
    for ax, label in zip(axes[1:, 0], labels[1:]):
        ax.format(ylabel=label)
    for i in range(n):
        axes[:-1, i].format(xticklabels=[])
        if i > 0:
            axes[i, 1:].format(yticklabels=[])
    if return_fig:
        return fig, axes
    return axes


def _corner_nodiag(
    image, 
    coords=None,
    labels=None, 
    frac_thresh=None,
    fig_kws=None, 
    prof=False,
    prof_kws=None,
    return_fig=False,
    **plot_kws
):
    n = image.ndim
    if labels is None:
        labels = n * ['']
    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault('figwidth', 1.5 * (n - 1))
    fig_kws.setdefault('aligny', True)
    plot_kws.setdefault('ec', 'None')
    
    fig, axes = pplt.subplots(
        nrows=n-1, ncols=n-1, 
        spanx=False, spany=False, **fig_kws
    )
    for i in range(n - 1):
        for j in range(n - 1):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            if prof == 'edges':
                profy = j == 0
                profx = i == n - 2
            else:
                profx = profy = prof
            H = project(image, (j, i + 1))

            x = coords[j]
            y = coords[i + 1]
            if x.ndim > 1:
                axis = [k for k in range(x.ndim) if k not in (j, i + 1)]
                ind = len(axis) * [0]
                idx = make_slice(x.ndim, axis, ind)
                x = x[idx]
                y = y[idx]

            plot_image(H, ax=ax, x=x, y=y,
                       profx=profx, profy=profy, prof_kws=prof_kws, 
                       frac_thresh=frac_thresh, **plot_kws)
    for ax, label in zip(axes[-1, :], labels):
        ax.format(xlabel=label)
    for ax, label in zip(axes[:, 0], labels[1:]):
        ax.format(ylabel=label)
    if return_fig:
        return fig, axes
    return axes


def interactive_proj2d(
    f, 
    coords=None,
    default_ind=(0, 1),
    slider_type='int',  # {'int', 'range'}
    dims=None,
    units=None,
    prof_kws=None,
    cmaps=None,
    **plot_kws,
):
    """Interactive plot of 2D projection of distribution `f`.
    
    The distribution is projected onto the specified axes. Sliders provide the
    option to slice the distribution before projecting.
    
    Parameters
    ----------
    f : ndarray
        An n-dimensional array.
    coords : list[ndarray]
        Coordinate arrays along each dimension. A square grid is assumed.
    default_ind : (i, j)
        Default x and y index to plot.
    slider_type : {'int', 'range'}
        Whether to slice one index along the axis or a range of indices.
    dims : list[str], shape (n,)
        Dimension names.
    units : list[str], shape (n,)
        Dimension units.
    prof_kws : dict
        Key word arguments for 1D profile plots.
    cmaps : dict
    
    Returns
    -------
    gui : ipywidgets.widgets.interaction.interactive
        This widget can be displayed by calling `IPython.display.display(gui)`. 
    """
    n = f.ndim
    if coords is None:
        coords = [np.arange(f.shape[k]) for k in range(n)]
    
    if dims is None:
        dims = [f'x{i}' for i in range(1, n + 1)]
    if units is None:
        units = n * ['']
    dims_units = []
    for dim, unit in zip(dims, units):
        dims_units.append(f'{dim}' + f' [{unit}]' if unit != '' else '')
    dim_to_int = {dim: i for i, dim in enumerate(dims)}
    if prof_kws is None:
        prof_kws = dict()
    prof_kws.setdefault('lw', 1.0)
    prof_kws.setdefault('alpha', 0.5)
    prof_kws.setdefault('color', 'white')
    prof_kws.setdefault('scale', 0.14)
    if cmaps is None:
        cmaps = ['viridis', 'dusk_r', 'mono_r', 'plasma']
    plot_kws.setdefault('colorbar', True)
    plot_kws['prof_kws'] = prof_kws
    
    # Widgets
    cmap = widgets.Dropdown(options=cmaps, description='cmap')
    discrete = widgets.Checkbox(value=False, description='discrete')
    log = widgets.Checkbox(value=False, description='log')
    contour = widgets.Checkbox(value=False, description='contour')
    profiles = widgets.Checkbox(value=True, description='profiles')
    scale = widgets.FloatSlider(value=0.15, min=0.0, max=1.0, step=0.01, description='scale',
                                continuous_update=True)
    dim1 = widgets.Dropdown(options=dims, index=default_ind[0], description='dim 1')
    dim2 = widgets.Dropdown(options=dims, index=default_ind[1], description='dim 2')
    vmax = widgets.FloatSlider(value=1.0, min=0.0, max=1.0, step=0.01, description='vmax',
                               continuous_update=True)
    fix_vmax = widgets.Checkbox(value=False, description='fix vmax')
    
    # Sliders
    sliders, checks = [], []
    for k in range(n):
        if slider_type == 'int':
            slider = widgets.IntSlider(
                min=0, max=f.shape[k], value=f.shape[k]//2,
                description=dims[k], 
                continuous_update=True,
            )
        elif slider_type == 'range':
            slider = widgets.IntRangeSlider(
                value=(0, f.shape[k]), min=0, max=f.shape[k],
                description=dims[k], 
                continuous_update=True,
            )
        else:
            raise ValueError('Invalid `slider_type`.')
        slider.layout.display = 'none'
        sliders.append(slider)
        checks.append(widgets.Checkbox(description=f'slice {dims[k]}'))
        
    # Hide/show sliders.
    def hide(button):
        for k in range(n):
            # Hide elements for dimensions being plotted.
            valid = dims[k] not in (dim1.value, dim2.value)
            disp = None if valid else 'none'
            for element in [sliders[k], checks[k]]:
                element.layout.display = disp
            # Uncheck boxes for dimensions being plotted. 
            if not valid and checks[k].value:
                checks[k].value = False
            # Make sliders respond to check boxes.
            if not checks[k].value:
                sliders[k].layout.display = 'none'
        # Hide vmax slider if fix_vmax checkbox is not checked.
        vmax.layout.display = None if fix_vmax.value else 'none' 
                    
    for element in (dim1, dim2, *checks, fix_vmax):
        element.observe(hide, names='value')
    # Initial hide
    for k in range(n):
        if k in default_ind:
            checks[k].layout.display = 'none'
            sliders[k].layout.display = 'none'
    vmax.layout.display = 'none'
                
    # I don't know how else to do this.
    def _update3(
        cmap, log, profiles, fix_vmax, vmax,
        dim1, dim2, 
        check1, check2, check3,
        slider1, slider2, slider3,
    ):
        checks = [check1, check2, check3]
        sliders = [slider1, slider2, slider3]
        for dim, check in zip(dims, checks):
            if check and dim in (dim1, dim2):
                return
        return _plot_figure(dim1, dim2, checks, sliders, log, profiles, cmap, fix_vmax, vmax)

    def _update4(
        cmap, log, profiles, fix_vmax, vmax,
        dim1, dim2, 
        check1, check2, check3, check4, 
        slider1, slider2, slider3, slider4,
    ):
        checks = [check1, check2, check3, check4]
        sliders = [slider1, slider2, slider3, slider4]
        for dim, check in zip(dims, checks):
            if check and dim in (dim1, dim2):
                return
        return _plot_figure(dim1, dim2, checks, sliders, log, profiles, cmap, fix_vmax, vmax)

    def _update5(
        cmap, log, profiles, fix_vmax, vmax,
        dim1, dim2, 
        check1, check2, check3, check4, check5,
        slider1, slider2, slider3, slider4, slider5,
    ):
        checks = [check1, check2, check3, check4, check5]
        sliders = [slider1, slider2, slider3, slider4, slider5]
        for dim, check in zip(dims, checks):
            if check and dim in (dim1, dim2):
                return
        return _plot_figure(dim1, dim2, checks, sliders, log, profiles, cmap, fix_vmax, vmax)

    def _update6(
        cmap, log, profiles, fix_vmax, vmax,
        dim1, dim2, 
        check1, check2, check3, check4, check5, check6,
        slider1, slider2, slider3, slider4, slider5, slider6,
    ):
        checks = [check1, check2, check3, check4, check5, check6]
        sliders = [slider1, slider2, slider3, slider4, slider5, slider6]
        for dim, check in zip(dims, checks):
            if check and dim in (dim1, dim2):
                return
        return _plot_figure(dim1, dim2, checks, sliders, log, profiles, cmap, fix_vmax, vmax)

    update = {
        3: _update3,
        4: _update4,
        5: _update5,
        6: _update6,
    }[n]
    
    def _plot_figure(dim1, dim2, checks, sliders, log, profiles, cmap, fix_vmax, vmax):
        if (dim1 == dim2):
            return
        axis_view = [dim_to_int[dim] for dim in (dim1, dim2)]
        axis_slice = [dim_to_int[dim] for dim, check in zip(dims, checks) if check]
        ind = sliders
        for k in range(n):
            if type(ind[k]) is int:
                ind[k] = (ind[k], ind[k] + 1)
        ind = [ind[k] for k in axis_slice]
        H = f[make_slice(f.ndim, axis_slice, ind)]
        H = project(H, axis_view)
        plot_kws.update({
            'profx': profiles,
            'profy': profiles,
            'cmap': cmap,
            'norm': 'log' if log else None,
            'vmax': vmax if fix_vmax else None,
        })
        fig, ax = pplt.subplots()
        plot_image(H, x=coords[axis_view[0]], y=coords[axis_view[1]], ax=ax, **plot_kws)
        ax.format(xlabel=dims_units[axis_view[0]], ylabel=dims_units[axis_view[1]])
        plt.show()
        
    kws = dict()
    kws['dim1'] = dim1
    kws['dim2'] = dim2
    for i, check in enumerate(checks, start=1):
        kws[f'check{i}'] = check
    for i, slider in enumerate(sliders, start=1):
        kws[f'slider{i}'] = slider
    kws['log'] = log
    kws['profiles'] = profiles
    kws['fix_vmax'] = fix_vmax
    kws['vmax'] = vmax
    kws['cmap'] = cmap
    gui = interactive(update, **kws)
    return gui