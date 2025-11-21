"""
Visualization utilities for boolean functions on 2D grids

Performance notes
-----------------
- The evaluators here call user-provided functions that often trigger heavy
    numerical work (e.g., solving evolutions and building joint distributions).
- For speed, both grid and curve evaluations support optional threaded
    parallelism via ``n_jobs`` using ``concurrent.futures.ThreadPoolExecutor``.
    Threads are chosen (vs processes) for robust use in notebooks and on Windows
    without pickling issues; NumPy/SciPy operations typically release the GIL so
    threads can still improve throughput.
"""

from typing import Callable, Iterable, Optional, Sequence, Tuple, List, Union
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba

__all__ = [
    "boolean_grid",
    "plot_boolean_region",
    "plot_multioutput_curves",
]

def _set_default_rcparams() -> None:
    """Apply project-wide default Matplotlib font/mathtext settings."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'sans'
    plt.rcParams['mathtext.it'] = 'sans:italic'
    plt.rcParams['mathtext.bf'] = 'sans:bold'

def _style_axes(ax: plt.Axes, *, x_label: Optional[str] = None, y_label: Optional[str] = None, title: Optional[str] = None) -> None:
    """Apply consistent axis labels, ticks, and optional title.

    - x_label/y_label default to 'x'/'y' when not provided
    - tick label size set to 15; axis label size set to 18
    - title is only set when provided (default: no title)
    """
    ax.set_xlabel(x_label if x_label is not None else 'x', fontsize=18)
    ax.set_ylabel(y_label if y_label is not None else 'y', fontsize=18)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    if title:
        ax.set_title(title)

def _auto_n_jobs() -> int:
    """Pick a conservative default thread count without extra deps.

    Heuristic:
    - If BLAS env vars suggest internal threading (>1), return 1 to avoid oversubscription.
    - Otherwise, use about half of logical CPUs minus one, capped at 16 and at least 1.
    """
    logical = os.cpu_count() or 1
    # Check common BLAS threading env vars
    blas_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    max_blas = 0
    for k in blas_keys:
        v = os.environ.get(k, "").strip()
        try:
            n = int(v)
            if n > max_blas:
                max_blas = n
        except Exception:
            pass
    if max_blas > 1:
        return 1
    base = max(1, logical // 2)
    return max(1, min(base - 1, 16))

def _resolve_n_jobs(n_jobs: Optional[int]) -> int:
    if n_jobs is None:
        return _auto_n_jobs()
    try:
        n = int(n_jobs)
    except Exception:
        return _auto_n_jobs()
    if n <= 0:
        return _auto_n_jobs()
    return n

def boolean_grid(func: Callable[[float, float], Union[bool, Sequence[bool]]],
                  x_range: Tuple[float, float],
                  y_range: Tuple[float, float],
                  n: int = 100,
                  *,
                  n_jobs: Optional[int] = None):
    """Evaluate a boolean function (single or multi-output) on an (x,y) grid.

    Parameters
    ----------
    func : Callable[[float, float], bool | Sequence[bool]]
        Function to evaluate. Can return a single bool or a sequence of bools.
    x_range : tuple of float
        (min, max) for the x-axis.
    y_range : tuple of float
        (min, max) for the y-axis.
    n : int, optional
        Number of grid points along each axis. Default is 100.
    n_jobs : int, optional
        Number of threads for parallel execution. If None, chosen automatically.

    Returns
    -------
    X_grid : ndarray
        Meshgrid X coordinates (n, n).
    Y_grid : ndarray
        Meshgrid Y coordinates (n, n).
    masks : ndarray
        Boolean mask(s). If func returns single bool, shape is (n, n).
        If func returns K bools, shape is (K, n, n).
    """
    x_min, x_max = map(float, x_range)
    y_min, y_max = map(float, y_range)
    n = int(n)
    x_vals = np.linspace(x_min, x_max, n)
    y_vals = np.linspace(y_min, y_max, n)
    Y_grid, X_grid = np.meshgrid(y_vals, x_vals)  # shape (n, n) each

    # Probe first point to determine arity
    first = func(float(X_grid.flat[0]), float(Y_grid.flat[0]))

    # Helper to evaluate all remaining points, optionally threaded
    nj = _resolve_n_jobs(n_jobs)

    def _get_coords():
        return [(float(X_grid.flat[i]), float(Y_grid.flat[i])) for i in range(1, X_grid.size)]

    def eval_points_single():
        mask = np.zeros((n, n), dtype=bool)
        mask.flat[0] = bool(first)
        if X_grid.size == 1:
            return mask
        coords = _get_coords()
        if int(nj) and nj > 1:
            n_workers = int(nj)
            # Chunking: split tasks to reduce ThreadPoolExecutor overhead
            n_chunks = n_workers * 4
            chunk_size = max(1, len(coords) // n_chunks)
            chunks = [coords[i:i + chunk_size] for i in range(0, len(coords), chunk_size)]
            print(f"Parallel execution: {len(chunks)} chunks (target {n_chunks}) on {n_workers} workers.")

            def _process_chunk(chunk):
                return [bool(func(x, y)) for (x, y) in chunk]

            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_process_chunk, chunk) for chunk in chunks]
                total = len(futures)
                completed = 0
                while completed < total:
                    completed = sum(1 for f in futures if f.done())
                    print(f"\rProgress: {completed}/{total} chunks", end="", flush=True)
                    time.sleep(0.1)
                print()
                chunk_results = [f.result() for f in futures]
            results = [item for sublist in chunk_results for item in sublist]
        else:
            results = [bool(func(x, y)) for (x, y) in coords]
        mask.flat[1:] = np.array(results, dtype=bool)
        return mask

    def eval_points_multi(K: int, first_vals: Sequence[bool]):
        masks = np.zeros((K, n, n), dtype=bool)
        for k in range(K):
            masks[k].flat[0] = bool(first_vals[k])
        if X_grid.size == 1:
            return masks
        coords = _get_coords()
        if int(nj) and nj > 1:
            n_workers = int(nj)
            # Chunking: split tasks to reduce ThreadPoolExecutor overhead
            n_chunks = n_workers * 4
            chunk_size = max(1, len(coords) // n_chunks)
            chunks = [coords[i:i + chunk_size] for i in range(0, len(coords), chunk_size)]
            print(f"Parallel execution: {len(chunks)} chunks (target {n_chunks}) on {n_workers} workers.")

            def _process_chunk(chunk):
                return [func(x, y) for (x, y) in chunk]

            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_process_chunk, chunk) for chunk in chunks]
                total = len(futures)
                completed = 0
                while completed < total:
                    completed = sum(1 for f in futures if f.done())
                    print(f"\rProgress: {completed}/{total} chunks", end="", flush=True)
                    time.sleep(0.1)
                print()
                chunk_results = [f.result() for f in futures]
            results = [item for sublist in chunk_results for item in sublist]
        else:
            results = [func(x, y) for (x, y) in coords]
        for idx, val in enumerate(results, start=1):
            for k in range(K):
                masks[k].flat[idx] = bool(val[k])
        return masks

    if isinstance(first, (list, tuple, np.ndarray)):
        outs = list(first)
        K = len(outs)
        masks = eval_points_multi(K, outs)
        return X_grid, Y_grid, masks
    else:
        mask = eval_points_single()
        return X_grid, Y_grid, mask

def plot_boolean_region(
    func: Callable[[float, float], Union[bool, Sequence[bool]]],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n: int = 100,
    *,
    n_jobs: Optional[int] = None,
    label: Optional[Union[str, Sequence[str]]] = None,
    color: Optional[Union[str, Sequence[str]]] = None,
    alpha: Union[float, Sequence[float]] = 0.4,
    mode: str = "overlay",
    ax: Optional[plt.Axes] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    save_data: Union[bool, str] = False,
):
    """Plot boolean region(s) for a single- or multi-output function.

    Parameters
    ----------
    func : Callable[[float, float], bool | Sequence[bool]]
        Function to evaluate. Can return a single bool or a sequence of bools.
    x_range : tuple of float
        (min, max) for the x-axis.
    y_range : tuple of float
        (min, max) for the y-axis.
    n : int, optional
        Number of grid points along each axis. Default is 100.
    n_jobs : int, optional
        Number of threads for parallel execution. If None, chosen automatically.
    label : str | Sequence[str], optional
        Label(s) for the region(s).
    color : str | Sequence[str], optional
        Color(s) for the region(s).
    alpha : float | Sequence[float], optional
        Transparency level(s). Default is 0.4.
    mode : {'overlay', 'separate', 'both'}, optional
        Plotting mode. 'overlay' draws all on one axes. 'separate' draws K figures.
        'both' does both. Default is 'overlay'.
    ax : matplotlib.axes.Axes, optional
        Target axes for overlay mode. If None, a new figure is created.
    x_label : str, optional
        Label for x-axis.
    y_label : str, optional
        Label for y-axis.
    title : str, optional
        Title for the plot(s).
    save_data : bool | str, optional
        If True, saves (X, Y, masks) to "{func.__name__}.npz".
        If a string, saves to that path. Default is False.

    Returns
    -------
    result : tuple or list
        - 'overlay': (fig, ax)
        - 'separate': list of (fig, ax)
        - 'both': (fig_overlay, ax_overlay, list of (fig, ax))
    """
    # Matplotlib style per project convention
    _set_default_rcparams()

    X_grid, Y_grid, masks = boolean_grid(func, x_range, y_range, n=n, n_jobs=n_jobs)

    if save_data:
        if isinstance(save_data, str):
            out_path = save_data
        else:
            out_path = getattr(func, "__name__", "boolean_region_data")
        
        if not out_path.endswith(".npz"):
            out_path += ".npz"
            
        # Ensure directory exists
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        np.savez_compressed(out_path, X=X_grid, Y=Y_grid, masks=masks)
        print(f"Boolean region data saved to: {os.path.abspath(out_path)}")

    # Normalize to multi-output representation
    if masks.ndim == 2:
        masks = masks[None, ...]
    K = masks.shape[0]

    # Normalize styling inputs
    def to_list(x, default, k):
        if x is None:
            return [default for _ in range(k)]
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x for _ in range(k)]

    labels = to_list(label, None, K)
    colors = to_list(color, None, K)
    alphas = to_list(alpha, 0.4, K)

    def draw_one(ax_obj, mk, lab, col, alp):
        # Pick color
        if col is None:
            col = next(ax_obj._get_lines.prop_cycler)['color']
        rgba = to_rgba(col)
        cmap = ListedColormap([(0, 0, 0, 0), (rgba[0], rgba[1], rgba[2], 1.0)])
        ax_obj.imshow(mk.astype(float),
                      origin='lower',
                      extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                      aspect='auto',
                      interpolation='nearest',
                      cmap=cmap,
                      alpha=float(alp))
        if lab:
            from matplotlib.patches import Patch
            proxy = Patch(facecolor=col, alpha=float(alp), label=lab)
            existing = ax_obj.get_legend()
            if existing is not None:
                old_handles = list(existing.legendHandles)
                old_labels = [t.get_text() for t in existing.texts]
            else:
                old_handles, old_labels = [], []
            ax_obj.legend(old_handles + [proxy], old_labels + [lab], loc='upper right', fontsize=15, labelspacing=0.35)
        # axis labels/ticks/title (title only when provided)
        _style_axes(ax_obj, x_label=x_label, y_label=y_label, title=None)

    mode = str(mode).lower()
    if mode not in ("overlay", "separate", "both"):
        raise ValueError("mode must be 'overlay', 'separate', or 'both'")

    results = None

    # Overlay plot
    if mode in ("overlay", "both"):
        if ax is None:
            fig_overlay, ax_overlay = plt.subplots(figsize=(6, 5))
        else:
            ax_overlay = ax
            fig_overlay = ax_overlay.figure
        for k in range(K):
            draw_one(ax_overlay, masks[k], labels[k], colors[k], alphas[k])
        ax_overlay.set_xlim(x_range)
        ax_overlay.set_ylim(y_range)
        # Optional title only if provided
        _style_axes(ax_overlay, x_label=x_label, y_label=y_label, title=title)
        if mode == "overlay":
            return fig_overlay, ax_overlay
        results = (fig_overlay, ax_overlay)

    # Separate plots
    if mode in ("separate", "both"):
        fig_list: List[plt.Figure] = []
        ax_list: List[plt.Axes] = []
        for k in range(K):
            fig_k, ax_k = plt.subplots(figsize=(5, 4))
            draw_one(ax_k, masks[k], labels[k], colors[k], alphas[k])
            ax_k.set_xlim(x_range)
            ax_k.set_ylim(y_range)
            # Optional per-axes title only if provided; otherwise none
            per_title = title if title is not None else None
            _style_axes(ax_k, x_label=x_label, y_label=y_label, title=per_title)
            fig_list.append(fig_k)
            ax_list.append(ax_k)
        if mode == "separate":
            return list(zip(fig_list, ax_list))
        else:
            return results + (list(zip(fig_list, ax_list)),)  # type: ignore


def plot_multioutput_curves(
    func: Callable[[float], Union[float, Sequence[float]]],
    x_values: np.ndarray,
    *,
    n_jobs: Optional[int] = None,
    label: Optional[Union[str, Sequence[str]]] = None,
    color: Optional[Union[str, Sequence[str]]] = None,
    linewidth: Union[float, Sequence[float]] = 1.5,
    ax: Optional[plt.Axes] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plot multiple y(x) curves returned by a single-input function on one axes.

    Parameters
    ----------
    func : Callable[[float], float | Sequence[float]]
        Function mapping a single float ``x`` to either a single float or a
        sequence of floats (multi-output). If single-output, one curve is drawn.
    x_values : np.ndarray
        1D array of x values at which to evaluate ``func``.
    n_jobs : int, optional
        Number of threads for parallel execution. If None, chosen automatically.
    label : str | Sequence[str], optional
        Label(s) for the curve(s); if multi-output, supply a list of length K.
    color : str | Sequence[str], optional
        Color(s) for the curve(s); if multi-output, supply a list of length K.
    linewidth : float | Sequence[float], optional
        Line width(s) for the curve(s). Defaults to ``1.5``.
    ax : matplotlib.axes.Axes, optional
        Target axes. If not provided, a new figure and axes are created.
    x_label : str, optional
        Label for x-axis.
    y_label : str, optional
        Label for y-axis.
    title : str, optional
        Title for the plot.

    Returns
    -------
    (fig, ax)
        The Matplotlib figure and axes containing the overlay plot.
    """
    x_values = np.asarray(x_values, dtype=float).ravel()
    if x_values.ndim != 1:
        raise ValueError("x_values must be a 1D array")

    # Probe first point to determine arity
    first = func(float(x_values[0]))

    nj = _resolve_n_jobs(n_jobs)

    def _get_xs():
        return [float(x) for x in x_values[1:]]

    if isinstance(first, (list, tuple, np.ndarray)):
        outs0 = list(first)
        K = len(outs0)
        Y = np.zeros((K, x_values.size), dtype=float)
        for k in range(K):
            Y[k, 0] = float(outs0[k])
        if x_values.size > 1:
            xs = _get_xs()
            if int(nj) and nj > 1:
                n_workers = int(nj)
                n_chunks = n_workers * 4
                chunk_size = max(1, len(xs) // n_chunks)
                chunks = [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]
                print(f"Parallel execution: {len(chunks)} chunks (target {n_chunks}) on {n_workers} workers.")

                def _process_chunk(chunk):
                    return [func(x) for x in chunk]

                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    chunk_results = list(ex.map(_process_chunk, chunks))
                results = [item for sublist in chunk_results for item in sublist]
            else:
                results = [func(x) for x in xs]
            for i, vals in enumerate(results, start=1):
                for k in range(K):
                    Y[k, i] = float(vals[k])
    else:
        K = 1
        Y = np.zeros((1, x_values.size), dtype=float)
        Y[0, 0] = float(first)
        if x_values.size > 1:
            xs = _get_xs()
            if int(nj) and nj > 1:
                n_workers = int(nj)
                n_chunks = n_workers * 4
                chunk_size = max(1, len(xs) // n_chunks)
                chunks = [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]
                print(f"Parallel execution: {len(chunks)} chunks (target {n_chunks}) on {n_workers} workers.")

                def _process_chunk(chunk):
                    return [func(x) for x in chunk]

                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    chunk_results = list(ex.map(_process_chunk, chunks))
                results = [item for sublist in chunk_results for item in sublist]
            else:
                results = [func(x) for x in xs]
            for i, val in enumerate(results, start=1):
                Y[0, i] = float(val)

    # Normalize styling inputs
    def to_list(x, default, k):
        if x is None:
            return [default for _ in range(k)]
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x for _ in range(k)]

    labels = to_list(label, None, K)
    colors = to_list(color, None, K)
    lws    = to_list(linewidth, 1.5, K)

    # Apply global style
    _set_default_rcparams()

    # Prepare axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    # Draw curves
    for k in range(K):
        col = colors[k]
        if col is None:
            # Fallback to Matplotlib's line color cycler
            try:
                col = ax._get_lines.get_next_color()
            except Exception:
                col = None
        ax.plot(x_values, Y[k], label=labels[k], color=col, linewidth=float(lws[k]))

    if any(lab is not None for lab in labels):
        ax.legend(loc='upper right', fontsize=15, labelspacing=0.35)
    _style_axes(ax, x_label=x_label, y_label=y_label, title=title)
    ax.grid(True, alpha=0.2)
    return fig, ax