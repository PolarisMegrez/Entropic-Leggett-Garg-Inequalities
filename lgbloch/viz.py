r"""lgbloch.viz: Visualization Tools for Leggett-Garg Inequality Analysis
=====================================================================
Provides visualization utilities for plotting boolean regions and multi-output curves used in Leggett-Garg inequality analysis. Includes parallel evaluation capabilities, data saving/loading, and multiple plotting modes. This module extends matplotlib with project-specific styling and provides efficient grid-based evaluation for parameter space exploration.

Public API
----------
- ``boolean_grid``: Evaluate boolean function on (x,y) grid with parallel execution.
- ``plot_boolean_region``: Plot boolean region(s) with customizable styling.
- ``replot_boolean_region``: Replot boolean region(s) from saved .npz data.
- ``plot_multioutput_curves``: Plot multiple curves from single-input function.

Notes
-----
- Parallelization: Supports both process-based (default, via joblib) and thread-based parallelism.
- Auto-detection: Automatically detects physical core count and respects OS limits (e.g., Windows 61-handle limit).
- Function support: Supports both single-output and multi-output boolean functions.
- Data persistence: Boolean region data can be saved to .npz files for later replotting.
- Styling: Implements project-wide default matplotlib styling (Arial font, custom math text).

"""

import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import joblib
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

__all__ = [
    "boolean_grid",
    "plot_boolean_region",
    "replot_boolean_region",
    "plot_multioutput_curves",
]


def _set_default_rcparams() -> None:
    """Apply project-wide default Matplotlib font/mathtext settings."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "sans"
    plt.rcParams["mathtext.it"] = "sans:italic"
    plt.rcParams["mathtext.bf"] = "sans:bold"
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 15


def _auto_n_jobs(backend: str = "threading") -> int:
    """Pick a default thread/process count based on backend.

    Heuristic:
    - If backend is 'loky' or 'multiprocessing':
      Try to use physical cores count (via psutil) if available, otherwise logical.
      On Windows, cap at 61 due to OS limitations.
    - If backend is 'threading':
      Use a conservative limit (half of logical CPUs, max 16) to avoid
      GIL contention and BLAS oversubscription.
    """
    # Try to get physical core count
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
    except ImportError:
        physical = None
        logical = os.cpu_count() or 1

    if physical is None:
        physical = logical

    if backend in ("loky", "multiprocessing"):
        # Aggressive for processes: prefer physical cores
        # On Windows, max_workers cannot exceed 61 due to WaitForMultipleObjects limit
        limit = 61 if os.name == "nt" else logical
        # Use physical cores as baseline, but respect Windows limit
        target = physical
        return max(1, min(target, limit))

    # Conservative for threads
    # Check common BLAS threading env vars
    blas_keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
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


def _resolve_n_jobs(n_jobs: int | None, backend: str = "threading") -> int:
    """Resolve number of jobs for parallel execution.

    Parameters
    ----------
    n_jobs : int or None
        Requested number of jobs. If None or invalid, uses auto-detected value.
    backend : str, optional
        Parallel backend ('loky', 'threading', etc.). Default is 'threading'.

    Returns
    -------
    int
        Number of jobs to use (always >= 1).

    """
    if n_jobs is None:
        return _auto_n_jobs(backend)
    try:
        n = int(n_jobs)
    except Exception:
        return _auto_n_jobs(backend)
    if n <= 0:
        return _auto_n_jobs(backend)
    return n


def boolean_grid(
    func: Callable[[float, float], bool | Sequence[bool]],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n: int = 100,
    *,
    n_jobs: int | None = None,
    backend: str = "auto",
):
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
        Number of threads/processes for parallel execution. If None, chosen automatically.
    backend : str, optional
        Parallel backend ('loky', 'threading', 'multiprocessing', 'auto').
        Default is 'auto' (which maps to 'loky' for process-based parallelism).

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
    if backend == "auto":
        backend = "loky"

    x_min, x_max = map(float, x_range)
    y_min, y_max = map(float, y_range)
    n = int(n)
    x_vals = np.linspace(x_min, x_max, n)
    y_vals = np.linspace(y_min, y_max, n)
    Y_grid, X_grid = np.meshgrid(y_vals, x_vals)

    # Probe first point to determine arity
    first = func(float(X_grid.flat[0]), float(Y_grid.flat[0]))

    # Helper to evaluate all remaining points, optionally threaded
    nj = _resolve_n_jobs(n_jobs, backend)

    def _get_coords():
        return [
            (float(X_grid.flat[i]), float(Y_grid.flat[i]))
            for i in range(1, X_grid.size)
        ]

    def _run_parallel(coords, process_chunk_func):
        """Common parallel execution logic."""
        n_workers = int(nj)
        # Chunking
        n_chunks = n_workers * 4
        chunk_size = max(1, len(coords) // n_chunks)
        chunks = [coords[i : i + chunk_size] for i in range(0, len(coords), chunk_size)]
        print(
            f"Parallel execution ({backend}): {len(chunks)} chunks on "
            f"{n_workers} workers."
        )

        if backend in ("loky", "multiprocessing"):
            # Process-based parallelism using joblib
            # We wrap the chunk processor to limit BLAS threads in each worker
            def _worker_wrapper(chunk):
                with threadpool_limits(limits=1, user_api="blas"):
                    return process_chunk_func(chunk)

            # verbose=5 provides a simple progress display to stderr
            chunk_results = joblib.Parallel(
                n_jobs=n_workers, backend=backend, verbose=5
            )(joblib.delayed(_worker_wrapper)(chunk) for chunk in chunks)
            return [item for sublist in chunk_results for item in sublist]

        else:
            # Thread-based parallelism using ThreadPoolExecutor (legacy/fallback)
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(process_chunk_func, chunk) for chunk in chunks]
                total = len(futures)
                completed = 0
                while completed < total:
                    completed = sum(1 for f in futures if f.done())
                    print(f"\rProgress: {completed}/{total} chunks", end="", flush=True)
                    time.sleep(0.1)
                print()
                chunk_results = [f.result() for f in futures]
            return [item for sublist in chunk_results for item in sublist]

    def eval_points_single():
        mask = np.zeros((n, n), dtype=bool)
        mask.flat[0] = bool(first)
        if X_grid.size == 1:
            return mask
        coords = _get_coords()
        if int(nj) and nj > 1:
            def _process_chunk(chunk):
                return [bool(func(x, y)) for (x, y) in chunk]

            results = _run_parallel(coords, _process_chunk)
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
            def _process_chunk(chunk):
                return [func(x, y) for (x, y) in chunk]

            results = _run_parallel(coords, _process_chunk)
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
    func: Callable[[float, float], bool | Sequence[bool]],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n: int = 100,
    *,
    n_jobs: int | None = None,
    backend: str = "auto",
    label: str | Sequence[str] | None = None,
    color: str | Sequence[str] | None = None,
    alpha: float | Sequence[float] = 0.4,
    linestyle: str | Sequence[str] | None = None,
    linewidth: float | Sequence[float] = 1.5,
    mode: str = "overlay",
    ax: plt.Axes | None = None,
    save_data: bool | str = False,
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
        Number of threads/processes for parallel execution. If None, chosen automatically.
    backend : str, optional
        Parallel backend ('loky', 'threading', 'multiprocessing', 'auto').
        Default is 'auto' (which maps to 'loky' for process-based parallelism).
    label : str | Sequence[str], optional
        Label(s) for the region(s).
    color : str | Sequence[str], optional
        Color(s) for the region(s).
    alpha : float | Sequence[float], optional
        Transparency level(s). Default is 0.4.
    linestyle : str | Sequence[str], optional
        Line style(s) for the region boundary. Defaults to ``None`` (no boundary).
    linewidth : float | Sequence[float], optional
        Line width(s) for the region boundary. Default is 1.5.
    mode : {'overlay', 'separate', 'both'}, optional
        Plotting mode. 'overlay' draws all on one axes. 'separate' draws K figures.
        'both' does both. Default is 'overlay'.
    ax : matplotlib.axes.Axes, optional
        Target axes for overlay mode. If None, a new figure is created.
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

    X_grid, Y_grid, masks = boolean_grid(
        func, x_range, y_range, n=n, n_jobs=n_jobs, backend=backend
    )

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
        print(f"Boolean region data saved to: {os.path.relpath(out_path)}")

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
    linestyles = to_list(linestyle, None, K)
    linewidths = to_list(linewidth, 1.5, K)

    def draw_one(ax_obj, mk, lab, col, alp, lst, lwd):
        # Pick color
        if col is None:
            col = next(ax_obj._get_lines.prop_cycler)["color"]

        # Convert mask to float for contourf
        Z_float = mk.astype(float)

        # Filled region using contourf (matching replot_from_data.ipynb)
        # levels=[0.5, 1.5] captures the '1's.
        ax_obj.contourf(
            X_grid, Y_grid, Z_float, levels=[0.5, 1.5], colors=[col], alpha=float(alp)
        )

        # Boundary contour if linestyle is provided
        if lst is not None:
            ax_obj.contour(
                X_grid,
                Y_grid,
                Z_float,
                levels=[0.5],
                colors=[col],
                linestyles=[lst],
                linewidths=[lwd],
            )

    mode = str(mode).lower()
    if mode not in ("overlay", "separate", "both"):
        raise ValueError("mode must be 'overlay', 'separate', or 'both'")

    results = None

    # Overlay plot
    if mode in ("overlay", "both"):
        if ax is None:
            fig_overlay, ax_overlay = plt.subplots(figsize=(7, 7))
        else:
            ax_overlay = ax
            fig_overlay = ax_overlay.figure

        for k in range(K):
            draw_one(
                ax_overlay,
                masks[k],
                labels[k],
                colors[k],
                alphas[k],
                linestyles[k],
                linewidths[k],
            )

        # Custom legend handling to match replot_from_data.ipynb
        # replace the default legend created by draw_one (if any) with a cleaner one
        if any(labels):
            from matplotlib.lines import Line2D

            legend_elements = []
            for k in range(K):
                if labels[k]:
                    # Use Line2D as in replot_from_data.ipynb
                    c = colors[k] if colors[k] is not None else "k"  # fallback
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            color=c,
                            lw=linewidths[k],
                            linestyle=linestyles[k],
                            label=labels[k],
                        )
                    )
            ax_overlay.legend(
                handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=15
            )

        ax_overlay.set_xlim(x_range)
        ax_overlay.set_ylim(y_range)
        if mode == "overlay":
            return fig_overlay, ax_overlay
        results = (fig_overlay, ax_overlay)

    # Separate plots
    if mode in ("separate", "both"):
        fig_list: list[plt.Figure] = []
        ax_list: list[plt.Axes] = []
        for k in range(K):
            fig_k, ax_k = plt.subplots(figsize=(5, 4))
            draw_one(
                ax_k,
                masks[k],
                labels[k],
                colors[k],
                alphas[k],
                linestyles[k],
                linewidths[k],
            )

            # Add legend for separate plots if label exists
            if labels[k]:
                from matplotlib.lines import Line2D

                c = colors[k] if colors[k] is not None else "k"
                le = Line2D(
                    [0],
                    [0],
                    color=c,
                    lw=linewidths[k],
                    linestyle=linestyles[k],
                    label=labels[k],
                )
                ax_k.legend(handles=[le], loc="upper right", fontsize=15)

            ax_k.set_xlim(x_range)
            ax_k.set_ylim(y_range)
            fig_list.append(fig_k)
            ax_list.append(ax_k)
        if mode == "separate":
            return list(zip(fig_list, ax_list, strict=True))
        else:
            return results + (list(zip(fig_list, ax_list, strict=True)),)  # type: ignore


def replot_boolean_region(
    filename: str,
    *,
    label: str | Sequence[str] | None = None,
    color: str | Sequence[str] | None = None,
    alpha: float | Sequence[float] = 0.4,
    linestyle: str | Sequence[str] | None = None,
    linewidth: float | Sequence[float] = 1.5,
    mode: str = "overlay",
    ax: plt.Axes | None = None,
):
    """Replot boolean region(s) from a saved .npz file.

    Parameters
    ----------
    filename : str
        Path to the .npz file containing 'X', 'Y', and 'masks' (or 'x', 'y', 'z').
    label : str | Sequence[str], optional
        Label(s) for the region(s).
    color : str | Sequence[str], optional
        Color(s) for the region(s).
    alpha : float | Sequence[float], optional
        Transparency level(s). Default is 0.4.
    linestyle : str | Sequence[str], optional
        Line style(s) for the region boundary.
    linewidth : float | Sequence[float], optional
        Line width(s) for the region boundary. Default is 1.5.
    mode : {'overlay', 'separate', 'both'}, optional
        Plotting mode. Default is 'overlay'.
    ax : matplotlib.axes.Axes, optional
        Target axes for overlay mode.

    Returns
    -------
    result : tuple or list
        Same as plot_boolean_region.

    """
    _set_default_rcparams()

    # Load data
    with np.load(filename) as data:
        if "x" in data and "y" in data and "z" in data:
            x = data["x"]
            y = data["y"]
            z = data["z"]
            if x.ndim == 1:
                X_grid, Y_grid = np.meshgrid(x, y)
            else:
                X_grid, Y_grid = x, y
            masks = z
        elif "X" in data and "Y" in data and "masks" in data:
            X_grid = data["X"]
            Y_grid = data["Y"]
            masks = data["masks"]
            if X_grid.ndim == 1:
                X_grid, Y_grid = np.meshgrid(X_grid, Y_grid)
        else:
            # Fallback
            keys = list(data.keys())
            v0 = data[keys[0]]
            v1 = data[keys[1]]
            masks = data[keys[2]]
            if v0.ndim == 2 and v1.ndim == 2:
                X_grid, Y_grid = v0, v1
            else:
                X_grid, Y_grid = np.meshgrid(v0, v1)

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
    linestyles = to_list(linestyle, None, K)
    linewidths = to_list(linewidth, 1.5, K)

    x_range = (X_grid.min(), X_grid.max())
    y_range = (Y_grid.min(), Y_grid.max())

    def draw_one(ax_obj, mk, lab, col, alp, lst, lwd):
        # Pick color
        if col is None:
            col = next(ax_obj._get_lines.prop_cycler)["color"]

        # Convert mask to float for contourf
        Z_float = mk.astype(float)

        # Filled region
        ax_obj.contourf(
            X_grid, Y_grid, Z_float, levels=[0.5, 1.5], colors=[col], alpha=float(alp)
        )

        # Boundary contour if linestyle is provided
        if lst is not None:
            ax_obj.contour(
                X_grid,
                Y_grid,
                Z_float,
                levels=[0.5],
                colors=[col],
                linestyles=[lst],
                linewidths=[lwd],
            )

    mode = str(mode).lower()
    if mode not in ("overlay", "separate", "both"):
        raise ValueError("mode must be 'overlay', 'separate', or 'both'")

    results = None

    # Overlay plot
    if mode in ("overlay", "both"):
        if ax is None:
            fig_overlay, ax_overlay = plt.subplots(figsize=(7, 7))
        else:
            ax_overlay = ax
            fig_overlay = ax_overlay.figure

        for k in range(K):
            draw_one(
                ax_overlay,
                masks[k],
                labels[k],
                colors[k],
                alphas[k],
                linestyles[k],
                linewidths[k],
            )

        # Custom legend
        if any(labels):
            from matplotlib.lines import Line2D

            legend_elements = []
            for k in range(K):
                if labels[k]:
                    c = colors[k] if colors[k] is not None else "k"
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            color=c,
                            lw=linewidths[k],
                            linestyle=linestyles[k],
                            label=labels[k],
                        )
                    )
            ax_overlay.legend(
                handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=15
            )

        ax_overlay.set_xlim(x_range)
        ax_overlay.set_ylim(y_range)

        if mode == "overlay":
            return fig_overlay, ax_overlay
        results = (fig_overlay, ax_overlay)

    # Separate plots
    if mode in ("separate", "both"):
        fig_list: list[plt.Figure] = []
        ax_list: list[plt.Axes] = []
        for k in range(K):
            fig_k, ax_k = plt.subplots(figsize=(5, 4))
            draw_one(
                ax_k,
                masks[k],
                labels[k],
                colors[k],
                alphas[k],
                linestyles[k],
                linewidths[k],
            )

            if labels[k]:
                from matplotlib.lines import Line2D

                c = colors[k] if colors[k] is not None else "k"
                le = Line2D(
                    [0],
                    [0],
                    color=c,
                    lw=linewidths[k],
                    linestyle=linestyles[k],
                    label=labels[k],
                )
                ax_k.legend(handles=[le], loc="upper right", fontsize=15)

            ax_k.set_xlim(x_range)
            ax_k.set_ylim(y_range)
            fig_list.append(fig_k)
            ax_list.append(ax_k)
        if mode == "separate":
            return list(zip(fig_list, ax_list, strict=True))
        else:
            return results + (list(zip(fig_list, ax_list, strict=True)),)  # type: ignore


def plot_multioutput_curves(
    func: Callable[[float], float | Sequence[float]],
    x_values: np.ndarray,
    *,
    n_jobs: int | None = None,
    backend: str = "auto",
    label: str | Sequence[str] | None = None,
    color: str | Sequence[str] | None = None,
    linewidth: float | Sequence[float] = 1.5,
    linestyle: str | Sequence[str] | None = None,
    ax: plt.Axes | None = None,
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
        Number of threads/processes for parallel execution. If None, chosen automatically.
    backend : str, optional
        Parallel backend ('loky', 'threading', 'multiprocessing', 'auto').
        Default is 'auto' (which maps to 'loky' for process-based parallelism).
    label : str | Sequence[str], optional
        Label(s) for the curve(s); if multi-output, supply a list of length K.
    color : str | Sequence[str], optional
        Color(s) for the curve(s); if multi-output, supply a list of length K.
    linewidth : float | Sequence[float], optional
        Line width(s) for the curve(s). Defaults to ``1.5``.
    linestyle : str | Sequence[str], optional
        Line style(s) for the curve(s). Defaults to ``'-'``.
    ax : matplotlib.axes.Axes, optional
        Target axes. If not provided, a new figure and axes are created.

    Returns
    -------
    (fig, ax)
        The Matplotlib figure and axes containing the overlay plot.

    """
    if backend == "auto":
        backend = "loky"

    x_values = np.asarray(x_values, dtype=float).ravel()
    if x_values.ndim != 1:
        raise ValueError("x_values must be a 1D array")

    # Probe first point to determine arity
    first = func(float(x_values[0]))

    # Helper to evaluate all remaining points, optionally threaded
    nj = _resolve_n_jobs(n_jobs, backend)

    def _get_xs():
        return [float(x) for x in x_values[1:]]

    def _run_parallel(xs, process_chunk_func):
        """Common parallel execution logic."""
        n_workers = int(nj)
        # Chunking
        n_chunks = n_workers * 4
        chunk_size = max(1, len(xs) // n_chunks)
        chunks = [xs[i : i + chunk_size] for i in range(0, len(xs), chunk_size)]
        print(
            f"Parallel execution ({backend}): {len(chunks)} chunks on "
            f"{n_workers} workers."
        )

        if backend in ("loky", "multiprocessing"):
            # Process-based parallelism using joblib
            # We wrap the chunk processor to limit BLAS threads in each worker
            def _worker_wrapper(chunk):
                with threadpool_limits(limits=1, user_api="blas"):
                    return process_chunk_func(chunk)

            # verbose=5 provides a simple progress display to stderr
            chunk_results = joblib.Parallel(
                n_jobs=n_workers, backend=backend, verbose=5
            )(joblib.delayed(_worker_wrapper)(chunk) for chunk in chunks)
            return [item for sublist in chunk_results for item in sublist]

        else:
            # Thread-based parallelism using ThreadPoolExecutor (legacy/fallback)
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(process_chunk_func, chunk) for chunk in chunks]
                total = len(futures)
                completed = 0
                while completed < total:
                    completed = sum(1 for f in futures if f.done())
                    print(f"\rProgress: {completed}/{total} chunks", end="", flush=True)
                    time.sleep(0.1)
                print()
                chunk_results = [f.result() for f in futures]
            return [item for sublist in chunk_results for item in sublist]

    if isinstance(first, (list, tuple, np.ndarray)):
        outs0 = list(first)
        K = len(outs0)
        Y = np.zeros((K, x_values.size), dtype=float)
        for k in range(K):
            Y[k, 0] = float(outs0[k])
        if x_values.size > 1:
            xs = _get_xs()
            if int(nj) and nj > 1:
                def _process_chunk(chunk):
                    return [func(x) for x in chunk]

                results = _run_parallel(xs, _process_chunk)
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
                def _process_chunk(chunk):
                    return [func(x) for x in chunk]

                results = _run_parallel(xs, _process_chunk)
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
    lws = to_list(linewidth, 1.5, K)
    lss = to_list(linestyle, "-", K)

    # Apply global style
    _set_default_rcparams()

    # Prepare axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
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
        ax.plot(
            x_values,
            Y[k],
            label=labels[k],
            color=col,
            linewidth=float(lws[k]),
            linestyle=lss[k],
        )

    if any(lab is not None for lab in labels):
        ax.legend(loc="upper right", fontsize=15, labelspacing=0.35)
    ax.grid(True, alpha=0.2)
    return fig, ax
