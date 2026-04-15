import os
import re
import sys
import inspect
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sgkit
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

import math

tsinfer_path = os.path.abspath("/well/kelleher/users/uuc395/tsinfer")
sys.path.append(tsinfer_path)
import tsinfer


def _extract_mispolarisation_rate(zarr_path, path_prefix):
    zarr_path = Path(zarr_path)
    prefix_name = Path(path_prefix).name
    match = re.fullmatch(
        rf"{re.escape(prefix_name)}mper(?P<rate>[-+0-9.eE]+)\.zarr",
        zarr_path.name,
    )
    if match is None:
        raise ValueError(
            f"Could not extract mispolarisation rate from {zarr_path.name!r}"
        )
    return float(match.group("rate"))


def _find_mispolarisation_zarrs(path_prefix):
    prefix_path = Path(path_prefix).expanduser()
    parent = prefix_path.parent
    if str(parent) == "":
        parent = Path(".")
    zarr_paths = sorted(parent.glob(f"{prefix_path.name}mper*.zarr"))
    if len(zarr_paths) == 0:
        raise FileNotFoundError(
            f"No zarrs matching {prefix_path}mper*.zarr were found"
        )
    return sorted(
        zarr_paths,
        key=lambda zarr_path: _extract_mispolarisation_rate(zarr_path, path_prefix),
    )


def _compute_mispolarised_derived_allele_count(ds):
    if "variant_allele_count" in ds:
        derived_allele_count = np.asarray(ds.variant_allele_count.values, dtype=int).copy()
    else:
        genotypes = np.asarray(ds.call_genotype.values, dtype=int)
        derived_allele_count = np.sum(genotypes, axis=(1, 2))
    if "call_genotype" in ds:
        total_allele_count = ds.call_genotype.shape[1] * ds.call_genotype.shape[2]
    else:
        total_allele_count = int(np.max(derived_allele_count))
    if "variant_mispolarisation_mask" not in ds:
        raise ValueError("Dataset is missing 'variant_mispolarisation_mask'")
    mispolarised = ~np.asarray(ds.variant_mispolarisation_mask.values, dtype=bool)
    derived_allele_count[mispolarised] = (
        total_allele_count - derived_allele_count[mispolarised]
    )
    return derived_allele_count


def _compute_mispolarised_daf(ds):
    derived_allele_count = _compute_mispolarised_derived_allele_count(ds)
    total_allele_count = ds.call_genotype.shape[1] * ds.call_genotype.shape[2]
    return derived_allele_count / total_allele_count


def plot_daf(path_prefix, ax=None, bw_adjust=0.6, plot_path=None):
    """
    Plot the derived allele frequency spectrum for each mispolarisation rate.

    Parameters
    ----------
    path_prefix : str or Path
        Path to the zarr dataset prefix excluding the final ``mper...`` suffix.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. A new figure and axes are created if omitted.
    bw_adjust : float, optional
        Multiplicative adjustment to the smoothing bandwidth.
    plot_path : str or Path, optional
        If provided, save the plot to this path before showing it.
    """
    zarr_paths = _find_mispolarisation_zarrs(path_prefix)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    colors = sns.color_palette("viridis", n_colors=len(zarr_paths))
    for color, zarr_path in zip(colors, zarr_paths):
        rate = _extract_mispolarisation_rate(zarr_path, path_prefix)
        ds = sgkit.load_dataset(str(zarr_path))
        daf = _compute_mispolarised_daf(ds)
        sns.kdeplot(
            daf,
            ax=ax,
            color=color,
            label=f"{rate:g}",
            bw_adjust=bw_adjust,
            common_norm=False,
            cut=0,
            clip=(0, 1),
        )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Derived allele frequency")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.set_title("Derived allele frequency spectrum")
    ax.legend(title="Mispolarisation rate")
    fig.tight_layout()
    if plot_path is not None:
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig, ax


def visualize_perf(
    folder, prefix, versions=["0.4", "1.0"], colors=["darkorange", "royalblue"]
):
    """
    Create a multi-panel plot visualizing tsinfer performance.

    Parameters:
    - folder: directory containing performance data files
    - prefix: prefix for the filenames
    - versions: list of tsinfer versions to compare
    - colors: list of colors for plotting each version
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

    ax_linesweep = fig.add_subplot(gs[0])

    df_list = []

    for i, version in enumerate(versions):
        perf_df = pd.read_csv(os.path.join(folder, f"{prefix}-{version}-perf.csv"))
        perf_df["version"] = version
        df_list.append(perf_df)

        linesweep_df = pd.read_csv(
            os.path.join(folder, f"{prefix}-{version}-linesweep.csv")
        )
        ancestors_per_epoch = linesweep_df.ancestors_per_epoch.values
        ax_linesweep.scatter(
            range(len(ancestors_per_epoch)),
            ancestors_per_epoch,
            color=colors[i],
            label=f"{version}",
            s=10,
            alpha=0.7,
        )

    perf_df = pd.concat(df_list)
    ax_linesweep.set_yscale("log")
    ax_linesweep.set_ylabel("Ancestors per epoch")
    ax_linesweep.set_xlabel("Epoch")
    ax_linesweep.set_title("Linesweep results: number of ancestors per epoch")
    ax_linesweep.legend(title="Version")

    steps = perf_df["step"].unique()
    bottom_gs = gs[1].subgridspec(1, 3, wspace=0.4)

    for i, step in enumerate(steps):
        step_data = perf_df[perf_df["step"] == step]
        ax = fig.add_subplot(bottom_gs[i])
        x = np.arange(2)
        width = 0.8 / len(versions)

        for j, version in enumerate(versions):
            version_data = step_data[step_data["version"] == version]
            if not version_data.empty:
                pos = x - 0.4 + width * (j + 0.5)
                values = [
                    version_data["wall_time"].values[0],
                    version_data["cpu_time"].values[0],
                ]
                bars = ax.bar(pos, values, width, label=f"{version}", color=colors[j])

                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        rotation=0,
                        fontsize=9,
                    )

        ax.set_title(f"{step}")
        ax.set_xticks(x)
        ax.set_xticklabels(["Wall time (s)", "CPU time (s)"])
        ax.set_ylabel("Time (s)")

        if i == 0:
            ax.legend(title="Version")

    plt.tight_layout()
    plt.show()


def plot_ancestor_boxplot(
    df,
    cutoffs=None,
    var="span",
    type="site",
    title="Ancestor lengths",
    y_log=False,
    plot_path=None,
    color_dict={"new": "#ea801c", "old": "#1a80bb", "true": "#b8b8b8"},
    version_dict={"new": "new", "old": "old"},
):
    if set(version_dict.keys()) != {"new", "old"}:
        raise ValueError("version_dict must contain exactly the keys 'new' and 'old'")

    df = df.copy()
    if cutoffs is None:
        cutoffs = np.unique(np.percentile(df["inferred_time"], np.linspace(0, 100, 9)))

    y_units = "sites" if type == "site" else "bp"
    if var == "span":
        var_col = "inferred_span"
        true_col = "true_span"
        var_labels = [
            "True",
            f"Inferred ({version_dict['new']})",
            f"Inferred ({version_dict['old']})",
        ]
        colors = [color_dict["true"], color_dict["new"], color_dict["old"]]
    elif var == "overshoot":
        var_col = "overshoot"
        true_col = None
        var_labels = [version_dict["new"], version_dict["old"]]
        colors = [color_dict["new"], color_dict["old"]]
    else:
        raise ValueError("var must be 'span' or 'overshoot'")

    df["frequency_bin"] = pd.cut(df["inferred_time"], bins=cutoffs, include_lowest=True)
    df["frequency_bin"] = df["frequency_bin"].apply(lambda x: f"({x.left:.2f}, {x.right:.2f}]")
    #return df
    parts = []
    if true_col is not None:
        parts.append(
            df[["frequency_bin", true_col]]
            .rename(columns={true_col: "value"})
            .assign(type="True")
        )
    for version_key in ["new", "old"]:
        version_label = version_dict[version_key]
        inferred_label = (
            f"Inferred ({version_label})" if var == "span" else version_label
        )
        part = (
            df.loc[df["version"] == version_label, ["frequency_bin", var_col]]
            .rename(columns={var_col: "value"})
            .assign(type=inferred_label)
        )
        assert len(part) > 0
        parts.append(part)
    lengths_df = pd.concat(parts, ignore_index=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [4, 1]})
    palette = {label: color for label, color in zip(var_labels, colors)}
    sns.boxplot(x="frequency_bin", y="value", hue="type", data=lengths_df, palette=palette, saturation=1, hue_order=var_labels, ax=ax1)
    ax1.legend(title="Version", loc="upper left", bbox_to_anchor=(1.02, 1))
    ax1.set_xlabel("Frequency range of focal site")
    ax1.set_ylabel(("Ancestor length" if var=="span" else "Ancestor length overshoot")+f" ({y_units})")
    ax1.set_title(title)
    if y_log:
        ax1.set_yscale("log")

    quantile_counts = df["frequency_bin"].value_counts(sort=False)
    sns.barplot(x=quantile_counts.index, y=quantile_counts.values, ax=ax2, color="#ced4da", linewidth=1, edgecolor="black")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax2.set_xlabel("")
    ax2.set_yscale("log")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if plot_path is not None:
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_ancestor_jitterplot(
    df,
    color_dict,
    var,
    freq_interval=[0, 1],
    group_by="sim",
    alpha=0.7,
    plot_size=(10, 5),
    y_lim=None,
):
    frame = pd.DataFrame(df, copy=True)
    if "version" not in frame.columns:
        raise ValueError("DataFrame must contain 'version'")
    if group_by not in frame.columns:
        raise ValueError(f"DataFrame must contain '{group_by}'")
    if len(freq_interval) != 2:
        raise ValueError("freq_interval must contain exactly two values")
    if var not in frame.columns:
        raise ValueError(f"DataFrame must contain '{var}'")

    if "allele_frequency" in frame.columns:
        freq_col = "allele_frequency"
    elif "inferred_time" in frame.columns:
        freq_col = "inferred_time"
    else:
        raise ValueError(
            "DataFrame must contain 'allele_frequency' or 'inferred_time'"
        )

    low, high = sorted(freq_interval)
    plot_df = frame.loc[
        frame["version"].notna()
        & frame[group_by].notna()
        & frame[var].notna()
        & frame[freq_col].between(low, high, inclusive="both")
    ].copy()
    if len(plot_df) == 0:
        raise ValueError("No rows remain after filtering")

    version_order = [key for key in color_dict if key in plot_df["version"].unique()]
    if len(version_order) == 0:
        version_order = plot_df["version"].drop_duplicates().tolist()

    def _sorted_values(values):
        unique_vals = pd.unique(values)
        try:
            return sorted(unique_vals, key=float)
        except (TypeError, ValueError):
            return list(unique_vals)

    group_order = _sorted_values(plot_df[group_by])
    fig_width, fig_height = plot_size
    fig, ax = plt.subplots(
        figsize=(max(fig_width, 1.8 * len(group_order)), fig_height)
    )

    def _overlay_boxplot(ax):
        positions = []
        values = []
        dodge_width = 0.8 / max(len(version_order), 1)
        offsets = (
            np.arange(len(version_order), dtype=float) - (len(version_order) - 1) / 2
        ) * dodge_width

        for group_idx, group_value in enumerate(group_order):
            group_subset = plot_df.loc[plot_df[group_by] == group_value]
            for version_idx, version_value in enumerate(version_order):
                group_values = (
                    group_subset.loc[group_subset["version"] == version_value, var]
                    .dropna()
                    .to_numpy()
                )
                if len(group_values) == 0:
                    continue
                positions.append(group_idx + offsets[version_idx])
                values.append(group_values)

        if len(values) == 0:
            return None

        box_width = min(0.18, 0.8 / max(len(version_order), 1) * 0.6)
        return ax.boxplot(
            values,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.0},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            medianprops={"color": "black", "linewidth": 1.2},
            zorder=3,
        )

    sns.stripplot(
        x=group_by,
        y=var,
        hue="version",
        data=plot_df,
        order=group_order,
        hue_order=version_order,
        palette=color_dict,
        dodge=True,
        jitter=0.25,
        alpha=alpha,
        size=4,
        ax=ax,
    )
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    boxplot = _overlay_boxplot(ax)
    if boxplot is not None:
        for box in boxplot["boxes"]:
            box.set_facecolor("white")
            box.set_edgecolor("black")
            box.set_zorder(3)
        for artist_key in ["whiskers", "caps", "medians"]:
            for artist in boxplot[artist_key]:
                artist.set_zorder(3)

    ax.set_xlabel(group_by.replace("_", " ").capitalize())
    ax.set_ylabel(var.replace("_", " ").capitalize())
    ax.set_yscale("log")
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_title(
        f"{var.replace('_', ' ').capitalize()} with "
        f"{freq_col.replace('_', ' ')} in [{low:g}, {high:g}]"
    )
    ax.legend(
        legend_handles[: len(version_order)],
        legend_labels[: len(version_order)],
        title="Version",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def _sorted_values(values):
    unique_vals = pd.unique(values)
    try:
        return sorted(unique_vals, key=float)
    except (TypeError, ValueError):
        return list(unique_vals)


def _get_frequency_column(frame):
    if "allele_frequency" in frame.columns:
        return "allele_frequency"
    if "inferred_time" in frame.columns:
        return "inferred_time"
    raise ValueError("DataFrame must contain 'allele_frequency' or 'inferred_time'")


def _build_version_palette(version_order, color_dict=None):
    default_colors = {"0.4": "#1a80bb", "1.0": "#ea801c"}
    fallback = sns.color_palette(n_colors=len(version_order))
    palette = {}
    for idx, version in enumerate(version_order):
        if color_dict is not None and version in color_dict:
            palette[version] = color_dict[version]
        elif color_dict is not None and str(version) in color_dict:
            palette[version] = color_dict[str(version)]
        else:
            palette[version] = default_colors.get(str(version), fallback[idx])
    return palette


def _plot_log10_violin_boxplot(
    ax,
    plot_df,
    group_col,
    value_col,
    group_order,
    version_order,
    palette,
    violin_bw_adjust=0.6,
    alpha=0.7,
    violin_width=0.65,
    box_width=0.12,
    empty_text="No positive values",
    log_y=True,
):
    if log_y:
        violin_df = plot_df.loc[plot_df[value_col] > 0].copy()
    else:
        violin_df = plot_df.loc[plot_df[value_col].notna()].copy()
    if len(violin_df) == 0:
        ax.text(
            0.5,
            0.5,
            empty_text,
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return violin_df

    plot_value_col = "_plot_value"
    if log_y:
        violin_df[plot_value_col] = np.log10(
            violin_df[value_col].to_numpy(dtype=float)
        )
    else:
        violin_df[plot_value_col] = violin_df[value_col].to_numpy(dtype=float)
    violin_ax = sns.violinplot(
        x=group_col,
        y=plot_value_col,
        hue="version",
        data=violin_df,
        order=group_order,
        hue_order=version_order,
        palette=palette,
        dodge=True,
        bw_adjust=violin_bw_adjust,
        cut=0,
        inner=None,
        linewidth=0,
        saturation=1,
        width=violin_width,
        ax=ax,
    )
    for collection in violin_ax.collections:
        collection.set_alpha(alpha)

    positions = []
    values = []
    dodge_width = violin_width / max(len(version_order), 1)
    offsets = (
        np.arange(len(version_order), dtype=float) - (len(version_order) - 1) / 2
    ) * dodge_width
    for group_idx, group_value in enumerate(group_order):
        subset = violin_df.loc[violin_df[group_col] == group_value]
        for version_idx, version_value in enumerate(version_order):
            group_values = (
                subset.loc[subset["version"] == version_value, plot_value_col]
                .dropna()
                .to_numpy(dtype=float)
            )
            if len(group_values) == 0:
                continue
            positions.append(group_idx + offsets[version_idx])
            values.append(group_values)

    if len(values) > 0:
        boxplot = ax.boxplot(
            values,
            positions=positions,
            widths=max(box_width, dodge_width * 0.72),
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.0},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            medianprops={"color": "black", "linewidth": 1.2},
            zorder=3,
        )
        for box in boxplot["boxes"]:
            box.set_facecolor("white")
            box.set_edgecolor("black")
            box.set_zorder(3)
        for artist_key in ["whiskers", "caps", "medians"]:
            for artist in boxplot[artist_key]:
                artist.set_zorder(3)
    return violin_df


def _set_log10_axis_ticks(ax, positive_values):
    if len(positive_values) == 0:
        return
    positive_values = np.asarray(positive_values, dtype=float)
    y_min = float(np.min(positive_values))
    y_max = float(np.max(positive_values))
    major_locator = LogLocator(base=10)
    major_values = major_locator.tick_values(y_min, y_max)
    major_values = major_values[(major_values >= y_min) & (major_values <= y_max)]
    if len(major_values) == 0:
        major_values = np.array([y_min, y_max], dtype=float)
    major_values = np.unique(major_values)

    minor_locator = LogLocator(base=10, subs=np.arange(2, 10, dtype=float))
    minor_values = minor_locator.tick_values(y_min, y_max)
    minor_values = minor_values[(minor_values >= y_min) & (minor_values <= y_max)]
    minor_values = np.setdiff1d(np.unique(minor_values), major_values)

    ax.set_yticks(np.log10(major_values))
    if len(minor_values) > 0:
        ax.set_yticks(np.log10(minor_values), minor=True)
    formatter = LogFormatterSciNotation(base=10)
    ax.set_yticklabels([formatter(value) for value in major_values])
    ax.set_yticklabels([], minor=True)


def _make_zero_gap_log_hist_layout(values, num_bins, gap_scale=0.5):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("No values provided")
    if np.any(values < 0):
        raise ValueError("Values must be non-negative")
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")

    positive_values = values[values > 0]
    if len(positive_values) == 0:
        zero_width = 1.0
        return {
            "positive_log_edges": np.array([], dtype=float),
            "positive_plot_edges": np.array([], dtype=float),
            "zero_left": 0.0,
            "zero_width": zero_width,
            "tick_positions": np.array([0.0]),
            "tick_labels": ["0"],
            "xlim": (0.0, 1.4 * zero_width),
        }

    log_positive = np.log10(positive_values)
    log_min = float(np.min(log_positive))
    log_max = float(np.max(log_positive))
    if np.isclose(log_min, log_max):
        log_min -= 0.5
        log_max += 0.5

    positive_log_edges = np.linspace(log_min, log_max, num_bins + 1)
    step = positive_log_edges[1] - positive_log_edges[0]
    zero_width = step
    gap_width = gap_scale * zero_width
    positive_plot_edges = (
        positive_log_edges - positive_log_edges[0] + zero_width + gap_width
    )

    # Start decade ticks at the first power of ten that lies within the
    # positive region. Using floor(log_min) can place the first positive tick
    # to the left of the zero gap when log_min falls between decades.
    tick_exponents = np.arange(
        int(np.ceil(log_min)),
        int(np.ceil(log_max)) + 1,
    )
    formatter = LogFormatterSciNotation(base=10)
    tick_positions = np.concatenate(
        (
            [0.0],
            tick_exponents.astype(float) - positive_log_edges[0] + zero_width + gap_width,
        )
    )
    tick_labels = ["0"] + [formatter(10.0**exp) for exp in tick_exponents]

    return {
        "positive_log_edges": positive_log_edges,
        "positive_plot_edges": positive_plot_edges,
        "zero_left": 0.0,
        "zero_width": zero_width,
        "tick_positions": tick_positions,
        "tick_labels": tick_labels,
        "xlim": (0.0, positive_plot_edges[-1]),
    }


def _count_zero_gap_log_hist(values, positive_log_edges):
    values = np.asarray(values, dtype=float)
    if np.any(values < 0):
        raise ValueError("Values must be non-negative")
    zero_count = int(np.sum(values == 0))
    if len(positive_log_edges) == 0:
        positive_counts = np.array([], dtype=int)
    else:
        positive_counts, _ = np.histogram(
            np.log10(values[values > 0]),
            bins=positive_log_edges,
        )
    return zero_count, positive_counts


def _map_zero_gap_log_value(value, layout):
    value = float(value)
    if value < 0:
        raise ValueError("Value must be non-negative")
    if value == 0:
        return layout["zero_left"] + 0.5 * layout["zero_width"]
    if len(layout["positive_log_edges"]) == 0:
        raise ValueError("Positive edges are required to map positive values")
    return (
        np.log10(value)
        - layout["positive_log_edges"][0]
        + layout["positive_plot_edges"][0]
    )


def plot_ancestor_hap_errors(
    df,
    color_dict=None,
    var="hap_error_rate",
    freq_interval=[0, 1],
    plot_path=None,
    num_bins=100,
    title=None,
    alpha=1.0,
    plot_size=(7, 7),
    version_x=0.04,
):
    frame = pd.DataFrame(df, copy=True)
    if "version" not in frame.columns:
        raise ValueError("DataFrame must contain 'version'")
    if var not in frame.columns:
        raise ValueError(f"DataFrame must contain '{var}'")
    if len(freq_interval) != 2:
        raise ValueError("freq_interval must contain exactly two values")
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")

    freq_col = _get_frequency_column(frame)

    low, high = sorted(freq_interval)
    plot_df = frame.loc[
        frame["version"].notna()
        & frame[var].notna()
        & frame[freq_col].between(low, high, inclusive="both")
    ].copy()
    if len(plot_df) == 0:
        raise ValueError("No rows remain after filtering")
    if np.any(plot_df[var].to_numpy(dtype=float) < 0):
        raise ValueError(f"{var} must be non-negative")

    version_order = _sorted_values(plot_df["version"])
    if len(version_order) != 2:
        raise ValueError(
            "plot_ancestor_hap_errors requires exactly two versions after filtering"
        )
    palette = _build_version_palette(version_order, color_dict)

    layout = _make_zero_gap_log_hist_layout(
        plot_df[var].to_numpy(dtype=float, copy=False),
        num_bins=num_bins,
    )
    max_count = 0
    counts_by_version = {}
    for version_value in version_order:
        version_values = plot_df.loc[plot_df["version"] == version_value, var].to_numpy(
            dtype=float, copy=False
        )
        zero_count, positive_counts = _count_zero_gap_log_hist(
            version_values,
            layout["positive_log_edges"],
        )
        counts_by_version[version_value] = (zero_count, positive_counts)
        max_count = max(
            max_count,
            zero_count,
            int(np.max(positive_counts)) if len(positive_counts) > 0 else 0,
        )
    if max_count == 0:
        raise ValueError("No histogram counts available after filtering")

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=plot_size,
        gridspec_kw={"hspace": 0.0},
    )
    axes = np.atleast_1d(axes).ravel()

    positive_plot_edges = layout["positive_plot_edges"]
    if len(positive_plot_edges) > 0:
        positive_widths = np.diff(positive_plot_edges)
    for ax_idx, (ax, version_value) in enumerate(zip(axes, version_order)):
        zero_count, positive_counts = counts_by_version[version_value]
        color = palette[version_value]
        bar_alpha = 1.0
        if zero_count > 0:
            ax.bar(
                layout["zero_left"],
                zero_count,
                width=layout["zero_width"],
                align="edge",
                color=color,
                alpha=bar_alpha,
                edgecolor=None,
                linewidth=0,
            )
        if len(positive_plot_edges) > 0 and np.sum(positive_counts) > 0:
            ax.bar(
                positive_plot_edges[:-1],
                positive_counts,
                width=positive_widths,
                align="edge",
                color=color,
                alpha=bar_alpha,
                edgecolor=None,
                linewidth=0,
            )
        version_values = plot_df.loc[plot_df["version"] == version_value, var].to_numpy(
            dtype=float, copy=False
        )
        mean_value = float(np.mean(version_values))
        mean_x = _map_zero_gap_log_value(mean_value, layout)
        ax.axvline(
            mean_x,
            color="black",
            linestyle="--",
            linewidth=1.5,
            zorder=20,
        )
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        ax.set_ylim(0.8, max_count * 1.05)
        ax.set_xlim(*layout["xlim"])
        max_power = int(np.floor(np.log10(max_count))) if max_count > 0 else 0
        major_ticks = [10**k for k in range(1, max_power + 1)]
        if len(major_ticks) > 0:
            ax.set_yticks(major_ticks)
        if ax_idx == 1:
            ax.invert_yaxis()
        x_span = layout["xlim"][1] - layout["xlim"][0]
        x_offset = 0.01 * x_span
        text_y = 0.94 if ax_idx == 0 else 0.06
        if mean_x >= layout["xlim"][0] + 0.85 * x_span:
            text_x = mean_x - x_offset
            text_ha = "right"
        else:
            text_x = mean_x + x_offset
            text_ha = "left"
        ax.text(
            text_x,
            text_y,
            f"mean = {mean_value:.1e}",
            color="black",
            transform=ax.get_xaxis_transform(),
            ha=text_ha,
            va="top" if ax_idx == 0 else "bottom",
            zorder=21,
        )

    axes[-1].set_xlabel(var.replace("_", " ").capitalize())
    axes[-1].set_xticks(layout["tick_positions"])
    axes[-1].set_xticklabels(layout["tick_labels"])
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", labelbottom=False)
    if title is not None:
        axes[0].set_title(title)
    axes[0].legend(
        handles=[
            Patch(facecolor=palette[version], edgecolor="none", label=str(version))
            for version in version_order
        ],
        title="Version",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

    fig.subplots_adjust(left=0.12, right=0.85, bottom=0.1, top=0.92, hspace=0.0)
    if title is not None:
        fig.subplots_adjust(top=0.88)
    if plot_path is not None:
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_ancestor_violin(df, var, cutoffs, violin_bw_adjust=0.6, log_y=False):
    frame = pd.DataFrame(df, copy=True)
    if "version" not in frame.columns:
        raise ValueError("DataFrame must contain 'version'")
    if var not in frame.columns:
        raise ValueError(f"DataFrame must contain '{var}'")
    if len(cutoffs) < 2:
        raise ValueError("cutoffs must contain at least two values")

    freq_col = _get_frequency_column(frame)
    bin_edges = np.asarray(cutoffs, dtype=float)
    if not np.all(np.diff(bin_edges) > 0):
        raise ValueError("cutoffs must be strictly increasing")

    plot_df = frame.loc[
        frame["version"].notna()
        & frame[var].notna()
        & frame[freq_col].notna()
    ].copy()
    if len(plot_df) == 0:
        raise ValueError("No rows remain after filtering")

    plot_df["frequency_bin"] = pd.cut(
        plot_df[freq_col], bins=bin_edges, include_lowest=True
    )
    plot_df = plot_df.loc[plot_df["frequency_bin"].notna()].copy()
    if len(plot_df) == 0:
        raise ValueError("No rows fall within the specified cutoffs")

    bin_labels = [
        f"[{bin_edges[idx]:g}, {bin_edges[idx + 1]:g})"
        for idx in range(len(bin_edges) - 1)
    ]
    interval_to_label = {
        interval: label
        for interval, label in zip(plot_df["frequency_bin"].cat.categories, bin_labels)
    }
    plot_df["frequency_bin_label"] = pd.Categorical(
        plot_df["frequency_bin"].map(interval_to_label),
        categories=bin_labels,
        ordered=True,
    )

    version_order = _sorted_values(plot_df["version"])
    palette = _build_version_palette(version_order)
    fig_width = max(8, 1.8 * len(bin_labels))
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(fig_width, 6),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.12},
    )

    violin_df = _plot_log10_violin_boxplot(
        ax_top,
        plot_df,
        "frequency_bin_label",
        var,
        bin_labels,
        version_order,
        palette,
        violin_bw_adjust=violin_bw_adjust,
        alpha=1.0,
        violin_width=0.58,
        box_width=0.1,
        empty_text="No positive values" if log_y else "No values",
        log_y=log_y,
    )
    ax_top.set_xlabel("")
    ax_top.set_ylabel(var.replace("_", " ").capitalize())
    if log_y and len(violin_df) > 0:
        _set_log10_axis_ticks(ax_top, violin_df[var].to_numpy(dtype=float))
    if ax_top.get_legend() is not None:
        ax_top.get_legend().remove()
    ax_top.legend(
        handles=[
            Patch(facecolor=palette[version], edgecolor="none", label=str(version))
            for version in version_order
        ],
        title="Version",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

    x = np.arange(len(bin_labels), dtype=float)
    counts = (
        plot_df.groupby("frequency_bin_label", sort=False, observed=False)
        .size()
        .reindex(bin_labels, fill_value=0)
    )
    max_count = int(counts.max()) if len(counts) > 0 else 0
    ax_bottom.bar(
        x,
        counts.to_numpy(dtype=float),
        width=0.55,
        color="#ced4da",
        edgecolor="black",
        linewidth=0.8,
    )

    ax_bottom.set_ylabel("Count")
    ax_bottom.set_xlabel("")
    ax_bottom.set_xticks(x)
    ax_bottom.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(bin_labels)
    if max_count > 0:
        ax_bottom.set_yscale("log")
        ax_bottom.set_ylim(0.8, max_count * 1.05)

    fig.subplots_adjust(left=0.08, right=0.85, bottom=0.1, top=0.96, hspace=0.06)
    plt.show()


def plot_ancestor_barchart(
    df,
    color_dict,
    var,
    freq_interval=[0, 1],
    group_by="sim",
):
    """
    Plot mean +/- SD of ``var`` by ``version`` within each ``group_by`` category.
    """
    frame = pd.DataFrame(df, copy=True)
    if "version" not in frame.columns:
        raise ValueError("DataFrame must contain 'version'")
    if group_by not in frame.columns:
        raise ValueError(f"DataFrame must contain '{group_by}'")
    if len(freq_interval) != 2:
        raise ValueError("freq_interval must contain exactly two values")
    if var not in frame.columns:
        raise ValueError(f"DataFrame must contain '{var}'")

    if "allele_frequency" in frame.columns:
        freq_col = "allele_frequency"
    elif "inferred_time" in frame.columns:
        freq_col = "inferred_time"
    else:
        raise ValueError(
            "DataFrame must contain 'allele_frequency' or 'inferred_time'"
        )

    low, high = sorted(freq_interval)
    base_df = frame.loc[
        frame["version"].notna()
        & frame[group_by].notna()
        & frame[var].notna()
    ].copy()
    if len(base_df) == 0:
        raise ValueError("No rows remain after removing missing group/version data")

    version_order = [key for key in color_dict if key in base_df["version"].unique()]
    if len(version_order) == 0:
        version_order = base_df["version"].drop_duplicates().tolist()

    def _sorted_values(values):
        unique_vals = pd.unique(values)
        try:
            return sorted(unique_vals, key=float)
        except (TypeError, ValueError):
            return list(unique_vals)

    group_order = _sorted_values(base_df[group_by])
    if len(version_order) == 0:
        raise ValueError("No rows found for the requested versions")

    plot_df = base_df.loc[
        base_df[freq_col].between(low, high, inclusive="both")
    ].copy()
    if len(plot_df) == 0:
        raise ValueError("No rows remain after filtering")

    summary = (
        plot_df.loc[plot_df["version"].isin(version_order)]
        .groupby([group_by, "version"], sort=False)[var]
        .agg(mean="mean", sd="std")
        .reset_index()
    )
    summary["sd"] = summary["sd"].fillna(0.0)

    fig_width = max(6, 1.8 * len(group_order))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    x = np.arange(len(group_order), dtype=float)
    width = 0.8 / max(len(version_order), 1)

    for idx, version in enumerate(version_order):
        version_summary = (
            summary.loc[summary["version"] == version, [group_by, "mean", "sd"]]
            .set_index(group_by)
            .reindex(group_order)
        )
        version_summary["sd"] = version_summary["sd"].fillna(0.0)
        offsets = (idx - (len(version_order) - 1) / 2) * width
        ax.bar(
            x + offsets,
            version_summary["mean"],
            width=width,
            color=color_dict.get(version),
            label=str(version),
            yerr=version_summary["sd"],
            capsize=4,
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in group_order])
    ax.set_xlabel(group_by.replace("_", " ").capitalize())
    ax.set_ylabel(var.replace("_", " ").capitalize())
    ax.set_title(
        f"{var.replace('_', ' ').capitalize()} with "
        f"{freq_col.replace('_', ' ')} in [{low:g}, {high:g}]"
    )
    ax.set_ylim(bottom=0)
    ax.legend(title="Version")
    plt.tight_layout()
    plt.show()

def plot_ancestor_scatter(
    df,
    freq_interval,
    color_dict,
    var="overshoot",
    alpha=0.1,
    num_bins=50,
    plot_path=None,
    error_profile=None,
    title=None,
):
    df = pd.DataFrame(df, copy=True)
    freq_col = _get_frequency_column(df)
    required_cols = {freq_col, "focal_position", "side", "version", var}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
    if len(freq_interval) != 2:
        raise ValueError("freq_interval must contain exactly two values")
    if len(color_dict) != 2:
        raise ValueError("color_dict must contain exactly two versions")
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")
    if error_profile is not None and "error_profile" not in df.columns:
        raise ValueError("DataFrame must contain 'error_profile' when filtering by it")

    low, high = sorted(freq_interval)
    version_keys = list(color_dict.keys())
    version_labels = [str(v) for v in version_keys]
    version_colors = [color_dict[v] for v in version_keys]

    plot_df = df.copy()
    if error_profile is not None:
        plot_df = plot_df.loc[plot_df["error_profile"] == error_profile].copy()
    plot_df = plot_df.loc[
        plot_df[freq_col].between(low, high, inclusive="both")
        & plot_df[var].notna()
    ].copy()
    if len(plot_df) == 0:
        raise ValueError("No rows remain after filtering")

    plot_df["version_label"] = plot_df["version"].astype(str)
    plot_df = plot_df.loc[plot_df["version_label"].isin(version_labels)].copy()
    if len(plot_df) == 0:
        raise ValueError("No rows found for the requested versions")

    grouped = (
        plot_df.groupby(
            ["focal_position", "side", "version_label"], as_index=False, sort=False
        )[var]
        .mean()
    )
    paired = grouped.pivot(
        index=["focal_position", "side"], columns="version_label", values=var
    ).reset_index()

    missing_versions = [label for label in version_labels if label not in paired.columns]
    if missing_versions:
        raise ValueError(
            f"No paired {var} values found for version(s): {missing_versions}"
        )

    paired = paired.dropna(subset=version_labels)
    if len(paired) == 0:
        raise ValueError("No focal_position/side groups contain both requested versions")

    paired = paired.sort_values(version_labels[0], kind="mergesort").reset_index(
        drop=True
    )

    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[4.5, 0.7],
        height_ratios=[0.7, 4.5],
        hspace=0.05,
        wspace=0.05,
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_legend = fig.add_subplot(gs[0, 1])

    var_label = var.replace("_", " ").capitalize()
    if var == "overshoot":
        x = paired[version_labels[0]] / 1e6
        y = paired[version_labels[1]] / 1e6
        axis_label = f"{var_label} (mbp"
    else:
        x = paired[version_labels[0]].to_numpy(dtype=float, copy=False)
        y = paired[version_labels[1]].to_numpy(dtype=float, copy=False)
        axis_label = var_label
    x_counts, _ = np.histogram(x, bins=num_bins)
    y_counts, _ = np.histogram(y, bins=num_bins)
    max_count = max(np.max(x_counts), np.max(y_counts))
    ax.scatter(
        x,
        y,
        color='black',
        alpha=alpha,
        s=10,
        edgecolors="none",
    )
    sns.histplot(
        x=x,
        ax=ax_histx,
        stat="count",
        bins=num_bins,
        linewidth=0,
        color=version_colors[0],
    )
    sns.histplot(
        y=y,
        ax=ax_histy,
        stat="count",
        bins=num_bins,
        linewidth=0,
        color=version_colors[1],
    )
    ax_histx.set_yscale("log")
    ax_histy.set_xscale("log")
    if max_count > 0:
        count_limits = (0.8, max_count * 1.05)
        ax_histx.set_ylim(count_limits)
        ax_histy.set_xlim(count_limits)

    if var == "overshoot":
        ax.set_xlabel(f"{axis_label}, {version_labels[0]})")
        ax.set_ylabel(f"{axis_label}, {version_labels[1]})")
    else:
        ax.set_xlabel(f"{axis_label} ({version_labels[0]})")
        ax.set_ylabel(f"{axis_label} ({version_labels[1]})")
    ax.set_aspect("equal", adjustable="box")

    ax_histx.set_ylabel("Count")
    ax_histx.set_xlabel("")
    ax_histx.tick_params(axis="x", labelbottom=False)
    if title is not None:
        ax_histx.set_title(title)
    ax_histy.set_xlabel("Count")
    ax_histy.set_ylabel("")
    ax_histy.tick_params(axis="y", labelleft=False)

    ax_legend.axis("off")
    ax_legend.legend(
        handles=[
            Patch(facecolor=version_colors[0], edgecolor="none", label=version_labels[0]),
            Patch(facecolor=version_colors[1], edgecolor="none", label=version_labels[1]),
        ],
        title="Version",
        loc="upper right",
        frameon=False,
    )

    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.92)
    if plot_path is not None:
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_overshoot_scatter(*args, **kwargs):
    return plot_ancestor_scatter(*args, var="overshoot", **kwargs)


def plot_overshoot_sign(
    df,
    cutoffs=[0, 0.5, 1.0],
    eps=100,
    select=[(0, 0), (2, 2), (2, 0)],
    plot_path=None,
):
    frame = pd.DataFrame(df, copy=True)
    required_cols = {"allele_frequency", "focal_position", "side", "version", "overshoot"}
    missing = required_cols.difference(frame.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
    if len(cutoffs) < 2:
        raise ValueError("cutoffs must contain at least two values")
    if eps < 0:
        raise ValueError("eps must be non-negative")

    version_colors = {"0.4": "#1a80bb", "1.0": "#ea801c"}
    try:
        version_order = sorted(frame["version"].dropna().unique().tolist(), key=float)
    except (TypeError, ValueError):
        version_order = frame["version"].dropna().unique().tolist()
    if len(version_order) != 2:
        raise ValueError("plot_overshoot_sign requires exactly two versions")

    grouped = (
        frame.loc[
            frame["allele_frequency"].notna()
            & frame["overshoot"].notna()
            & frame["version"].notna()
            & frame["focal_position"].notna()
            & frame["side"].notna()
        ]
        .groupby(["focal_position", "side", "version"], as_index=False, sort=False)
        .agg(
            overshoot=("overshoot", "mean"),
            allele_frequency=("allele_frequency", "mean"),
        )
    )
    paired = grouped.pivot(
        index=["focal_position", "side"],
        columns="version",
        values=["overshoot", "allele_frequency"],
    ).reset_index()
    paired.columns = [
        col
        if not isinstance(col, tuple)
        else col[0]
        if col[1] == ""
        else f"{col[0]}__{col[1]}"
        for col in paired.columns
    ]

    overshoot_cols = [f"overshoot__{version}" for version in version_order]
    missing_versions = [version for version in version_order if f"overshoot__{version}" not in paired.columns]
    if missing_versions:
        raise ValueError(
            f"No paired overshoot values found for version(s): {missing_versions}"
        )
    paired = paired.dropna(subset=overshoot_cols)
    if len(paired) == 0:
        raise ValueError("No focal_position/side groups contain both requested versions")

    af_cols = [
        col
        for col in [f"allele_frequency__{version}" for version in version_order]
        if col in paired.columns
    ]
    paired["allele_frequency_pair"] = paired.loc[:, af_cols].mean(axis=1)

    def _sign(values):
        values = np.asarray(values, dtype=float)
        out = np.zeros(len(values), dtype=int)
        out[values > eps] = 1
        out[values < -eps] = -1
        return out

    paired["sign_0"] = _sign(paired[f"overshoot__{version_order[0]}"])
    paired["sign_1"] = _sign(paired[f"overshoot__{version_order[1]}"])

    bin_edges = np.asarray(cutoffs, dtype=float)
    if not np.all(np.diff(bin_edges) > 0):
        raise ValueError("cutoffs must be strictly increasing")
    paired["frequency_bin"] = pd.cut(
        paired["allele_frequency_pair"], bins=bin_edges, include_lowest=True
    )
    paired = paired.loc[paired["frequency_bin"].notna()].copy()
    if len(paired) == 0:
        raise ValueError("No paired rows fall within the specified cutoffs")

    if select is None:
        select_per_bin = [None] * len(cutoffs[:-1])
    elif (
        isinstance(select, tuple)
        and len(select) == 2
        and all(isinstance(value, (int, np.integer)) for value in select)
    ):
        select_per_bin = [tuple(select)] * len(cutoffs[:-1])
    else:
        select_per_bin = [tuple(item) for item in select]
        if len(select_per_bin) == 0:
            select_per_bin = [None] * len(cutoffs[:-1])
        elif len(select_per_bin) < len(cutoffs) - 1:
            select_per_bin.extend([select_per_bin[-1]] * (len(cutoffs) - 1 - len(select_per_bin)))
        else:
            select_per_bin = select_per_bin[: len(cutoffs) - 1]
    for coords in select_per_bin:
        if coords is None:
            continue
        if len(coords) != 2 or any(coord not in (0, 1, 2) for coord in coords):
            raise ValueError("select entries must be 2-tuples of heatmap coordinates in {0, 1, 2}")

    sign_order = [-1, 0, 1]
    sign_labels = {-1: "Negative", 0: "Within ε", 1: "Positive"}
    interval_order = paired["frequency_bin"].cat.categories
    n_bins = len(interval_order)
    fig_width = max(3.8 * n_bins + 1.2, 7.5)
    fig_height = 7.8
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        2,
        n_bins + 1,
        width_ratios=[1] * n_bins + [0.22],
        height_ratios=[1, 0.3],
        wspace=0.35,
        hspace=0.06,
    )
    heatmap_axes = [fig.add_subplot(gs[0, idx]) for idx in range(n_bins)]
    hist_top_axes = []
    hist_bottom_axes = []
    for idx in range(n_bins):
        hist_gs = gs[1, idx].subgridspec(2, 1, hspace=0.0, height_ratios=[1, 1])
        ax_top = fig.add_subplot(hist_gs[0, 0])
        ax_bottom = fig.add_subplot(hist_gs[1, 0], sharex=ax_top)
        hist_top_axes.append(ax_top)
        hist_bottom_axes.append(ax_bottom)
    cbar_ax = fig.add_subplot(gs[0, -1])
    legend_ax = fig.add_subplot(gs[1, -1])
    cmap = plt.get_cmap("viridis")
    matrices = []
    for idx, interval in enumerate(interval_order):
        subset = paired.loc[paired["frequency_bin"] == interval]
        counts = pd.crosstab(
            pd.Categorical(subset["sign_0"], categories=sign_order, ordered=True),
            pd.Categorical(subset["sign_1"], categories=sign_order, ordered=True),
            dropna=False,
        ).reindex(index=sign_order, columns=sign_order, fill_value=0)
        total = int(counts.to_numpy().sum())
        props = counts / total if total > 0 else counts.astype(float)
        matrices.append((idx, interval, counts, props, total, subset))
    vmax = max((float(props.to_numpy().max()) for _, _, _, props, _, _ in matrices), default=0.01)
    vmax = max(vmax, 0.01)

    cbar_mappable = None
    for idx, (interval_idx, interval, counts, props, total, subset) in enumerate(matrices):
        ax = heatmap_axes[idx]

        cbar_mappable = sns.heatmap(
            props,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            annot=False,
            cbar=False,
            square=True,
            linewidths=1,
            linecolor="white",
            xticklabels=[sign_labels[value] for value in sign_order],
            yticklabels=[sign_labels[value] for value in sign_order],
        )
        for row_idx in range(counts.shape[0]):
            for col_idx in range(counts.shape[1]):
                value = float(props.iat[row_idx, col_idx])
                color_position = 0 if vmax == 0 else value / vmax
                ax.text(
                    col_idx + 0.5,
                    row_idx + 0.42,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    color="black" if color_position > 0.85 else "white",
                    fontsize=12,
                    fontweight="semibold",
                )
                ax.text(
                    col_idx + 0.5,
                    row_idx + 0.67,
                    f"n={counts.iat[row_idx, col_idx]}",
                    ha="center",
                    va="center",
                    color="black" if color_position > 0.85 else "white",
                    fontsize=8,
                )
        ax.set_title(f"f \u2208 [{cutoffs[interval_idx]:g}, {cutoffs[interval_idx + 1]:g}) (N = {total})")
        ax.set_xlabel(
            f"Version {version_order[1]} overshoot sign",
            color=version_colors.get(str(version_order[1]), "black"),
            labelpad=12,
        )
        ax.set_ylabel("")
        ax.yaxis.label.set_color(
            version_colors.get(str(version_order[0]), "black")
        )
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=90, labelleft=True)

        selected_cell = select_per_bin[idx] if idx < len(select_per_bin) else None
        if selected_cell is not None:
            row_idx, col_idx = selected_cell
            ax.add_patch(
                plt.Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2.5,
                )
            )
            selected_subset = subset.loc[
                (subset["sign_0"] == sign_order[row_idx])
                & (subset["sign_1"] == sign_order[col_idx])
            ].copy()
        else:
            selected_subset = subset.iloc[0:0].copy()

        hist_ax_top = hist_top_axes[idx]
        hist_ax_bottom = hist_bottom_axes[idx]
        if len(selected_subset) > 0:
            values_top = (
                selected_subset[f"overshoot__{version_order[0]}"]
                .to_numpy(dtype=float)
                / 1e6
            )
            values_bottom = (
                selected_subset[f"overshoot__{version_order[1]}"]
                .to_numpy(dtype=float)
                / 1e6
            )
            combined_values = np.concatenate([values_top, values_bottom])
            hist_min = float(np.min(combined_values))
            hist_max = float(np.max(combined_values))
            if hist_min == hist_max:
                x_pad = max(abs(hist_min) * 0.05, 1e-6)
            else:
                x_pad = (hist_max - hist_min) * 0.03
            x_left = hist_min - x_pad
            x_right = hist_max + x_pad
            hist_edges = np.linspace(x_left, x_right, 101)
            counts_top, _ = np.histogram(values_top, bins=hist_edges)
            counts_bottom, _ = np.histogram(values_bottom, bins=hist_edges)
            max_count = int(max(np.max(counts_top), np.max(counts_bottom)))
            if np.sum(counts_top) > 0:
                hist_ax_top.bar(
                    hist_edges[:-1],
                    counts_top,
                    width=np.diff(hist_edges),
                    align="edge",
                    color=version_colors.get(str(version_order[0]), "#1a80bb"),
                    alpha=0.7,
                    edgecolor=None,
                    linewidth=0,
                )
            if np.sum(counts_bottom) > 0:
                hist_ax_bottom.bar(
                    hist_edges[:-1],
                    counts_bottom,
                    width=np.diff(hist_edges),
                    align="edge",
                    color=version_colors.get(str(version_order[1]), "#ea801c"),
                    alpha=0.7,
                    edgecolor=None,
                    linewidth=0,
                )
            if max_count > 0:
                hist_ax_top.set_yscale("log")
                hist_ax_bottom.set_yscale("log")
                hist_ax_top.set_ylim(0.8, max_count * 1.05)
                hist_ax_bottom.set_ylim(0.8, max_count * 1.05)
                max_power = int(np.floor(np.log10(max_count))) if max_count > 0 else 0
                major_ticks = [10**k for k in range(0, max_power + 1)]
                hist_ax_top.set_yticks(major_ticks)
                hist_ax_bottom.set_yticks(major_ticks)
                tick_labels = [""] + [rf"$10^{k}$" for k in range(1, max_power + 1)]
                hist_ax_top.set_yticklabels(tick_labels)
                hist_ax_bottom.set_yticklabels(tick_labels)
            hist_ax_bottom.invert_yaxis()
            hist_ax_top.set_xlim(x_left, x_right)
        else:
            hist_ax_top.text(
                0.5,
                0.5,
                "No paired points\nin selected cell",
                ha="center",
                va="center",
                transform=hist_ax_top.transAxes,
            )
        for mini_ax in [hist_ax_top, hist_ax_bottom]:
            for spine in mini_ax.spines.values():
                spine.set_edgecolor("#4d4d4d")
                spine.set_linewidth(1.5)
            mini_ax.tick_params(axis="x", colors="black")
            mini_ax.tick_params(axis="y", colors="black")

        hist_ax_top.set_xlabel("")
        hist_ax_bottom.set_xlabel("Overshoot (mbp)", color="black", labelpad=10)
        hist_ax_top.set_ylabel("")
        hist_ax_bottom.set_ylabel("")
        hist_ax_top.tick_params(axis="x", labelbottom=False)

    if cbar_mappable is not None:
        colorbar = fig.colorbar(
            cbar_mappable.collections[0], cax=cbar_ax, label="Proportion"
        )
        colorbar.ax.tick_params(labelsize=8)
        colorbar.set_label("Proportion", size=9)
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Patch(
                facecolor=version_colors.get(str(version_order[0]), "#1a80bb"),
                edgecolor="none",
                label=str(version_order[0]),
            ),
            Patch(
                facecolor=version_colors.get(str(version_order[1]), "#ea801c"),
                edgecolor="none",
                label=str(version_order[1]),
            ),
        ],
        title="Version",
        loc="center left",
        frameon=False,
    )

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.08, top=0.94)
    heat_left = min(ax.get_position().x0 for ax in heatmap_axes)
    heat_right = max(ax.get_position().x1 for ax in heatmap_axes)
    heat_bottom = min(ax.get_position().y0 for ax in heatmap_axes)
    heat_top = max(ax.get_position().y1 for ax in heatmap_axes)
    heat_center_x = 0.5 * (heat_left + heat_right)
    heat_center_y = 0.5 * (heat_bottom + heat_top)
    fig.text(
        heat_left - 0.045,
        heat_center_y,
        f"Version {version_order[0]} overshoot sign",
        rotation=90,
        ha="center",
        va="center",
        color=version_colors.get(str(version_order[0]), "black"),
    )
    cbar_width = cbar_ax.get_position().width
    cbar_ax.set_position(
        [
            cbar_ax.get_position().x0 + 0.015,
            heat_bottom,
            cbar_width * 0.35,
            heat_top - heat_bottom,
        ]
    )
    legend_pos = legend_ax.get_position()
    legend_ax.set_position(
        [
            legend_pos.x0 - 0.03,
            legend_pos.y0,
            legend_pos.width,
            legend_pos.height,
        ]
    )
    hist_left = min(ax.get_position().x0 for ax in hist_bottom_axes)
    hist_right = max(ax.get_position().x1 for ax in hist_bottom_axes)
    hist_bottom = min(ax.get_position().y0 for ax in hist_bottom_axes)
    hist_split_y = 0.5 * (
        min(ax.get_position().y0 for ax in hist_top_axes)
        + max(ax.get_position().y1 for ax in hist_bottom_axes)
    )
    hist_center_x = 0.5 * (hist_left + hist_right)
    fig.text(
        hist_left - 0.045,
        hist_split_y,
        "Count",
        rotation=90,
        ha="center",
        va="center",
        color="black",
    )
    if plot_path is not None:
        plot_path = Path(plot_path)
        if plot_path.exists():
            plot_path.unlink()
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_overshoot_cdf(
    df,
    freq_interval,
    by,
    panel,
    plot_path=None,
):
    df = pd.DataFrame(df, copy=True)
    required_cols = {"allele_frequency", "overshoot", by, panel}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
    if len(freq_interval) != 2:
        raise ValueError("freq_interval must contain exactly two values")

    low, high = sorted(freq_interval)
    plot_df = df.loc[
        df["allele_frequency"].between(low, high, inclusive="both")
        & df["overshoot"].notna()
        & df[by].notna()
        & df[panel].notna()
    ].copy()
    if len(plot_df) == 0:
        raise ValueError("No rows remain after filtering")

    plot_df["overshoot_mbp"] = plot_df["overshoot"] / 1e6
    panel_order = plot_df[panel].drop_duplicates().tolist()
    by_order = plot_df[by].drop_duplicates().tolist()
    palette = dict(zip(by_order, sns.color_palette(n_colors=len(by_order))))
    n_panels = len(panel_order)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 4 * nrows),
        sharex=False,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, panel_value in zip(axes, panel_order):
        subset = plot_df.loc[plot_df[panel] == panel_value]
        panel_x = subset["overshoot_mbp"].to_numpy(dtype=float, copy=False)
        panel_min = float(np.min(panel_x))
        panel_max = float(np.max(panel_x))
        if panel_min == panel_max:
            x_pad = max(abs(panel_min) * 0.05, 1e-6)
        else:
            x_pad = (panel_max - panel_min) * 0.03
        x_left = panel_min - x_pad
        x_right = panel_max + x_pad
        ax.set_xlim(x_left, x_right)

        for by_value in by_order:
            values = (
                subset.loc[subset[by] == by_value, "overshoot_mbp"]
                .sort_values()
                .to_numpy(dtype=float)
            )
            if len(values) == 0:
                continue
            cdf = np.arange(1, len(values) + 1, dtype=float) / len(values)
            x_step = np.concatenate(([x_left], values, [x_right]))
            y_step = np.concatenate(([0.0], cdf, [1.0]))
            ax.step(
                x_step,
                y_step,
                where="post",
                color=palette[by_value],
                linewidth=1.8,
            )
        ax.set_title(str(panel_value))
        ax.set_xlabel("Overshoot (mbp)")
        ax.set_ylabel("Empirical CDF")
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    legend_ax = axes[0]
    legend_handles = [
        Line2D([0], [0], color=palette[label], lw=1.8, label=str(label))
        for label in by_order
    ]
    if legend_handles:
        legend_ax.legend(
            handles=legend_handles,
            title=by.replace("_", " ").capitalize(),
            loc="lower right",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if plot_path is not None:
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_overshoot_hist(
    df,
    by,
    freq_interval,
    plot_path=None,
    num_bins=100,
    alpha=0.5,
    reverse_order=False,
    legend_title=None,
    version_x=0.04,
    plot_size=(7, 7),
    title=None,
):
    df = pd.DataFrame(df, copy=True)
    required_cols = {"allele_frequency", "overshoot", "version", by}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
    if len(freq_interval) != 2:
        raise ValueError("freq_interval must contain exactly two values")
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")

    low, high = sorted(freq_interval)
    plot_df = df.loc[
        df["allele_frequency"].between(low, high, inclusive="both")
        & df["overshoot"].notna()
        & df["version"].notna()
        & df[by].notna()
    ].copy()
    if len(plot_df) == 0:
        raise ValueError("No rows remain after filtering")

    plot_df["overshoot_mbp"] = plot_df["overshoot"] / 1e6
    try:
        version_order = sorted(plot_df["version"].drop_duplicates().tolist(), key=float)
    except (TypeError, ValueError):
        version_order = plot_df["version"].drop_duplicates().tolist()
    if len(version_order) != 2:
        raise ValueError(
            "plot_overshoot_hist requires exactly two versions after filtering"
        )
    try:
        by_order = sorted(plot_df[by].drop_duplicates().tolist(), key=float)
    except (TypeError, ValueError):
        by_order = plot_df[by].drop_duplicates().tolist()
    if len(by_order) > 4:
        raise ValueError("plot_overshoot_hist supports at most 4 values of 'by'")
    if legend_title is None:
        legend_title = by.replace("_", " ").capitalize()

    tab20c = list(plt.get_cmap("tab20c").colors)

    def _version_palette(version_value):
        version_label = str(version_value)
        if version_label == "0.4":
            base_indices = [3, 2, 1, 0]
        elif version_label == "1.0":
            base_indices = [7, 6, 5, 4]
        else:
            base_indices = [3, 2, 1, 0]
        return {
            by_value: tab20c[idx]
            for by_value, idx in zip(by_order, base_indices[-len(by_order):])
        }

    palette_by_version = {
        version_value: _version_palette(version_value) for version_value in version_order
    }
    all_values = plot_df["overshoot_mbp"].to_numpy(dtype=float, copy=False)
    global_min = float(np.min(all_values))
    global_max = float(np.max(all_values))
    if global_min == global_max:
        x_pad = max(abs(global_min) * 0.05, 1e-6)
        global_min -= x_pad
        global_max += x_pad
    tick_min = int(np.floor(global_min))
    tick_max = int(np.ceil(global_max))
    if tick_min == tick_max:
        tick_min -= 1
        tick_max += 1
    bin_edges = np.linspace(tick_min, tick_max, num_bins + 1)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=plot_size,
        sharex=True,
        gridspec_kw={"hspace": 0.0},
    )
    axes = np.atleast_1d(axes).ravel()

    max_count = 0
    counts_by_version = {version_value: {} for version_value in version_order}
    for version_value in version_order:
        subset = plot_df.loc[plot_df["version"] == version_value]
        for by_value in by_order:
            values = subset.loc[subset[by] == by_value, "overshoot_mbp"]
            counts, _ = np.histogram(values, bins=bin_edges)
            counts_by_version[version_value][by_value] = counts
            if len(counts) > 0:
                max_count = max(max_count, int(np.max(counts)))

    if max_count == 0:
        raise ValueError("No histogram counts available after filtering")

    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + 0.5 * bin_widths
    negative_mask = bin_centers < 0
    nonnegative_mask = ~negative_mask

    for ax_idx, (ax, version_value) in enumerate(zip(axes, version_order)):
        version_palette = palette_by_version[version_value]
        for by_value in reversed(by_order):
            counts = counts_by_version[version_value][by_value]
            if np.sum(counts) == 0:
                continue
            if np.any(negative_mask & (counts > 0)) or not reverse_order:
                negative_counts = counts.copy()
                if reverse_order:
                    negative_counts[nonnegative_mask] = 0
                ax.bar(
                    bin_edges[:-1],
                    negative_counts,
                    width=bin_widths,
                    align="edge",
                    color=version_palette[by_value],
                    alpha=alpha,
                    edgecolor=None,
                    linewidth=0,
                )
        if reverse_order:
            for by_value in by_order:
                counts = counts_by_version[version_value][by_value]
                if np.sum(counts) == 0:
                    continue
                if np.any(nonnegative_mask & (counts > 0)):
                    nonnegative_counts = counts.copy()
                    nonnegative_counts[negative_mask] = 0
                    ax.bar(
                        bin_edges[:-1],
                        nonnegative_counts,
                        width=bin_widths,
                        align="edge",
                        color=version_palette[by_value],
                        alpha=alpha,
                        edgecolor=None,
                        linewidth=0,
                    )
        ax.set_ylabel("Count")
        ax.set_xlim(tick_min, tick_max)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0, zorder=9)
        ax.set_yscale("log")
        ax.set_ylim(0.8, max_count * 1.05)
        max_power = int(np.floor(np.log10(max_count))) if max_count > 0 else 0
        major_ticks = [10**k for k in range(1, max_power + 1)]
        ax.set_yticks(major_ticks)
        if ax_idx == 1:
            ax.invert_yaxis()
        legend_handles = [
            Patch(facecolor=version_palette[label], edgecolor="none", label=str(label))
            for label in by_order
        ]
        ax.legend(
            handles=legend_handles,
            title=legend_title,
            loc="upper right" if ax_idx == 0 else "lower right",
        )
        text_x = version_x
        text_y = 0.96 if ax_idx == 0 else 0.04
        text_va = "top" if ax_idx == 0 else "bottom"
        ax.text(
            text_x,
            text_y,
            f"Version {version_value}",
            transform=ax.transAxes,
            ha="left",
            va=text_va,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "#cccccc",
            },
        )

    axes[-1].set_xlabel("Overshoot (mbp)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", labelbottom=False)
    middle_lw = axes[0].spines["left"].get_linewidth()
    axes[0].spines["bottom"].set_visible(False)
    axes[1].spines["top"].set_visible(False)

    top_margin = 0.92 if title is not None else 0.95
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.08, top=top_margin, hspace=0.0)
    fig.canvas.draw()
    for ax in axes:
        ticks = ax.get_xticks()
        ticks = ticks[(ticks >= tick_min) & (ticks <= tick_max)]
        for endpoint in (tick_min, tick_max):
            if not np.any(np.isclose(ticks, endpoint)):
                ticks = np.append(ticks, endpoint)
        ax.set_xticks(np.sort(np.unique(ticks)))
    divider_y = axes[1].get_position().y1
    fig.add_artist(
        Line2D(
            [axes[0].get_position().x0, axes[0].get_position().x1],
            [divider_y, divider_y],
            transform=fig.transFigure,
            color="black",
            linewidth=middle_lw,
            zorder=10,
        )
    )
    if title is not None:
        plot_center_x = (axes[0].get_position().x0 + axes[0].get_position().x1) / 2
        fig.text(plot_center_x, 0.975, title, ha="center", va="top")
    if plot_path is not None:
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_path_likelihood(
    df,
    path_id,
    likelihood_threshold=1e-13,
    cmap="viridis",
    figsize=(14, 8),
    weight_by_n=None,
    mu=None,
    rho=None,
    compared_by=None,
    long_df=None,
):
    """
    Plot per-site likelihood structure for a single path.

    If ``long_df`` is supplied, it is used directly (after filtering). Otherwise
    likelihood blocks are built directly from compact per-site arrays in ``df``.
    """

    def _filter_by_params(frame):
        if weight_by_n is not None and "weight_by_n" in frame.columns:
            frame = frame.loc[frame["weight_by_n"] == weight_by_n]
        if mu is not None and "mu" in frame.columns:
            mu_values = pd.to_numeric(frame["mu"], errors="coerce")
            frame = frame.loc[
                np.isclose(mu_values, float(mu), equal_nan=False)
            ]
        if rho is not None and "rho" in frame.columns:
            rho_values = pd.to_numeric(frame["rho"], errors="coerce")
            frame = frame.loc[
                np.isclose(rho_values, float(rho), equal_nan=False)
            ]
        if "likelihood_threshold" in frame.columns:
            threshold_values = pd.to_numeric(
                frame["likelihood_threshold"], errors="coerce"
            )
            frame = frame.loc[
                np.isclose(
                    threshold_values, float(likelihood_threshold), equal_nan=False
                )
            ]
        return frame

    df = pd.DataFrame(df, copy=False)
    if "path_id" not in df.columns:
        raise ValueError("DataFrame must contain 'path_id'")

    filtered_df = _filter_by_params(df)
    subset_df = filtered_df.loc[filtered_df["path_id"] == path_id].copy()
    if len(subset_df) == 0:
        raise ValueError(f"No rows found for path_id={path_id}")

    site_rows = (
        subset_df[["site", "k"]]
        .drop_duplicates(subset=["site"], keep="last")
        .sort_values("site")
        .reset_index(drop=True)
    )
    if len(site_rows) == 0:
        raise ValueError(f"No sites found for path_id={path_id}")

    if long_df is None:
        grouped_likelihoods = {}
        site_payload = (
            subset_df[["site", "k", "likelihoods", "likelihood_nodes"]]
            .drop_duplicates(subset=["site"], keep="last")
            .sort_values("site")
        )
        for row in site_payload.itertuples(index=False):
            k = int(row.k)
            likelihood_values = np.asarray(row.likelihoods, dtype=np.float64)
            if len(likelihood_values) != k:
                raise ValueError(
                    f"Expected {k} likelihoods at site={row.site}, found {len(likelihood_values)}"
                )
            likelihood_node_ids = np.asarray(row.likelihood_nodes, dtype=np.int64)
            if len(likelihood_node_ids) != k:
                raise ValueError(
                    f"Expected {k} likelihood node ids at site={row.site}, found {len(likelihood_node_ids)}"
                )
            if k > 1:
                order = np.lexsort((likelihood_node_ids, likelihood_values))
                likelihood_values = likelihood_values[order]
            grouped_likelihoods[row.site] = likelihood_values
    else:
        long_all = pd.DataFrame(long_df, copy=False)
        if "path_id" not in long_all.columns:
            raise ValueError("Provided long_df must contain 'path_id'")
        long_subset = long_all.loc[long_all["path_id"] == path_id]
        long_subset = _filter_by_params(long_subset)
        long_subset = long_subset.sort_values(
            ["site", "full_likelihood", "likelihood_node_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        grouped_likelihoods = {
            site: grp["full_likelihood"].to_numpy(dtype=np.float64, copy=False)
            for site, grp in long_subset.groupby("site", sort=False)
        }

    site_values = site_rows["site"].to_numpy(copy=False)
    k_values = site_rows["k"].to_numpy(dtype=np.int64, copy=False)
    total_blocks = int(k_values[k_values > 0].sum())
    if total_blocks == 0:
        raise ValueError(f"No likelihood values available for path_id={path_id}")

    polygons = np.empty((total_blocks, 4, 2), dtype=np.float32)
    values = np.empty(total_blocks, dtype=np.float64)
    offset = 0

    for col, (site, k) in enumerate(zip(site_values, k_values)):
        if k <= 0:
            continue
        site_likelihoods = grouped_likelihoods.get(site)
        if site_likelihoods is None:
            raise ValueError(
                f"Missing likelihood entries for site={site} in path_id={path_id}"
            )
        if len(site_likelihoods) != k:
            raise ValueError(
                f"Expected {k} likelihoods at site={site}, found {len(site_likelihoods)}"
            )

        y_edges = np.linspace(0.0, 1.0, k + 1, dtype=np.float32)
        next_offset = offset + k
        block = polygons[offset:next_offset]
        block[:, 0, 0] = col
        block[:, 0, 1] = y_edges[:-1]
        block[:, 1, 0] = col + 1
        block[:, 1, 1] = y_edges[:-1]
        block[:, 2, 0] = col + 1
        block[:, 2, 1] = y_edges[1:]
        block[:, 3, 0] = col
        block[:, 3, 1] = y_edges[1:]
        values[offset:next_offset] = site_likelihoods
        offset = next_offset

    if not np.all(np.isfinite(values)):
        raise ValueError("Likelihood values must be finite")

    threshold_mask = np.equal(values, likelihood_threshold)
    non_threshold_mask = ~threshold_mask
    mapped_values = np.zeros_like(values, dtype=np.float64)

    threshold_eps = max(np.finfo(np.float64).eps, abs(likelihood_threshold) * 1e-12)
    legend_min = likelihood_threshold + threshold_eps

    if np.any(non_threshold_mask):
        non_threshold_values = values[non_threshold_mask]
        max_non_threshold = float(np.max(non_threshold_values))
        if max_non_threshold <= legend_min:
            mapped_values[non_threshold_mask] = 0.4
        else:
            scaled = (
                np.clip(non_threshold_values, legend_min, max_non_threshold)
                - legend_min
            ) / (max_non_threshold - legend_min)
            mapped_values[non_threshold_mask] = 0.4 + 0.6 * scaled
    else:
        max_non_threshold = legend_min * (1 + 1e-12)

    mapped_values[threshold_mask] = 0.0

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
    ax_heat = fig.add_subplot(gs[0])
    ax_k = fig.add_subplot(gs[1], sharex=ax_heat)

    collection = PolyCollection(
        polygons,
        array=mapped_values,
        cmap=cmap,
        edgecolors="none",
        antialiased=False,
    )
    collection.set_clim(0.0, 1.0)
    ax_heat.add_collection(collection)
    ax_heat.set_xlim(0, len(site_rows))
    ax_heat.set_ylim(0.0, 1.0)
    ax_heat.set_ylabel("Likelihood blocks")
    ax_heat.set_yticks([])

    child_ids = subset_df["child_id"].dropna().unique()
    child_label = child_ids[0] if len(child_ids) > 0 else "NA"

    def _format_param(name, value):
        if isinstance(value, (float, np.floating)):
            value_str = f"{float(value):.3g}"
        else:
            value_str = str(value)
        label = f"{name}={value_str}"
        if compared_by == name:
            math_label = label.replace("_", r"\_")
            return rf"$\bf{{{math_label}}}$"
        return label

    title_parts = [f"Path {path_id}"]
    if "weight_by_n" in subset_df.columns:
        title_parts.append(_format_param("weight_by_n", bool(subset_df["weight_by_n"].iloc[0])))
    if "mu" in subset_df.columns:
        title_parts.append(_format_param("mu", float(subset_df["mu"].iloc[0])))
    if "rho" in subset_df.columns:
        title_parts.append(_format_param("rho", float(subset_df["rho"].iloc[0])))
    if "likelihood_threshold" in subset_df.columns:
        title_parts.append(_format_param("likelihood_threshold", float(subset_df["likelihood_threshold"].iloc[0])))
    ax_heat.set_title(" | ".join(title_parts))

    site_to_x = {site: idx + 0.5 for idx, site in enumerate(site_values)}
    mismatch_sites = (
        subset_df.loc[subset_df["selected_mismatch"] > 0, "site"]
        .drop_duplicates()
        .to_numpy(copy=False)
    )
    recombination_sites = (
        subset_df.loc[subset_df["selected_recombination"] > 0, "site"]
        .drop_duplicates()
        .to_numpy(copy=False)
    )
    for site in mismatch_sites:
        x = site_to_x.get(site)
        if x is not None:
            ax_heat.axvline(x, color="red", linewidth=1.2, alpha=0.95)
    for site in recombination_sites:
        x = site_to_x.get(site)
        if x is not None:
            ax_heat.axvline(x, color="orange", linewidth=1.2, alpha=0.95)

    ax_k.bar(
        np.arange(len(site_rows)),
        k_values,
        width=1.0,
        align="edge",
        color="#4c4c4c",
        edgecolor="none",
    )
    ax_k.set_ylabel("Num. tracked nodes")
    ax_k.set_xlabel("Site")
    positive_k = k_values[k_values > 0]
    if len(positive_k) > 0:
        ax_k.set_ylim(bottom=max(0.8, float(np.min(positive_k)) * 0.8))

    max_ticks = min(12, len(site_rows))
    tick_idx = np.linspace(0, len(site_rows) - 1, num=max_ticks, dtype=int)
    ax_k.set_xticks(tick_idx)
    ax_k.set_xticklabels(site_values[tick_idx])

    plt.setp(ax_heat.get_xticklabels(), visible=False)

    fig.subplots_adjust(right=0.84)
    top = ax_heat.get_position().y1
    bottom = ax_k.get_position().y0
    right = max(ax_heat.get_position().x1, ax_k.get_position().x1)
    total_height = top - bottom

    legend_x = right + 0.02
    legend_width = 0.022
    main_height = total_height * 0.75
    gap_height = total_height * 0.05
    threshold_height = total_height * 0.06

    main_y = top - main_height
    threshold_y = max(bottom, main_y - gap_height - threshold_height)

    base_cmap = plt.get_cmap(cmap)
    high_cmap = LinearSegmentedColormap.from_list(
        f"{cmap}_high", base_cmap(np.linspace(0.4, 1.0, 256))
    )
    high_vmax = max(max_non_threshold, legend_min * (1 + 1e-12))

    main_cax = fig.add_axes([legend_x, main_y, legend_width, main_height])
    high_sm = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=legend_min, vmax=high_vmax),
        cmap=high_cmap,
    )
    high_sm.set_array([])
    high_cbar = fig.colorbar(high_sm, cax=main_cax)

    threshold_label = np.format_float_scientific(likelihood_threshold, precision=0)
    mid_ticks = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
    valid_mid_ticks = mid_ticks[
        np.logical_and(mid_ticks > legend_min, mid_ticks < high_vmax)
    ]
    tick_values = np.concatenate(
        [np.array([legend_min], dtype=np.float64), valid_mid_ticks, np.array([high_vmax])]
    )
    high_cbar.set_ticks(tick_values.tolist())
    high_cbar.set_ticklabels(
        [f"{threshold_label} + ε"]
        + [f"{tick:.1f}" for tick in valid_mid_ticks]
        + [f"{high_vmax:.2g}"]
    )

    threshold_ax = fig.add_axes([legend_x, threshold_y, legend_width, threshold_height])
    threshold_ax.add_patch(
        plt.Rectangle((0.0, 0.0), 1.0, 1.0, color=base_cmap(0.0), linewidth=0)
    )
    threshold_ax.set_xlim(0.0, 1.0)
    threshold_ax.set_ylim(0.0, 1.0)
    threshold_ax.set_xticks([])
    threshold_ax.set_yticks([])
    threshold_ax.text(
        1.25,
        0.5,
        threshold_label,
        ha="left",
        va="center",
        transform=threshold_ax.transAxes,
    )

    num_mismatches = int((subset_df["selected_mismatch"] > 0).sum())
    num_recombinations = int((subset_df["selected_recombination"] > 0).sum())
    handles = [
        Line2D([0], [0], color="red", lw=1.8, label=f"mismatch (n={num_mismatches})"),
        Line2D(
            [0],
            [0],
            color="orange",
            lw=1.8,
            label=f"recombination (n={num_recombinations})",
        ),
    ]
    event_height = total_height * 0.14
    event_y = max(bottom, threshold_y - total_height * 0.20)
    event_ax = fig.add_axes([legend_x, event_y, 0.20, event_height])
    event_ax.set_axis_off()
    event_ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )

    plt.show()
    plt.close(fig)


def plot_paths_interactively(df):
    """
    Interactive viewer over a combined dataframe with multiple parameter settings.

    - Builds path summary sorted by num_errors.
    - Slider is global path_id (0..max path_id) and does not change with params.
    - Parameter selectors fix non-compared parameters.
    - Compare-by selector plots one column per value of selected compare variable.

    To reduce memory use on large data, plotting reads compact per-site likelihood
    arrays directly instead of building cached long-form tables by default.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError(
            "plot_paths_interactively requires ipywidgets and IPython display"
        ) from exc

    from tsinfer.logging import make_path_df

    combined_df = pd.DataFrame(df, copy=False)
    required = {
        "path_id",
        "weight_by_n",
        "mu",
        "rho",
        "likelihood_threshold",
        "k",
    }
    missing = required.difference(combined_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "DataFrame missing required columns for interactive view: "
            f"{missing_str}"
        )

    path_df = make_path_df(combined_df).sort_values(
        ["num_errors", "path_id"], kind="mergesort"
    ).reset_index(drop=True)

    def _fmt_value(v):
        if isinstance(v, (float, np.floating)):
            return np.format_float_scientific(float(v), precision=2)
        return str(v)

    weight_values = [bool(v) for v in pd.unique(combined_df["weight_by_n"])]
    ordered_weights = [v for v in [True, False] if v in set(weight_values)]
    ordered_weights.extend([v for v in weight_values if v not in set(ordered_weights)])

    mu_values = sorted(float(v) for v in pd.unique(combined_df["mu"]))
    rho_values = sorted(float(v) for v in pd.unique(combined_df["rho"]))
    threshold_values = sorted(
        float(v) for v in pd.unique(combined_df["likelihood_threshold"])
    )

    weight_buttons = widgets.ToggleButtons(
        options=[(str(v), v) for v in ordered_weights],
        description="weight_by_n",
    )
    mu_buttons = widgets.ToggleButtons(
        options=[(_fmt_value(v), v) for v in mu_values],
        description="mu",
    )
    rho_buttons = widgets.ToggleButtons(
        options=[(_fmt_value(v), v) for v in rho_values],
        description="rho",
    )
    threshold_buttons = widgets.ToggleButtons(
        options=[(_fmt_value(v), v) for v in threshold_values],
        description="likelihood_threshold",
    )
    compare_buttons = widgets.ToggleButtons(
        options=[
            ("weight_by_n", "weight_by_n"),
            ("mu", "mu"),
            ("rho", "rho"),
            ("likelihood_threshold", "likelihood_threshold"),
        ],
        description="Compare by",
    )

    max_path_id = int(pd.to_numeric(combined_df["path_id"], errors="coerce").max())
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=max_path_id,
        step=1,
        description="path",
        continuous_update=False,
        readout=True,
        layout=widgets.Layout(width="82%"),
    )
    path_input = widgets.BoundedIntText(
        value=0,
        min=0,
        max=max_path_id,
        step=1,
        description="path_id",
        layout=widgets.Layout(width="18%"),
    )

    status = widgets.HTML()
    output = widgets.Output()

    path_row = widgets.HBox([slider, path_input], layout=widgets.Layout(width="100%"))
    controls_row = widgets.HBox(
        [weight_buttons, mu_buttons, rho_buttons, threshold_buttons],
        layout=widgets.Layout(width="100%"),
    )
    compare_row = widgets.HBox([compare_buttons], layout=widgets.Layout(width="100%"))
    ui = widgets.VBox([output, path_row, controls_row, compare_row, status])

    def _get_compare_values(name):
        if name == "weight_by_n":
            return ordered_weights
        if name == "mu":
            return mu_values
        if name == "rho":
            return rho_values
        if name == "likelihood_threshold":
            return threshold_values
        raise ValueError(f"Unknown compare variable: {name}")

    def _parameter_set(compare_name, compare_value):
        params = {
            "weight_by_n": weight_buttons.value,
            "mu": float(mu_buttons.value),
            "rho": float(rho_buttons.value),
            "likelihood_threshold": float(threshold_buttons.value),
        }
        params[compare_name] = compare_value
        return params

    def _find_summary_row(path_id, params):
        subset = path_df.loc[path_df["path_id"] == path_id]
        subset = subset.loc[subset["weight_by_n"] == params["weight_by_n"]]
        subset = subset.loc[
            np.isclose(
                pd.to_numeric(subset["mu"], errors="coerce"),
                float(params["mu"]),
                equal_nan=False,
            )
        ]
        subset = subset.loc[
            np.isclose(
                pd.to_numeric(subset["rho"], errors="coerce"),
                float(params["rho"]),
                equal_nan=False,
            )
        ]
        subset = subset.loc[
            np.isclose(
                pd.to_numeric(subset["likelihood_threshold"], errors="coerce"),
                float(params["likelihood_threshold"]),
                equal_nan=False,
            )
        ]
        if len(subset) == 0:
            return None
        return subset.iloc[0]

    def draw_current():
        path_id = int(slider.value)
        compare_name = compare_buttons.value
        compare_values = _get_compare_values(compare_name)
        n_cols = max(1, len(compare_values))

        # Shrink per-panel plot width as the number of compared columns increases.
        panel_fig_width = max(4.2, 13.0 / n_cols)
        panel_fig_height = max(4.8, 8.0 - 0.6 * (n_cols - 1))
        panel_figsize = (panel_fig_width, panel_fig_height)

        panel_outputs = []
        status_parts = []
        panel_width = max(20, int(98 / n_cols))

        for value in compare_values:
            params = _parameter_set(compare_name, value)
            label = f"{compare_name}={_fmt_value(value)}"
            panel = widgets.Output(
                layout=widgets.Layout(width=f"{panel_width}%")
            )

            with panel:
                row = _find_summary_row(path_id, params)
                if row is None:
                    print(f"{label}: no row for path_id={path_id}")
                else:
                    status_parts.append(
                        f"{label}: errors={int(row['num_errors'])}, "
                        f"switches={int(row['num_switches'])}, "
                        f"mismatches={int(row['num_mismatches'])}"
                    )
                    plot_path_likelihood(
                        combined_df,
                        path_id=path_id,
                        likelihood_threshold=float(params["likelihood_threshold"]),
                        weight_by_n=bool(params["weight_by_n"]),
                        mu=float(params["mu"]),
                        rho=float(params["rho"]),
                        compared_by=compare_name,
                        figsize=panel_figsize,
                    )
            panel_outputs.append(panel)

        with output:
            output.clear_output(wait=True)
            display(widgets.HBox(panel_outputs, layout=widgets.Layout(width="100%")))

        if len(status_parts) == 0:
            status.value = f"<b>path_id {path_id}</b>: no matching rows"
        else:
            status.value = (
                f"<b>path_id {path_id}</b> &nbsp;|&nbsp; "
                + " &nbsp;|&nbsp; ".join(status_parts)
            )

    def _on_change(change):
        if change.get("name") == "value":
            draw_current()

    def _on_slider_change(change):
        if change.get("name") != "value":
            return
        new_path = int(change["new"])
        if int(path_input.value) != new_path:
            path_input.value = new_path
        draw_current()

    def _on_path_input_change(change):
        if change.get("name") != "value":
            return
        new_path = int(change["new"])
        if int(slider.value) != new_path:
            slider.value = new_path
        else:
            draw_current()

    slider.observe(_on_slider_change, names="value")
    path_input.observe(_on_path_input_change, names="value")
    weight_buttons.observe(_on_change, names="value")
    mu_buttons.observe(_on_change, names="value")
    rho_buttons.observe(_on_change, names="value")
    threshold_buttons.observe(_on_change, names="value")
    compare_buttons.observe(_on_change, names="value")

    draw_current()
    display(ui)

    return {
        "ui": ui,
        "path_df": path_df,
        "slider": slider,
        "path_input": path_input,
        "weight_buttons": weight_buttons,
        "mu_buttons": mu_buttons,
        "rho_buttons": rho_buttons,
        "threshold_buttons": threshold_buttons,
        "compare_buttons": compare_buttons,
    }


def plot_grouped_barchart(
    ax, path_df, variable, fixed="likelihood_threshold", group_by=None
):
    """
    Plot grouped boxplots of ``variable`` for one fixed parameter setting.

    The fixed parameter is held at its first observed value.
    Grouping uses ``group_by`` on the x-axis, and the other unfixed parameter
    is used as color hue.
    """
    df = pd.DataFrame(path_df, copy=False)
    parameter_order = ["weight_by_n", "mu", "rho", "likelihood_threshold"]

    if variable not in df.columns:
        raise ValueError(f"Column '{variable}' not found in path_df")
    if fixed not in parameter_order:
        raise ValueError(
            f"'fixed' must be one of {parameter_order}, got '{fixed}'"
        )
    if fixed not in df.columns:
        raise ValueError(f"Column '{fixed}' not found in path_df")

    unfixed = [col for col in parameter_order if col != fixed and col in df.columns]
    if len(unfixed) < 2:
        raise ValueError(
            "Need at least two unfixed parameter columns in path_df; "
            f"found {unfixed}"
        )
    if group_by is None:
        x_col = unfixed[0]
    else:
        if group_by not in parameter_order:
            raise ValueError(
                f"'group_by' must be one of {parameter_order}, got '{group_by}'"
            )
        if group_by == fixed:
            raise ValueError(
                f"'group_by' cannot equal fixed ('{fixed}')"
            )
        if group_by not in unfixed:
            raise ValueError(
                f"Column '{group_by}' not available for grouping in path_df"
            )
        x_col = group_by
    remaining = [col for col in unfixed if col != x_col]
    hue_col = remaining[0]
    extra_fixed_cols = remaining[1:]

    fixed_candidates = df[fixed].dropna().to_numpy(copy=False)
    if len(fixed_candidates) == 0:
        raise ValueError(f"No non-null values found in fixed column '{fixed}'")
    fixed_value = fixed_candidates[0]
    subset = df.loc[df[fixed] == fixed_value].copy()
    if len(subset) == 0:
        raise ValueError(
            f"No rows available after filtering {fixed}={fixed_value!r}"
        )
    extra_fixed_values = {}
    for col in extra_fixed_cols:
        col_candidates = subset[col].dropna().to_numpy(copy=False)
        if len(col_candidates) == 0:
            continue
        col_value = col_candidates[0]
        subset = subset.loc[subset[col] == col_value]
        extra_fixed_values[col] = col_value

    def _sorted_values(values, col):
        unique_vals = list(pd.unique(values))
        if col == "weight_by_n":
            ordered = [v for v in [True, False] if v in set(unique_vals)]
            ordered.extend([v for v in unique_vals if v not in set(ordered)])
            return ordered
        numeric_vals = pd.to_numeric(pd.Series(unique_vals), errors="coerce")
        if numeric_vals.notna().all():
            return sorted(float(v) for v in unique_vals)
        return sorted(unique_vals, key=lambda x: str(x))

    x_order = _sorted_values(subset[x_col], x_col)
    hue_order = _sorted_values(subset[hue_col], hue_col)

    boxplot_kwargs = dict(
        data=subset,
        x=x_col,
        y=variable,
        hue=hue_col,
        order=x_order,
        hue_order=hue_order,
        ax=ax,
        palette="tab10",
        width=0.65,
        dodge=True,
        showfliers=True,
    )
    if "gap" in inspect.signature(sns.boxplot).parameters:
        boxplot_kwargs["gap"] = 0.12

    sns.boxplot(**boxplot_kwargs)
    ax.set_title(variable)
    ax.set_xlabel(x_col)
    ax.set_ylabel(variable)
    ax.legend(
        title=hue_col,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    return {
        "fixed": fixed,
        "fixed_value": fixed_value,
        "x": x_col,
        "hue": hue_col,
        "extra_fixed": extra_fixed_values,
    }


def plot_grouped_hist_by_site(
    df, variable, group_by, fixed, bins=15, y_log=True
):
    """
    Plot binned site histograms of ``variable`` over a parameter grid.

    - ``fixed`` parameter is held at its first observed value.
    - ``group_by`` parameter defines subplot columns.
    - The remaining unfixed parameter defines subplot rows.
    """
    frame = pd.DataFrame(df, copy=False)
    parameter_order = ["weight_by_n", "mu", "rho", "likelihood_threshold"]

    if variable not in frame.columns:
        raise ValueError(f"Column '{variable}' not found in df")
    if "site" not in frame.columns:
        raise ValueError("Column 'site' not found in df")
    if fixed not in parameter_order:
        raise ValueError(f"'fixed' must be one of {parameter_order}, got '{fixed}'")
    if group_by not in parameter_order:
        raise ValueError(
            f"'group_by' must be one of {parameter_order}, got '{group_by}'"
        )
    if fixed == group_by:
        raise ValueError("'fixed' and 'group_by' must be different")
    if fixed not in frame.columns:
        raise ValueError(f"Column '{fixed}' not found in df")
    if group_by not in frame.columns:
        raise ValueError(f"Column '{group_by}' not found in df")

    row_candidates = [
        col for col in parameter_order if col not in {fixed, group_by} and col in frame.columns
    ]
    if len(row_candidates) == 0:
        raise ValueError(
            "Could not determine row parameter from remaining unfixed columns; "
            f"found {row_candidates}"
        )
    row_by = row_candidates[0]
    extra_fixed_cols = row_candidates[1:]

    fixed_candidates = frame[fixed].dropna().to_numpy(copy=False)
    if len(fixed_candidates) == 0:
        raise ValueError(f"No non-null values found in fixed column '{fixed}'")
    fixed_value = fixed_candidates[0]
    subset = frame.loc[frame[fixed] == fixed_value].copy()
    if len(subset) == 0:
        raise ValueError(f"No rows available after filtering {fixed}={fixed_value!r}")
    extra_fixed_values = {}
    for col in extra_fixed_cols:
        col_candidates = subset[col].dropna().to_numpy(copy=False)
        if len(col_candidates) == 0:
            continue
        col_value = col_candidates[0]
        subset = subset.loc[subset[col] == col_value]
        extra_fixed_values[col] = col_value

    if int(bins) <= 0:
        raise ValueError("'bins' must be a positive integer")
    bins = int(bins)

    variable_values = pd.to_numeric(subset[variable], errors="coerce")
    if variable_values.isna().all():
        raise ValueError(f"Column '{variable}' has no numeric values after filtering")
    subset = subset.loc[variable_values.notna()].copy()
    subset[variable] = variable_values.loc[subset.index].to_numpy(dtype=np.float64)
    subset["site"] = pd.to_numeric(subset["site"], errors="coerce")
    subset = subset.loc[subset["site"].notna()].copy()
    if len(subset) == 0:
        raise ValueError("No finite numeric site values after filtering")

    def _sorted_values(values, col):
        unique_vals = list(pd.unique(values))
        if col == "weight_by_n":
            ordered = [v for v in [True, False] if v in set(unique_vals)]
            ordered.extend([v for v in unique_vals if v not in set(ordered)])
            return ordered
        numeric_vals = pd.to_numeric(pd.Series(unique_vals), errors="coerce")
        if numeric_vals.notna().all():
            return sorted(float(v) for v in unique_vals)
        return sorted(unique_vals, key=lambda x: str(x))

    col_values = _sorted_values(subset[group_by], group_by)
    row_values = _sorted_values(subset[row_by], row_by)
    if len(col_values) == 0 or len(row_values) == 0:
        raise ValueError("No values available to plot after filtering")

    site_values = subset["site"].to_numpy(dtype=np.float64, copy=False)
    site_min = float(np.min(site_values))
    site_max = float(np.max(site_values))
    if not np.isfinite(site_min) or not np.isfinite(site_max):
        raise ValueError("Site values must be finite")
    if site_min == site_max:
        site_max = site_min + 1.0
    bin_edges = np.linspace(site_min, site_max, bins + 1, dtype=np.float64)

    n_rows = len(row_values)
    n_cols = len(col_values)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        squeeze=False,
        sharex=True,
        sharey=True,
        figsize=(max(4.0 * n_cols, 8.0), max(2.8 * n_rows, 4.0)),
    )

    palette = sns.color_palette("tab10", n_colors=n_cols)
    col_colors = {value: palette[i] for i, value in enumerate(col_values)}

    for row_idx, row_value in enumerate(row_values):
        for col_idx, col_value in enumerate(col_values):
            ax = axes[row_idx, col_idx]
            panel_df = subset.loc[
                (subset[row_by] == row_value) & (subset[group_by] == col_value)
            ]
            if len(panel_df) == 0:
                ax.set_axis_off()
                continue

            panel_sites = panel_df["site"].to_numpy(dtype=np.float64, copy=False)
            panel_weights = panel_df[variable].to_numpy(dtype=np.float64, copy=False)
            valid = np.isfinite(panel_sites) & np.isfinite(panel_weights)
            if not np.any(valid):
                ax.set_axis_off()
                continue

            counts, _ = np.histogram(
                panel_sites[valid], bins=bin_edges, weights=panel_weights[valid]
            )
            ax.bar(
                bin_edges[:-1],
                counts,
                width=np.diff(bin_edges),
                align="edge",
                color=col_colors[col_value],
                edgecolor="none",
                alpha=0.95,
            )
            if y_log:
                ax.set_yscale("log")
                positive_counts = counts[counts > 0]
                if len(positive_counts) > 0:
                    ax.set_ylim(bottom=max(0.8, float(np.min(positive_counts)) * 0.8))

            if row_idx == 0:
                ax.set_title(f"{group_by}={col_value}")
            if col_idx == 0:
                ax.set_ylabel(f"{variable}\n{row_by}={row_value}")
            if row_idx == n_rows - 1:
                ax.set_xlabel("site")

    fig.suptitle(f"{variable} by site ({fixed}={fixed_value})", y=1.02)
    fig.tight_layout()
    return {
        "fig": fig,
        "axes": axes,
        "fixed": fixed,
        "fixed_value": fixed_value,
        "group_by": group_by,
        "row_by": row_by,
        "extra_fixed": extra_fixed_values,
    }


def plot_hist_by_path(path_df, variable, group_by, fixed, bins=15, y_log=True):
    """
    Plot histograms of a path-level variable over a parameter grid.

    - ``fixed`` parameter is held at its first observed value.
    - ``group_by`` parameter defines subplot columns.
    - The remaining unfixed parameter defines subplot rows.
    """
    frame = pd.DataFrame(path_df, copy=False)
    parameter_order = ["weight_by_n", "mu", "rho", "likelihood_threshold"]

    if variable not in frame.columns:
        raise ValueError(f"Column '{variable}' not found in path_df")
    if fixed not in parameter_order:
        raise ValueError(f"'fixed' must be one of {parameter_order}, got '{fixed}'")
    if group_by not in parameter_order:
        raise ValueError(
            f"'group_by' must be one of {parameter_order}, got '{group_by}'"
        )
    if fixed == group_by:
        raise ValueError("'fixed' and 'group_by' must be different")
    if fixed not in frame.columns:
        raise ValueError(f"Column '{fixed}' not found in path_df")
    if group_by not in frame.columns:
        raise ValueError(f"Column '{group_by}' not found in path_df")
    if int(bins) <= 0:
        raise ValueError("'bins' must be a positive integer")
    bins = int(bins)

    row_candidates = [
        col
        for col in parameter_order
        if col not in {fixed, group_by} and col in frame.columns
    ]
    if len(row_candidates) == 0:
        raise ValueError(
            "Could not determine row parameter from remaining unfixed columns; "
            f"found {row_candidates}"
        )
    row_by = row_candidates[0]
    extra_fixed_cols = row_candidates[1:]

    fixed_candidates = frame[fixed].dropna().to_numpy(copy=False)
    if len(fixed_candidates) == 0:
        raise ValueError(f"No non-null values found in fixed column '{fixed}'")
    fixed_value = fixed_candidates[0]
    subset = frame.loc[frame[fixed] == fixed_value].copy()
    if len(subset) == 0:
        raise ValueError(f"No rows available after filtering {fixed}={fixed_value!r}")
    extra_fixed_values = {}
    for col in extra_fixed_cols:
        col_candidates = subset[col].dropna().to_numpy(copy=False)
        if len(col_candidates) == 0:
            continue
        col_value = col_candidates[0]
        subset = subset.loc[subset[col] == col_value]
        extra_fixed_values[col] = col_value

    variable_values = pd.to_numeric(subset[variable], errors="coerce")
    if variable_values.isna().all():
        raise ValueError(f"Column '{variable}' has no numeric values after filtering")
    subset = subset.loc[variable_values.notna()].copy()
    subset[variable] = variable_values.loc[subset.index].to_numpy(dtype=np.float64)

    def _sorted_values(values, col):
        unique_vals = list(pd.unique(values))
        if col == "weight_by_n":
            ordered = [v for v in [True, False] if v in set(unique_vals)]
            ordered.extend([v for v in unique_vals if v not in set(ordered)])
            return ordered
        numeric_vals = pd.to_numeric(pd.Series(unique_vals), errors="coerce")
        if numeric_vals.notna().all():
            return sorted(float(v) for v in unique_vals)
        return sorted(unique_vals, key=lambda x: str(x))

    col_values = _sorted_values(subset[group_by], group_by)
    row_values = _sorted_values(subset[row_by], row_by)
    if len(col_values) == 0 or len(row_values) == 0:
        raise ValueError("No values available to plot after filtering")

    n_rows = len(row_values)
    n_cols = len(col_values)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        squeeze=False,
        sharex=True,
        sharey=True,
        figsize=(max(4.0 * n_cols, 8.0), max(2.8 * n_rows, 4.0)),
    )

    palette = sns.color_palette("tab10", n_colors=n_cols)
    col_colors = {value: palette[i] for i, value in enumerate(col_values)}

    for row_idx, row_value in enumerate(row_values):
        for col_idx, col_value in enumerate(col_values):
            ax = axes[row_idx, col_idx]
            panel_df = subset.loc[
                (subset[row_by] == row_value) & (subset[group_by] == col_value)
            ]
            if len(panel_df) == 0:
                ax.set_axis_off()
                continue

            values = panel_df[variable].to_numpy(dtype=np.float64, copy=False)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                ax.set_axis_off()
                continue

            ax.hist(
                values,
                bins=bins,
                color=col_colors[col_value],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.95,
            )
            if y_log:
                ax.set_yscale("log")

            if row_idx == 0:
                ax.set_title(f"{group_by}={col_value}")
            if col_idx == 0:
                ax.set_ylabel(f"count\n{row_by}={row_value}")
            if row_idx == n_rows - 1:
                ax.set_xlabel(variable)

    fig.suptitle(f"{variable} by path ({fixed}={fixed_value})", y=1.02)
    fig.tight_layout()
    return {
        "fig": fig,
        "axes": axes,
        "fixed": fixed,
        "fixed_value": fixed_value,
        "group_by": group_by,
        "row_by": row_by,
        "extra_fixed": extra_fixed_values,
    }

def _positive_floor(values):
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return 1e-3
    return min(vals)

def plot_perf(df, plot_path=None):
    required_cols = {"elapsed_time", "user_time", "command", "num_samples", "version"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    plot_df = df.loc[:, ["elapsed_time", "user_time", "command", "num_samples", "version"]].copy()
    plot_df = plot_df.dropna(
        subset=["elapsed_time", "user_time", "command", "num_samples", "version"]
    )
    if len(plot_df) == 0:
        raise ValueError("No rows remain after removing missing performance data")

    plot_df["version"] = plot_df["version"].astype(str)

    old_version = "0.4.1" if "0.4.1" in plot_df["version"].unique() else "0.4"
    new_version = "0.5.0" if "0.5.0" in plot_df["version"].unique() else "1.0"
    version_set = set(plot_df["version"].unique())
    if old_version not in version_set or new_version not in version_set:
        raise ValueError("plot_perf requires rows for both old and new versions")

    label_map = {
        "match_ancestors": "Match ancestors",
        "match_samples": "Match samples",
    }
    command_order = [cmd for cmd in ["match_ancestors", "match_samples"] if cmd in plot_df["command"].unique()]
    command_order.extend(
        cmd for cmd in plot_df["command"].drop_duplicates().tolist() if cmd not in command_order
    )
    hue_order = [label_map.get(cmd, cmd.replace("_", " ").title()) for cmd in command_order]
    palette = {
        "Match ancestors": "#f2a65a",
        "Match samples": "#ea801c",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    metric_specs = [
        ("elapsed_time", "Elapsed time"),
        ("user_time", "User time"),
    ]

    for ax, (metric, title) in zip(axes, metric_specs):
        agg = (
            plot_df.groupby(["command", "num_samples", "version"], as_index=False)[metric]
            .mean()
        )
        wide = agg.pivot_table(
            index=["command", "num_samples"],
            columns="version",
            values=metric,
        )
        wide = wide.dropna(subset=[old_version, new_version]).reset_index()
        if len(wide) == 0:
            raise ValueError(f"No paired rows found for {metric}")

        wide["speedup"] = wide[old_version] / wide[new_version]
        sample_order = sorted(wide["num_samples"].unique())
        wide["num_samples"] = pd.Categorical(
            wide["num_samples"],
            categories=sample_order,
            ordered=True,
        )
        wide["command"] = wide["command"].map(
            lambda value: label_map.get(value, value.replace("_", " ").title())
        )

        sns.barplot(
            data=wide,
            x="num_samples",
            y="speedup",
            hue="command",
            hue_order=[label for label in hue_order if label in wide["command"].unique()],
            palette=palette,
            dodge=True,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("Speedup")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, title="Inference step", loc="best", frameon=False)
    if axes[1].legend_ is not None:
        axes[1].legend_.remove()

    plt.tight_layout()
    if plot_path is not None:
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig, axes
