import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

tsinfer_path = os.path.abspath("/well/kelleher/users/uuc395/tsinfer")
sys.path.append(tsinfer_path)
import tsinfer


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
    color_dict={"new": "#ea801c","old": "#1a80bb","true": "#b8b8b8"},
):
    #df = df.copy()
    #df = df.drop_duplicates(subset=["inferred_node", "version"], keep="first")
    if cutoffs is None:
        cutoffs = np.unique(np.percentile(df["inferred_time"], np.linspace(0, 100, 9)))

    y_units = "sites" if type == "site" else "bp"
    if var == "span":
        var_col = "inferred_span"
        true_col = "true_span"
        var_labels = ["True", "Inferred (0.5.0)", "Inferred (0.4.1)"]
        colors = [color_dict["true"], color_dict["new"], color_dict["old"]]
    elif var == "overshoot":
        var_col = "overshoot"
        true_col = None
        var_labels = ["0.5.0", "0.4.1"]
        colors = [color_dict["new"], color_dict["old"]]
    else:
        raise ValueError("var must be 'span' or 'overshoot'")

    df["frequency_bin"] = pd.cut(df["inferred_time"], bins=cutoffs, include_lowest=True)
    df["frequency_bin"] = df["frequency_bin"].apply(lambda x: f"({x.left:.2f}, {x.right:.2f}]")
    #return df
    parts = []
    if true_col is not None:
        parts.append(df[["frequency_bin", true_col]].rename(columns={true_col: "value"}).assign(type="True"))
    for version in ["0.4.1", "0.5.0"]:
        part = df.loc[df["version"]==version, ["frequency_bin", var_col]].rename(columns={var_col: "value"}).assign(type=f"Inferred ({version})")
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
        plt.close(fig)
    else:
        plt.show()




def plot_path_likelihood(
    df, path_id, likelihood_threshold=1e-13, cmap="viridis", figsize=(14, 8)
):
    """
    Plot per-site likelihood structure for a single path.

    The main pane is a rectangular heatmap where each site column is split into
    k blocks (one per likelihood value). Block heights vary as 1/k so each column
    spans the same total height. A lower pane shows k per site as a simple box/bar
    track at half height relative to the heatmap.
    """
    from likelihoods import make_long_df

    df = pd.DataFrame(df, copy=False)
    if "path_id" not in df.columns:
        raise ValueError("DataFrame must contain 'path_id'")

    subset_df = df.loc[df["path_id"] == path_id].copy()
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

    long_df = make_long_df(subset_df)
    long_df = long_df.sort_values(
        ["site", "full_likelihood", "likelihood_node_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped_likelihoods = {
        site: grp["full_likelihood"].to_numpy(dtype=np.float64, copy=False)
        for site, grp in long_df.groupby("site", sort=False)
    }

    polygons = []
    values = []
    site_values = site_rows["site"].to_numpy(copy=False)
    k_values = site_rows["k"].to_numpy(dtype=np.int64, copy=False)

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

        y_edges = np.linspace(0.0, 1.0, k + 1)
        for row, value in enumerate(site_likelihoods):
            polygons.append(
                [
                    (col, y_edges[row]),
                    (col + 1, y_edges[row]),
                    (col + 1, y_edges[row + 1]),
                    (col, y_edges[row + 1]),
                ]
            )
            values.append(value)

    if len(values) == 0:
        raise ValueError(f"No likelihood values available for path_id={path_id}")

    values = np.asarray(values, dtype=np.float64)
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
    ax_heat.set_title(f"Path {path_id} (child_id={child_label})")

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
    high_cbar.set_ticks([legend_min, high_vmax])
    high_cbar.set_ticklabels([f"{threshold_label} + ε", f"{high_vmax:.2g}"])

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

    handles = [
        Line2D([0], [0], color="red", lw=1.8, label="mismatch"),
        Line2D([0], [0], color="orange", lw=1.8, label="recombination"),
    ]
    event_height = total_height * 0.14
    event_y = max(bottom, threshold_y - total_height * 0.20)
    event_ax = fig.add_axes([legend_x, event_y, 0.16, event_height])
    event_ax.set_axis_off()
    event_ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )

    plt.show()
