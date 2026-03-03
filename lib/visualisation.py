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
    df,
    path_id,
    likelihood_threshold=1e-13,
    cmap="viridis",
    figsize=(14, 8),
    weight_by_n=None,
    mismatch_ratio=None,
    compared_by=None,
    long_df=None,
):
    """
    Plot per-site likelihood structure for a single path.

    If ``long_df`` is supplied, it is used directly (after filtering) to avoid
    rebuilding long-form likelihood rows on every redraw.
    """
    from likelihoods import make_long_df

    def _filter_by_params(frame):
        frame = frame.copy()
        if weight_by_n is not None and "weight_by_n" in frame.columns:
            frame = frame.loc[frame["weight_by_n"] == weight_by_n]
        if mismatch_ratio is not None and "mismatch_ratio" in frame.columns:
            mismatch_values = pd.to_numeric(frame["mismatch_ratio"], errors="coerce")
            frame = frame.loc[
                np.isclose(mismatch_values, float(mismatch_ratio), equal_nan=False)
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
        long_subset = make_long_df(subset_df)
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

    title_parts = [f"Path {path_id} (child_id={child_label})"]
    if "weight_by_n" in subset_df.columns:
        title_parts.append(_format_param("weight_by_n", bool(subset_df["weight_by_n"].iloc[0])))
    if "mismatch_ratio" in subset_df.columns:
        title_parts.append(_format_param("mismatch_ratio", float(subset_df["mismatch_ratio"].iloc[0])))
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
            ax_heat.axvline(x, color="#ff69b4", linewidth=1.2, alpha=0.95)

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
    ax_k.set_yscale("log")
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
    high_cbar.set_ticklabels([f"{threshold_label}\n+ ε", f"{high_vmax:.2g}"])

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
            color="#ff69b4",
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


def plot_paths_interactively(df):
    """
    Interactive viewer over a combined dataframe with multiple parameter settings.

    - Builds path summary sorted by num_errors.
    - Slider is global path_id (0..max path_id) and does not change with params.
    - Parameter selectors fix non-compared parameters.
    - Compare-by selector plots one column per value of selected compare variable.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError(
            "plot_paths_interactively requires ipywidgets and IPython display"
        ) from exc

    from likelihoods import make_long_df, summarise_paths

    combined_df = pd.DataFrame(df, copy=False)
    required = {
        "path_id",
        "weight_by_n",
        "mismatch_ratio",
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

    path_df = summarise_paths(combined_df).sort_values(
        ["num_errors", "path_id"], kind="mergesort"
    ).reset_index(drop=True)

    # Build long dataframe once for all updates.
    long_base = make_long_df(combined_df)
    repeats = combined_df["k"].to_numpy(dtype=np.int64, copy=False)
    long_base["path_id"] = np.repeat(
        combined_df["path_id"].to_numpy(copy=False), repeats
    )
    for col in ["weight_by_n", "mismatch_ratio", "likelihood_threshold"]:
        if col in combined_df.columns:
            long_base[col] = np.repeat(combined_df[col].to_numpy(copy=False), repeats)

    def _fmt_value(v):
        if isinstance(v, (float, np.floating)):
            return np.format_float_scientific(float(v), precision=2)
        return str(v)

    weight_values = [bool(v) for v in pd.unique(combined_df["weight_by_n"])]
    ordered_weights = [v for v in [True, False] if v in set(weight_values)]
    ordered_weights.extend([v for v in weight_values if v not in set(ordered_weights)])

    mismatch_values = sorted(float(v) for v in pd.unique(combined_df["mismatch_ratio"]))
    threshold_values = sorted(
        float(v) for v in pd.unique(combined_df["likelihood_threshold"])
    )

    weight_buttons = widgets.ToggleButtons(
        options=[(str(v), v) for v in ordered_weights],
        description="weight_by_n",
    )
    mismatch_buttons = widgets.ToggleButtons(
        options=[(_fmt_value(v), v) for v in mismatch_values],
        description="mismatch_ratio",
    )
    threshold_buttons = widgets.ToggleButtons(
        options=[(_fmt_value(v), v) for v in threshold_values],
        description="likelihood_threshold",
    )
    compare_buttons = widgets.ToggleButtons(
        options=[
            ("weight_by_n", "weight_by_n"),
            ("mismatch_ratio", "mismatch_ratio"),
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
        description="path_id",
        continuous_update=False,
        readout=True,
        layout=widgets.Layout(width="95%"),
    )

    status = widgets.HTML()
    output = widgets.Output()

    controls_row = widgets.HBox(
        [weight_buttons, mismatch_buttons, threshold_buttons],
        layout=widgets.Layout(width="100%"),
    )
    compare_row = widgets.HBox([compare_buttons], layout=widgets.Layout(width="100%"))
    ui = widgets.VBox([output, slider, controls_row, compare_row, status])

    def _get_compare_values(name):
        if name == "weight_by_n":
            return ordered_weights
        if name == "mismatch_ratio":
            return mismatch_values
        if name == "likelihood_threshold":
            return threshold_values
        raise ValueError(f"Unknown compare variable: {name}")

    def _parameter_set(compare_name, compare_value):
        params = {
            "weight_by_n": weight_buttons.value,
            "mismatch_ratio": float(mismatch_buttons.value),
            "likelihood_threshold": float(threshold_buttons.value),
        }
        params[compare_name] = compare_value
        return params

    def _find_summary_row(path_id, params):
        subset = path_df.loc[path_df["path_id"] == path_id]
        subset = subset.loc[subset["weight_by_n"] == params["weight_by_n"]]
        subset = subset.loc[
            np.isclose(
                pd.to_numeric(subset["mismatch_ratio"], errors="coerce"),
                float(params["mismatch_ratio"]),
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

        panel_outputs = []
        status_parts = []

        for value in compare_values:
            params = _parameter_set(compare_name, value)
            label = f"{compare_name}={_fmt_value(value)}"
            panel = widgets.Output(layout=widgets.Layout(width=f"{max(30, int(98 / max(1, len(compare_values))))}%"))

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
                        mismatch_ratio=float(params["mismatch_ratio"]),
                        compared_by=compare_name,
                        long_df=long_base,
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

    slider.observe(_on_change, names="value")
    weight_buttons.observe(_on_change, names="value")
    mismatch_buttons.observe(_on_change, names="value")
    threshold_buttons.observe(_on_change, names="value")
    compare_buttons.observe(_on_change, names="value")

    draw_current()
    display(ui)

    return {
        "ui": ui,
        "path_df": path_df,
        "long_df": long_base,
        "slider": slider,
        "weight_buttons": weight_buttons,
        "mismatch_buttons": mismatch_buttons,
        "threshold_buttons": threshold_buttons,
        "compare_buttons": compare_buttons,
    }

