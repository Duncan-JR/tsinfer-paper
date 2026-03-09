import sgkit
import pandas as pd
import numpy as np
import os
import sys
import concurrent.futures
import itertools
tsinfer_path = os.path.abspath("/home/duncan/trees/tsinfer")
modules_path = os.path.abspath("/home/duncan/trees/tsinfer-paper/lib")
sys.path.append(modules_path)
sys.path.append(tsinfer_path)
import tsinfer
import likelihoods
import real_data
import struct
import msprime
import math


SWITCH_CAUSE_NAMES = {
    0: "none",
    1: "forced_end",
    2: "score_driven",
    3: "ambiguous",
}


def load_hmm_log(path, likelihood_threshold=1e-13):
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 12:
        raise ValueError("HMM log is truncated")
    if data[:8] != b"TSILHMML":
        raise ValueError(f"Bad magic: {data[:8]!r}")
    version = struct.unpack_from("<I", data, 8)[0]
    if version != 7:
        raise ValueError(f"Unsupported version: {version}. Expected version 7.")

    view = memoryview(data)
    path_begin_child_id = {}
    path_begin_child_time = {}
    selected_map = {}

    site_path_ids = []
    site_sites = []
    site_k = []
    site_likelihoods = []
    site_likelihood_nodes = []
    site_recombination_required = []
    prop_min_likelihood = []
    prop_max_likelihood = []

    pos = 12
    size = len(data)
    try:
        while pos < size:
            rec_type = data[pos]
            pos += 1
            if rec_type == 1:  # PATH_BEGIN
                path_id, _, _, child_id, child_time = struct.unpack_from(
                    "<Qiiid", data, pos
                )
                pos += 28
                path_begin_child_id[path_id] = child_id
                path_begin_child_time[path_id] = child_time
            elif rec_type == 2:  # SITE_VALUES
                path_id, site, k = struct.unpack_from("<QiI", data, pos)
                pos += 16
                likelihoods = np.frombuffer(view, dtype="<f8", count=k, offset=pos)
                pos += 8 * k
                likelihood_nodes = np.frombuffer(view, dtype="<i4", count=k, offset=pos)
                pos += 4 * k
                recombination_required = np.frombuffer(
                    view, dtype=np.int8, count=k, offset=pos
                )
                pos += k

                site_path_ids.append(path_id)
                site_sites.append(site)
                site_k.append(k)
                site_likelihoods.append(likelihoods)
                site_likelihood_nodes.append(likelihood_nodes)
                site_recombination_required.append(recombination_required)

                if k == 0:
                    prop_min_likelihood.append(0.0)
                    prop_max_likelihood.append(0.0)
                else:
                    prop_min_likelihood.append(
                        np.mean(np.equal(likelihoods, likelihood_threshold))
                    )
                    prop_max_likelihood.append(np.mean(np.equal(likelihoods, 1.0)))
            elif rec_type == 3:  # PATH_END
                _, _, _ = struct.unpack_from("<QiQ", data, pos)
                pos += 20
            elif rec_type == 4:  # SELECTED_NODE
                (
                    path_id,
                    site,
                    selected_node,
                    selected_mismatch,
                    selected_recombination,
                    selected_switch_cause,
                ) = struct.unpack_from("<Qiibbb", data, pos)
                pos += 19
                selected_map[(path_id, site)] = (
                    selected_node,
                    selected_mismatch,
                    selected_recombination,
                    selected_switch_cause,
                )
            else:
                raise ValueError(f"Unknown record type: {rec_type}")

        if pos != size:
            raise ValueError("Malformed HMM log")
    except struct.error as exc:
        raise ValueError("HMM log is truncated or malformed") from exc

    n_sites = len(site_path_ids)
    if n_sites == 0:
        df = pd.DataFrame(
            {
                "path_id": pd.Series(dtype=np.uint64),
                "site": pd.Series(dtype=np.int32),
                "k": pd.Series(dtype=np.int64),
                "likelihoods": pd.Series(dtype=object),
                "likelihood_nodes": pd.Series(dtype=object),
                "recombination_required": pd.Series(dtype=object),
                "selected_node": pd.Series(dtype=np.int32),
                "selected_mismatch": pd.Series(dtype=np.int8),
                "selected_recombination": pd.Series(dtype=np.int8),
                "selected_switch_cause": pd.Series(dtype=object),
                "child_id": pd.Series(dtype=np.int32),
                "child_time": pd.Series(dtype=np.float64),
                "prop_min_likelihood": pd.Series(dtype=np.float64),
                "prop_max_likelihood": pd.Series(dtype=np.float64),
            }
        )
    else:
        selected_node = np.full(n_sites, -1, dtype=np.int32)
        selected_mismatch = np.full(n_sites, -1, dtype=np.int8)
        selected_recombination = np.full(n_sites, -1, dtype=np.int8)
        selected_switch_cause_code = np.zeros(n_sites, dtype=np.uint8)
        for index, (path_id, site) in enumerate(zip(site_path_ids, site_sites)):
            values = selected_map.get((path_id, site))
            if values is not None:
                selected_node[index] = values[0]
                selected_mismatch[index] = values[1]
                selected_recombination[index] = values[2]
                selected_switch_cause_code[index] = values[3]

        selected_switch_cause = np.fromiter(
            (
                SWITCH_CAUSE_NAMES.get(int(code), f"unknown_{int(code)}")
                for code in selected_switch_cause_code
            ),
            dtype=object,
            count=n_sites,
        )

        child_id = np.fromiter(
            (path_begin_child_id.get(path_id, -1) for path_id in site_path_ids),
            dtype=np.int32,
            count=n_sites,
        )
        child_time = np.fromiter(
            (path_begin_child_time.get(path_id, np.nan) for path_id in site_path_ids),
            dtype=np.float64,
            count=n_sites,
        )

        df = pd.DataFrame(
            {
                "path_id": site_path_ids,
                "site": site_sites,
                "k": site_k,
                "likelihoods": site_likelihoods,
                "likelihood_nodes": site_likelihood_nodes,
                "recombination_required": site_recombination_required,
                "selected_node": selected_node,
                "selected_mismatch": selected_mismatch,
                "selected_recombination": selected_recombination,
                "selected_switch_cause": selected_switch_cause,
                "child_id": child_id,
                "child_time": child_time,
                "prop_min_likelihood": np.asarray(
                    prop_min_likelihood, dtype=np.float64
                ),
                "prop_max_likelihood": np.asarray(
                    prop_max_likelihood, dtype=np.float64
                ),
            }
        )

    df["num_min_likelihood"] = (df.prop_min_likelihood * df.k).astype(np.int64)
    df["num_max_likelihood"] = (df.prop_max_likelihood * df.k).astype(np.int64)
    return df


def summarise_paths(df):
    """
    Aggregate per-site HMM log rows into one row per path_id and parameter combo.

    If present in the input, parameter columns
    ("weight_by_n", "mu", "rho", "likelihood_threshold")
    are included in grouping and output.
    """
    df = pd.DataFrame(df, copy=False)
    required = {
        "path_id",
        "child_id",
        "selected_recombination",
        "selected_switch_cause",
        "selected_mismatch",
        "num_max_likelihood",
        "num_min_likelihood",
        "prop_max_likelihood",
        "prop_min_likelihood",
        "k",
    }
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame missing required columns: {missing_str}")

    parameter_columns = [
        col
        for col in ["weight_by_n", "mu", "rho", "likelihood_threshold"]
        if col in df.columns
    ]
    group_columns = ["path_id", *parameter_columns]

    if len(df) == 0:
        out = {
            "path_id": pd.Series(dtype=np.uint64),
            "child_id": pd.Series(dtype=np.int32),
            "num_switches": pd.Series(dtype=np.int64),
            "num_switches_forced_end": pd.Series(dtype=np.int64),
            "num_switches_score_driven": pd.Series(dtype=np.int64),
            "num_switches_ambiguous": pd.Series(dtype=np.int64),
            "num_mismatches": pd.Series(dtype=np.int64),
            "num_errors": pd.Series(dtype=np.int64),
            "prop_tie_breaks": pd.Series(dtype=np.float64),
            "mean_num_max_likelihood": pd.Series(dtype=np.float64),
            "mean_num_min_likelihood": pd.Series(dtype=np.float64),
            "mean_prop_max_likelihood": pd.Series(dtype=np.float64),
            "mean_prop_min_likelihood": pd.Series(dtype=np.float64),
            "mean_k": pd.Series(dtype=np.float64),
            "max_k": pd.Series(dtype=np.int64),
        }
        for col in parameter_columns:
            out[col] = pd.Series(dtype=df[col].dtype)
        ordered = [
            "path_id",
            *parameter_columns,
            "child_id",
            "num_switches",
            "num_switches_forced_end",
            "num_switches_score_driven",
            "num_switches_ambiguous",
            "num_mismatches",
            "num_errors",
            "prop_tie_breaks",
            "mean_num_max_likelihood",
            "mean_num_min_likelihood",
            "mean_prop_max_likelihood",
            "mean_prop_min_likelihood",
            "mean_k",
            "max_k",
        ]
        return pd.DataFrame(out)[ordered]

    child_counts = df.groupby(group_columns, sort=False)["child_id"].nunique(
        dropna=False
    )
    bad = child_counts[child_counts > 1]
    if len(bad) > 0:
        bad_ids = ", ".join(str(x) for x in bad.index[:10])
        raise ValueError(
            "Each (path_id, parameter combination) must map to exactly one child_id; "
            f"inconsistent groups include: {bad_ids}"
        )

    grouped = df.groupby(group_columns, sort=False)

    switch_flag = pd.Series(
        df["selected_recombination"].to_numpy(dtype=np.int64, copy=False) > 0,
        index=df.index,
    )
    mismatch_flag = pd.Series(
        df["selected_mismatch"].to_numpy(dtype=np.int64, copy=False) > 0,
        index=df.index,
    )
    tie_flag = pd.Series(
        df["num_max_likelihood"].to_numpy(dtype=np.int64, copy=False) > 1,
        index=df.index,
    )
    switch_cause = pd.Series(
        df["selected_switch_cause"].to_numpy(dtype=object, copy=False),
        index=df.index,
    )

    group_keys = [df[col] for col in group_columns]

    summary = grouped.agg(
        child_id=("child_id", "first"),
        mean_num_max_likelihood=("num_max_likelihood", "mean"),
        mean_num_min_likelihood=("num_min_likelihood", "mean"),
        mean_prop_max_likelihood=("prop_max_likelihood", "mean"),
        mean_prop_min_likelihood=("prop_min_likelihood", "mean"),
        mean_k=("k", "mean"),
        max_k=("k", "max"),
    )
    summary["num_switches"] = (
        switch_flag.groupby(group_keys, sort=False).sum().astype(np.int64)
    )
    summary["num_switches_forced_end"] = (
        ((switch_cause == "forced_end") & switch_flag)
        .groupby(group_keys, sort=False)
        .sum()
        .astype(np.int64)
    )
    summary["num_switches_score_driven"] = (
        ((switch_cause == "score_driven") & switch_flag)
        .groupby(group_keys, sort=False)
        .sum()
        .astype(np.int64)
    )
    summary["num_switches_ambiguous"] = (
        ((switch_cause == "ambiguous") & switch_flag)
        .groupby(group_keys, sort=False)
        .sum()
        .astype(np.int64)
    )
    summary["num_mismatches"] = (
        mismatch_flag.groupby(group_keys, sort=False).sum().astype(np.int64)
    )
    summary["num_errors"] = summary["num_switches"] + summary["num_mismatches"]
    summary["prop_tie_breaks"] = tie_flag.groupby(group_keys, sort=False).mean()

    summary = summary.reset_index()
    ordered = [
        "path_id",
        *parameter_columns,
        "child_id",
        "num_switches",
        "num_switches_forced_end",
        "num_switches_score_driven",
        "num_switches_ambiguous",
        "num_mismatches",
        "num_errors",
        "prop_tie_breaks",
        "mean_num_max_likelihood",
        "mean_num_min_likelihood",
        "mean_prop_max_likelihood",
        "mean_prop_min_likelihood",
        "mean_k",
        "max_k",
    ]
    return summary[ordered]


def run_1kgp_inference(subset_num_sites,
                        subset_num_samples,
                        output_path,
                        weight_by_n=True,
                        mu=1e-20,
                        rho=1e-2,
                        likelihood_threshold=1e-13,
                        zarr_path="/home/duncan/trees/tsinfer-paper/data/chr17/data.zarr/",
                        region_mask_name="variant_all_subset_chr17p_region_filterNton23_site_density_threshold_sites_per_kbp_5_window_size_100000_mask",
                        store_likelihoods=False,
                        run_label=None,
                        likelihoods_csv_path=None,
                        return_df=True,
                        ):
    

    wbn = "on" if weight_by_n else "off"
    run_label_str = "" if run_label is None else f"_{str(run_label).replace(os.sep, '_')}"
    def make_path(filename):
        name = (
            f"subset_s{subset_num_sites}_n{subset_num_samples}_mu{mu}_rho{rho}"
            f"_wbn_{wbn}{run_label_str}_{filename}"
        )
        return os.path.join(output_path, name)
    
    variant_data = real_data.make_1kgp_variant_data(subset_num_sites,
                           subset_num_samples,
                           zarr_path,
                           region_mask_name,
                        )
    anc_data =  tsinfer.generate_ancestors(variant_data,
                                           progress_monitor=True,
                                           path=make_path("ancestors.zarr"),
    )

    ancestors_ts = tsinfer.match_ancestors(
        variant_data,
        anc_data,
        progress_monitor=True,
        hmm_likelihood_log=make_path("match_ancestors.bin"),
    )
    ancestors_ts.dump(make_path("ancestors.trees"))

    num_sites = ancestors_ts.num_sites
    recombination = np.full(num_sites-1, rho)
    mismatch = np.full(num_sites, mu)
    ts = tsinfer.match_samples(
        variant_data,
        ancestors_ts,
        progress_monitor=True,
        hmm_likelihood_log=make_path("match_samples.bin"),
        weight_by_n=weight_by_n,
        likelihood_threshold=likelihood_threshold,
        recombination=recombination,
        mismatch=mismatch
    )
    ts.dump(make_path("final.trees"))

    print('Generating dataframe')
    df = likelihoods.load_hmm_log(
        make_path("match_samples.bin"),
        likelihood_threshold=likelihood_threshold,
    )
    df["mu"] = float(mu)
    df["rho"] = float(rho)
    df["weight_by_n"] = weight_by_n
    df["likelihood_threshold"] = likelihood_threshold
    csv_path = (
        make_path("likelihoods.csv")
        if likelihoods_csv_path is None
        else likelihoods_csv_path
    )
    if store_likelihoods:
        print('Writing dataframe to disk')
        csv_dir = os.path.dirname(csv_path)
        if csv_dir != "":
            os.makedirs(csv_dir, exist_ok=True)
        df.to_csv(csv_path, index=None)

    if return_df:
        return df
    if store_likelihoods:
        return csv_path
    return None


def run_1kgp_comparison(subset_num_sites,
                        subset_num_samples,
                        output_path,
                        weight_by_n_array=None,
                        mu_array=None,
                        rho_array=None,
                        likelihood_threshold_array=None,
                        zarr_path="/home/duncan/trees/tsinfer-paper/data/chr17/data.zarr/",
                        region_mask_name="variant_all_subset_chr17p_region_filterNton23_site_density_threshold_sites_per_kbp_5_window_size_100000_mask",
                        store_likelihoods=False,
                        ):
    if weight_by_n_array is None:
        weight_by_n_array = [True]
    if mu_array is None:
        mu_array = [1e-20]
    if rho_array is None:
        rho_array = [1e-2]
    if likelihood_threshold_array is None:
        likelihood_threshold_array = [1e-13]
    if store_likelihoods:
        raise ValueError(
            "run_1kgp_comparison currently runs in-memory only: "
            "set store_likelihoods=False"
        )

    weight_by_n_values = list(weight_by_n_array)
    mu_values = list(mu_array)
    rho_values = list(rho_array)
    likelihood_threshold_values = list(likelihood_threshold_array)

    if len(weight_by_n_values) == 0:
        raise ValueError("weight_by_n_array must contain at least one value")
    if len(mu_values) == 0:
        raise ValueError("mu_array must contain at least one value")
    if len(rho_values) == 0:
        raise ValueError("rho_array must contain at least one value")
    if len(likelihood_threshold_values) == 0:
        raise ValueError("likelihood_threshold_array must contain at least one value")

    dfs = []
    for weight_by_n in weight_by_n_values:
        for mu in mu_values:
            for rho in rho_values:
                for likelihood_threshold in likelihood_threshold_values:
                    print(
                        f"Running inference with mu={mu}, rho={rho}, "
                        f"likelihood_threshold={likelihood_threshold}, "
                        f"weight_by_n={weight_by_n}"
                    )
                    df = run_1kgp_inference(
                        subset_num_sites=subset_num_sites,
                        subset_num_samples=subset_num_samples,
                        output_path=output_path,
                        weight_by_n=weight_by_n,
                        mu=mu,
                        rho=rho,
                        likelihood_threshold=likelihood_threshold,
                        zarr_path=zarr_path,
                        region_mask_name=region_mask_name,
                        store_likelihoods=False,
                    )
                    dfs.append(df)

    if len(dfs) == 0:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def _run_1kgp_inference_to_disk(task):
    csv_path = run_1kgp_inference(
        subset_num_sites=task["subset_num_sites"],
        subset_num_samples=task["subset_num_samples"],
        output_path=task["output_path"],
        weight_by_n=task["weight_by_n"],
        mu=task["mu"],
        rho=task["rho"],
        likelihood_threshold=task["likelihood_threshold"],
        zarr_path=task["zarr_path"],
        region_mask_name=task["region_mask_name"],
        store_likelihoods=True,
        run_label=task["run_label"],
        likelihoods_csv_path=task["csv_path"],
        return_df=False,
    )
    return task["task_index"], csv_path


def run_1kgp_comparison_in_parallel(subset_num_sites,
                                    subset_num_samples,
                                    output_path,
                                    weight_by_n_array=None,
                                    mu_array=None,
                                    rho_array=None,
                                    likelihood_threshold_array=None,
                                    zarr_path="/home/duncan/trees/tsinfer-paper/data/chr17/data.zarr/",
                                    region_mask_name="variant_all_subset_chr17p_region_filterNton23_site_density_threshold_sites_per_kbp_5_window_size_100000_mask",
                                    store_likelihoods=False,
                                    max_workers=None,
                                    ):
    if weight_by_n_array is None:
        weight_by_n_array = [True]
    if mu_array is None:
        mu_array = [1e-20]
    if rho_array is None:
        rho_array = [1e-2]
    if likelihood_threshold_array is None:
        likelihood_threshold_array = [1e-13]

    weight_by_n_values = list(weight_by_n_array)
    mu_values = list(mu_array)
    rho_values = list(rho_array)
    likelihood_threshold_values = list(likelihood_threshold_array)

    if len(weight_by_n_values) == 0:
        raise ValueError("weight_by_n_array must contain at least one value")
    if len(mu_values) == 0:
        raise ValueError("mu_array must contain at least one value")
    if len(rho_values) == 0:
        raise ValueError("rho_array must contain at least one value")
    if len(likelihood_threshold_values) == 0:
        raise ValueError("likelihood_threshold_array must contain at least one value")

    os.makedirs(output_path, exist_ok=True)

    tasks = []
    combos = itertools.product(
        weight_by_n_values, mu_values, rho_values, likelihood_threshold_values
    )
    for task_index, (weight_by_n, mu, rho, likelihood_threshold) in enumerate(combos):
        run_label = f"cmp_{task_index:04d}"
        csv_name = f"comparison_{run_label}_likelihoods.csv"
        csv_path = os.path.join(output_path, csv_name)
        tasks.append(
            {
                "task_index": task_index,
                "subset_num_sites": subset_num_sites,
                "subset_num_samples": subset_num_samples,
                "output_path": output_path,
                "weight_by_n": weight_by_n,
                "mu": mu,
                "rho": rho,
                "likelihood_threshold": likelihood_threshold,
                "zarr_path": zarr_path,
                "region_mask_name": region_mask_name,
                "run_label": run_label,
                "csv_path": csv_path,
            }
        )

    if len(tasks) == 0:
        return pd.DataFrame()

    completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_1kgp_inference_to_disk, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                completed.append(future.result())
            except Exception as exc:
                raise RuntimeError("A parallel 1KGP inference task failed") from exc

    completed.sort(key=lambda x: x[0])
    csv_paths = [path for _, path in completed]
    combined_df = pd.concat(
        (pd.read_csv(path) for path in csv_paths),
        ignore_index=True,
    )

    if not store_likelihoods:
        for path in csv_paths:
            if os.path.exists(path):
                os.remove(path)

    return combined_df


def make_long_df(df):
    """
    Flatten per-site likelihood arrays into one row per likelihood value.
    """
    df = pd.DataFrame(df, copy=False)
    required = {
        "path_id",
        "child_id",
        "child_time",
        "site",
        "k",
        "likelihoods",
        "likelihood_nodes",
        "recombination_required",
        "selected_node",
        "selected_mismatch",
        "selected_recombination",
        "selected_switch_cause",
        "num_min_likelihood",
        "num_max_likelihood",
    }
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame missing required columns: {missing_str}")

    if len(df) == 0:
        raise ValueError("Input DataFrame is empty")

    k = df["k"].to_numpy(dtype=np.int64, copy=False)
    if np.any(k < 0):
        raise ValueError("Column 'k' must be non-negative")

    likelihoods = df["likelihoods"].to_numpy(dtype=object, copy=False)
    likelihood_nodes = df["likelihood_nodes"].to_numpy(dtype=object, copy=False)
    recombination_required = df["recombination_required"].to_numpy(
        dtype=object, copy=False
    )

    # Validate payload lengths once to fail early on malformed rows.
    like_lens = np.fromiter(
        (len(x) for x in likelihoods), dtype=np.int64, count=len(df)
    )
    node_lens = np.fromiter(
        (len(x) for x in likelihood_nodes), dtype=np.int64, count=len(df)
    )
    recombination_lens = np.fromiter(
        (len(x) for x in recombination_required), dtype=np.int64, count=len(df)
    )
    if not np.array_equal(like_lens, k):
        raise ValueError("Lengths in 'likelihoods' do not match column 'k'")
    if not np.array_equal(node_lens, k):
        raise ValueError("Lengths in 'likelihood_nodes' do not match column 'k'")
    if not np.array_equal(recombination_lens, k):
        raise ValueError("Lengths in 'recombination_required' do not match column 'k'")

    if k.sum() == 0:
        return pd.DataFrame(
            {
                "path_id": pd.Series(dtype=np.int32),
                "child_id": pd.Series(dtype=np.int32),
                "child_time": pd.Series(dtype=np.float64),
                "prop_min_likelihood": pd.Series(dtype=np.float64),
                "prop_max_likelihood": pd.Series(dtype=np.float64),
                "site": pd.Series(dtype=df["site"].dtype),
                "k": pd.Series(dtype=np.int64),
                "likelihood_node_id": pd.Series(dtype=np.int32),
                "likelihood": pd.Series(dtype=np.float64),
                "recombination_required": pd.Series(dtype=np.int8),
                "selected_node": pd.Series(dtype=np.int32),
                "selected_mismatch": pd.Series(dtype=np.int8),
                "selected_recombination": pd.Series(dtype=np.int8),
                "selected_switch_cause": pd.Series(dtype=object),
                "num_min_likelihood": pd.Series(dtype=np.int64),
                "num_max_likelihood": pd.Series(dtype=np.int64),
            }
        )

    likelihood_flat = np.concatenate(likelihoods)
    return pd.DataFrame(
        {
            "path_id": np.repeat(df["path_id"].to_numpy(dtype=np.int32, copy=False), k),
            "child_id": np.repeat(df["child_id"].to_numpy(dtype=np.int32, copy=False), k),
            "child_time": np.repeat(
                df["child_time"].to_numpy(dtype=np.float64, copy=False), k
            ),
            "site": np.repeat(df["site"].to_numpy(copy=False), k),
            "k": np.repeat(k, k),
            "likelihood": likelihood_flat,
            "likelihood_node_id": np.concatenate(likelihood_nodes),
            "recombination_required": np.concatenate(recombination_required),
            "selected_node": np.repeat(
                df["selected_node"].to_numpy(dtype=np.int32, copy=False), k
            ),
            "selected_mismatch": np.repeat(
                df["selected_mismatch"].to_numpy(dtype=np.int8, copy=False), k
            ),
            "selected_recombination": np.repeat(
                df["selected_recombination"].to_numpy(dtype=np.int8, copy=False), k
            ),
            "selected_switch_cause": np.repeat(
                df["selected_switch_cause"].to_numpy(dtype=object, copy=False), k
            ),
            "num_min_likelihood": np.repeat(
                df["num_min_likelihood"].to_numpy(dtype=np.int64, copy=False), k
            ),
            "num_max_likelihood": np.repeat(
                df["num_max_likelihood"].to_numpy(dtype=np.int64, copy=False), k
            ),
        }
    )

def adjust_rho(rho, n):
    return (rho/n)/(1 - rho + rho/n)

def calculate_upper_k(m, eps, mu, rho, n=1):
    a = math.log(mu/(1 - mu))
    b = math.log(adjust_rho(rho, n))
    e = math.log(eps)

    return math.ceil((e - m*b)/a)

def calculate_upper_m(k, eps, mu, rho, n=1):

    a = math.log(mu/(1 - mu))
    b = math.log(adjust_rho(rho, n))
    e = math.log(eps)

    return math.ceil((e - k*a)/b)

def likelihood(k, m, mu, rho, n=1):
    return (mu / (1 - mu))**k * (adjust_rho(rho, n))**m

def enumerate_possible_likelihoods(eps, mu, rho, n=1):
    """
    Given a likelihood threshold eps, recombination prob. rho and mutation prob. mu,
    calculate the range of all possible combinations of k (num. mismatches) and m
    (num. switches) relative to a perfect path that result in a likelihood bigger
    than the threshold. All paths with k or m values outside of this range will
    have the same likelihood equal to eps. Also, for each combination of k and m,
    calculate the associated likelihood.
    """
    if mu < 0 or mu >= 0.5:
        raise ValueError("mu must be in the interval [0, 0.5)")
    if rho < 0 or rho > 1:
        raise ValueError("rho must be in the interval [0, 1]")
    if n < 1 or not isinstance(n, int):
        raise ValueError("n must be an integer >= 1")
    
    k_upper = calculate_upper_k(0, eps, mu, rho, n)
    m_upper = calculate_upper_m(0, eps, mu, rho, n)
    lik_mat = np.full([k_upper, m_upper], eps)
    range_pairs = []
    for k in range(0, k_upper):
        m_upper = calculate_upper_m(k, eps, mu, rho, n)
        range_pairs.append(((k, 0), (k, m_upper - 1)))
        for m in range(0, m_upper):
            lik_mat[k, m] = likelihood(k, m, mu, rho, n)

    return range_pairs, lik_mat
