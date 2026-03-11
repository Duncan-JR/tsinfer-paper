import sgkit
import pandas as pd
import numpy as np
import os
import sys
import concurrent.futures
import itertools
tsinfer_path = os.path.abspath("/home/duncan/trees/tsinfer")
if tsinfer_path not in sys.path:
    sys.path.insert(0, tsinfer_path)
import tsinfer
import tsinfer.logging as tsinfer_logging
import msprime
import math
try:
    from . import real_data
except ImportError:
    from lib import real_data


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
    df = tsinfer_logging.load_hmm_log(
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
