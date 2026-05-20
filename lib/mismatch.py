from concurrent.futures import ThreadPoolExecutor
import os
import threading
from lib import utils
import importlib
importlib.reload(utils)
import numpy as np
import pandas as pd
import tskit
import tsinfer
from tsinfer.inference import SampleMatcher
import msprime
import sgkit
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns

def create_sample_mask(num_samples, subset_size):
    subset_size = min(num_samples, subset_size)
    selected_samples = np.random.choice(range(num_samples), size=subset_size, replace=False)
    sample_mask = np.ones(num_samples, dtype=bool)
    sample_mask[selected_samples] = False
    assert np.sum(~sample_mask) == subset_size

    return sample_mask

def match_ancestors_with_subsample(zarr_path, sample_mask=None, num_threads=190):
    vdata = tsinfer.VariantData(
        zarr_path,
        ancestral_state="variant_mispolarised_ancestral_state",
        sample_mask=sample_mask,
    )
    anc_data = tsinfer.generate_ancestors(vdata,
                                          progress_monitor=True,
                                           num_threads=num_threads)
    
    ancestors_ts = tsinfer.match_ancestors(vdata,
                                           anc_data,
                                           progress_monitor=True,
                                           num_threads=num_threads,
                                           path_compression=False)
    return vdata, ancestors_ts



def _get_subset_doubletons(G, sites_position, sample_mask, window):
    subset_samples = np.flatnonzero(~sample_mask)

    ac = G[:, subset_samples, :].sum(axis=(1, 2))
    dbtn_sites = np.where(ac == 2)[0]
    dbtn_pos = sites_position[dbtn_sites]

    left_pos = dbtn_pos - window / 2
    right_pos = dbtn_pos + window / 2
    left_site = np.searchsorted(sites_position, left_pos, side="left")
    right_site = np.searchsorted(sites_position, right_pos, side="right")

    num_dbtns = len(dbtn_pos)
    ploidy = G.shape[2]
    if num_dbtns == 0:
        dbtn_samples = np.empty((0, 2), dtype=np.int32)
        dbtn_ploidies = np.empty((0, 2), dtype=np.int32)
    else:
        dbtn_G = G[dbtn_sites][:, subset_samples, :]
        _, dbtn_samples_long, dbtn_ploidies_long = np.nonzero(dbtn_G)
        dbtn_samples = subset_samples[dbtn_samples_long.reshape(num_dbtns, 2)].astype(
            np.int32
        )
        dbtn_ploidies = dbtn_ploidies_long.reshape(num_dbtns, 2).astype(np.int32)

    dbtn_haplotype_ids = (dbtn_samples * ploidy + dbtn_ploidies).astype(np.int32)
    return {
        "dbtn_sites": dbtn_sites,
        "dbtn_pos": dbtn_pos,
        "left_pos": left_pos,
        "right_pos": right_pos,
        "left_site": left_site,
        "right_site": right_site,
        "dbtn_samples": dbtn_samples,
        "dbtn_ploidies": dbtn_ploidies,
        "dbtn_haplotype_ids": dbtn_haplotype_ids,
    }


def _resolve_recomb_map(recomb_map):
    if os.path.exists(recomb_map) or os.path.isabs(recomb_map):
        return recomb_map
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [os.path.join(repo_root, recomb_map)]
    if recomb_map.startswith("../"):
        candidates.append(os.path.join(repo_root, recomb_map[3:]))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _recombination_rate_to_dist(rho, positions):
    try:
        return np.diff(rho.get_cumulative_mass(positions))
    except AttributeError:
        return np.diff(positions) * rho


def _recombination_cumsum(rho, positions):
    try:
        return rho.get_cumulative_mass(positions)
    except AttributeError:
        return positions * rho


def _site_recombination_cumsum(sites_position, recomb_map):
    rate_map = msprime.RateMap.read_hapmap(
        _resolve_recomb_map(recomb_map),
        position_col=1,
        rate_col=2,
    )
    genetic_dists = _recombination_rate_to_dist(rate_map, sites_position)
    recomb_cumsum = np.zeros(len(sites_position), dtype=np.float64)
    if len(genetic_dists) > 0:
        recomb_cumsum[1:] = np.cumsum(genetic_dists)
    return recomb_cumsum


def _empty_hmm_df():
    empty_df = pd.DataFrame(
        columns=[
            "match_left",
            "match_right",
            "match_pos_window",
            "match_site_window",
            "num_mismatches",
            "mismatch_rate",
        ]
    )
    empty_df.index.name = "focal_position"
    return empty_df

def count_hmm_mismatches(
    zarr_path,
    sample_mask,    
    window,
    rho=1e-8,
    mu=1e-3,
    num_threads=190,
    vdata=None,
    ancestors_ts=None,
    ds=None,
    global_non_singleton_mask=None,
):
    if vdata is None or ancestors_ts is None:
        vdata, ancestors_ts = match_ancestors_with_subsample(
            zarr_path,
            sample_mask,
            num_threads=num_threads,
        )
        dbtn_pos = []
        dbtn_carriers = []

        for variant in vdata.variants(recode_ancestral=True):
            if variant.site.ancestral_allele == tskit.MISSING_DATA:
                continue
            if len(variant.alleles) != 2:
                continue
            g = variant.genotypes
            carriers = np.flatnonzero(g == 1)
            if carriers.size != 2:
                continue
            if np.any((g != 0) & (g != 1) & (g != tskit.MISSING_DATA)):
                continue
            dbtn_pos.append(variant.site.position)
            dbtn_carriers.extend(carriers.tolist())

        if len(dbtn_pos) == 0:
            return _empty_hmm_df()

        task_site_pos = np.repeat(np.asarray(dbtn_pos), 2)
        task_sample_id = np.asarray(dbtn_carriers, dtype=np.int32)
        left_pos = task_site_pos - window / 2
        right_pos = task_site_pos + window / 2
    else:
        if ds is None:
            ds = sgkit.load_dataset(zarr_path)
        G = ds.call_genotype.values
        sites_position = ds.variant_position.values
        if global_non_singleton_mask is not None:
            G = G[global_non_singleton_mask]
            sites_position = sites_position[global_non_singleton_mask]
        doubleton_data = _get_subset_doubletons(
            G,
            sites_position,
            sample_mask,
            window,
        )
        if len(doubleton_data["dbtn_pos"]) == 0:
            return _empty_hmm_df()

        task_site_pos = np.repeat(doubleton_data["dbtn_pos"], 2)
        task_sample_id = doubleton_data["dbtn_haplotype_ids"].reshape(-1)
        left_pos = np.repeat(doubleton_data["left_pos"], 2)
        right_pos = np.repeat(doubleton_data["right_pos"], 2)

    ancestors_sites_pos = ancestors_ts.sites_position
    task_left = np.searchsorted(ancestors_sites_pos, left_pos, side="left")
    task_right = np.searchsorted(ancestors_sites_pos, right_pos, side="right")

    keep = task_right > task_left
    task_left = task_left[keep]
    task_right = task_right[keep]
    task_site_pos = task_site_pos[keep]
    task_sample_id = task_sample_id[keep]

    num_sites = ancestors_ts.num_sites
    recombination = np.full(num_sites - 1, rho, dtype=np.float64)
    mismatch = np.full(num_sites, mu, dtype=np.float64)

    matcher = SampleMatcher(
        vdata,
        ancestors_ts,
        recombination=recombination,
        mismatch=mismatch,
        engine=tsinfer.C_ENGINE,
        num_threads=0,
        path_compression=False,
    )

    unique_sample_ids = np.unique(task_sample_id)
    hap_cache = {}
    for sid, hap in matcher.variant_data.haplotypes(
        unique_sample_ids,
        sites=matcher.inference_site_id,
        recode_ancestral=True,
    ):
        hap_cache[int(sid)] = np.asarray(hap, dtype=np.int8)

    local = threading.local()

    def get_worker_state():
        if not hasattr(local, "c_matcher"):
            local.c_matcher = matcher.create_matcher_instance()
            local.match = np.empty(matcher.num_sites, dtype=np.int8)
        return local.c_matcher, local.match

    def run_one(i):
        c_matcher, match = get_worker_state()
        left = int(task_left[i])
        right = int(task_right[i])
        hap = hap_cache[int(task_sample_id[i])]
        match.fill(tskit.MISSING_DATA)
        c_matcher.find_path(hap, left, right, match)
        h = hap[left:right]
        m = match[left:right]
        return int(np.count_nonzero((h != m) & (h != tskit.MISSING_DATA)))

    n = len(task_sample_id)
    if num_threads is None:
        num_threads = min(32, os.cpu_count() or 1)

    if num_threads <= 1:
        mismatch_counts = [run_one(i) for i in range(n)]
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            mismatch_counts = list(pool.map(run_one, range(n)))
            
    hmm_df = pd.DataFrame(
        {
            "focal_position": task_site_pos,
            "match_left": ancestors_sites_pos[task_left],
            "match_right": ancestors_sites_pos[task_right - 1],
            "match_pos_window": ancestors_sites_pos[task_right - 1] - ancestors_sites_pos[task_left],
            "match_site_window": task_right - task_left,
            "num_mismatches": mismatch_counts,
        }
    )
    hmm_df = hmm_df.groupby("focal_position", sort=False).agg(
        {
            "match_left": "first",
            "match_right": "first",
            "match_pos_window": "first",
            "match_site_window": "first",
            "num_mismatches": "sum",
        }
    )
    hmm_df["mismatch_rate"] = np.divide(
        hmm_df["num_mismatches"],
        hmm_df["match_site_window"],
        out=np.zeros(len(hmm_df), dtype=float),
        where=hmm_df["match_site_window"].to_numpy() > 0,
    )
    hmm_df.index.name = "focal_position"
    return hmm_df

def count_haplotype_mismatches(
    G,
    G_error_mask,
    sites_position,
    sample_mask,
    window,
    recomb_map=None,
):
    doubleton_data = _get_subset_doubletons(
        G,
        sites_position,
        sample_mask,
        window,
    )

    dbtns = doubleton_data["dbtn_pos"]
    num_dbtns = len(dbtns)
    num_sites = len(sites_position)
    left_pos = doubleton_data["left_pos"]
    right_pos = doubleton_data["right_pos"]
    left_site = doubleton_data["left_site"]
    right_site = doubleton_data["right_site"]
    dbtn_samples = doubleton_data["dbtn_samples"]
    dbtn_ploidies = doubleton_data["dbtn_ploidies"]
    dbtn_haplotype_ids = doubleton_data["dbtn_haplotype_ids"]
    num_samples = np.sum(~sample_mask)
    hap_site_count = np.zeros(num_dbtns, dtype=np.int32)
    num_true_errors = np.zeros(num_dbtns, dtype=np.int32)
    num_obs_errors = np.zeros(num_dbtns, dtype=np.int32)
    num_tp_errors = np.zeros(num_dbtns, dtype=np.int32)
    num_fn_errors = np.zeros(num_dbtns, dtype=np.int32)
    num_fp_errors = np.zeros(num_dbtns, dtype=np.int32)
    num_tn_errors = np.zeros(num_dbtns, dtype=np.int32)
    tpr = np.full(num_dbtns, np.nan, dtype=float)
    fpr = np.full(num_dbtns, np.nan, dtype=float)
    site_true_errors = np.zeros((num_sites, num_samples, 2), dtype=np.int32)
    G_cov = np.zeros((num_sites, num_samples, 2), dtype=np.int32)
    site_obs_errors = np.zeros(num_sites, dtype=np.int32)
    G_error_select = ~G_error_mask
    if recomb_map is None:
        D_left = np.full(num_dbtns, np.nan, dtype=np.float64)
        D_right = np.full(num_dbtns, np.nan, dtype=np.float64)
    else:
        rate_map = msprime.RateMap.read_hapmap(
            _resolve_recomb_map(recomb_map),
            position_col=1,
            rate_col=2,
        )
        map_length = rate_map.sequence_length
        clipped_left_pos = np.clip(left_pos, 0, map_length)
        clipped_dbtns = np.clip(dbtns, 0, map_length)
        clipped_right_pos = np.clip(right_pos, 0, map_length)
        left_D = _recombination_cumsum(rate_map, clipped_left_pos)
        dbtn_D = _recombination_cumsum(rate_map, clipped_dbtns)
        right_D = _recombination_cumsum(rate_map, clipped_right_pos)
        D_left = dbtn_D - left_D
        D_right = right_D - dbtn_D
    D_window = D_left + D_right
    
    for i in tqdm(range(num_dbtns), desc="Haplotype mismatches", leave=False):
        left = left_site[i]
        right = right_site[i]
        samples = dbtn_samples[i]
        ploidies = dbtn_ploidies[i]
        region = slice(left,right)
        hap_site_count[i] = right - left
        local_haps = G[region, samples, ploidies]
        G_cov[region, samples, ploidies] += 1
        true_error_select_full = G_error_select[
            region, samples, ploidies
        ]
        true_error_select = true_error_select_full.max(axis=1)
        obs_error_select = ~(local_haps[:, 0] == local_haps[:, 1])

        num_true = np.sum(true_error_select)
        num_tp = np.sum(np.logical_and(true_error_select, obs_error_select))
        num_obs = np.sum(obs_error_select)
        num_fp = np.sum(obs_error_select > true_error_select)
        site_true_errors[region, samples, ploidies] += true_error_select_full
        site_obs_errors[region] += obs_error_select

        num_true_errors[i] = num_true
        num_obs_errors[i] = num_obs
        num_tp_errors[i] = num_tp
        num_fn_errors[i] = num_true - num_tp
        num_fp_errors[i] = num_fp
        num_tn_errors[i] = np.sum(~obs_error_select == ~true_error_select)
        tpr[i] = num_tp / num_true if num_true > 0 else np.nan
        fpr[i] = num_fp / num_obs if num_obs > 0 else np.nan

    df = pd.DataFrame(
        {
            "hap_site_count": hap_site_count,
            "left_hap_boundary": left_pos,
            "right_hap_boundary": right_pos,
            "num_true_errors": num_true_errors,
            "num_obs_errors": num_obs_errors,
            "obs_error_rate": num_obs_errors/(hap_site_count*2),
            "true_error_rate": num_true_errors/(hap_site_count*2),
            "obs_error_rate_per_bp": (
                num_obs_errors / window if window > 0 else np.nan
            ),
            "true_error_rate_per_bp": (
                num_true_errors / window if window > 0 else np.nan
            ),
            "D_left": D_left,
            "D_right": D_right,
            "D_window": D_window,
            "num_tp_errors": num_tp_errors,
            "num_fn_errors": num_fn_errors,
            "num_fp_errors": num_fp_errors,
            "num_tn_errors": num_tn_errors,
            "tpr": tpr,
            "fpr": fpr,
            "carrier_0": dbtn_haplotype_ids[:, 0],
            "carrier_1": dbtn_haplotype_ids[:, 1],
        },
        index=pd.Index(dbtns, name="focal_position"),
    )

    genotype_coverage = np.sum(G_cov > 0)/G_cov.size
    genotype_prop_duplicate = np.sum(G_cov[G_cov > 0] - 1)/np.sum(G_cov)

    return df, genotype_coverage, genotype_prop_duplicate


def load_ancestor_df(prefix, error, old_sim=False):
    anc_df_path = f"../data/anc_eval/dataframes/{prefix}-gerr_{error}-ser0-mper0-ancestors.csv"
    anc_df = pd.read_csv(anc_df_path)
    if old_sim:
        anc_df = anc_df[anc_df.version == 1.0]
    anc_df = utils.reshape_side_dependent_columns(anc_df)
    anc_df.set_index("focal_position", inplace=True)
    return anc_df


def measure_mismatches(prefix,
                       subset_sizes,
                       window_sizes,
                       seed=1,
                       include_hmm=False,
                       include_anc_df=False,
                       rho=0,
                       hmm_reference_error="disabled",
                       recomb_map="data/HapMapII_GRCh38/genetic_map_Hg38_chr20.txt"):
    np.random.seed(seed)

    zarr_path = f"../data/anc_eval/zarr_vcfs/{prefix}-gerr_disabled-ser0-mper0.zarr"
    ds = sgkit.load_dataset(zarr_path)
    num_samples = ds.sizes["samples"]
    singleton_mask = np.sum(ds.call_genotype.values, axis=(1, 2)) > 1
    sites_position = ds.variant_position.values[singleton_mask]
    
    dfs = []
    meta_rows = []
    meta_df = pd.DataFrame(
        columns=[
            "window_size",
            "subset_size",
            "error",
            "true_error_rate_per_site",
            "true_error_rate_per_bp",
            "genotype_coverage",
            "genotype_prop_duplicate",
        ]
    )
    hmm_vdata = None
    hmm_ancestors_ts = None

    if include_hmm:
        hmm_reference_path = (
            f"../data/anc_eval/zarr_vcfs/{prefix}-gerr_{hmm_reference_error}-ser0-mper0.zarr"
        )
        hmm_vdata, hmm_ancestors_ts = match_ancestors_with_subsample(
            hmm_reference_path,
            sample_mask=None,
        )

    for window in tqdm(window_sizes, desc="Windows"):
        for n in tqdm(subset_sizes, desc=f"Subset sizes (window={window})", leave=False):
            sample_mask = create_sample_mask(num_samples, subset_size=n)
            for error in tqdm(["enabled", "disabled"], desc=f"Errors (n={n})", leave=False):
                zarr_path = f"../data/anc_eval/zarr_vcfs/{prefix}-gerr_{error}-ser0-mper0.zarr"
                ds = sgkit.load_dataset(zarr_path)
                G = ds.call_genotype.values[singleton_mask, :, :]
                G_error_mask = ds.call_genotype_error_mask.values[singleton_mask, :, :]

                num_true_errors = (~G_error_mask).sum()
                true_error_rate_per_site = (
                    num_true_errors / G_error_mask.size
                    if G_error_mask.size > 0
                    else np.nan
                )
                site_span = (
                    np.max(sites_position) - np.min(sites_position)
                    if len(sites_position) > 0
                    else 0
                )
                true_error_rate_per_bp = (
                    num_true_errors / site_span
                    if site_span > 0
                    else np.nan
                )
                df, genotype_coverage, genotype_prop_duplicate = count_haplotype_mismatches(
                    G,
                    G_error_mask,
                    sites_position,
                    sample_mask,
                    window,
                    recomb_map=recomb_map,
                )

                if include_anc_df:
                    anc_df = load_ancestor_df(prefix, error)
                    assert len(anc_df) > 0
                    df = df.join(anc_df, how="left")

                # if include_hmm:
                #     hmm_df = count_hmm_mismatches(
                #         zarr_path,
                #         sample_mask,
                #         window,
                #         vdata=hmm_vdata,
                #         ancestors_ts=hmm_ancestors_ts,
                #         ds=ds,
                #         rho=rho,
                #     )
                #     df = df.join(hmm_df, how="left")

                df["window_size"] = window
                df["sample_size"] = n
                df["genotyping_error"] = error
                dfs.append(df)
                meta_rows.append(
                    {
                        "window_size": window,
                        "subset_size": n,
                        "error": error,
                        "true_error_rate_per_site": true_error_rate_per_site,
                        "true_error_rate_per_bp": true_error_rate_per_bp,
                        "genotype_coverage": genotype_coverage,
                        "genotype_prop_duplicate": genotype_prop_duplicate,
                    }
                )

    meta_df = pd.DataFrame.from_records(meta_rows, columns=meta_df.columns)
    return pd.concat(dfs), meta_df
