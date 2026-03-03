import sgkit
import numpy as np
import os
import sys
tsinfer_path = os.path.abspath("/home/duncan/trees/tsinfer")
sys.path.append(tsinfer_path)
import tsinfer

def make_subset_masks(subset_num_sites, subset_num_samples, region_mask_name, ds, seed=1):
    """
    Make a mask with exactly subset_num_sites sites that are unmasked, centered in the 
    region defined by site_mask (e.g. the p-arm of a chromosome). Return
    this and a sample_mask including a random sample of subset_num_samples individuals.
    """
    np.random.seed(seed)
    dupe_mask = ds.variant_duplicate_position_mask
    region_mask = ds.data_vars[region_mask_name].values
    site_mask = region_mask | dupe_mask

    combined_mask = np.asarray(site_mask, dtype=bool)
    num_sites = combined_mask.shape[0]
    if subset_num_sites <= 0:
        raise ValueError("num_sites must be > 0")

    ids = np.flatnonzero(~combined_mask)
    n_avail = ids.size

    if n_avail == 0:
        return np.ones(num_sites, dtype=bool)
    if n_avail <= subset_num_sites:
        return combined_mask.copy()

    mid = n_avail // 2
    start = max(0, mid - subset_num_sites // 2)
    end = start + subset_num_sites
    if end > n_avail:
        end = n_avail
        start = end - subset_num_sites

    chunk_left = ids[start]
    chunk_right = ids[end - 1] + 1
    window_mask = np.ones(num_sites, dtype=bool)
    window_mask[chunk_left:chunk_right] = False
    subset_site_mask = combined_mask | window_mask

    num_samples = ds.sizes["samples"]
    subset_num_samples = min(num_samples, subset_num_samples)
    sample_indices = np.random.choice(num_samples, size=subset_num_samples, replace=False)
    sample_mask = np.ones(ds.sizes["samples"], dtype=bool)
    sample_mask[sample_indices] = False

    return subset_site_mask, sample_mask

def make_1kgp_variant_data(
                           subset_num_sites,
                           subset_num_samples,
                           zarr_path,
                           region_mask_name,
                        ):

    ds = sgkit.load_dataset(zarr_path)
    subset_site_mask, subset_sample_mask = make_subset_masks(subset_num_sites=subset_num_sites,
                                                    subset_num_samples=subset_num_samples,
                                                    region_mask_name=region_mask_name,
                                                    ds=ds)
    variant_data = tsinfer.VariantData(zarr_path,
                                      site_mask=subset_site_mask,
                                      sample_mask=subset_sample_mask,
                                       ancestral_state="variant_ancestral_allele")
    return variant_data