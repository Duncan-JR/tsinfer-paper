import json
import os
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tskit
import zarr


DEFAULT_MASK_PATH = "/well/kelleher/users/uuc395/tsinfer-dev/data/1kgp_chr20p_full_mask.zarr"
DEFAULT_ZARR_PATH = "/well/kelleher/users/uuc395/tsinfer-dev/data/1kgp_chr20.vcz"
TSINFER_PATHS = {
    "1.0": "/well/kelleher/users/uuc395/tsinfer",
    "0.4": "/well/kelleher/users/uuc395/tsinfer-old",
}


def resolve_tsinfer_path(version):
    try:
        return os.path.abspath(TSINFER_PATHS[version])
    except KeyError as exc:
        valid = ", ".join(sorted(TSINFER_PATHS))
        raise ValueError(f"Unsupported version '{version}'. Expected one of: {valid}") from exc


def load_variant_data(zarr_path, mask_path):
    mask_root = zarr.open_group(mask_path, mode="r")
    return tsinfer.VariantData(
        zarr_path,
        ancestral_state=np.asarray(mask_root["ancestral_state"][:]).astype(str),
        sample_mask=np.asarray(mask_root["sample_mask"][:], dtype=bool),
        site_mask=np.asarray(mask_root["site_mask"][:], dtype=bool),
    )


def extract_computation_time(provenance_source, commands):
    matches = []
    try:
        provenances = provenance_source.provenances()
    except AttributeError:
        return pd.DataFrame()
    for prov in provenances:
        record = json.loads(prov.record)
        command = record.get("parameters", {}).get("command")
        if command in commands:
            resources = record.get("resources", {})
            matches.append({**resources, "command": command})
    return pd.DataFrame(matches)


def load_ancestor_data(path):
    if hasattr(tsinfer, "AncestorData") and hasattr(tsinfer.AncestorData, "load"):
        return tsinfer.AncestorData.load(str(path))
    if hasattr(tsinfer, "load"):
        return tsinfer.load(str(path))
    raise RuntimeError("Could not find a tsinfer API to load ancestor data")


def run_1kgp_inference(output_folder, zarr_path, mask_path, version, num_threads):
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    ancestor_data_path = output_dir / f"ancestor_data_version{version}.zarr"
    ancestors_ts_path = output_dir / f"ancestors_version{version}.trees"
    inferred_ts_path = output_dir / f"ts_version{version}.trees"
    perf_path = output_dir / f"performance_version{version}.csv"

    variant_data = load_variant_data(zarr_path=zarr_path, mask_path=mask_path)
    if inferred_ts_path.exists():
        click.echo(f"Loading existing inferred tree sequence: {inferred_ts_path}")
        inferred_ts = tskit.load(str(inferred_ts_path))
    else:
        if ancestors_ts_path.exists():
            click.echo(f"Loading existing ancestors tree sequence: {ancestors_ts_path}")
            ancestors_ts = tskit.load(str(ancestors_ts_path))
        else:
            if ancestor_data_path.exists():
                click.echo(f"Loading existing ancestor data: {ancestor_data_path}")
                ancestors = load_ancestor_data(ancestor_data_path)
            else:
                click.echo(f"Generating ancestor data: {ancestor_data_path}")
                ancestors = tsinfer.generate_ancestors(
                    variant_data,
                    path=str(ancestor_data_path),
                    progress_monitor=True,
                    num_threads=num_threads,
                )
            click.echo(f"Matching ancestors: {ancestors_ts_path}")
            ancestors_ts = tsinfer.match_ancestors(
                variant_data,
                ancestors,
                progress_monitor=True,
                num_threads=num_threads,
            )
            ancestors_ts.dump(str(ancestors_ts_path))

        click.echo(f"Matching samples: {inferred_ts_path}")
        raw_ts = tsinfer.match_samples(
            variant_data,
            ancestors_ts,
            post_process=False,
            progress_monitor=True,
            num_threads=num_threads,
        )
        inferred_ts = tsinfer.post_process(raw_ts)
        inferred_ts.dump(str(inferred_ts_path))

    perf_df = extract_computation_time(
        inferred_ts,
        ["generate_ancestors", "match_ancestors", "match_samples"],
    )
    if perf_df.empty:
        perf_df = pd.DataFrame(columns=["command"])

    perf_df["version"] = version
    perf_df["num_samples"] = variant_data.num_samples
    perf_df["num_sites"] = variant_data.num_sites
    perf_df["num_mutations"] = inferred_ts.num_mutations
    perf_df["num_edges"] = inferred_ts.num_edges
    perf_df["mask_path"] = str(mask_path)
    perf_df["zarr_path"] = str(zarr_path)
    perf_df.to_csv(perf_path, index=False)

    return perf_df


@click.command()
@click.argument("output_folder", type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.option("--version", required=True, type=str, help="tsinfer version label to run.")
@click.option(
    "--zarr-path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_ZARR_PATH,
    show_default=True,
    help="Path to the 1KGP variant data.",
)
@click.option(
    "--mask-path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_MASK_PATH,
    show_default=True,
    help="Path to the mask zarr containing ancestral state and masks.",
)
@click.option(
    "--num-threads",
    type=int,
    default=190,
    show_default=True,
    help="Number of tsinfer worker threads.",
)
def main(output_folder, version, zarr_path, mask_path, num_threads):
    tsinfer_path = resolve_tsinfer_path(version)
    sys.path.insert(0, tsinfer_path)
    global tsinfer
    import tsinfer

    run_1kgp_inference(
        output_folder=output_folder,
        zarr_path=zarr_path,
        mask_path=mask_path,
        version=version,
        num_threads=num_threads,
    )


if __name__ == "__main__":
    main()
