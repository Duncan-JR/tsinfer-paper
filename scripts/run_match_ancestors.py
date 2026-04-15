import click
import os
import sys
tsinfer_path = os.path.abspath("/well/kelleher/users/uuc395/tsinfer")
sys.path.append(tsinfer_path)
import tsinfer

def run_match_ancestors(
    zarr_path,
    anc_data_path,
    site_mask="variant_all_subset_chr17p_10k",
    weight_by_n=False,
):
    variant_data = tsinfer.VariantData(
        zarr_path,
        site_mask=site_mask,
        sample_mask="sample_all_subset_mask",
        ancestral_state="variant_ancestral_allele",
    )
    anc_data = tsinfer.AncestorData.load(anc_data_path)
    return tsinfer.match_ancestors(
        variant_data,
        anc_data,
        num_threads=0,
        progress_monitor=True,
        weight_by_n=weight_by_n,
    )


@click.command()
@click.option("--zarr-path", default="data/tsinfer-snakemake/chr17/data.zarr", type=click.Path(exists=True))
@click.option("--anc-data-path", default="data/weight_by_n/ancestors_10k.zarr", type=click.Path(exists=True))
@click.option(
    "--site-mask",
    default="variant_all_subset_chr17p_10k",
    show_default=True,
)
@click.option(
    "--weight-by-n/--no-weight-by-n",
    default=False,
    show_default=True,
)
def main(zarr_path, anc_data_path, site_mask, weight_by_n):
    ts = run_match_ancestors(
        zarr_path=zarr_path,
        anc_data_path=anc_data_path,
        site_mask=site_mask,
        weight_by_n=weight_by_n,
    )
    click.echo(
        "Matched ancestors TS: "
        f"nodes={ts.num_nodes} edges={ts.num_edges} "
        f"sites={ts.num_sites} mutations={ts.num_mutations}"
    )


if __name__ == "__main__":
    main()
