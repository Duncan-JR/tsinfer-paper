import msprime
import stdpopsim
import warnings


def simulate(model, contig, samples, left, right, seed):
    if model == "Neutral":
        ts = msprime.sim_ancestry(
            samples["all"],
            sequence_length=right - left,
            random_seed=seed,
            recombination_rate=1e-8,
            population_size=1e4,
            model='smc_prime'
        )
        mutation_rate = 1e-8
    else:
        species = stdpopsim.get_species("HomSap")
        demographic_model = species.get_demographic_model(model)
        contig = species.get_contig(
            contig,
            mutation_rate=0,
            left=float(left),
            right=float(right),
            genetic_map="HapMapII_GRCh38",
        )
        engine = stdpopsim.get_engine("msprime")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=r"The demographic model has mutation rate .* but this simulation used the contig's mutation rate 0.*",
            )
            ts = engine.simulate(
                demographic_model,
                contig,
                samples,
                seed=seed,
                msprime_model="smc_prime",
            )
        mutation_rate = demographic_model.mutation_rate
    return msprime.sim_mutations(
        ts,
        rate=mutation_rate,
        random_seed=seed,
        model=msprime.BinaryMutationModel(),
    )
