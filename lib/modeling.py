import math
import numpy as np
import pandas as pd
import msprime
import tskit
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import scipy.optimize as optimize
import scipy.stats as stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import special
from numpy.polynomial.hermite import hermgauss
from lib import mismatch

VALID_TIME_VARS = {"T_mrca", "T_dbtn"}


def _validate_time_var(df, var):
    if var not in VALID_TIME_VARS:
        raise ValueError("var must be 'T_mrca' or 'T_dbtn'")
    if "n" not in df or var not in df:
        raise ValueError(f"df must contain 'n' and '{var}' columns")


def _positive_times(data, var):
    times = data[var].dropna().to_numpy()
    return times[np.isfinite(times) & (times > 0)]


def _mean_axis_label(var):
    return "Mean carrier TMRCA" if var == "T_mrca" else f"Mean {var}"


def _mean_title(method, var):
    time_label = "TMRCA" if var == "T_mrca" else var
    return f"{method}: mean {time_label}"


def _draw_mean_panel(
    ax,
    fits,
    sample_sizes,
    method,
    var,
    observed_color,
    predicted_color,
):
    observed_means = [float(fits.loc[n, "empirical_mean"]) for n in sample_sizes]
    predicted_means = [float(fits.loc[n, "fitted_mean"]) for n in sample_sizes]
    ax.plot(
        sample_sizes,
        observed_means,
        color=observed_color,
        marker="o",
        label="Observed",
    )
    ax.plot(
        sample_sizes,
        predicted_means,
        color=predicted_color,
        marker="o",
        label="Predicted",
    )
    ax.set_title(_mean_title(method, var))
    ax.set_xlabel("n")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(_mean_axis_label(var))
    ax.legend()


def estimate_gamma(
    df,
    quantile_spacing=0.1,
    method="quantile",
    max_shape=1e8,
    var="T_mrca",
):
    """
    Estimate gamma parameters for doubleton ages, grouped by sample size.

    Parameters
    ----------
    df : pandas.DataFrame
        Doubleton age table with columns ``n`` and the selected time variable.
    quantile_spacing : float
        Spacing for the quantile grid used when ``method="quantile"``.
        The endpoints 0 and 1 are excluded before fitting.
    method : {"quantile", "two-quantile", "moments"}
        Fitting method. ``"quantile"`` minimizes raw quantile differences.
        ``"two-quantile"`` matches the 0.25 and 0.75 raw-time quantiles,
        following the tsdate interquantile gamma-matching approach.
    max_shape : float
        Maximum shape parameter used by ``method="two-quantile"``.
    var : {"T_mrca", "T_dbtn"}
        Time variable to fit. Defaults to carrier TMRCA.

    Returns
    -------
    pandas.DataFrame
        One row per sample size, with shape ``alpha`` and rate ``beta``.
    """
    if method not in {"quantile", "two-quantile", "moments"}:
        raise ValueError("method must be 'quantile', 'two-quantile', or 'moments'")
    if not 0 < quantile_spacing < 1:
        raise ValueError("quantile_spacing must be between 0 and 1")
    if max_shape <= 0:
        raise ValueError("max_shape must be positive")
    _validate_time_var(df, var)

    def _moment_fit(times):
        mean = np.mean(times)
        variance = np.var(times)
        if mean <= 0 or variance <= 0:
            raise ValueError(f"{var} values must have positive mean and variance")
        alpha = mean**2 / variance
        beta = mean / variance
        return alpha, beta

    def _quantile_fit(times):
        probs = np.arange(0, 1 + quantile_spacing / 2, quantile_spacing)
        if probs[-1] < 1:
            probs = np.append(probs, 1)
        probs = probs[(0 < probs) & (probs < 1)]
        if len(probs) == 0:
            raise ValueError("quantile_spacing produced no inner quantiles")
        observed = np.quantile(times, probs)
        alpha0, beta0 = _moment_fit(times)

        def objective(log_params):
            alpha = np.exp(log_params[0])
            beta = np.exp(log_params[1])
            fitted = stats.gamma.ppf(probs, a=alpha, scale=1 / beta)
            return np.mean((fitted - observed) ** 2)

        result = optimize.minimize(
            objective,
            x0=np.log([alpha0, beta0]),
            method="Nelder-Mead",
        )
        if not result.success:
            raise RuntimeError(f"Gamma quantile fit failed: {result.message}")
        alpha, beta = np.exp(result.x)
        return alpha, beta

    def _two_quantile_fit(times):
        q1, q2 = 0.25, 0.75
        x1, x2 = np.quantile(times, [q1, q2])

        if x1 <= 0 or x2 <= 0:
            raise ValueError(f"{var} quantiles must be positive")
        if x2 == x1:
            alpha = max_shape
            beta = special.gammaincinv(alpha, q1) / x1
            return alpha, beta
        if x2 < x1:
            raise ValueError("Empirical quantiles must be sorted")

        target_ratio = x2 / x1

        def fitted_ratio(alpha):
            y1 = special.gammaincinv(alpha, q1)
            y2 = special.gammaincinv(alpha, q2)
            return y2 / y1

        # Same lower-bound idea as tsdate.approximate_gamma_iqr.
        lower = np.log(q2 / q1) / np.log(x2 / x1)
        lower = max(lower, np.finfo(float).tiny)

        if lower >= max_shape:
            alpha = max_shape
            beta = special.gammaincinv(alpha, q1) / x1
            return alpha, beta

        def objective(alpha):
            return fitted_ratio(alpha) - target_ratio

        lo = lower
        hi = max_shape
        f_lo = objective(lo)
        f_hi = objective(hi)

        if not np.isfinite(f_lo) or not np.isfinite(f_hi):
            raise RuntimeError("Gamma two-quantile fit failed with non-finite objective")

        if f_lo == 0:
            alpha = lo
        elif f_hi == 0:
            alpha = hi
        elif f_lo * f_hi < 0:
            alpha = optimize.brentq(objective, lo, hi)
        else:
            # Fallback to bounded scalar minimization if numerical roundoff prevents
            # a clean bracket.
            result = optimize.minimize_scalar(
                lambda a: objective(a) ** 2,
                bounds=(lo, hi),
                method="bounded",
            )
            if not result.success:
                raise RuntimeError(f"Gamma two-quantile fit failed: {result.message}")
            alpha = result.x

        beta = special.gammaincinv(alpha, q1) / x1
        return alpha, beta

    records = []
    for n, n_df in df.groupby("n", sort=True):
        times = _positive_times(n_df, var)
        if len(times) == 0:
            continue

        if method == "moments":
            alpha, beta = _moment_fit(times)
        elif method == "two-quantile":
            alpha, beta = _two_quantile_fit(times)
        else:
            alpha, beta = _quantile_fit(times)

        records.append({
            "n": n,
            "var": var,
            "method": method,
            "quantile_spacing": quantile_spacing if method == "quantile" else np.nan,
            "alpha": alpha,
            "beta": beta,
            "scale": 1 / beta,
            "empirical_mean": np.mean(times),
            "fitted_mean": alpha / beta,
            "num_observations": len(times),
        })

    if len(records) == 0:
        raise ValueError(f"No valid {var} values")
    return pd.DataFrame.from_records(records)


def estimate_lognormal(df, var="T_mrca"):
    """
    Estimate lognormal parameters for doubleton ages, grouped by sample size.

    The fit is the maximum-likelihood estimate for a two-parameter lognormal
    with location fixed at zero, equivalent to matching the mean and variance
    of log time.
    """
    _validate_time_var(df, var)

    records = []
    for n, n_df in df.groupby("n", sort=True):
        times = _positive_times(n_df, var)
        if len(times) == 0:
            continue

        log_times = np.log(times)
        mu = np.mean(log_times)
        sigma = np.std(log_times)
        if sigma <= 0:
            raise ValueError(f"log({var}) values must have positive variance")
        fitted_mean = np.exp(mu + sigma**2 / 2)
        records.append({
            "n": n,
            "var": var,
            "distribution": "lognormal",
            "method": "mle",
            "mu": mu,
            "sigma": sigma,
            "scale": np.exp(mu),
            "empirical_mean": np.mean(times),
            "fitted_mean": fitted_mean,
            "num_observations": len(times),
        })

    if len(records) == 0:
        raise ValueError(f"No valid {var} values")
    return pd.DataFrame.from_records(records)


    
def extract_doubleton_times(ts):
    """
    Return one row per true doubleton mutation.
    """
    records = []
    for tree in ts.trees(sample_lists=True):
        for site in tree.sites():
            for mut in site.mutations:
                if tree.num_samples(mut.node) != 2:
                    continue

                carriers = sorted(tree.samples(mut.node))
                carrier_1, carrier_2 = carriers

                pair_mrca = tree.mrca(carrier_1, carrier_2)
                T_mrca = tree.time(pair_mrca)
                T_dbtn = np.nan if tskit.is_unknown_time(mut.time) else mut.time

                records.append({
                    "site": site.id,
                    "position": site.position,
                    "carrier_1": carrier_1,
                    "carrier_2": carrier_2,
                    "T_dbtn": T_dbtn,
                    "T_mrca": T_mrca,
                })

    return pd.DataFrame.from_records(records)

def run_simulations(sample_sizes, N_e, seq_length=1e6, rate=1e-8):
    ts_dict = {}
    dfs = []
    for n in sample_sizes:
        base_ts = msprime.sim_ancestry(
            samples=n,
            sequence_length=seq_length,
            recombination_rate=rate,
            population_size=N_e,
            random_seed=n,
        )
        ts = msprime.sim_mutations(base_ts, rate=rate, random_seed=n)
        ts_dict[n] = ts
        df = extract_doubleton_times(ts)
        df["n"] = n
        df["N_e"] = N_e
        dfs.append(df)

    df = pd.concat(dfs)
    return df, ts_dict


def _tree_sequence_genotypes(ts):
    G = np.asarray(ts.genotype_matrix(), dtype=np.int8)
    return G.reshape(G.shape[0], ts.num_samples // 2, 2)


def count_haplotype_matches(
    ts,
    window_sizes,
    recomb_map=None,
):
    """
    Count haplotype mismatches around doubletons in a tree sequence.
    """
    G_full = _tree_sequence_genotypes(ts)
    singleton_mask = np.sum(G_full, axis=(1, 2)) > 1
    G = G_full[singleton_mask, :, :]
    sites_position = ts.sites_position[singleton_mask]
    G_error_mask = np.ones_like(G, dtype=bool)
    sample_mask = np.zeros(G.shape[1], dtype=bool)

    dfs = []
    assert len(window_sizes) > 0
    for window in window_sizes:
        df, _, _ = mismatch.count_haplotype_mismatches(
            G,
            G_error_mask,
            sites_position,
            sample_mask,
            window,
            recomb_map=recomb_map,
        )
        df["window_size"] = window
        dfs.append(df)

    return pd.concat(dfs)


def expected_hap_differences_gamma(alpha, beta, pi, L, r):
    L = float(L)
    r = np.asarray(r, dtype=float)
    rate = 2 * r
    p_recomb = 1 - (1 + rate * L / beta) ** (-alpha)
    expected = pi * (
        L
        - beta
        / (rate * (alpha - 1))
        * (1 - (1 + rate * L / beta) ** (1 - alpha))
    )
    if expected.ndim == 0:
        return expected.item(), p_recomb.item()
    return expected, p_recomb


def expected_hap_differences_lognormal(mu, sigma, pi, L, r, n_quad=32):
    nodes, weights = hermgauss(n_quad)

    # Convert Hermite nodes for exp(-x^2) to standard normal nodes.
    t = np.exp(mu + sigma * np.sqrt(2) * nodes)
    L = float(L)
    r = np.asarray(r, dtype=float)
    rate = 2 * r[..., np.newaxis] * t
    p_values = -np.expm1(-L * rate)
    values = p_values / rate
    surv_int = np.sum(weights * values, axis=-1) / np.sqrt(np.pi)
    p_recomb = np.sum(weights * p_values, axis=-1) / np.sqrt(np.pi)
    expected = pi * (L - surv_int)
    if expected.ndim == 0:
        return expected.item(), p_recomb.item()
    return expected, p_recomb


def fit_error_rate(dbtn_df, ts_dict, window_sizes, n, r):
    ts = ts_dict[n]
    mm_df = count_haplotype_matches(ts, window_sizes)
    gamma_df = estimate_gamma(dbtn_df)
    lognormal_df = estimate_lognormal(dbtn_df)
    gamma_row = gamma_df.loc[gamma_df.n == n]
    lognormal_row = lognormal_df.loc[lognormal_df.n == n]
    alpha = float(gamma_row["alpha"].iloc[0])
    beta = float(gamma_row["beta"].iloc[0])
    mu = float(lognormal_row["mu"].iloc[0])
    sigma = float(lognormal_row["sigma"].iloc[0])
    lognormal_scale = float(lognormal_row["scale"].iloc[0])
    lognormal_fitted_mean = float(lognormal_row["fitted_mean"].iloc[0])
    pi = ts.diversity()
    observed = mm_df.groupby("window_size", sort=False)["obs_error_rate_per_bp"].mean()

    records = []
    for window in window_sizes:
        L = window / 2
        gamma_differences, gamma_p_recomb = expected_hap_differences_gamma(
            alpha,
            beta,
            pi,
            L,
            r,
        )
        lognormal_differences, lognormal_p_recomb = (
            expected_hap_differences_lognormal(mu, sigma, pi, L, r)
        )
        records.append({
            "window_size": window,
            "L": L,
            "n": n,
            "alpha": alpha,
            "beta": beta,
            "lognormal_mu": mu,
            "lognormal_sigma": sigma,
            "lognormal_scale": lognormal_scale,
            "lognormal_fitted_mean": lognormal_fitted_mean,
            "pi": pi,
            "r": r,
            "pred_gamma_differences": 2 * gamma_differences,
            "pred_lognormal_differences": 2 * lognormal_differences,
            "pred_gamma_error_rate_per_bp": gamma_differences / L,
            "pred_lognormal_error_rate_per_bp": lognormal_differences / L,
            "pred_gamma_p_recomb": 2 * gamma_p_recomb - gamma_p_recomb**2,
            "pred_lognormal_p_recomb": 2 * lognormal_p_recomb - lognormal_p_recomb**2,
            "observed_error_rate_per_bp": observed.get(window, np.nan),
        })
    fit_df = pd.DataFrame.from_records(records)
    return mm_df, fit_df


def fit_error_rate_map(dbtn_df, ts_dict, window_sizes, n, recomb_map):
    ts = ts_dict[n]
    mm_df = count_haplotype_matches(ts, window_sizes, recomb_map=recomb_map)
    gamma_df = estimate_gamma(dbtn_df)
    lognormal_df = estimate_lognormal(dbtn_df)
    gamma_row = gamma_df.loc[gamma_df.n == n]
    lognormal_row = lognormal_df.loc[lognormal_df.n == n]
    alpha = float(gamma_row["alpha"].iloc[0])
    beta = float(gamma_row["beta"].iloc[0])
    mu = float(lognormal_row["mu"].iloc[0])
    sigma = float(lognormal_row["sigma"].iloc[0])
    pi = ts.diversity()

    mm_df = mm_df.copy()
    mm_df["observed_error_rate_per_bp"] = mm_df["obs_error_rate_per_bp"]
    mm_df["pred_gamma_differences"] = np.nan
    mm_df["pred_lognormal_differences"] = np.nan
    mm_df["pred_gamma_error_rate_per_bp"] = np.nan
    mm_df["pred_lognormal_error_rate_per_bp"] = np.nan
    mm_df["pred_gamma_p_recomb"] = np.nan
    mm_df["pred_lognormal_p_recomb"] = np.nan

    for window_size, index in mm_df.groupby("window_size", sort=False).groups.items():
        L = window_size / 2
        D = mm_df.loc[index, ["D_left", "D_right"]].to_numpy(dtype=float)
        r = D / L
        gamma_differences, gamma_p_recomb = expected_hap_differences_gamma(
            alpha,
            beta,
            pi,
            L,
            r,
        )
        lognormal_differences, lognormal_p_recomb = expected_hap_differences_lognormal(
            mu,
            sigma,
            pi,
            L,
            r,
        )
        gamma_differences = gamma_differences.sum(axis=1)
        lognormal_differences = lognormal_differences.sum(axis=1)
        gamma_p_recomb = gamma_p_recomb.sum(axis=1) - gamma_p_recomb.prod(axis=1)
        lognormal_p_recomb = lognormal_p_recomb.sum(axis=1) - lognormal_p_recomb.prod(axis=1)
        mm_df.loc[index, "left_recomb_rate"] = r[:, 0]
        mm_df.loc[index, "right_recomb_rate"] = r[:, 1]
        mm_df.loc[index, "pred_gamma_differences"] = gamma_differences
        mm_df.loc[index, "pred_lognormal_differences"] = lognormal_differences
        mm_df.loc[index, "pred_gamma_p_recomb"] = gamma_p_recomb
        mm_df.loc[index, "pred_lognormal_p_recomb"] = lognormal_p_recomb
        if window_size > 0:
            mm_df.loc[index, "pred_gamma_error_rate_per_bp"] = (
                gamma_differences / window_size
            )
            mm_df.loc[index, "pred_lognormal_error_rate_per_bp"] = (
                lognormal_differences / window_size
            )
    return mm_df


def plot_doubleton_ages(df, var="T_mrca"):
    """
    Plot empirical doubleton carrier coalescence times and allele ages.
    """
    _validate_time_var(df, var)

    sample_sizes = sorted(df["n"].dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    palette = sns.color_palette("viridis", n_colors=len(sample_sizes))
    colors = dict(zip(sample_sizes, palette))

    def _positive_log_time(data, col):
        data = data[np.isfinite(data[col]) & (data[col] > 0)].copy()
        data[f"log10_{col}"] = np.log10(data[col])
        return data

    def _format_log10_ticks(ax):
        lower, upper = ax.get_xlim()
        lower_tick = math.ceil(lower)
        upper_tick = math.floor(upper)
        if lower_tick <= upper_tick:
            ax.set_xticks(np.arange(lower_tick, upper_tick + 1))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: rf"$10^{{{value:g}}}$")
        )

    titles = {
        "T_mrca": "Doubleton carrier TMRCA distribution",
        "T_dbtn": "Doubleton allele age distribution",
    }
    plot_vars = [var, next(col for col in ["T_mrca", "T_dbtn"] if col != var)]
    for ax, col in zip(axes, plot_vars):
        plot_df = _positive_log_time(df, col)
        if plot_df.empty:
            ax.text(0.5, 0.5, f"No valid {col} values", ha="center", va="center")
        else:
            sns.kdeplot(
                data=plot_df,
                x=f"log10_{col}",
                hue="n",
                palette=colors,
                common_norm=False,
                cut=0,
                ax=ax,
            )
        ax.set_title(titles[col])
        _format_log10_ticks(ax)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")

    plt.show()


def plot_gamma_fit(df, quantile_spacing=0.1, var="T_mrca"):
    """
    Plot moment-matched and quantile-matched gamma fits for doubleton ages.
    """
    _validate_time_var(df, var)
    sample_sizes = sorted(df["n"].dropna().unique())
    if len(sample_sizes) == 0:
        raise ValueError("df must contain at least one non-null sample size in 'n'")

    moment_fits = estimate_gamma(df, method="moments", var=var)
    quantile_fits = estimate_gamma(
        df,
        quantile_spacing=quantile_spacing,
        method="quantile",
        var=var,
    )
    two_quantile_fits = estimate_gamma(
        df,
        quantile_spacing=quantile_spacing,
        method="two-quantile",
        var=var,
    )
    fit_df = pd.concat(
        [moment_fits, quantile_fits, two_quantile_fits],
        ignore_index=True,
    )
    fit_df = fit_df.set_index(["method", "n"])

    n_cols = max(1, len(sample_sizes))
    fig, axes = plt.subplots(
        3,
        n_cols + 1,
        figsize=(4.8 * (n_cols + 1), 13.2),
        constrained_layout=True,
        squeeze=False,
    )
    palette = sns.color_palette("viridis", n_colors=len(sample_sizes))
    colors = dict(zip(sample_sizes, palette))
    methods = ["moments", "quantile", "two-quantile"]
    observed_color = "0.25"
    predicted_color = "red"

    def _format_log10_ticks(ax):
        lower, upper = ax.get_xlim()
        lower_tick = math.ceil(lower)
        upper_tick = math.floor(upper)
        if lower_tick <= upper_tick:
            ax.set_xticks(np.arange(lower_tick, upper_tick + 1))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: rf"$10^{{{value:g}}}$")
        )

    for row, method in enumerate(methods):
        for col, n in enumerate(sample_sizes):
            ax = axes[row, col]
            n_df = df[df["n"] == n]
            obs = _positive_times(n_df, var)
            if len(obs) == 0:
                ax.text(0.5, 0.5, f"No valid {var} values", ha="center", va="center")
                continue

            fit = fit_df.loc[(method, n)]
            alpha = float(fit["alpha"])
            beta = float(fit["beta"])
            scale = float(fit["scale"])
            empirical_mean = float(fit["empirical_mean"])
            fitted_mean = float(fit["fitted_mean"])

            log_obs = np.log10(obs)
            sns.kdeplot(
                x=log_obs,
                ax=ax,
                color="black",
                label="Empirical KDE",
                common_norm=False,
                cut=0,
            )
            log_grid = np.linspace(np.min(log_obs), np.max(log_obs), 400)
            x_grid = 10 ** log_grid
            gamma_log_density = (
                stats.gamma.pdf(x_grid, a=alpha, scale=scale)
                * np.log(10)
                * x_grid
            )
            ax.plot(
                log_grid,
                gamma_log_density,
                color=colors[n],
                linewidth=2,
                label=f"Gamma alpha={alpha:.2f}, beta={beta:.3g}",
            )
            ax.axvline(
                np.log10(empirical_mean),
                color=observed_color,
                linestyle="--",
                linewidth=1,
            )
            ax.axvline(
                np.log10(fitted_mean),
                color=predicted_color,
                linestyle="--",
                linewidth=1,
            )
            ax.text(
                0.07,
                0.49,
                f"Observed mean = {empirical_mean:.3g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color=observed_color,
                fontsize=9,
            )
            ax.text(
                0.07,
                0.45,
                f"Predicted mean = {fitted_mean:.3g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color=predicted_color,
                fontsize=9,
            )
            _format_log10_ticks(ax)
            ax.set_title(f"{method}: {var}, n={n}")
            ax.set_xlabel(var)
            ax.set_ylabel("Density")
            ax.legend(fontsize=8, loc="lower left")

            inset = inset_axes(
                ax,
                width="38%",
                height="38%",
                loc="upper left",
                bbox_to_anchor=(0.10, 0, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0.8,
            )
            obs_quantiles = np.sort(obs)
            probs = (np.arange(1, len(obs_quantiles) + 1) - 0.5) / len(obs_quantiles)
            gamma_quantiles = stats.gamma.ppf(probs, a=alpha, scale=scale)
            log_obs_quantiles = np.log10(obs_quantiles)
            log_gamma_quantiles = np.log10(gamma_quantiles)
            inset.scatter(
                log_gamma_quantiles,
                log_obs_quantiles,
                s=6,
                alpha=0.35,
                color="black",
            )
            qq_min = min(np.min(log_gamma_quantiles), np.min(log_obs_quantiles))
            qq_max = max(np.max(log_gamma_quantiles), np.max(log_obs_quantiles))
            inset.plot([qq_min, qq_max], [qq_min, qq_max], color=colors[n], linewidth=1)
            inset.set_xlabel("Gamma", fontsize=7)
            inset.set_ylabel("Empirical", fontsize=7)
            inset.tick_params(axis="both", labelsize=6)

        mean_ax = axes[row, n_cols]
        method_fits = fit_df.xs(method, level="method")
        _draw_mean_panel(
            mean_ax,
            method_fits,
            sample_sizes,
            method,
            var,
            observed_color,
            predicted_color,
        )

    plt.show()


def plot_multifit(df, var="T_mrca"):
    """
    Plot two-quantile gamma and lognormal fits for doubleton ages.
    """
    _validate_time_var(df, var)
    sample_sizes = sorted(df["n"].dropna().unique())
    if len(sample_sizes) == 0:
        raise ValueError("df must contain at least one non-null sample size in 'n'")

    gamma_fits = estimate_gamma(df, method="two-quantile", var=var)
    lognormal_fits = estimate_lognormal(df, var=var)
    gamma_fits = gamma_fits.set_index("n")
    lognormal_fits = lognormal_fits.set_index("n")

    n_cols = max(1, len(sample_sizes))
    fig, axes = plt.subplots(
        2,
        n_cols + 1,
        figsize=(4.8 * (n_cols + 1), 8.8),
        constrained_layout=True,
        squeeze=False,
    )
    palette = sns.color_palette("viridis", n_colors=len(sample_sizes))
    colors = dict(zip(sample_sizes, palette))
    observed_color = "0.25"
    predicted_color = "red"

    def _format_log10_ticks(ax):
        lower, upper = ax.get_xlim()
        lower_tick = math.ceil(lower)
        upper_tick = math.floor(upper)
        if lower_tick <= upper_tick:
            ax.set_xticks(np.arange(lower_tick, upper_tick + 1))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: rf"$10^{{{value:g}}}$")
        )

    def _draw_fit(ax, n, obs, fit_name, pdf_func, ppf_func, fitted_mean, label):
        log_obs = np.log10(obs)
        sns.kdeplot(
            x=log_obs,
            ax=ax,
            color="black",
            label="Empirical KDE",
            common_norm=False,
            cut=0,
        )
        log_grid = np.linspace(np.min(log_obs), np.max(log_obs), 400)
        x_grid = 10 ** log_grid
        log_density = pdf_func(x_grid) * np.log(10) * x_grid
        ax.plot(
            log_grid,
            log_density,
            color=colors[n],
            linewidth=2,
            label=label,
        )

        empirical_mean = np.mean(obs)
        ax.axvline(
            np.log10(empirical_mean),
            color=observed_color,
            linestyle="--",
            linewidth=1,
        )
        ax.axvline(
            np.log10(fitted_mean),
            color=predicted_color,
            linestyle="--",
            linewidth=1,
        )
        ax.text(
            0.07,
            0.49,
            f"Observed mean = {empirical_mean:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=observed_color,
            fontsize=9,
        )
        ax.text(
            0.07,
            0.45,
            f"Predicted mean = {fitted_mean:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=predicted_color,
            fontsize=9,
        )
        _format_log10_ticks(ax)
        ax.set_title(f"{fit_name}: {var}, n={n}")
        ax.set_xlabel(var)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc="lower left")

        inset = inset_axes(
            ax,
            width="38%",
            height="38%",
            loc="upper left",
            bbox_to_anchor=(0.10, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0.8,
        )
        obs_quantiles = np.sort(obs)
        probs = (np.arange(1, len(obs_quantiles) + 1) - 0.5) / len(obs_quantiles)
        fitted_quantiles = ppf_func(probs)
        log_obs_quantiles = np.log10(obs_quantiles)
        log_fitted_quantiles = np.log10(fitted_quantiles)
        inset.scatter(
            log_fitted_quantiles,
            log_obs_quantiles,
            s=6,
            alpha=0.35,
            color="black",
        )
        qq_min = min(np.min(log_fitted_quantiles), np.min(log_obs_quantiles))
        qq_max = max(np.max(log_fitted_quantiles), np.max(log_obs_quantiles))
        inset.plot([qq_min, qq_max], [qq_min, qq_max], color=colors[n], linewidth=1)
        inset.set_xlabel("Fitted", fontsize=7)
        inset.set_ylabel("Empirical", fontsize=7)
        inset.tick_params(axis="both", labelsize=6)

    for col, n in enumerate(sample_sizes):
        n_df = df[df["n"] == n]
        obs = _positive_times(n_df, var)
        if len(obs) == 0:
            for row in range(2):
                axes[row, col].text(
                    0.5,
                    0.5,
                    f"No valid {var} values",
                    ha="center",
                    va="center",
                )
            continue

        gamma_fit = gamma_fits.loc[n]
        gamma_alpha = float(gamma_fit["alpha"])
        gamma_scale = float(gamma_fit["scale"])
        _draw_fit(
            axes[0, col],
            n,
            obs,
            "Gamma",
            lambda x, a=gamma_alpha, s=gamma_scale: stats.gamma.pdf(x, a=a, scale=s),
            lambda p, a=gamma_alpha, s=gamma_scale: stats.gamma.ppf(p, a=a, scale=s),
            float(gamma_fit["fitted_mean"]),
            f"Gamma alpha={gamma_alpha:.2f}, beta={float(gamma_fit['beta']):.3g}",
        )

        lognormal_fit = lognormal_fits.loc[n]
        lognormal_sigma = float(lognormal_fit["sigma"])
        lognormal_scale = float(lognormal_fit["scale"])
        _draw_fit(
            axes[1, col],
            n,
            obs,
            "Lognormal",
            lambda x, s=lognormal_sigma, sc=lognormal_scale: stats.lognorm.pdf(
                x,
                s=s,
                scale=sc,
            ),
            lambda p, s=lognormal_sigma, sc=lognormal_scale: stats.lognorm.ppf(
                p,
                s=s,
                scale=sc,
            ),
            float(lognormal_fit["fitted_mean"]),
            f"Lognormal mu={float(lognormal_fit['mu']):.2f}, sigma={lognormal_sigma:.2f}",
        )

    _draw_mean_panel(
        axes[0, n_cols],
        gamma_fits,
        sample_sizes,
        "Gamma",
        var,
        observed_color,
        predicted_color,
    )
    _draw_mean_panel(
        axes[1, n_cols],
        lognormal_fits,
        sample_sizes,
        "Lognormal",
        var,
        observed_color,
        predicted_color,
    )

    plt.show()


def plot_error_rate_fit(fit_df, ax=None, p_ax=None):
    if ax is None and p_ax is None:
        fig, (ax, p_ax) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    elif ax is None or p_ax is None:
        raise ValueError("ax and p_ax must be provided together")
    else:
        fig = ax.figure

    ax.plot(
        fit_df["window_size"],
        fit_df["pred_gamma_error_rate_per_bp"],
        marker="o",
        color="red",
        label="Lomax",
    )
    ax.plot(
        fit_df["window_size"],
        fit_df["pred_lognormal_error_rate_per_bp"],
        marker="o",
        color="tab:blue",
        label="Lognormal",
    )
    ax.plot(
        fit_df["window_size"],
        fit_df["observed_error_rate_per_bp"],
        marker="o",
        color="0.25",
        label="Observed",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Window size")
    ax.set_ylabel("Error rate per bp")
    ax.legend()

    p_ax.plot(
        fit_df["window_size"],
        fit_df["pred_gamma_p_recomb"],
        marker="o",
        color="red",
        label="Lomax",
    )
    p_ax.plot(
        fit_df["window_size"],
        fit_df["pred_lognormal_p_recomb"],
        marker="o",
        color="tab:blue",
        label="Lognormal",
    )
    p_ax.set_xscale("log")
    p_ax.set_xlabel("Window size")
    p_ax.set_ylabel("Probability(recombination)")
    p_ax.legend()
    return fig, (ax, p_ax)


def plot_error_rate_map(df):
    required_cols = {
        "window_size",
        "observed_error_rate_per_bp",
        "pred_gamma_error_rate_per_bp",
        "pred_lognormal_error_rate_per_bp",
        "pred_gamma_p_recomb",
        "pred_lognormal_p_recomb",
    }
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"df is missing columns: {sorted(missing_cols)}")

    frame = pd.DataFrame(df, copy=False)
    window_sizes = sorted(frame["window_size"].dropna().unique())
    if len(window_sizes) == 0:
        raise ValueError("df must contain at least one window size")

    distribution_specs = [
        ("Observed", "observed_error_rate_per_bp", "0.25"),
        ("Lognormal", "pred_lognormal_error_rate_per_bp", "tab:blue"),
        ("Gamma", "pred_gamma_error_rate_per_bp", "red"),
    ]
    scatter_specs = [
        ("Gamma", "pred_gamma_error_rate_per_bp", "red"),
        ("Lognormal", "pred_lognormal_error_rate_per_bp", "tab:blue"),
    ]

    fig = plt.figure(
        figsize=(max(4.0 * len(window_sizes), 12.0), 12.5),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[3.0, 1.35],
    )
    hist_grid = outer_grid[0].subgridspec(
        len(distribution_specs),
        len(window_sizes),
        wspace=0.2,
        hspace=0.32,
    )
    bottom_grid = outer_grid[1].subgridspec(1, 4, wspace=0.28)

    def _finite_values(values):
        values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
            dtype=float,
            copy=False,
        )
        return values[np.isfinite(values) & (values >= 0)]

    all_positive = []
    for _, col, _ in distribution_specs:
        values = _finite_values(frame[col])
        all_positive.extend(values[values > 0])
    all_positive = np.asarray(all_positive, dtype=float)
    if len(all_positive) == 0:
        positive_edges = None
    else:
        log_min = np.floor(np.log10(np.min(all_positive)))
        log_max = np.ceil(np.log10(np.max(all_positive)))
        if log_min == log_max:
            log_min -= 0.5
            log_max += 0.5
        positive_edges = np.logspace(log_min, log_max, 40)

    def _plot_distribution(ax, values, color):
        values = _finite_values(values)
        zero_count = np.count_nonzero(values == 0)
        positive = values[values > 0]
        if len(positive) > 0 and positive_edges is not None:
            ax.hist(positive, bins=positive_edges, color=color, alpha=0.75)
            ax.set_xscale("log")
        if zero_count > 0:
            ax.text(
                0.98,
                0.92,
                f"zero = {zero_count}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color=color,
            )
        if len(positive) == 0:
            ax.text(0.5, 0.5, "No positive values", ha="center", va="center")
        ax.set_yscale("log")

    for row, (label, col, color) in enumerate(distribution_specs):
        for col_idx, window_size in enumerate(window_sizes):
            ax = fig.add_subplot(hist_grid[row, col_idx])
            values = frame.loc[frame["window_size"] == window_size, col]
            _plot_distribution(ax, values, color)
            if row == 0:
                ax.set_title(f"window_size = {window_size:g}")
            if col_idx == 0:
                ax.set_ylabel(label)
            if row == len(distribution_specs) - 1:
                ax.set_xlabel("Error rate per bp")
            else:
                ax.set_xlabel("")

    def _plot_observed_vs_fitted(ax, fitted_col, label, color):
        x = pd.to_numeric(frame["observed_error_rate_per_bp"], errors="coerce").to_numpy(
            dtype=float,
            copy=False,
        )
        y = pd.to_numeric(frame[fitted_col], errors="coerce").to_numpy(
            dtype=float,
            copy=False,
        )
        mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
        ax.scatter(x[mask], y[mask], s=16, color=color, alpha=0.3, linewidths=0)
        fit_mask = mask & (x > 0) & (y > 0)
        if np.count_nonzero(fit_mask) >= 2:
            slope, intercept = np.polyfit(np.log10(x[fit_mask]), np.log10(y[fit_mask]), 1)
            x_line = np.logspace(
                np.log10(np.min(x[fit_mask])),
                np.log10(np.max(x[fit_mask])),
                100,
            )
            y_line = 10 ** (intercept + slope * np.log10(x_line))
            ax.plot(x_line, y_line, color=color, linewidth=2)
        positive = np.concatenate([x[mask & (x > 0)], y[mask & (y > 0)]])
        if len(positive) > 0:
            lo = np.min(positive)
            hi = np.max(positive)
            ax.plot([lo, hi], [lo, hi], color="0.6", linestyle="--", linewidth=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_title(f"Observed vs {label}")
        ax.set_xlabel("Observed error rate per bp")
        ax.set_ylabel(f"{label} error rate per bp")

    for idx, (label, col, color) in enumerate(scatter_specs):
        ax = fig.add_subplot(bottom_grid[0, idx])
        _plot_observed_vs_fitted(ax, col, label, color)

    mean_fit_df = (
        frame.groupby("window_size", sort=True)[
            [
                "observed_error_rate_per_bp",
                "pred_gamma_error_rate_per_bp",
                "pred_lognormal_error_rate_per_bp",
                "pred_gamma_p_recomb",
                "pred_lognormal_p_recomb",
            ]
        ]
        .mean()
        .reset_index()
    )
    ax = fig.add_subplot(bottom_grid[0, 2])
    p_ax = fig.add_subplot(bottom_grid[0, 3])
    plot_error_rate_fit(mean_fit_df, ax=ax, p_ax=p_ax)

    plt.show()
