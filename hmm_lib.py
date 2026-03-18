import numpy as np
from IPython.display import HTML, Math, display
import ipywidgets as widgets
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import msprime
from numba import njit
    
def randomise_haplotypes(num_nodes, num_sites, seed=1):
    np.random.seed(seed)
    reference = np.random.randint(0, 2, (num_nodes, num_sites))
    query = np.random.randint(0, 2, num_sites)
    return reference, query


@njit
def _update_site_numba(
    site,
    L,
    reference,
    query,
    recomb_required,
    mismatch,
    L_mat,
    L_norm_mat,
    max_likelihood_node,
    max_likelihood,
    rho,
    mu,
    n,
    eps,
):
    max_L = -1.0
    max_L_node = -1
    p_recomb = rho / n
    query_site = query[site]

    for u in range(L.shape[0]):
        recomb_required[u, site] = False
        mismatch[u, site] = True

        p_last = L[u]
        p_no_recomb = p_last * (1 - rho + p_recomb)
        if p_no_recomb > p_recomb:
            p_transition = p_no_recomb
        else:
            p_transition = p_recomb
            recomb_required[u, site] = True

        ref_allele = reference[u, site]
        p_emission = mu
        if query_site == ref_allele:
            mismatch[u, site] = False
            p_emission = 1 - mu
        elif ref_allele == -1:
            p_emission = 0.0

        likelihood = p_transition * p_emission
        L[u] = likelihood
        L_mat[u, site] = likelihood
        if likelihood > max_L:
            max_L = likelihood
            max_L_node = u

    if max_L == 0.0:
        return site

    max_likelihood_node[site] = max_L_node
    max_likelihood[site] = max_L
    for u in range(L.shape[0]):
        L_norm = max(L[u] / max_L, eps)
        L[u] = L_norm
        L_norm_mat[u, site] = L_norm
    return -1


@njit
def _run_numba(
    L,
    reference,
    query,
    recomb_required,
    mismatch,
    L_mat,
    L_norm_mat,
    max_likelihood_node,
    max_likelihood,
    path,
    rho,
    mu,
    n,
    eps,
):
    num_nodes = reference.shape[0]
    num_sites = reference.shape[1]

    for u in range(num_nodes):
        L[u] = 1.0

    for u in range(num_nodes):
        for site in range(num_sites):
            recomb_required[u, site] = False
            mismatch[u, site] = True
            L_mat[u, site] = 0.0
            L_norm_mat[u, site] = 0.0

    for site in range(num_sites):
        max_likelihood_node[site] = -1
        max_likelihood[site] = 0.0
        path[site] = -1

    for site in range(num_sites):
        failed_site = _update_site_numba(
            site,
            L,
            reference,
            query,
            recomb_required,
            mismatch,
            L_mat,
            L_norm_mat,
            max_likelihood_node,
            max_likelihood,
            rho,
            mu,
            n,
            eps,
        )
        if failed_site != -1:
            return failed_site, 0, 0

    num_switches = 0
    num_mismatches = 0
    u = max_likelihood_node[num_sites - 1]
    for site in range(num_sites - 1, -1, -1):
        path[site] = u
        num_mismatches += mismatch[u, site]
        if recomb_required[u, site]:
            num_switches += 1
            if site == 0:
                return site, 0, 0
            new_u = max_likelihood_node[site - 1]
            if u == new_u:
                return site, 0, 0
            u = new_u

    return -1, num_mismatches, num_switches

class LSHMM:
    def __init__(self, reference, query, mu, rho, eps=1e-8, scale_by_n=False):
        num_nodes = reference.shape[0]
        num_sites = reference.shape[1]
        self.num_nodes = num_nodes
        self.num_sites = num_sites
        self.mu = mu
        self.rho = rho
        self.eps = eps
        self.scale_by_n = scale_by_n
        self.L_norm_mat = np.zeros((num_nodes, num_sites), dtype=float)
        self.L_mat = np.zeros((num_nodes, num_sites), dtype=float)
        self.L = np.full(num_nodes, 1.0)
        self.reference = reference
        self.query = query
        self.recomb_required = np.zeros((num_nodes, num_sites), dtype=bool)
        self.mismatch = np.ones((num_nodes, num_sites), dtype=bool)
        self.max_likelihood_node = np.full(num_sites, -1)
        self.max_likelihood = np.zeros(num_sites)
        self.num_switches = 0
        self.num_mismatches = 0
        self.path_likelihood = 0.0
        self.path = np.full(num_sites, -1, dtype=np.int64)
        if self.scale_by_n:
            self.n = self.num_nodes
        else:
            self.n = 1
    

    def update_site(self, site):
        failed_site = _update_site_numba(
            site,
            self.L,
            self.reference,
            self.query,
            self.recomb_required,
            self.mismatch,
            self.L_mat,
            self.L_norm_mat,
            self.max_likelihood_node,
            self.max_likelihood,
            self.rho,
            self.mu,
            self.n,
            self.eps,
        )
        if failed_site != -1:
            raise Exception(f"All likelihoods are zero at site {site}")

    def run(self):
        failed_site, num_mismatches, num_switches = _run_numba(
            self.L,
            self.reference,
            self.query,
            self.recomb_required,
            self.mismatch,
            self.L_mat,
            self.L_norm_mat,
            self.max_likelihood_node,
            self.max_likelihood,
            self.path,
            self.rho,
            self.mu,
            self.n,
            self.eps,
        )
        if failed_site != -1:
            raise Exception(f"All likelihoods are zero at site {failed_site}")

        self.num_switches = num_switches
        self.num_mismatches = num_mismatches
        self.path_likelihood = np.log10(self.max_likelihood).sum()
        return num_mismatches, num_switches

class AncestorHMM:
    def __init__(self, reference, query):
        num_nodes = reference.shape[0]
        num_sites = reference.shape[1]
        self.num_nodes = num_nodes
        self.num_sites = num_sites
        self.L_mat = np.zeros((num_nodes, num_sites), dtype=int)
        self.L_norm_mat = np.zeros((num_nodes, num_sites), dtype=int)
        self.L = np.ones(num_nodes, dtype=int)
        self.reference = reference
        self.query = query
        self.recomb_required = np.zeros((num_nodes, num_sites), dtype=bool)
        self.mismatch = np.ones((num_nodes, num_sites), dtype=bool)
        self.max_likelihood_node = np.full(num_sites, -1)
        self.num_switches = 0
        self.num_mismatches = 0
        self.path_likelihood = 0.0
        self.path = []
    
    def update_site(self, site):
        max_L = 0
        max_L_node = 0

        for u in range(self.num_nodes):
            last_L = self.L[u]
            mismatch = True
            if self.query[site] == self.reference[u, site]:
                mismatch = False
            self.mismatch[u, site] = mismatch
            if mismatch:
                if last_L == 0:
                    self.L[u] = 0
                else:
                    self.recomb_required[u, site] = True
                    self.L[u] = 1
            else:
                if last_L == 0:
                    self.L[u] = 2
                    self.recomb_required[u, site] = True
                else:
                    self.L[u] = 3
            self.L_mat[u, site] = self.L[u]
            if self.L[u] > max_L:
                max_L = self.L[u]
                max_L_node = u
        
        if max_L == 0:
            raise Exception(f"All likelihoods are zero at site {site}")
        
        for u in range(self.num_nodes):
            if self.L[u] == max_L:
                self.L[u] = 1
            else:
                self.L[u] = 0
            self.L_norm_mat[u,site] = self.L[u]
        self.max_likelihood_node[site] = max_L_node

    
    def run(self):
        for site in range(0, self.num_sites):
            self.update_site(site)
        path = []
        num_switches = 0
        num_mismatches = 0
        u = int(self.max_likelihood_node[self.num_sites - 1])
        path = np.full(self.num_sites, -1)
        for site in range(self.num_sites - 1, -1, -1):
            path[site] = int(u)
            num_mismatches += self.mismatch[u, site]
            if self.recomb_required[u, site]:
                num_switches += 1                
                u = self.max_likelihood_node[site-1]
        self.path = path
        self.num_switches = num_switches
        self.num_mismatches = num_mismatches
        return num_mismatches, num_switches
        


def calculate_rho(mu, k):
    rho_sc2ts = mu**k / (mu**k + (1 - mu)**k)
    rho_adj = mu**k / (1 - mu)**k
    dict = {"rho_sc2ts": rho_sc2ts, "rho_adj": rho_adj}
    return dict


def simulate_genotypes(num_samples, subset_size, sequence_length=2e5, seed=1, rate=1e-8):
    assert subset_size <= num_samples
    ts_base = msprime.sim_ancestry(
        num_samples+1,
        ploidy=1,
        sequence_length=sequence_length,
        random_seed=seed,
        recombination_rate=rate,
        population_size=1e4,
    )
    ts = msprime.sim_mutations(ts_base, rate=rate, random_seed=seed)
    genotype_matrix = ts.genotype_matrix(samples=range(subset_size+1)).transpose()
    reference = genotype_matrix[:subset_size]
    query = genotype_matrix[subset_size]
    return reference, query



def simulate_k(num_sims, num_samples, k_min, k_max, mu=0.25, sequence_length=2e5):
    records = []

    for n in range(num_sims):
        for k in range(k_min, k_max+1):
            rho_dict = calculate_rho(mu, k)
            reference, query = simulate_genotypes(num_samples=num_samples,
                                                  subset_size=num_samples,
                                                  sequence_length=sequence_length,
                                                  seed=n+1)
            for (rho_formula, rho) in rho_dict.items():
                hmm = LSHMM(reference, query, mu, rho, scale_by_n=False)
                num_mismatches, num_switches = hmm.run()
                mm_switch_ratio = np.nan
                if num_switches > 0:
                    mm_switch_ratio = num_mismatches / num_switches
                records.append({"index": n,
                                "k": k,
                                "rho": rho,
                                "rho_formula": rho_formula,
                                "num_mismatches": num_mismatches,
                                "num_switches": num_switches,
                                "mm_switch_ratio": mm_switch_ratio,
                                })
    df = pd.DataFrame.from_records(records)
    return df

def simulate_scale_by_n(num_sims,
                        num_samples,
                        subset_sizes,
                        mu=0.25,
                        rho=0.1,
                        sequence_length=2e5):
    records = []

    for n in range(num_sims):
        for scale_by_n in [True, False]:
            full_reference, query = simulate_genotypes(num_samples=num_samples,
                                                    subset_size=num_samples,
                                                    sequence_length=sequence_length,
                                                    seed=n+1)
            for subset_size in subset_sizes:
                reference = full_reference[0:subset_size,:]
                assert reference.shape[0] == subset_size
                hmm = LSHMM(reference, query, mu, rho, scale_by_n=scale_by_n)
                num_mismatches, num_switches = hmm.run()
                mm_switch_ratio = np.nan
                if num_switches > 0:
                    mm_switch_ratio = num_mismatches / num_switches
                records.append({"index": n,
                                "scale_by_n": scale_by_n,
                                "subset_size": subset_size,
                                "num_mismatches": num_mismatches,
                                "num_switches": num_switches,
                                "mm_switch_ratio": mm_switch_ratio,
                                })
    df = pd.DataFrame.from_records(records)
    return df


def plot_mismatch_by_k(df, 
                       ycol = "mm_switch_ratio",
                       fcol = "rho_formula",
                       ):
    
    label_map = {
        "rho_sc2ts": "sc2ts ρ formula",
        "rho_adj": "Adjusted ρ formula",
    }
    colors = {
        "rho_sc2ts": "#4c78a8",
        "rho_adj": "#f58518",
    }

    ks = sorted(df["k"].unique())
    positions = []
    data = []
    box_colors = []
    for i, k in enumerate(ks):
        for offset, key in [(-0.2, "rho_sc2ts"), (0.2, "rho_adj")]:
            values = df[(df["k"] == k) & (df[fcol] == key)][ycol].dropna().values
            if values.size == 0:
                continue
            positions.append(i + offset)
            data.append(values)
            box_colors.append(colors[key])

    fig, ax = plt.subplots()
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.35,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
    for median in bp["medians"]:
        median.set_color("black")

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Expected no. mismatches per switch (k)")
    ax.set_ylabel("No. mismatches / No. switches")
    ax.legend(
        handles=[
            Patch(facecolor=colors["rho_sc2ts"], label=label_map["rho_sc2ts"]),
            Patch(facecolor=colors["rho_adj"], label=label_map["rho_adj"]),
        ],
        loc="best",
    )
    plt.show()


def plot_scale_by_n(df):
    label_map = {
        True: "scale_by_n=True",
        False: "scale_by_n=False",
    }
    colors = {
        True: "#4c78a8",
        False: "#f58518",
    }
    subset_sizes = sorted(df["subset_size"].unique())
    positions = []
    data = []
    box_colors = []
    for i, subset_size in enumerate(subset_sizes):
        for offset, key in [(-0.2, True), (0.2, False)]:
            values = df[
                (df["subset_size"] == subset_size) & (df["scale_by_n"] == key)
            ]["num_switches"].dropna().values
            if values.size == 0:
                continue
            positions.append(i + offset)
            data.append(values)
            box_colors.append(colors[key])

    fig, ax = plt.subplots()
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.35,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
    for median in bp["medians"]:
        median.set_color("black")

    ax.set_xticks(range(len(subset_sizes)))
    ax.set_xticklabels([str(x) for x in subset_sizes])
    ax.set_xlabel("subset_size")
    ax.set_ylabel("num_switches")
    ax.legend(
        handles=[
            Patch(facecolor=colors[True], label=label_map[True]),
            Patch(facecolor=colors[False], label=label_map[False]),
        ],
        loc="best",
    )
    plt.show()


def _fmt(x):
    if not np.isfinite(x):
        return str(x)
    s = f"{x:.5f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _fmt_sci(x):
    if not np.isfinite(x):
        return str(x)
    return f"{x:.3e}"


def _text_color_for_fill(fill):
    if not fill or not fill.startswith("#"):
        return "black"
    hex_str = fill.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(ch * 2 for ch in hex_str)
    try:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
    except ValueError:
        return "black"
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    return "white" if luminance < 0.5 else "black"


def _colormap_hex(cmap_name, x, vmin, vmax):
    if not np.isfinite(x):
        return "#ffffff"
    if vmin == vmax:
        t = 0.5
    else:
        t = (x - vmin) / (vmax - vmin)
    t = float(np.clip(t, 0.0, 1.0))
    if cm is None:
        return "#ffffff"
    r, g, b, _ = cm.get_cmap(cmap_name)(t)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _draw_cell(parts, x, y, w, h, text=None, font_size=22, stroke_width=1, text_color="black", fill="white"):
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
        f'stroke="black" stroke-width="{stroke_width}" />'
    )
    if text is not None:
        if text_color is None:
            text_color = _text_color_for_fill(fill)
        parts.append(
            f'<text x="{x + w/2}" y="{y + h/2}" text-anchor="middle" '
            f'alignment-baseline="middle" font-size="{font_size}" fill="{text_color}">{text}</text>'
        )


def plot_hmm(
    hmm,
    label_col_width=100,
    top_axis_height=42,
    top_margin=12,
    cell_width=67,
    cell_height=67,
    like_height=19,
    like_text_height=11,
    like_text_gap=13,
    like_geno_gap=7,
    row_gap=18,
    col_gap=None,
    query_gap=34,
    path_stroke_width=6,
    font_family="sans-serif",
    cmap_name="Blues",
):
    col_gap = row_gap if col_gap is None else col_gap
    col_span = cell_width + col_gap
    row_span = like_text_gap + like_height + like_geno_gap + cell_height + row_gap

    path = getattr(hmm, "path", None)
    has_path = path is not None and len(path) > 0

    extra_right = 50
    if has_path:
        box_gap = col_gap
        summary_w = cell_width * 3.8
        extra_right = box_gap + summary_w
    total_width = label_col_width + hmm.num_sites * col_span - col_gap + extra_right
    nodes_h = hmm.num_nodes * row_span - row_gap
    nodes_top = top_axis_height + top_margin
    query_y = nodes_top + nodes_h + query_gap
    total_height = query_y + cell_height

    parts = []
    parts.append(
        f'<svg width="{total_width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:{font_family};">'
    )

    # Axis labels
    x_center = label_col_width + (hmm.num_sites * col_span - col_gap) / 2
    parts.append(
        f'<text x="{x_center}" y="{top_axis_height * 0.40}" text-anchor="middle" '
        f'alignment-baseline="middle" font-size="22">Site</text>'
    )
    y_center = nodes_top + nodes_h / 2
    parts.append(
        f'<text x="{label_col_width * 0.10}" y="{y_center}" text-anchor="middle" '
        f'alignment-baseline="middle" font-size="22" transform="rotate(-90 {label_col_width * 0.10} {y_center})">Ancestor</text>'
    )

    # Site tick labels (no tick lines)
    label_y = top_axis_height * 0.82
    for s in range(hmm.num_sites):
        x0 = label_col_width + s * col_span
        xc = x0 + cell_width / 2
        parts.append(
            f'<text x="{xc}" y="{label_y}" text-anchor="middle" alignment-baseline="middle" font-size="20">{s}</text>'
        )

    # Nodes (likelihoods + reference genotypes)
    row_label_x = label_col_width - 28
    lvals = hmm.L_norm_mat[np.isfinite(hmm.L_norm_mat)]
    lmin = float(np.min(lvals)) if lvals.size else 0.0
    lmax = float(np.max(lvals)) if lvals.size else 1.0
    for u in range(hmm.num_nodes):
        y_like = nodes_top + u * row_span + like_text_gap
        y_geno = y_like + like_height + like_geno_gap
        yc = y_geno + cell_height / 2

        parts.append(
            f'<text x="{row_label_x}" y="{yc}" text-anchor="end" alignment-baseline="middle" font-size="22">{u}</text>'
        )

        for s in range(hmm.num_sites):
            x = label_col_width + s * col_span

            unnorm = hmm.L_mat0[u, s] if hasattr(hmm, "L_mat0") else hmm.L_mat[u, s]
            parts.append(
                f'<text x="{x + cell_width/2}" y="{y_like - like_text_gap/2}" '
                f'text-anchor="middle" alignment-baseline="middle" '
                f'font-size="{int(like_text_height*1.25)}" fill="black">{_fmt(unnorm)}</text>'
            )
            _draw_cell(
                parts,
                x=x,
                y=y_like,
                w=cell_width,
                h=like_height,
                text=_fmt(hmm.L_norm_mat[u, s]),
                font_size=int(like_height * 0.65),
                stroke_width=1,
                fill=_colormap_hex(cmap_name, hmm.L_norm_mat[u, s], lmin, lmax),
                text_color=None,
            )

            if hmm.recomb_required[u, s]:
                parts.append(
                    f'<rect x="{x}" y="{y_like + like_height}" '
                    f'width="{cell_width}" height="{like_geno_gap}" '
                    f'fill="orange" stroke="orange" stroke-width="1" />'
                )

            is_mismatch = bool(hmm.mismatch[u, s])
            g_fill = "white" if is_mismatch else "#dddddd"
            g_color = "red" if is_mismatch else "black"
            _draw_cell(
                parts,
                x=x,
                y=y_geno,
                w=cell_width,
                h=cell_height,
                text=str(int(hmm.reference[u, s])),
                font_size=int(cell_height * 0.65),
                stroke_width=1,
                fill=g_fill,
                text_color=g_color,
            )

    # Query row (genotypes only)
    query_x0 = label_col_width
    query_w = hmm.num_sites * col_span - col_gap
    query_pad = cell_height * 0.08
    parts.append(
        f'<rect x="{query_x0 - query_pad}" y="{query_y - query_pad}" '
        f'width="{query_w + 2 * query_pad}" height="{cell_height + 2 * query_pad}" '
        f'fill="black" stroke="none" />'
    )
    parts.append(
        f'<text x="{row_label_x}" y="{query_y + cell_height/2}" text-anchor="end" '
        f'alignment-baseline="middle" font-size="22">Query</text>'
    )
    for s in range(hmm.num_sites):
        x = label_col_width + s * col_span
        _draw_cell(
            parts,
            x=x,
            y=query_y,
            w=cell_width,
            h=cell_height,
            text=str(int(hmm.query[s])),
            font_size=int(cell_height * 0.65),
            stroke_width=1,
            fill="#dddddd",
        )

    # Viterbi path overlay (genotype cells)
    if has_path:
        half_gap = col_gap / 2
        for s, u in enumerate(path):
            if u is None or u < 0:
                continue
            x = label_col_width + s * col_span
            y = nodes_top + u * row_span + like_text_gap + like_height + like_geno_gap
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" '
                f'fill="none" stroke="black" stroke-width="{path_stroke_width}" />'
            )
            if s == hmm.num_sites - 1:
                continue
            u_next = path[s + 1]
            if u_next is None or u_next < 0:
                continue
            y_next = nodes_top + u_next * row_span + like_text_gap + like_height + like_geno_gap
            yc = y + cell_height / 2
            yc_next = y_next + cell_height / 2
            x_right = x + cell_width
            x_next = label_col_width + (s + 1) * col_span
            x_mid = x_right + half_gap
            if u_next == u:
                parts.append(
                    f'<line x1="{x_right}" y1="{yc}" x2="{x_next}" y2="{yc}" '
                    f'stroke="black" stroke-width="{path_stroke_width}" '
                    f'stroke-linecap="square" />'
                )
            else:
                parts.append(
                    f'<line x1="{x_right}" y1="{yc}" x2="{x_mid}" y2="{yc}" '
                    f'stroke="black" stroke-width="{path_stroke_width}" '
                    f'stroke-linecap="square" />'
                )
                parts.append(
                    f'<line x1="{x_mid}" y1="{yc}" x2="{x_mid}" y2="{yc_next}" '
                    f'stroke="black" stroke-width="{path_stroke_width}" '
                    f'stroke-linecap="square" />'
                )
                parts.append(
                    f'<line x1="{x_mid}" y1="{yc_next}" x2="{x_next}" y2="{yc_next}" '
                    f'stroke="black" stroke-width="{path_stroke_width}" '
                    f'stroke-linecap="square" />'
                )
        # Summary box to the right of the final path node
        last_idx = len(path) - 1
        u_last = path[last_idx]
        if u_last is not None and u_last >= 0:
            x_last = label_col_width + last_idx * col_span
            y_last = nodes_top + u_last * row_span + like_text_gap + like_height + like_geno_gap
            yc_last = y_last + cell_height / 2
            box_gap = col_gap
            summary_w = cell_width * 3.8
            line_h = cell_height * 0.40
            pad_y = cell_height * 0.12
            summary_h = line_h * 3 + pad_y * 2
            summary_x = x_last + cell_width + box_gap
            summary_y = yc_last - summary_h / 2
            parts.append(
                f'<line x1="{x_last + cell_width}" y1="{yc_last}" '
                f'x2="{summary_x}" y2="{yc_last}" '
                f'stroke="black" stroke-width="{path_stroke_width}" '
                f'stroke-linecap="square" />'
            )
            parts.append(
                f'<rect x="{summary_x}" y="{summary_y}" width="{summary_w}" height="{summary_h}" '
                f'fill="white" stroke="black" stroke-width="{path_stroke_width}" />'
            )
            title_margin = pad_y * 1.4
            parts.append(
                f'<text x="{summary_x}" y="{summary_y - title_margin}" text-anchor="start" '
                f'alignment-baseline="middle" font-size="{int(line_h * 0.84)}">Path summary</text>'
            )
            summary_rows = [
                ("num_switches", str(hmm.num_switches)),
                ("num_mismatches", str(hmm.num_mismatches)),
                ("path_likelihood", _fmt_sci(hmm.path_likelihood)),
            ]
            text_x = summary_x + summary_w * 0.06
            for i, (label, value) in enumerate(summary_rows):
                y_text = summary_y + pad_y + (i + 0.5) * line_h
                parts.append(
                    f'<text x="{text_x}" y="{y_text}" text-anchor="start" '
                    f'alignment-baseline="middle" font-size="{int(line_h * 0.6)}">{label}: {value}</text>'
                )

    parts.append("</svg>")
    return "".join(parts)


def show_hmm(hmm, **kwargs):
    display(HTML(plot_hmm(hmm, **kwargs)))


def plot_hmm_heatmap(
    hmm,
    likelihood_type="norm",
    cmap="viridis",
    figsize=(14, 8),
    missing_color="#f0f0f0",
):
    if likelihood_type == "norm":
        likelihood_mat = np.asarray(hmm.L_norm_mat, dtype=np.float64)
        title_suffix = "normalised likelihood"
    elif likelihood_type in {"raw", "unnorm"}:
        likelihood_mat = np.asarray(hmm.L_mat, dtype=np.float64)
        title_suffix = "likelihood"
    else:
        raise ValueError("likelihood_type must be 'norm', 'unnorm', or 'raw'")

    reference = np.asarray(hmm.reference)
    missing_mask = reference == -1
    valid_mask = np.isfinite(likelihood_mat) & ~missing_mask
    if not np.any(valid_mask):
        raise ValueError("No finite likelihood values available to plot")

    valid_values = likelihood_mat[valid_mask]

    base_cmap = plt.get_cmap(cmap)
    heat_cmap = LinearSegmentedColormap.from_list(
        f"{cmap}_high", base_cmap(np.linspace(0.4, 1.0, 256))
    )
    base_cmap = base_cmap.with_extremes(bad=missing_color)
    heat_cmap.set_bad(missing_color)
    plot_mask = missing_mask | ~np.isfinite(likelihood_mat)
    use_eps_floor = likelihood_type == "norm" and getattr(hmm, "eps", 0) > 0

    if use_eps_floor:
        eps_value = float(hmm.eps)
        threshold_mask = valid_mask & np.isclose(likelihood_mat, eps_value)
        non_threshold_mask = valid_mask & ~threshold_mask
        mapped_values = np.zeros_like(likelihood_mat, dtype=np.float64)

        threshold_eps = max(np.finfo(np.float64).eps, abs(eps_value) * 1e-12)
        legend_min = eps_value + threshold_eps

        if np.any(non_threshold_mask):
            non_threshold_values = likelihood_mat[non_threshold_mask]
            max_non_threshold = float(np.max(non_threshold_values))
            if max_non_threshold <= legend_min:
                mapped_values[non_threshold_mask] = 0.4
            else:
                scaled = (
                    np.clip(non_threshold_values, legend_min, max_non_threshold) - legend_min
                ) / (max_non_threshold - legend_min)
                mapped_values[non_threshold_mask] = 0.4 + 0.6 * scaled
        else:
            max_non_threshold = legend_min * (1 + 1e-12)

        mapped_values[threshold_mask] = 0.0
        plot_values = np.ma.array(mapped_values, mask=plot_mask)
        image_cmap = base_cmap
        image_vmin = 0.0
        image_vmax = 1.0
    else:
        vmin = float(np.min(valid_values))
        vmax = float(np.max(valid_values))
        if vmax <= vmin:
            vmax = vmin + 1e-12
        plot_values = np.ma.array(likelihood_mat, mask=plot_mask)
        image_cmap = heat_cmap
        image_vmin = vmin
        image_vmax = vmax

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        plot_values,
        aspect="auto",
        interpolation="nearest",
        cmap=image_cmap,
        vmin=image_vmin,
        vmax=image_vmax,
        origin="upper",
    )

    ax.set_xlabel("Site")
    ax.set_ylabel("Ancestor")
    ax.set_title(f"LSHMM {title_suffix} heatmap")

    path = getattr(hmm, "path", None)
    has_path = path is not None and len(path) > 0
    if has_path:
        path = np.asarray(path, dtype=np.int64)
        valid_path_mask = path >= 0
        if np.any(valid_path_mask):
            x = np.arange(len(path), dtype=np.float64)
            y = path.astype(np.float64)
            ax.plot(x[valid_path_mask], y[valid_path_mask], color="red", linewidth=1.6)

            site_idx = np.flatnonzero(valid_path_mask)
            mismatch_sites = site_idx[hmm.mismatch[path[site_idx], site_idx]]
            recomb_sites = site_idx[hmm.recomb_required[path[site_idx], site_idx]]

            if mismatch_sites.size > 0:
                ax.scatter(
                    mismatch_sites,
                    path[mismatch_sites],
                    marker="x",
                    s=36,
                    color="red",
                    linewidths=1.4,
                    zorder=4,
                )
            if recomb_sites.size > 0:
                ax.scatter(
                    recomb_sites,
                    path[recomb_sites],
                    marker="s",
                    s=38,
                    facecolors="none",
                    edgecolors="orange",
                    linewidths=1.6,
                    zorder=4,
                )

    handles = [
        Line2D([0], [0], color="red", lw=1.8, label="path"),
        Line2D(
            [0],
            [0],
            color="red",
            marker="x",
            linestyle="None",
            markersize=7,
            markeredgewidth=1.4,
            label=f"mismatch (n={int(hmm.num_mismatches)})",
        ),
        Line2D(
            [0],
            [0],
            color="orange",
            marker="s",
            markerfacecolor="none",
            linestyle="None",
            markersize=7,
            markeredgewidth=1.4,
            label=f"recombination (n={int(hmm.num_switches)})",
        ),
        Patch(facecolor=missing_color, edgecolor="none", label="missing"),
    ]

    max_xticks = min(12, hmm.num_sites)
    if max_xticks > 0:
        xticks = np.linspace(0, hmm.num_sites - 1, num=max_xticks, dtype=int)
        ax.set_xticks(np.unique(xticks))

    if hmm.num_nodes > 30:
        ax.set_yticks(np.linspace(0, hmm.num_nodes - 1, num=10, dtype=int))

    fig.subplots_adjust(right=0.80)
    ax_pos = ax.get_position()
    top = ax_pos.y1
    bottom = ax_pos.y0
    total_height = top - bottom
    legend_x = ax_pos.x1 + 0.02
    cbar_width = 0.022
    legend_box_h = total_height * 0.18
    legend_gap = total_height * 0.04
    legend_box_y = bottom
    colorbar_bottom = legend_box_y + legend_box_h + legend_gap

    if use_eps_floor:
        gap_height = total_height * 0.04
        threshold_height = total_height * 0.06
        main_height = top - colorbar_bottom - gap_height - threshold_height
        main_y = top - main_height
        threshold_y = colorbar_bottom
        high_vmax = max(max_non_threshold, legend_min * (1 + 1e-12))

        main_cax = fig.add_axes([legend_x, main_y, cbar_width, main_height])
        high_sm = plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=legend_min, vmax=high_vmax),
            cmap=heat_cmap,
        )
        high_sm.set_array([])
        high_cbar = fig.colorbar(high_sm, cax=main_cax)
        high_cbar.set_label(title_suffix)

        eps_label = np.format_float_scientific(eps_value, precision=0)
        mid_ticks = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
        valid_mid_ticks = mid_ticks[
            np.logical_and(mid_ticks > legend_min, mid_ticks < high_vmax)
        ]
        tick_values = np.concatenate(
            [np.array([legend_min], dtype=np.float64), valid_mid_ticks, np.array([high_vmax])]
        )
        high_cbar.set_ticks(tick_values.tolist())
        high_cbar.set_ticklabels(
            [f"{eps_label} + ε"]
            + [f"{tick:.1g}" for tick in valid_mid_ticks]
            + [f"{high_vmax:.2g}"]
        )

        threshold_ax = fig.add_axes([legend_x, threshold_y, cbar_width, threshold_height])
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
            eps_label,
            ha="left",
            va="center",
            transform=threshold_ax.transAxes,
        )
    else:
        main_cax = fig.add_axes([legend_x, colorbar_bottom, cbar_width, top - colorbar_bottom])
        cbar = fig.colorbar(image, cax=main_cax)
        cbar.set_label(title_suffix)

    legend_box_x = legend_x
    legend_box_w = 0.18
    legend_ax = fig.add_axes([legend_box_x, legend_box_y, legend_box_w, legend_box_h])
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        fancybox=False,
        borderaxespad=0.0,
    )

    return fig, ax


class HMMViz:
    def __init__(self, num_nodes, num_sites, seed=1):
        self.num_nodes = num_nodes
        self.num_sites = num_sites
        rng = np.random.default_rng(seed)
        self.reference = rng.integers(0, 2, size=(num_nodes, num_sites))
        self.query = rng.integers(0, 2, size=num_sites)


    def _make_hmm(self, mu, rho, scale_by_n, k=0, use_sc2ts_formula=True):
        rho_dict = {"rho_sc2ts": 0, "rho_adj": 0}
        if k > 0:
            rho_dict = calculate_rho(0.25, k)
            if use_sc2ts_formula:
                rho = rho_dict["rho_sc2ts"]
            else:
                rho = rho_dict["rho_adj"]
        hmm = LSHMM(
            reference=self.reference,
            query=self.query,
            mu=mu,
            rho=rho,
            scale_by_n=scale_by_n,
        )
        
        hmm.rho_sc2ts = rho_dict["rho_sc2ts"]
        hmm.rho_adj = rho_dict["rho_adj"]
        num_mismatches, num_switches = hmm.run()
        return hmm

    def visualise(self, **plot_kwargs):
        mu_text = widgets.FloatText(value=0.2, description="μ")
        mu_slider = widgets.FloatSlider(
            value=0.2, min=0.0, max=1.0, step=0.001, readout=False, layout=widgets.Layout(width="200px")
        )
        rho_text = widgets.FloatText(value=0.1, description="ρ")
        rho_slider = widgets.FloatSlider(
            value=0.1, min=0.0, max=1.0, step=0.001, readout=False, layout=widgets.Layout(width="200px")
        )
        scale_toggle = widgets.Checkbox(value=False, description="scale_by_n")
        k_text = widgets.BoundedIntText(value=0, min=0, max=10**9, step=1, description="k")
        k_slider = widgets.IntSlider(
            value=0, min=0, max=100, step=1, readout=False, layout=widgets.Layout(width="200px")
        )
        use_sc2ts_toggle = widgets.Checkbox(value=True, description="use_sc2ts_formula")

        widgets.jslink((mu_text, "value"), (mu_slider, "value"))
        widgets.jslink((rho_text, "value"), (rho_slider, "value"))
        widgets.jslink((k_text, "value"), (k_slider, "value"))

        svg_out = widgets.HTML(value="")
        scroll_container = widgets.Box(
            [svg_out],
            layout=widgets.Layout(overflow_x="auto", border="1px solid gray", width="2000px"),
        )
        rho_mu_heading = widgets.HTML("<b style='font-size:18px'>tsinfer parameterisation</b>")
        sc2ts_heading = widgets.HTML("<b style='font-size:18px'>sc2ts parameterisation</b>")
        formulas = widgets.Output()

        for w in (mu_text, rho_text, k_text):
            w.style = {"description_width": "initial"}
            w.layout = widgets.Layout(width="140px")
        for w in (mu_slider, rho_slider, k_slider):
            w.layout = widgets.Layout(width="220px")
        for w in (scale_toggle, use_sc2ts_toggle):
            w.layout = widgets.Layout(width="220px")

        style_tag = widgets.HTML(
            "<style>.widget-label{font-size:15px;}"
            ".widget-text input,.widget-int input,.widget-float input{font-size:15px;}"
            ".widget-slider input[type=range]{height:18px;}</style>"
        )

        controls_tsinfer = widgets.HBox([mu_text, mu_slider, rho_text, rho_slider, scale_toggle])
        controls_sc2ts = widgets.HBox([k_text, k_slider, use_sc2ts_toggle])
        container = widgets.VBox(
            [scroll_container, style_tag, rho_mu_heading, controls_tsinfer, sc2ts_heading, controls_sc2ts, formulas]
        )

        def update_svg(*args):
            hmm = self._make_hmm(
                mu_text.value,
                rho_text.value,
                scale_toggle.value,
                k=k_text.value,
                use_sc2ts_formula=use_sc2ts_toggle.value,
            )
            svg_out.value = plot_hmm(hmm, **plot_kwargs)
            formulas.clear_output(wait=True)
            with formulas:
                display(
                    Math(
                        r"\rho_{\mathrm{sc2ts}} = \frac{\mu^k}{\mu^k + (1-\mu)^k}"
                        + r" = "
                        + f"{hmm.rho_sc2ts:.6g}"
                    )
                )
                display(
                    Math(
                        r"\rho_{\mathrm{adj}} = \frac{\mu^k}{(1-\mu)^k}"
                        + r" = "
                        + f"{hmm.rho_adj:.6g}"
                    )
                )

        mu_text.observe(lambda change: update_svg() if change["name"] == "value" else None, names="value")
        rho_text.observe(lambda change: update_svg() if change["name"] == "value" else None, names="value")
        scale_toggle.observe(
            lambda change: update_svg() if change["name"] == "value" else None, names="value"
        )
        k_text.observe(lambda change: update_svg() if change["name"] == "value" else None, names="value")
        use_sc2ts_toggle.observe(
            lambda change: update_svg() if change["name"] == "value" else None, names="value"
        )

        display(container)
        update_svg()

        
