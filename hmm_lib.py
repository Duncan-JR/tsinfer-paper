import numpy as np
from IPython.display import HTML, display
import ipywidgets as widgets
try:
    from matplotlib import cm
except Exception:
    cm = None


def randomise_haplotypes(num_nodes, num_sites, seed=1):
    np.random.seed(seed)
    reference = np.random.randint(0, 2, (num_nodes, num_sites))
    query = np.random.randint(0, 2, num_sites)
    return reference, query

class LSHMM:
    def __init__(self, reference, query, mu, rho, scale_by_n=False):
        num_nodes = reference.shape[0]
        num_sites = reference.shape[1]
        self.num_nodes = num_nodes
        self.num_sites = num_sites
        self.mu = mu
        self.rho = rho
        self.scale_by_n = scale_by_n
        self.L_norm_mat = np.zeros((num_nodes, num_sites), dtype=float)
        self.L_mat = np.zeros((num_nodes, num_sites), dtype=float)
        self.L = np.full(num_nodes, 1.0)
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
        max_L = -1
        max_L_node = -1
        rho = self.rho
        mu = self.mu
        if self.scale_by_n:
            n = self.num_nodes
        else:
            n = 1
        
        for u in range(self.num_nodes):
            p_last = self.L[u]
            p_no_recomb = p_last * (1 - rho + rho / n)
            p_recomb = rho / n
            if p_no_recomb > p_recomb:
                p_transition = p_no_recomb
            else:
                p_transition = p_recomb
                self.recomb_required[u, site] = True

            p_emission = mu
            if self.query[site] == self.reference[u, site]:
                self.mismatch[u, site] = False
                p_emission = 1 - self.mu
            self.L[u] = p_transition * p_emission
            self.L_mat[u, site] = self.L[u]
            if self.L[u] > max_L:
                max_L = self.L[u]
                max_L_node = u
        
        if max_L == 0:
            raise Exception(f"All likelihoods are zero at site {site}")
        
        self.max_likelihood_node[site] = max_L_node
        for u in range(self.num_nodes):
            L_norm = self.L[u] / max_L
            self.L[u] = L_norm
            self.L_norm_mat[u, site] = L_norm

    def run(self):
        for site in range(0, self.num_sites):
            self.update_site(site)
        path = []
        num_switches = 0
        num_mismatches = 0
        u = int(self.max_likelihood_node[self.num_sites - 1])
        for site in range(self.num_sites - 1, -1, -1):
            path.append(int(u))
            num_mismatches += self.mismatch[u, site]
            if self.recomb_required[u, site]:
                num_switches += 1
                assert site > 0
                new_u = self.max_likelihood_node[site-1]
                assert u != new_u
                u = new_u
        path.reverse()
        self.path = path
        self.num_switches = num_switches
        self.num_mismatches = num_mismatches
        self.path_likelihood = self.rho**num_switches * self.mu**num_mismatches
    

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
    row_span = like_text_height + like_height + like_geno_gap + cell_height + row_gap

    extra_right = 50
    if getattr(hmm, "path", None):
        box_gap = col_gap * 0.6
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
        f'alignment-baseline="middle" font-size="22" transform="rotate(-90 {label_col_width * 0.10} {y_center})">Sample</text>'
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
        y_like = nodes_top + u * row_span + like_text_height
        y_geno = y_like + like_height + like_geno_gap
        yc = y_geno + cell_height / 2

        parts.append(
            f'<text x="{row_label_x}" y="{yc}" text-anchor="end" alignment-baseline="middle" font-size="22">{u}</text>'
        )

        for s in range(hmm.num_sites):
            x = label_col_width + s * col_span

            unnorm = hmm.L_mat0[u, s] if hasattr(hmm, "L_mat0") else hmm.L_mat[u, s]
            parts.append(
                f'<text x="{x + cell_width/2}" y="{y_like - like_text_height/2}" '
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
    if getattr(hmm, "path", None):
        half_gap = col_gap / 2
        for s, u in enumerate(hmm.path):
            if u is None or u < 0:
                continue
            x = label_col_width + s * col_span
            y = nodes_top + u * row_span + like_text_height + like_height + like_geno_gap
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" '
                f'fill="none" stroke="black" stroke-width="{path_stroke_width}" />'
            )
            if s == hmm.num_sites - 1:
                continue
            u_next = hmm.path[s + 1]
            if u_next is None or u_next < 0:
                continue
            y_next = nodes_top + u_next * row_span + like_text_height + like_height + like_geno_gap
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
        last_idx = len(hmm.path) - 1
        u_last = hmm.path[last_idx]
        if u_last is not None and u_last >= 0:
            x_last = label_col_width + last_idx * col_span
            y_last = nodes_top + u_last * row_span + like_text_height + like_height + like_geno_gap
            yc_last = y_last + cell_height / 2
            box_gap = col_gap * 0.6
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


class HMMViz:
    def __init__(self, num_nodes, num_sites, seed=1):
        self.num_nodes = num_nodes
        self.num_sites = num_sites
        rng = np.random.default_rng(seed)
        self.reference = rng.integers(0, 2, size=(num_nodes, num_sites))
        self.query = rng.integers(0, 2, size=num_sites)

    def _make_hmm(self, mu, rho, scale_by_n):
        hmm = LSHMM(
            reference=self.reference,
            query=self.query,
            mu=mu,
            rho=rho,
            scale_by_n=scale_by_n,
        )
        hmm.run()
        return hmm

    def visualise(self, **plot_kwargs):
        mu_text = widgets.FloatText(value=0.2, description="mu")
        mu_slider = widgets.FloatSlider(
            value=0.2, min=0.0, max=1.0, step=0.001, readout=False, layout=widgets.Layout(width="200px")
        )
        rho_text = widgets.FloatText(value=0.1, description="rho")
        rho_slider = widgets.FloatSlider(
            value=0.1, min=0.0, max=1.0, step=0.001, readout=False, layout=widgets.Layout(width="200px")
        )
        scale_toggle = widgets.ToggleButton(value=False, description="scale_by_n")

        widgets.jslink((mu_text, "value"), (mu_slider, "value"))
        widgets.jslink((rho_text, "value"), (rho_slider, "value"))

        svg_out = widgets.HTML(value="")
        scroll_container = widgets.Box(
            [svg_out],
            layout=widgets.Layout(overflow_x="auto", border="1px solid gray", width="2000px"),
        )
        controls = widgets.HBox([mu_text, mu_slider, rho_text, rho_slider, scale_toggle])
        container = widgets.VBox([scroll_container, controls])

        def update_svg(*args):
            hmm = self._make_hmm(mu_text.value, rho_text.value, scale_toggle.value)
            svg_out.value = plot_hmm(hmm, **plot_kwargs)

        mu_text.observe(lambda change: update_svg() if change["name"] == "value" else None, names="value")
        rho_text.observe(lambda change: update_svg() if change["name"] == "value" else None, names="value")
        scale_toggle.observe(
            lambda change: update_svg() if change["name"] == "value" else None, names="value"
        )

        display(container)
        update_svg()

        
