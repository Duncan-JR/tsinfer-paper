import numpy as np
from IPython.display import HTML, display
try:
    from matplotlib import cm
except Exception:
    cm = None

class LSHMM:
    def __init__(self, num_nodes, num_sites, mu, rho, scale_by_n=False):
        self.num_nodes = num_nodes
        self.num_sites = num_sites
        self.mu = mu
        self.rho = rho
        self.scale_by_n = scale_by_n
        self.L_norm_mat = np.zeros((num_nodes, num_sites), dtype=float)
        self.L_mat = np.zeros((num_nodes, num_sites), dtype=float)
        self.L = np.full(self.num_nodes, 1.0)
        self.reference = np.zeros((num_nodes, num_sites))
        self.query = np.zeros(num_sites)
        self.recomb_required = np.zeros((num_nodes, num_sites), dtype=bool)
        self.mismatch = np.ones((num_nodes, num_sites), dtype=bool)
        self.max_likelihood_node = np.full(num_sites, -1)
        self.num_switches = 0
        self.num_mismatches = 0
        self.path_likelihood = 0.0
        self.path = []
    
    def randomise_haplotypes(self, seed=1):
        np.random.seed(seed)
        self.reference = np.random.randint(0, 2, (self.num_nodes, self.num_sites))
        self.query = np.random.randint(0, 2, self.num_sites)

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
    

class HMMViz:
    def __init__(
        self,
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
    ):
        self.hmm = hmm
        self.label_col_width = label_col_width
        self.top_axis_height = top_axis_height
        self.top_margin = top_margin
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.like_height = like_height
        self.like_text_height = like_text_height
        self.like_geno_gap = like_geno_gap
        self.row_gap = row_gap
        self.col_gap = row_gap if col_gap is None else col_gap
        self.query_gap = query_gap
        self.path_stroke_width = path_stroke_width
        self.font_family = font_family

        self.num_nodes = hmm.num_nodes
        self.num_sites = hmm.num_sites

    def _fmt(self, x):
        if not np.isfinite(x):
            return str(x)
        s = f"{x:.5f}".rstrip("0").rstrip(".")
        return s if s else "0"

    def _fmt_sci(self, x):
        if not np.isfinite(x):
            return str(x)
        return f"{x:.3e}"

    def _text_color_for_fill(self, fill):
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

    def _colormap_hex(self, cmap_name, x, vmin, vmax):
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

    def draw_cell(self, parts, x, y, w, h, text=None, font_size=22, stroke_width=1, text_color="black", fill="white"):
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
            f'stroke="black" stroke-width="{stroke_width}" />'
        )
        if text is not None:
            if text_color is None:
                text_color = self._text_color_for_fill(fill)
            parts.append(
                f'<text x="{x + w/2}" y="{y + h/2}" text-anchor="middle" '
                f'alignment-baseline="middle" font-size="{font_size}" fill="{text_color}">{text}</text>'
            )

    def visualise(self):
        col_span = self.cell_width + self.col_gap
        row_span = (
            self.like_text_height + self.like_height + self.like_geno_gap + self.cell_height + self.row_gap
        )

        extra_right = 50
        if getattr(self.hmm, "path", None):
            box_gap = self.col_gap * 0.6
            summary_w = self.cell_width * 3.8
            extra_right = box_gap + summary_w
        total_width = (
            self.label_col_width + self.num_sites * col_span - self.col_gap + extra_right
        )
        nodes_h = self.num_nodes * row_span - self.row_gap
        nodes_top = self.top_axis_height + self.top_margin
        query_y = nodes_top + nodes_h + self.query_gap
        total_height = query_y + self.cell_height

        parts = []
        parts.append(
            f'<svg width="{total_width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg" '
            f'style="font-family:{self.font_family};">'
        )

        # Axis labels
        x_center = self.label_col_width + (self.num_sites * col_span - self.col_gap) / 2
        parts.append(
            f'<text x="{x_center}" y="{self.top_axis_height * 0.40}" text-anchor="middle" '
            f'alignment-baseline="middle" font-size="22">Site</text>'
        )
        y_center = nodes_top + nodes_h / 2
        parts.append(
            f'<text x="{self.label_col_width * 0.10}" y="{y_center}" text-anchor="middle" '
            f'alignment-baseline="middle" font-size="22" transform="rotate(-90 {self.label_col_width * 0.10} {y_center})">Sample</text>'
        )

        # Site tick labels (no tick lines)
        label_y = self.top_axis_height * 0.82
        for s in range(self.num_sites):
            x0 = self.label_col_width + s * col_span
            xc = x0 + self.cell_width / 2
            parts.append(
                f'<text x="{xc}" y="{label_y}" text-anchor="middle" alignment-baseline="middle" font-size="20">{s}</text>'
            )

        # Nodes (likelihoods + reference genotypes)
        row_label_x = self.label_col_width - 28
        lvals = self.hmm.L_norm_mat[np.isfinite(self.hmm.L_norm_mat)]
        lmin = float(np.min(lvals)) if lvals.size else 0.0
        lmax = float(np.max(lvals)) if lvals.size else 1.0
        for u in range(self.num_nodes):
            y_like = nodes_top + u * row_span + self.like_text_height
            y_geno = y_like + self.like_height + self.like_geno_gap
            yc = y_geno + self.cell_height / 2

            parts.append(
                f'<text x="{row_label_x}" y="{yc}" text-anchor="end" alignment-baseline="middle" font-size="22">{u}</text>'
            )

            for s in range(self.num_sites):
                x = self.label_col_width + s * col_span

                unnorm = (
                    self.hmm.L_mat0[u, s]
                    if hasattr(self.hmm, "L_mat0")
                    else self.hmm.L_mat[u, s]
                )
                parts.append(
                    f'<text x="{x + self.cell_width/2}" y="{y_like - self.like_text_height/2}" '
                    f'text-anchor="middle" alignment-baseline="middle" '
                    f'font-size="{int(self.like_text_height*1.25)}" fill="black">{self._fmt(unnorm)}</text>'
                )
                self.draw_cell(
                    parts,
                    x=x,
                    y=y_like,
                    w=self.cell_width,
                    h=self.like_height,
                    text=self._fmt(self.hmm.L_norm_mat[u, s]),
                    font_size=int(self.like_height * 0.65),
                    stroke_width=1,
                    fill=self._colormap_hex("Blues", self.hmm.L_norm_mat[u, s], lmin, lmax),
                    text_color=None,
                )

                if self.hmm.recomb_required[u, s]:
                    parts.append(
                        f'<rect x="{x}" y="{y_like + self.like_height}" '
                        f'width="{self.cell_width}" height="{self.like_geno_gap}" '
                        f'fill="orange" stroke="orange" stroke-width="1" />'
                    )

                is_mismatch = bool(self.hmm.mismatch[u, s])
                g_fill =  "white" if is_mismatch else "#dddddd"
                g_color = "red" if is_mismatch else "black"
                self.draw_cell(
                    parts,
                    x=x,
                    y=y_geno,
                    w=self.cell_width,
                    h=self.cell_height,
                    text=str(int(self.hmm.reference[u, s])),
                    font_size=int(self.cell_height * 0.65),
                    stroke_width=1,
                    fill=g_fill,
                    text_color=g_color,
                )

        # Query row (genotypes only)
        query_x0 = self.label_col_width
        query_w = self.num_sites * col_span - self.col_gap
        query_pad = self.cell_height * 0.08
        parts.append(
            f'<rect x="{query_x0 - query_pad}" y="{query_y - query_pad}" '
            f'width="{query_w + 2 * query_pad}" height="{self.cell_height + 2 * query_pad}" '
            f'fill="black" stroke="none" />'
        )
        parts.append(
            f'<text x="{row_label_x}" y="{query_y + self.cell_height/2}" text-anchor="end" '
            f'alignment-baseline="middle" font-size="22">Query</text>'
        )
        for s in range(self.num_sites):
            x = self.label_col_width + s * col_span
            self.draw_cell(
                parts,
                x=x,
                y=query_y,
                w=self.cell_width,
                h=self.cell_height,
                text=str(int(self.hmm.query[s])),
                font_size=int(self.cell_height * 0.65),
                stroke_width=1,
                fill="#dddddd",
            )

        # Viterbi path overlay (genotype cells)
        if getattr(self.hmm, "path", None):
            half_gap = self.col_gap / 2
            for s, u in enumerate(self.hmm.path):
                if u is None or u < 0:
                    continue
                x = self.label_col_width + s * col_span
                y = nodes_top + u * row_span + self.like_text_height + self.like_height + self.like_geno_gap
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{self.cell_width}" height="{self.cell_height}" '
                    f'fill="none" stroke="black" stroke-width="{self.path_stroke_width}" />'
                )
                if s == self.num_sites - 1:
                    continue
                u_next = self.hmm.path[s + 1]
                if u_next is None or u_next < 0:
                    continue
                y_next = nodes_top + u_next * row_span + self.like_text_height + self.like_height + self.like_geno_gap
                yc = y + self.cell_height / 2
                yc_next = y_next + self.cell_height / 2
                x_right = x + self.cell_width
                x_next = self.label_col_width + (s + 1) * col_span
                x_mid = x_right + half_gap
                if u_next == u:
                    parts.append(
                        f'<line x1="{x_right}" y1="{yc}" x2="{x_next}" y2="{yc}" '
                        f'stroke="black" stroke-width="{self.path_stroke_width}" '
                        f'stroke-linecap="square" />'
                    )
                else:
                    parts.append(
                        f'<line x1="{x_right}" y1="{yc}" x2="{x_mid}" y2="{yc}" '
                        f'stroke="black" stroke-width="{self.path_stroke_width}" '
                        f'stroke-linecap="square" />'
                    )
                    parts.append(
                        f'<line x1="{x_mid}" y1="{yc}" x2="{x_mid}" y2="{yc_next}" '
                        f'stroke="black" stroke-width="{self.path_stroke_width}" '
                        f'stroke-linecap="square" />'
                    )
                    parts.append(
                        f'<line x1="{x_mid}" y1="{yc_next}" x2="{x_next}" y2="{yc_next}" '
                        f'stroke="black" stroke-width="{self.path_stroke_width}" '
                        f'stroke-linecap="square" />'
                    )
            # Summary box to the right of the final path node
            last_idx = len(self.hmm.path) - 1
            u_last = self.hmm.path[last_idx]
            if u_last is not None and u_last >= 0:
                x_last = self.label_col_width + last_idx * col_span
                y_last = nodes_top + u_last * row_span + self.like_text_height + self.like_height + self.like_geno_gap
                yc_last = y_last + self.cell_height / 2
                box_gap = self.col_gap * 0.6
                summary_w = self.cell_width * 3.8
                line_h = self.cell_height * 0.40
                pad_y = self.cell_height * 0.12
                summary_h = line_h * 3 + pad_y * 2
                summary_x = x_last + self.cell_width + box_gap
                summary_y = yc_last - summary_h / 2
                parts.append(
                    f'<line x1="{x_last + self.cell_width}" y1="{yc_last}" '
                    f'x2="{summary_x}" y2="{yc_last}" '
                    f'stroke="black" stroke-width="{self.path_stroke_width}" '
                    f'stroke-linecap="square" />'
                )
                parts.append(
                    f'<rect x="{summary_x}" y="{summary_y}" width="{summary_w}" height="{summary_h}" '
                    f'fill="white" stroke="black" stroke-width="{self.path_stroke_width}" />'
                )
                summary_rows = [
                    ("num_switches", str(self.hmm.num_switches)),
                    ("num_mismatches", str(self.hmm.num_mismatches)),
                    ("path_likelihood", self._fmt_sci(self.hmm.path_likelihood)),
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

    def show(self):
        display(HTML(self.visualise()))


        
