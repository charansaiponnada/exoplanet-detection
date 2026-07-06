"""
Generate polished infographic-style pipeline diagrams as draw.io (mxGraph) XML
and render them to high-resolution PNG with the draw.io desktop CLI.

Why draw.io instead of matplotlib: draw.io produces clean, presentation-grade
flowchart graphics (rounded nodes, proper arrow routing, consistent styling)
that read as a designed infographic rather than a plotted figure.

Usage (Windows, draw.io desktop installed):
    uv run python helper_scripts/visualize/drawio_diagrams.py
"""

import subprocess
from pathlib import Path

OUT_DIR = Path("output/diagrams")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DRAWIO_EXE = r"C:\Program Files\draw.io\draw.io.exe"

# --- Branded palette (fill, stroke) — tied to ISRO orange + template blue ---
BLUE = ("#DAE8FC", "#4C63E0")      # data / input
ORANGE = ("#FFE6CC", "#FF6600")    # processing
PURPLE = ("#E1D5E7", "#9673A6")    # features / ML
GREEN = ("#D5E8D4", "#82B366")     # output
AMBER = ("#FFF2CC", "#D6B656")     # callout / stat
NAVY_TEXT = "#16213E"
MUTED_TEXT = "#5A6472"


def xml_esc(s):
    """Escape a string so it is safe inside an XML double-quoted attribute.
    draw.io stores HTML labels this way: the raw '<br>' becomes '&lt;br&gt;',
    and draw.io un-escapes + renders it as HTML because the cell has html=1."""
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def html_esc(s):
    """Escape user text for HTML and turn newlines into <br> line breaks."""
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace("\n", "<br>"))


def titled(title, subtitle, sub_color=MUTED_TEXT, title_size=17, sub_size=13):
    """Build a raw HTML label: bold title over a smaller muted subtitle.
    Single-quoted HTML attributes; the whole thing gets xml_esc'd by the box."""
    return (
        f"<b style='font-size:{title_size}px'>{html_esc(title)}</b>"
        f"<br><br><span style='font-size:{sub_size}px;color:{sub_color}'>{html_esc(subtitle)}</span>"
    )


class Diagram:
    """Minimal mxGraph XML builder."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = []
        self._id = 10

    def _next(self):
        self._id += 1
        return f"n{self._id}"

    def box(self, x, y, w, h, text, fill, stroke, font_size=15, bold=True,
            rounded=True, font_color=NAVY_TEXT, align="center", valign="middle",
            shadow=True, raw_html=None):
        cid = self._next()
        style = (
            f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth=2;"
            f"fontSize={font_size};fontColor={font_color};"
            f"fontStyle={1 if bold else 0};align={align};verticalAlign={valign};"
            f"arcSize=12;shadow={1 if shadow else 0};spacingLeft=6;spacingRight=6;"
            f"spacingTop=6;spacingBottom=6;"
        )
        raw = raw_html if raw_html is not None else html_esc(text)
        self.cells.append(
            f'<mxCell id="{cid}" value="{xml_esc(raw)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return cid

    def label(self, x, y, w, h, text, font_size=13, bold=False,
              font_color=MUTED_TEXT, align="center"):
        cid = self._next()
        style = (
            f"text;html=1;strokeColor=none;fillColor=none;align={align};"
            f"verticalAlign=middle;whiteSpace=wrap;fontSize={font_size};"
            f"fontColor={font_color};fontStyle={1 if bold else 0};"
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{xml_esc(html_esc(text))}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return cid

    def badge(self, cx, cy, r, text, fill, stroke, font_color="#FFFFFF", font_size=14):
        cid = self._next()
        style = (
            f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"strokeWidth=2;fontSize={font_size};fontColor={font_color};fontStyle=1;"
            f"align=center;verticalAlign=middle;shadow=0;"
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{xml_esc(html_esc(text))}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{cx - r}" y="{cy - r}" width="{2 * r}" height="{2 * r}" as="geometry"/></mxCell>'
        )
        return cid

    def arrow(self, src, dst, color="#8A93A3"):
        cid = self._next()
        style = (
            f"edgeStyle=none;rounded=0;html=1;strokeColor={color};strokeWidth=2.5;"
            f"endArrow=block;endFill=1;endSize=8;"
        )
        self.cells.append(
            f'<mxCell id="{cid}" style="{style}" edge="1" parent="1" source="{src}" target="{dst}">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>'
        )
        return cid

    def arrow_xy(self, x1, y1, x2, y2, color="#8A93A3"):
        """Draw an arrow between absolute (x,y) coordinates using dummy vertex cells."""
        src_id = self._next()
        dst_id = self._next()
        cid = self._next()
        self.cells.append(
            f'<mxCell id="{src_id}" style="rounded=0;html=1;whiteSpace=wrap;" vertex="1" parent="1">'
            f'<mxGeometry x="{x1}" y="{y1}" width="1" height="1" as="geometry"/></mxCell>'
        )
        self.cells.append(
            f'<mxCell id="{dst_id}" style="rounded=0;html=1;whiteSpace=wrap;" vertex="1" parent="1">'
            f'<mxGeometry x="{x2}" y="{y2}" width="1" height="1" as="geometry"/></mxCell>'
        )
        style = (
            f"edgeStyle=none;rounded=0;html=1;strokeColor={color};strokeWidth=2.5;"
            f"endArrow=block;endFill=1;endSize=8;"
        )
        self.cells.append(
            f'<mxCell id="{cid}" style="{style}" edge="1" parent="1" source="{src_id}" target="{dst_id}">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>'
        )
        return cid

    def to_xml(self):
        root = (
            '<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(self.cells)
        )
        model = (
            f'<mxGraphModel dx="900" dy="600" grid="0" gridSize="10" guides="1" '
            f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{self.width}" pageHeight="{self.height}" math="0" shadow="0">'
            f'<root>{root}</root></mxGraphModel>'
        )
        return f'<mxfile><diagram id="d1" name="d1">{model}</diagram></mxfile>'


def render(diagram, name):
    xml_path = OUT_DIR / f"{name}.drawio"
    png_path = OUT_DIR / f"{name}.png"
    xml_path.write_text(diagram.to_xml(), encoding="utf-8")
    subprocess.run(
        [DRAWIO_EXE, "--export", "--format", "png", "--scale", "2.5",
         "--border", "8", "--output", str(png_path), str(xml_path)],
        check=True, capture_output=True,
    )
    print(f"  [OK] {png_path}")


# ======================================================================
# 1. Pipeline flow — 7 stages, horizontal, number badges + arrows
# ======================================================================
def pipeline_flow():
    W, H = 1160, 430
    d = Diagram(W, H)
    stages = [
        ("TESS Data", "MAST archive\nFITS light curves", BLUE),
        ("Preprocess", "Sigma-clip 5σ\nSavitzky-Golay", ORANGE),
        ("BLS Search", "5000 trial\nperiods", ORANGE),
        ("Features", "17-dim\nfeature vector", PURPLE),
        ("Classify", "Vetting +\nRandomForest", PURPLE),
        ("Validate", "Plausibility +\nconfidence", GREEN),
        ("Dashboard", "Streamlit +\nexport", GREEN),
    ]
    n = len(stages)
    margin = 30
    gap = 24
    bw = (W - 2 * margin - (n - 1) * gap) / n
    bh = 150
    y = 150
    boxes = []
    for i, (title, sub, (fill, stroke)) in enumerate(stages):
        x = margin + i * (bw + gap)
        b = d.box(x, y, bw, bh, "", fill, stroke, raw_html=titled(title, sub))
        d.badge(x + bw / 2, y, 20, str(i + 1), stroke, stroke, font_size=15)
        boxes.append(b)
    for i in range(n - 1):
        d.arrow(boxes[i], boxes[i + 1])
    d.label(0, 40, W, 40, "End-to-End Detection Pipeline", font_size=24,
            bold=True, font_color=NAVY_TEXT)
    d.label(0, 82, W, 26,
            "Raw flux  →  confidence-scored report  ·  every stage inspectable, no black box",
            font_size=15, font_color=MUTED_TEXT)
    render(d, "flow_pipeline")


# ======================================================================
# 2. Interim classifier architecture (running now) + Phase-2 note
# ======================================================================
def architecture():
    W, H = 1160, 480
    d = Diagram(W, H)
    stages = [
        ("Synthetic Data", "300 labelled\nexamples · 4 classes", BLUE),
        ("Layer 4 Features", "17-dim vector\nper candidate", PURPLE),
        ("Random Forest", "300 trees\ndepth 6 · balanced", ORANGE),
        ("Output", "Class probabilities\n+ top features", GREEN),
    ]
    n = len(stages)
    margin = 40
    gap = 55
    bw = (W - 2 * margin - (n - 1) * gap) / n
    bh = 150
    y = 130
    boxes = []
    for i, (title, sub, (fill, stroke)) in enumerate(stages):
        x = margin + i * (bw + gap)
        b = d.box(x, y, bw, bh, "", fill, stroke, raw_html=titled(title, sub))
        boxes.append(b)
    for i in range(n - 1):
        d.arrow(boxes[i], boxes[i + 1])

    d.label(0, 34, W, 40, "Layer 5 Today — Interim Classifier (running now)",
            font_size=23, bold=True, font_color=NAVY_TEXT)

    # Two stat callouts underneath
    sy = 320
    sw = (W - 2 * margin - 30) / 2
    d.box(margin, sy, sw, 110, "", *AMBER,
          raw_html=titled("71% held-out test accuracy",
                          "4 classes · 300 synthetic examples · eclipsing-binary F1 = 0.91",
                          sub_color="#7A5C00"))
    d.box(margin + sw + 30, sy, sw, 110, "", *AMBER,
          raw_html=titled("Top contributing features",
                          "red-noise ratio · transit depth · depth SNR · kurtosis · duration",
                          sub_color="#7A5C00"))
    render(d, "arch_classifier")


# ======================================================================
# 3. Tech stack — 5 grouped cards with BUILT / PLANNED badges
# ======================================================================
def tech_stack():
    W, H = 1160, 470
    d = Diagram(W, H)
    cols = [
        ("Data Access", "lightkurve\nastropy (FITS)\nMAST · astroquery", BLUE, "BUILT", GREEN),
        ("Processing + ML", "NumPy · SciPy\npandas\nscikit-learn", ORANGE, "BUILT", GREEN),
        ("Dashboard", "Streamlit\nmatplotlib", PURPLE, "BUILT", GREEN),
        ("Deep Learning", "PyTorch\nMamba SSM\nHuggingFace", ORANGE, "PLANNED", AMBER),
        ("Infrastructure", "uv\nPython 3.12+\nGit", BLUE, "BUILT", GREEN),
    ]
    n = len(cols)
    margin = 34
    gap = 22
    bw = (W - 2 * margin - (n - 1) * gap) / n
    bh = 250
    y = 130
    for i, (title, items, (fill, stroke), badge_txt, (bfill, bstroke)) in enumerate(cols):
        x = margin + i * (bw + gap)
        card_html = (
            f"<b style='font-size:17px'>{html_esc(title)}</b><br><br><br>"
            f"<span style='font-size:14px;color:{MUTED_TEXT}'>{html_esc(items)}</span>"
        )
        d.box(x, y, bw, bh, "", fill, stroke, valign="top", raw_html=card_html)
        d.box(x + bw / 2 - 48, y - 18, 96, 30, badge_txt, bfill, bstroke,
              font_size=12, rounded=True, shadow=False, font_color=NAVY_TEXT)
    d.label(0, 40, W, 40, "Technology Stack", font_size=24, bold=True, font_color=NAVY_TEXT)
    d.label(0, 82, W, 26, "100% open-source  ·  green = built & running today  ·  amber = planned for the 30-hour finale",
            font_size=14, font_color=MUTED_TEXT)
    render(d, "tech_stack")


# ======================================================================
# 4. Dashboard mockup (browser frame) — slide 6
# ======================================================================
def dashboard_mockup():
    W, H = 1160, 560
    d = Diagram(W, H)
    d.label(0, 24, W, 36, "Interactive Dashboard  ·  app.py (Streamlit)", font_size=22,
            bold=True, font_color=NAVY_TEXT)

    # browser window frame
    d.box(30, 80, W - 60, H - 110, "", "#FFFFFF", "#B8C0CC", font_size=1, rounded=True, shadow=True)
    d.box(30, 80, W - 60, 34, "", "#EEF1F5", "#B8C0CC", rounded=True, shadow=False)
    for i, c in enumerate(["#FF6058", "#FEBC2E", "#28C840"]):
        d.badge(56 + i * 22, 97, 7, "", c, c)

    # sidebar
    sx, sy = 50, 128
    d.box(sx, sy, 250, H - 170, "Input\n", "#F0F2F6", "#D6DBE3", font_size=15, valign="top")
    radio = ["Synthetic: planet", "Synthetic: eclipsing binary",
             "Local TESS file", "Upload CSV", "Real TIC ID (MAST)"]
    for i, r in enumerate(radio):
        ry = sy + 50 + i * 46
        d.badge(sx + 24, ry + 10, 8, "", "#FF6600" if i == 0 else "#FFFFFF", "#9AA3B0")
        d.label(sx + 40, ry, 210, 22, r, font_size=13, font_color=NAVY_TEXT, align="left")
    d.box(sx + 20, sy + H - 240, 210, 42, "Run pipeline", *ORANGE, font_size=15, font_color=NAVY_TEXT)

    # main metrics row
    mx = 320
    mw = (W - mx - 60 - 3 * 14) / 4
    metrics = [
        ("Classification", "candidate_\nplanetary_transit"),
        ("Confidence", "0.98\nhigh confidence"),
        ("Detection SNR", "37.0"),
        ("Plausible?", "Yes"),
    ]
    for i, (k, v) in enumerate(metrics):
        x = mx + i * (mw + 14)
        d.box(x, 132, mw, 92, f"{k}\n\n{v}", "#F8FAFC", "#D6DBE3", font_size=12,
              bold=False, valign="top")

    # plot placeholder
    d.box(mx, 238, W - mx - 60, 200,
          "Raw · Detrended · BLS Periodogram · Phase-Fold\n(matplotlib figure — shared with the CLI)",
          "#F4F7FB", "#C7D0DC", font_size=15, bold=False, font_color=MUTED_TEXT)

    # tabs
    tabs = ["Parameters", "Vetting", "ML classifier", "Features", "Export"]
    tw = (W - mx - 60 - 4 * 8) / 5
    for i, t in enumerate(tabs):
        x = mx + i * (tw + 8)
        active = i == 2
        d.box(x, 452, tw, 40, t, ("#E1D5E7" if active else "#FFFFFF"),
              ("#9673A6" if active else "#D6DBE3"), font_size=12,
              bold=active, shadow=False)
    render(d, "dashboard_mockup")


# ======================================================================
# 5. Full Architecture Diagram (7-layer vertical, with icons) — 
#    Matches the style of pipeline.png but in draw.io
# ======================================================================
def architecture_detailed():
    W, H = 1300, 820
    d = Diagram(W, H)

    # Title
    d.label(W / 2 - 300, 16, 600, 42,
            "🏗️  AI-Enabled Exoplanet Transit Detection — System Architecture",
            font_size=22, bold=True, font_color=NAVY_TEXT)
    d.label(W / 2 - 300, 56, 600, 24,
            "7-layer pipeline · every stage inspectable · no black box",
            font_size=14, font_color=MUTED_TEXT)

    # ---- Layer definitions ----
    # Each: (title, icon, items, fill, stroke)
    layers = [
        ("LAYER 1  DATA INGESTION", "📡",
         ["TESS FITS  (lightkurve / MAST)",
          "Curated ISRO Dataset  (labels)",
          "Synthetic Generator  (src/data_io.py)"],
         BLUE),
        ("LAYER 2  PREPROCESSING", "🧹",
         ["Sigma Clip  (5σ outlier removal)",
          "Savitzky-Golay Detrend  (0.5-day)",
          "Flux Normalization  →  relative flux"],
         ORANGE),
        ("LAYER 3  SIGNAL DETECTION", "🔍",
         ["BLS Periodogram  (5000 periods, 0.5–15 d)",
          "Peak Selection  (max SNR above noise floor)",
          "Phase Folding  (stack all transits)"],
         ORANGE),
        ("LAYER 4  FEATURE ENGINEERING", "📊",
         ["Transit Shape  (ingress/egress slope, V/U ratio)",
          "Noise Stats  (skew, kurtosis, entropy, red-noise)",
          "Depth Consistency  (per-transit spread, variation)"],
         PURPLE),
        ("LAYER 5  CLASSIFICATION ENGINE", "🤖",
         ["Classical Vetting  (odd-even, secondary, shape)",
          "Random Forest  (300 trees, 4 classes, running now)",
          "Mamba SSM  (bidirectional, 6 classes, Phase 2)"],
         PURPLE),
        ("LAYER 6  SCIENTIFIC VALIDATION", "✅",
         ["Physical Plausibility  (duty cycle, depth, transit count)",
          "Confidence Score  (0–1 weighted: SNR + consistency)",
          "Verdict  (high_confidence / human_review / false_positive)"],
         GREEN),
        ("LAYER 7  OUTPUT & PRESENTATION", "📋",
         ["JSON Report  (parameters, classification, confidence)",
          "PNG Figure  (2×2: raw, detrended, BLS, folded)",
          "Streamlit Dashboard  (metric cards, 5 tabs, export)"],
         GREEN),
    ]

    layer_h = 78
    label_w = 200
    items_w = 940
    gap = 12
    margin_x = 60
    margin_y = 90

    start_y = margin_y
    prev_box_id = None

    for i, (title, icon, items, (fill, stroke)) in enumerate(layers):
        y = start_y + i * (layer_h + gap)

        # --- Layer label box (left) ---
        label_html = (
            f"<span style='font-size:28px'>{html_esc(icon)}</span>"
            f"<br><b style='font-size:11px'>{html_esc(title)}</b>"
        )
        label_box = d.box(margin_x, y, label_w, layer_h, "",
                          fill, stroke, valign="center",
                          raw_html=label_html, font_color="#FFFFFF")

        # --- Items box (right) ---
        item_text = "  │  ".join(items)
        item_html = (
            f"<span style='font-size:18px'>{html_esc(icon)}</span>"
            f"&nbsp;&nbsp;<b style='font-size:14px'>{html_esc(items[0])}</b>"
            f"<br><span style='font-size:12px;color:{MUTED_TEXT}'>"
            f"&nbsp;&nbsp;&nbsp;{html_esc(items[1])}</span>"
            f"<br><span style='font-size:12px;color:{MUTED_TEXT}'>"
            f"&nbsp;&nbsp;&nbsp;{html_esc(items[2])}</span>"
        )
        item_box = d.box(margin_x + label_w + 14, y, items_w, layer_h, "",
                         "#FFFFFF", stroke, align="left", valign="center",
                         raw_html=item_html, font_color=NAVY_TEXT)

        # --- Arrow from previous layer ---
        if prev_box_id is not None:
            d.arrow(prev_box_id, label_box, color=stroke)

        prev_box_id = label_box

    # --- Legend / Status badge row at bottom ---
    legend_y = start_y + len(layers) * (layer_h + gap) + 16
    d.box(margin_x, legend_y, label_w + items_w + 14, 50,
          "✅  All 7 layers built & tested  ·  Live: Layers 1–7  ·  "
          "ML Enhancement (Phase 2): Mamba SSM on ISRO curated data planned for 30-hr finale",
          *AMBER, font_size=12, bold=False, font_color="#7A5C00")

    render(d, "architecture_detailed")


# ======================================================================
# 6. Horizontal Pipeline — 2 rows × 4 stages, each with output sub-box
#    16:9 ratio, BAH template style
# ======================================================================
def pipeline_horizontal_detailed():
    W, H = 1600, 900
    d = Diagram(W, H)

    # ---------- Title ----------
    d.label(W / 2 - 350, 14, 700, 40,
            "🏗️  AI-Enabled Exoplanet Detection Pipeline",
            font_size=24, bold=True, font_color=NAVY_TEXT)
    d.label(W / 2 - 350, 50, 700, 22,
            "TESS Light Curve  →  Confidence-Scored Report  ·  Every Stage Inspectable",
            font_size=13, font_color=MUTED_TEXT)

    # ---------- Stage definitions ----------
    # (icon, title, bullets, output_text, color)
    stages = [
        # Row 1
        ("📡", "Data Ingestion",
         ["MAST Archive (lightkurve)", "Curated ISRO Dataset", "Synthetic Generator"],
         "DataFrame(time, flux, err)", BLUE),
        ("🧹", "Preprocessing",
         ["Sigma Clip (5σ outliers)", "S-G Detrend (0.5d)", "Flux Normalization"],
         "Detrended Flux", ORANGE),
        ("🔍", "BLS Period Search",
         ["Box Least Squares", "5000 periods × 20 durations", "Peak SNR selection"],
         "{period, depth, t0, SNR}", ORANGE),
        ("⏱", "Phase Folding",
         ["Fold on best period", "Stack all transits", "30-bin binned curve"],
         "Phase-Folded LC", ORANGE),
        # Row 2
        ("📊", "Feature Engineering",
         ["Transit shape (slope, V/U)", "Noise stats (skew, entropy)", "Depth consistency"],
         "17-dim Feature Vector", PURPLE),
        ("🤖", "Classification",
         ["Odd-even + secondary + shape", "Random Forest (4 classes)", "Mamba SSM (Phase 2)"],
         "Label + Flags", PURPLE),
        ("✅", "Validation & Scoring",
         ["Physical plausibility checks", "Weighted 0-1 score", "Verdict assignment"],
         "Confidence {score, verdict}", GREEN),
        ("📋", "Output & Presentation",
         ["JSON results report", "2×2 PNG diagnostic figure", "Streamlit dashboard"],
         "JSON + PNG + Dashboard", GREEN),
    ]

    # Layout
    margin_x = 30
    gap_x = 20
    main_h = 155
    sub_h = 48
    gap_main_sub = 10
    row_gap = 60

    sw = (W - 2 * margin_x - 3 * gap_x) / 4  # stage width

    # Row Y positions
    title_row1_y = 100
    sub_row1_y = title_row1_y + main_h + gap_main_sub
    title_row2_y = title_row1_y + main_h + sub_h + gap_main_sub + row_gap
    sub_row2_y = title_row2_y + main_h + gap_main_sub

    row_starts = [title_row1_y, title_row2_y]
    sub_starts = [sub_row1_y, sub_row2_y]

    all_main_boxes = []
    all_sub_boxes = []

    # Pass 1: draw all boxes
    for idx, (icon, title, bullets, output, (fill, stroke)) in enumerate(stages):
        row = 0 if idx < 4 else 1
        col = idx % 4
        x = margin_x + col * (sw + gap_x)
        y = row_starts[row]
        sub_y = sub_starts[row]

        bullet_html = "<br>".join([f"  • {html_esc(b)}" for b in bullets])
        main_html = (
            f"<span style='font-size:26px'>{html_esc(icon)}</span>"
            f"&nbsp;<b style='font-size:15px'>{html_esc(title)}</b>"
            f"<br><span style='font-size:11px;color:{MUTED_TEXT}'>{bullet_html}</span>"
        )
        main_box = d.box(x, y, sw, main_h, "", "#FFFFFF", stroke,
                          align="left", valign="top", raw_html=main_html,
                          font_color=NAVY_TEXT)
        all_main_boxes.append(main_box)

        sub_html = (
            f"<span style='font-size:13px;color:{stroke}'><b>→ {html_esc(output)}</b></span>"
        )
        sub_box = d.box(x, sub_y, sw, sub_h, "", "#F8FAFC", stroke,
                         align="center", valign="middle", raw_html=sub_html,
                         font_color=stroke)
        all_sub_boxes.append(sub_box)

    # Pass 2: draw all arrows
    for idx, (icon, title, bullets, output, (fill, stroke)) in enumerate(stages):
        row = 0 if idx < 4 else 1
        col = idx % 4

        # Vertical: main box → sub box (same stage)
        d.arrow(all_main_boxes[idx], all_sub_boxes[idx])

        # Horizontal: stage → next stage (within row)
        if col < 3:
            arrow_id = d._next()
            style = (
                f"edgeStyle=none;rounded=0;html=1;strokeColor={stroke};strokeWidth=2.5;"
                f"endArrow=block;endFill=1;endSize=8;"
            )
            d.cells.append(
                f'<mxCell id="{arrow_id}" style="{style}" edge="1" parent="1" '
                f'source="{all_main_boxes[idx]}" target="{all_main_boxes[idx + 1]}">'
                f'<mxGeometry relative="1" as="geometry"/></mxCell>'
            )

    # Arrow from row 1, stage 4 output → row 2, stage 1
    last_row1_sub = all_sub_boxes[3]
    first_row2_main = all_main_boxes[4]
    arrow_id = d._next()
    style = (
        f"edgeStyle=elbow;elbow=vertical;rounded=0;html=1;strokeColor={ORANGE[1]};strokeWidth=2.5;"
        f"endArrow=block;endFill=1;endSize=8;dashPattern=8 4;"
    )
    d.cells.append(
        f'<mxCell id="{arrow_id}" style="{style}" edge="1" parent="1" '
        f'source="{last_row1_sub}" target="{first_row2_main}">'
        f'<mxGeometry relative="1" as="geometry">'
        f'<Array as="points"><mxPoint x="{margin_x + 3.5 * sw + 3 * gap_x - (sw + gap_x)/2}" y="{row_starts[1]}"/></Array>'
        f'</mxGeometry></mxCell>'
    )

    # ---------- Legend / Footer ----------
    footer_y = sub_row2_y + sub_h + 40
    footer_text = (
        "▪  Built & validated 7-layer pipeline  ▪  "
        "Layer 5: rule-based + Random Forest (live)  ▪  "
        "Layer 5 enhancement: Mamba SSM on ISRO curated data (Phase 2)"
    )
    d.box(margin_x, footer_y, W - 2 * margin_x, 42, "",
          *AMBER, raw_html=titled("", footer_text, sub_color="#7A5C00"),
          font_size=12, bold=False)

    render(d, "pipeline_horizontal_detailed")


# ======================================================================
# 7. Deep Learning Model Architecture — Mamba SSM (Phase 2)
#     16:9, BAH template style
# ======================================================================
def model_architecture_dl():
    W, H = 1600, 900
    d = Diagram(W, H)

    d.label(W / 2 - 350, 14, 700, 38,
            "🧠  Deep Learning Architecture — Mamba SSM Encoder (Planned)",
            font_size=22, bold=True, font_color=NAVY_TEXT)
    d.label(W / 2 - 350, 48, 700, 20,
            "Phase 2  ·  pending ISRO curated dataset  ·  replaces Random Forest in Layer 5",
            font_size=12, font_color=MUTED_TEXT)

    # ---- Top badge: status ----
    d.box(W / 2 - 100, 74, 200, 28, "PLANNED — 30-hour Finale",
          *AMBER, font_size=11, font_color="#7A5C00", bold=True, rounded=True)

    # ---- Row 1: Input pipeline ----
    row1_y = 120
    bw1, bh1 = 280, 100

    def lbl(title, body):
        return (f"<b style='font-size:14px'>{html_esc(title)}</b>"
                f"<br><span style='font-size:11px;color:{MUTED_TEXT}'>{html_esc(body)}</span>")

    # Input
    in_box = d.box(50, row1_y, bw1, bh1, "",
                   BLUE[0], BLUE[1], raw_html=lbl("📥 Input", "Phase-folded light curve\n(time, flux, flux_err)\nBLS periodogram features"),
                   align="left", valign="top")

    # Feature extraction
    fe_box = d.box(50 + bw1 + 30, row1_y, bw1, bh1, "",
                   "#E8F4FD", "#4A90D9", raw_html=lbl("📊 Feature Extraction", "Transit shape metrics\nNoise statistics\nDepth/duration/SNR\nPeriod aliases"),
                   align="left", valign="top")

    # Normalization
    norm_box = d.box(50 + 2 * (bw1 + 30), row1_y, bw1, bh1, "",
                     "#FFF0E0", "#FF6600", raw_html=lbl("⚙ Normalization", "LayerNorm over sequence\nMin-max scaling\nZ-score standardization\nMasked padding"),
                     align="left", valign="top")

    # Sequence preparation
    seq_box = d.box(50 + 3 * (bw1 + 30), row1_y, bw1, bh1, "",
                    "#F0E6FF", "#9673A6", raw_html=lbl("🔗 Sequence Prep", "Fixed-length windows\nSliding window stride=1\nSequence length: 2048\nBatch size: 64"),
                    align="left", valign="top")

    # Horizontal arrows row 1
    for src, dst in [(in_box, fe_box), (fe_box, norm_box), (norm_box, seq_box)]:
        d.arrow(src, dst)

    # ---- Row 2: Mamba SSM Encoder ----
    row2_y = 280

    # Large bounding box for Mamba block
    mamba_x = 100
    mamba_w = W - 200
    mamba_h = 220

    d.box(mamba_x, row2_y, mamba_w, mamba_h, "",
          "#F5F0FF", "#9673A6", raw_html="", bold=False, font_size=1)

    d.label(mamba_x + 20, row2_y + 6, 400, 24,
            "🔄  Bidirectional Mamba SSM Encoder", font_size=15, bold=True, font_color=NAVY_TEXT, align="left")

    # Sub-blocks inside Mamba
    mb_y = row2_y + 40
    mb_w = (mamba_w - 4 * 20) / 4
    mb_h = 120

    mamba_blocks = [
        ("Forward SSM\nLayer 1", "Selective scan\nHidden dim: 512\nState dim: 16\nActivation: SiLU"),
        ("Forward SSM\nLayer 2", "Selective scan\nHidden dim: 512\nState dim: 16\nResidual connect"),
        ("Backward SSM\nLayer 1", "Reverse sequence\nHidden dim: 512\nState dim: 16\nActivation: SiLU"),
        ("Backward SSM\nLayer 2", "Reverse sequence\nHidden dim: 512\nState dim: 16\nResidual connect"),
    ]

    mamba_boxes = []
    for i, (title, body) in enumerate(mamba_blocks):
        x = mamba_x + 20 + i * (mb_w + 20)
        b = d.box(x, mb_y, mb_w, mb_h, "",
                  "#FFFFFF", "#9673A6", raw_html=lbl(title, body),
                  align="left", valign="top")
        mamba_boxes.append(b)
        if i > 0:
            d.arrow(mamba_boxes[i - 1], b)

    # Concatenation box
    concat_x = mamba_x + 20
    concat_y = mb_y + mb_h + 14
    concat_w = mamba_w - 40
    concat_h = 36
    d.box(concat_x, concat_y, concat_w, concat_h, "",
          "#E8FDE0", "#82B366", raw_html=lbl("", "Concatenate forward + backward outputs  →  Sequence representation (batch, seq_len, 1024)"),
          align="center", valign="center", font_size=12, bold=False)

    # ---- Row 3: Classification Head ----
    row3_y = 550
    head_w = 280
    head_h = 110
    gap_h = 20

    mlp_boxes = []
    mlp_layers = [
        ("MLP Layer 1", "Linear 1024 → 512\nReLU + Dropout 0.3"),
        ("MLP Layer 2", "Linear 512 → 256\nReLU + Dropout 0.3"),
        ("MLP Layer 3", "Linear 256 → 128\nReLU + Dropout 0.3"),
    ]
    head_start_x = (W - (3 * head_w + 2 * gap_h)) / 2

    for i, (title, body) in enumerate(mlp_layers):
        x = head_start_x + i * (head_w + gap_h)
        b = d.box(x, row3_y, head_w, head_h, "",
                  "#F0E6FF", "#9673A6", raw_html=lbl(title, body),
                  align="left", valign="top")
        mlp_boxes.append(b)
        if i > 0:
            d.arrow(mlp_boxes[i - 1], b)

    # Output
    out_x = head_start_x + 3 * (head_w + gap_h)
    out_box = d.box(out_x, row3_y, head_w, head_h, "",
                    GREEN[0], GREEN[1], raw_html=lbl("🎯 Output (Softmax)", "6 classes:\nplanet_transit / EB / blend\nstarspot / noise / unknown\nConfidence calibration"),
                    align="left", valign="top")
    d.arrow(mlp_boxes[-1], out_box)

    # ---- Training details row ----
    row4_y = 700
    card_w = (W - 120) / 3
    card_h = 130

    cards = [
        ("📋 Training Config",
         "Loss: Cross-entropy\nOptimizer: AdamW (lr=1e-4)\nBatch size: 64\nEpochs: 100 (early stop@10)\nGradient clipping: 1.0",
         "#FFF8E0", "#D6B656"),
        ("📈 Data Augmentation",
         "Noise injection (Gaussian)\nTransit depth scaling\nPeriod jitter / phase shift\nMissing data masking\nClass balancing (weighted sampler)",
         "#FFF8E0", "#D6B656"),
        ("📊 Evaluation",
         "Accuracy, Precision, Recall, F1\nConfusion matrix\nROC-AUC per class\nCalibration plot (ECE)\nk-fold cross-validation (5)",
         "#FFF8E0", "#D6B656"),
    ]

    for i, (title, body, bg, stroke) in enumerate(cards):
        x = 60 + i * (card_w + 20)
        d.box(x, row4_y, card_w, card_h, "",
              bg, stroke, raw_html=lbl(title, body),
              align="left", valign="top")

    # ---- Arrow from Sequence Prep to Mamba ----
    d.arrow(seq_box, mamba_boxes[0])

    # ---- Arrow from Mamba concat area to MLP ----
    mid_x = concat_x + concat_w / 2
    d.arrow_xy(concat_x + concat_w - 10, concat_y + concat_h,
               head_start_x + head_w / 2, row3_y, color="#9673A6")

    # ---- Current classifier comparison card ----
    d.box(W - 300, 74, 250, 28, "Current: Random Forest (live)",
          *GREEN, font_size=11, font_color="#0B8457", bold=True, rounded=True)

    render(d, "model_architecture_dl")


def main():
    print("Rendering draw.io diagrams...\n")
    pipeline_flow()
    architecture()
    tech_stack()
    dashboard_mockup()
    architecture_detailed()
    pipeline_horizontal_detailed()
    model_architecture_dl()
    print(f"\nAll diagrams saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
