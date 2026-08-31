"""Render subset of draw.io XML to black-and-white, publication-grade SVG.

Follows the visual system in .opencode/skills/drawio-skill:
white fill, black 2px borders, square corners, large text, minimal decoration.

Supported drawio constructs (authored deliberately to stay in this subset):
  * vertices: rectangle (square or rounded), ellipse, rhombus, cylinder3
  * styles: strokeWidth, fillColor, strokeColor, fontSize, fontFamily,
    fontStyle (1=bold, 2=italic), align, verticalAlign, spacingLeft,
    rounded, dashed, whiteSpace=wrap, html=1 with <br>/<b>/<i>
  * edges without source/target using an explicit <Array as="points">
    of absolute mxPoint waypoints; endArrow/startArrow=block, dashed.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import xml.etree.ElementTree as ET

_HTML5 = {k.strip("&;"): v for k, v in html.entities.html5.items() if len(v) == 1}


def xml_safe(raw: str) -> str:
    """Replace common HTML entities (&mdash; etc.) that are illegal in XML."""
    def repl(m):
        ent = m.group(1)
        if ent in _HTML5:
            return _HTML5[ent]
        return m.group(0)
    return re.sub(r"&([a-zA-Z][a-zA-Z0-9]*);", repl, raw)

STYLE_DEFAULTS = dict(
    strokeWidth=2,
    fillColor="#ffffff",
    strokeColor="#000000",
    fontColor="#000000",
    fontSize=16,
    fontStyle=0,
    align="center",
    verticalAlign="middle",
    spacingLeft=0,
    fontFamily="sans-serif",
)

CX = "http://r.jina.ai/http://www.w3.org/1999/xhtml"


def parse_style(style: str) -> dict:
    s = dict(STYLE_DEFAULTS)
    if not style:
        return s
    for tok in style.split(";"):
        if not tok:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            s[k] = v
    conv = int
    for k in ("strokeWidth", "fontSize", "fontStyle", "spacingLeft", "arcSize"):
        if k in s:
            try:
                s[k] = conv(s[k])
            except (TypeError, ValueError):
                s[k] = STYLE_DEFAULTS[k]
    if s.get("rounded") == "0":
        s["rounded"] = False
    elif s.get("rounded") == "1":
        s["rounded"] = True
    else:
        s["rounded"] = False
    return s


TAG_RE = re.compile(r"<br\s*/?>|<b>|</b>|<i>|</i>|<[^>]+>", re.IGNORECASE)


def plain_lines(value: str) -> list[tuple[str, int]]:
    """Split a drawio label into (text, style_flag) lines. style flag: 0 normal, 1 bold."""
    if value is None:
        return []
    value = html.unescape(value)
    out: list[tuple[str, int]] = []
    if "<" not in value or ">" not in value:
        for line in value.split("\n"):
            out.append((line, 0))
        return out
    # Mini HTML: track <b> state, split on <br>.
    bold = False
    buf = ""
    for part in TAG_RE.split(value):
        pass
    # Rebuild via tokens to handle nested <b> tags.
    tokens = re.split(r"(<br\s*/?>)", value, flags=re.IGNORECASE)
    tok_re = re.compile(r"<b>|</b>|<i>|</i>|</?[^>]*>", re.IGNORECASE)
    for line_seg in re.split(r"<br\s*/?>", value, flags=re.IGNORECASE):
        bold = False
        buf = ""
        for tok in tok_re.split(line_seg):
            t = tok.strip()
            if t.lower() == "<b>":
                bold = True
            elif t.lower() == "</b>":
                bold = False
            elif t == "" or tok_re.fullmatch(t):
                continue
            else:
                buf += tok
        out.append((buf, 1 if bold else 0))
    return out


def esc(t: str) -> str:
    return html.escape(t, quote=True)


def svg_for_diagram(diagram: ET.Element) -> str:
    cells = {c.get("id"): c for c in diagram.iter("mxCell")}
    vertices, edges = [], []
    for c in cells.values():
        if c.get("vertex") == "1":
            vertices.append(c)
        elif c.get("edge") == "1":
            edges.append(c)

    minx = min(float(c.find("mxGeometry").get("x", 0)) for c in vertices)
    miny = min(float(c.find("mxGeometry").get("y", 0)) for c in vertices)
    maxx = max(
        float(c.find("mxGeometry").get("x", 0)) + float(c.find("mxGeometry").get("width", 0))
        for c in vertices
    )
    maxy = max(
        float(c.find("mxGeometry").get("y", 0)) + float(c.find("mxGeometry").get("height", 0))
        for c in vertices
    )
    for e in edges:
        geo = e.find("mxGeometry")
        if geo is None:
            continue
        pts = geo.find("Array")
        if pts is not None:
            xs = [float(p.get("x")) for p in pts.iter("mxPoint")]
            ys = [float(p.get("y")) for p in pts.iter("mxPoint")]
            if xs:
                minx, maxx = min(minx, min(xs)), max(maxx, max(xs))
            if ys:
                miny, maxy = min(miny, min(ys)), max(maxy, max(ys))

    pad = 16
    W, H = int(maxx - minx + 2 * pad), int(maxy - miny + 2 * pad)
    P = []
    for e in edges:
        geo = e.find("mxGeometry")
        if geo is None:
            continue
        pts = geo.find("Array")
        if pts is None:
            continue
        coords = [(float(p.get("x")), float(p.get("y"))) for p in pts.iter("mxPoint")]
        P.append((e, coords))

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    parts.append(
        "<style>text{font-family:DejaVu Sans,Arial,Helvetica,sans-serif;}"
        "line,path,polyline,rect,ellipse{shape-rendering:geometricPrecision;}</style>"
    )

    def xf(x: float) -> float:
        return x - minx + pad

    def yf(y: float) -> float:
        return y - miny + pad

    # ---- vertices ----
    for c in vertices:
        geo = c.find("mxGeometry")
        if geo is None:
            continue
        st = parse_style(c.get("style", ""))
        x, y = xf(float(geo.get("x", 0))), yf(float(geo.get("y", 0)))
        width, height = float(geo.get("width", 0)), float(geo.get("height", 0))
        sw = st["strokeWidth"]
        fill = st["fillColor"]
        if fill not in ("none", "None"):
            fill = fill or "#ffffff"
        stroke = st.get("strokeColor", "#000000")
        shape = st.get("shape", "rectangle").lower()
        label = c.get("value", "")
        lines = plain_lines(label)
        body = ""
        if shape == "ellipse":
            body = (
                f'<ellipse cx="{x+width/2:.1f}" cy="{y+height/2:.1f}" '
                f'rx="{width/2:.1f}" ry="{height/2:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />'
            )
        elif shape in ("rhombus", "triangle"):
            path = (
                f"M{x+width/2:.1f},{y}L{x+width:.1f},{y+height/2:.1f}"
                f"L{x+width/2:.1f},{y+height:.1f}L{x},{y+height/2:.1f}Z"
            )
            body = f'<path d="{path}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" fill-rule="evenodd" />'
        elif shape == "cylinder3":
            top = y + height * 0.18
            body = (
                f'<path d="M{x},{top} L{x},{y+height} A{width/2:.1f},{height*0.18:.1f} 0 0 0 {x+width},{y+height} '
                f'L{x+width},{top} A{width/2:.1f},{height*0.18:.1f} 0 0 1 {x},{top} Z" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />'
            )
            body += (
                f'<ellipse cx="{x+width/2:.1f}" cy="{top:.1f}" rx="{width/2:.1f}" '
                f'ry="{height*0.18:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />'
            )
        else:
            rx = height * 0.08 if st["rounded"] else 0
            r = f' rx="{rx:.1f}"' if rx > 0 else ""
            body = (
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{r} />'
            )
        parts.append(body)

        # text
        if lines:
            fs = st["fontSize"]
            fw = 800 if any(flag == 1 for _, flag in lines) else 400
            anchor = "middle"
            tx = x + width / 2
            if st["align"] == "left":
                anchor = "start"
                tx = x + 6 + st["spacingLeft"]
            elif st["align"] == "right":
                anchor = "end"
                tx = x + width - 6 - st["spacingLeft"]
            line_h = fs * 1.28
            n = len(lines)
            if st["verticalAlign"] == "top":
                ty0 = y + 4 + fs
            elif st["verticalAlign"] == "bottom":
                ty0 = y + height - 4 - line_h * (n - 1) - fs * 0.4
            else:
                ty0 = y + height / 2 - line_h * (n - 1) / 2
            for i, (txt, flag) in enumerate(lines):
                if not txt:
                    continue
                fwb = 800 if flag else fw
                parts.append(
                    f'<text x="{tx:.1f}" y="{ty0 + i*line_h:.1f}" font-size="{fs}" '
                    f'font-weight="{fwb}" fill="{st["fontColor"]}" '
                    f'text-anchor="{anchor}">{esc(txt)}</text>'
                )

    # ---- edges ----
    for e, coords in P:
        st = parse_style(e.get("style", ""))
        swap = {e: g for g, (edge, _) in enumerate(P)}
        sw = st["strokeWidth"]
        stroke = st.get("strokeColor", "#000000")
        dash = f' stroke-dasharray="{st.get("dashPattern", "4 3")}"' if st.get("dashed") == "1" else ""
        poly = " ".join(f"{xf(x):.1f},{yf(y):.1f}" for x, y in coords)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{stroke}" stroke-width="{sw}"{dash} />')

        def arrow(end: bool):
            a, b = (coords[-2], coords[-1]) if end else (coords[1], coords[0])
            dx, dy = b[0] - a[0], b[1] - a[1]
            ln = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
            ux, uy = dx / ln, dy / ln
            L = 9 * sw ** 0.5 + 4
            Wd = L * 0.7
            bx, by = b[0], b[1]
            tail = (bx - ux * L, by - uy * L)
            px, py = -uy, ux
            c1 = (tail[0] + px * Wd, tail[1] + py * Wd)
            c2 = (tail[0] - px * Wd, tail[1] - py * Wd)
            return f'<polygon points="{bx:.1f},{by:.1f} {c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f}" fill="{stroke}" stroke="{stroke}" stroke-width="1" />'

        if st.get("endArrow") not in (None, "none"):
            parts.append(arrow(end=True))
        if st.get("startArrow") not in (None, "none"):
            parts.append(arrow(end=False))

        # edge label
        val = e.get("value", "")
        if val:
            lines = plain_lines(val)
            ms = [(xf(x), yf(y)) for x, y in coords]
            mx = sum(p[0] for p in ms) / len(ms)
            my = sum(p[1] for p in ms) / len(ms)
            fs = st.get("fontSize", 15)
            for i, (txt, flag) in enumerate(lines):
                parts.append(
                    f'<text x="{mx:.1f}" y="{my + i*fs*1.2 - (len(lines)-1)*fs*0.6:.1f}" '
                    f'font-size="{fs}" font-weight="{800 if flag else 400}" fill="{st["fontColor"]}" '
                    f'text-anchor="middle" stroke="white" stroke-width="0">{esc(txt)}</text>'
                )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw = f.read()
    tree = ET.fromstring(xml_safe(raw))
    root = tree
    # opencode uses classic mxGraphModel
    for d in root.iter("diagram"):
        graphs = list(d.iter("mxGraphModel"))
        if not graphs:
            continue
        model = graphs[0]
        svg = svg_for_diagram(model)
        with open(args.output, "w") as f:
            f.write(svg)
        print(f"wrote {args.output}")
        return
    sys.exit("no <diagram> with mxGraphModel found")


if __name__ == "__main__":
    main()