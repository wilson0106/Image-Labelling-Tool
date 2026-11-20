"""Figuras exportáveis e cores consistentes para a análise temporal/espacial.

Este módulo não modifica CSVs, embeddings ou coordenadas t-SNE.
"""
from __future__ import annotations

import colorsys
import math
import textwrap
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

INK = "#FFFFFF"
BG = "#0B0B0D"
GRID = "#2A2A30"

# Density-stain recipe shared by the Tk panel, the web page, the exports and
# the GIF: 2-D histogram on the fixed frame bounds, Gaussian smoothing, three
# bands at fixed fractions of a reference peak.
DENSITY_BINS = 55
DENSITY_SIGMA = 1.4
DENSITY_MIN_POINTS = 20
DENSITY_MIN_PEAK = 0.035
DENSITY_LEVELS = (0.17, 0.38, 0.64)
DENSITY_ALPHAS = (0.22, 0.37, 0.56)


def class_density(xy, bounds):
    """Smoothed 2-D histogram of *xy* on *bounds*; ``None`` when unavailable."""
    if gaussian_filter is None or not bounds:
        return None
    xmin, xmax, ymin, ymax = bounds
    hist, xedge, yedge = np.histogram2d(xy[:, 0], xy[:, 1], bins=DENSITY_BINS,
                                        range=[[xmin, xmax], [ymin, ymax]])
    return gaussian_filter(hist, DENSITY_SIGMA), xedge, yedge


def density_reference(view, frames, bounds):
    """Per-class reference peak shared by every frame of the current view.

    The stain bands are drawn at 17/38/64 % of a peak.  Measured against each
    frame's own peak, a frame with 20 images looks as dense as one with 2 000
    and frames cannot be compared (Individual, Comparar, Animação, GIF).  The
    reference returned here is the highest smoothed density any frame reaches
    for that class, so the densest frame keeps its full bands and the others
    are proportionally fainter — a common scale, as recommended for small
    multiples (Tufte 1983, *The Visual Display of Quantitative Information*;
    Cleveland 1993, *Visualizing Data*).  Classes below ``DENSITY_MIN_POINTS``
    in a frame are not drawn as stains there and do not contribute.
    """
    out = {}
    if view is None or view.empty or not bounds or gaussian_filter is None:
        return out
    xy_all = view[["x", "y"]].to_numpy(dtype=float)
    classes = view["classe"].to_numpy()
    for _label, mask in frames:
        sel = np.asarray(mask.fillna(False), dtype=bool) if hasattr(mask, "fillna") else np.asarray(mask, dtype=bool)
        if sel.shape[0] != len(view) or not sel.any():
            continue
        frame_classes = classes[sel]
        frame_xy = xy_all[sel]
        for name in np.unique(frame_classes):
            pick = frame_classes == name
            if int(pick.sum()) < DENSITY_MIN_POINTS:
                continue
            dens = class_density(frame_xy[pick], bounds)
            if dens is None:
                continue
            peak = float(dens[0].max())
            if peak > out.get(name, 0.0):
                out[name] = peak
    return out


def density_levels(peak, settings, name):
    """Band thresholds for one class in one frame (``None`` = draw nothing)."""
    ref = peak
    if settings.get("density_common", False):
        common = (settings.get("density_ref") or {}).get(name)
        if common:
            ref = max(float(common), peak)
    if ref <= DENSITY_MIN_PEAK:
        return None
    return [ref * f for f in DENSITY_LEVELS]


def _lab(rgb):
    """sRGB -> CIELAB (D65), para afastar perceptualmente as cores."""
    rgb = np.asarray(rgb, dtype=float)
    linear = np.where(rgb > .04045, ((rgb + .055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = linear @ np.array([[.4124564, .3575761, .1804375],
                             [.2126729, .7151522, .0721750],
                             [.0193339, .1191920, .9503041]]).T
    xyz /= np.array([.95047, 1.0, 1.08883])
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    return np.column_stack((116 * f[:, 1] - 16,
                            500 * (f[:, 0] - f[:, 1]),
                            200 * (f[:, 1] - f[:, 2])))


# Hand-picked qualitative colours for dark backgrounds, ordered so that the
# first few are the most distinct (cyan, orange, green, rose, violet, yellow...).
# Minimum pairwise CIELAB distance between any two of them is > 27.
CURATED_CLASS_COLORS = (
    "#22D3EE", "#FF8C1A", "#4ADE50", "#FF3D71", "#B983FF", "#FFE14D",
    "#3B82F6", "#F472D0", "#C08457", "#E5E7EB", "#14B8A6", "#D9F99D",
)


def make_class_palette(names):
    """Deterministic colour map; independent of the order of *names*.

    Up to 12 classes use a curated high-contrast palette.  Larger sets keep
    those 12 and extend them by maximising the minimum CIELAB distance to the
    colours already chosen.  The panel computes the palette from the classes
    being displayed, so a small selection always gets the most distinct colours.
    """
    ordered = sorted({str(name) for name in names}, key=lambda n: (n.casefold(), n))
    if not ordered:
        return {}
    curated = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
                        for c in CURATED_CLASS_COLORS])
    if len(ordered) <= len(curated):
        return {name: CURATED_CLASS_COLORS[i] for i, name in enumerate(ordered)}
    generated = np.array([
        colorsys.hsv_to_rgb(h / 72, sat, value)
        for h in range(72)
        for sat, value in ((.90, 1.0), (.67, .99), (.82, .82))
    ])
    candidates = np.vstack([curated, generated])
    labs = _lab(candidates)
    picked = list(range(len(curated)))
    distances = np.full(len(candidates), np.inf)
    for k in picked:
        distances = np.minimum(distances, np.sum((labs - labs[k]) ** 2, axis=1))
    for _ in range(min(len(ordered), len(candidates)) - len(picked)):
        distances[picked] = -1
        choice = int(np.argmax(distances))
        picked.append(choice)
        distances = np.minimum(distances, np.sum((labs - labs[choice]) ** 2, axis=1))
    colors = ["#{:02X}{:02X}{:02X}".format(*(int(round(255 * v)) for v in candidates[i]))
              for i in picked]
    return {name: colors[i % len(colors)] for i, name in enumerate(ordered)}


# Visualização na dashboard mantém o tema escuro; exportação é independente.
DEFAULT_EXPORT_STYLE = {
    "background": "#FFFFFF",
    "title_template": "{intervalo}",
    "title_family": "DejaVu Sans", "title_size": 16, "title_color": "#131B2A",
    "xlabel": "t-SNE 1", "ylabel": "t-SNE 2",
    "axis_family": "DejaVu Sans", "axis_size": 13, "axis_color": "#131B2A",
    "tick_family": "DejaVu Sans", "tick_size": 11, "tick_color": "#131B2A",
    "legend_title": "CLASSES",
    "legend_family": "DejaVu Sans", "legend_size": 11, "legend_color": "#131B2A",
    "legend_title_size": 12, "legend_title_color": "#131B2A",
    "legend_names": {},
    "panel_title_a": "", "panel_title_b": "", "panel_title_c": "",
    "figure_width": 16.5, "figure_height": 8.5,
    "auto_expand": True,
}
def export_style(**overrides):
    """Cópia mutável da configuração de exportação — sem tocar o gráfico em tela."""
    result = {**DEFAULT_EXPORT_STYLE, "legend_names": {}}
    result.update(overrides)
    result["legend_names"] = dict(result.get("legend_names") or {})
    return result

def _plot_theme(style):
    if style is None:
        return BG, INK, "#5A5A63", "white"
    bg = style["background"]
    return bg, style["axis_color"], style["tick_color"], (
        "#CAD4DF" if bg.upper() in ("#FFFFFF", "#FFF") else "#58718A")

def _panel_title(template, interval, index, style=None):
    specific = (style or {}).get("panel_title_" + "abc"[index], "")
    text = str(specific).strip() or str(template)
    return text.replace("{intervalo}", str(interval)).replace("{painel}", "ABC"[index])

def _plot(ax, rows, title, bounds, settings, palette, style=None, index=0):
    bg, ink, spine, grid = _plot_theme(style)
    ax.set_facecolor(bg)
    if style:
        title_text = _panel_title(style["title_template"], title, index, style)
        line_chars = int(style.get("_title_chars", 100))
        title_text = textwrap.fill(title_text, width=max(10,line_chars),
                                  break_long_words=True,break_on_hyphens=False)
        ax.set_title(title_text, color=style["title_color"], fontweight="bold",
                     fontsize=float(style["title_size"]), family=style["title_family"], pad=12)
        ax.set_xlabel(style["xlabel"], color=style["axis_color"],
                      fontsize=float(style["axis_size"]), family=style["axis_family"])
        ax.set_ylabel(style["ylabel"], color=style["axis_color"],
                      fontsize=float(style["axis_size"]), family=style["axis_family"])
        ax.tick_params(colors=style["tick_color"], labelsize=float(style["tick_size"]))
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily(style["tick_family"])
        ax.grid(color=grid, alpha=.85 if bg.upper() == "#FFFFFF" else .28, linewidth=.6)
    else:
        ax.set_title(title, color=INK, fontweight="bold", fontsize=13, pad=12)
        ax.set_xlabel(settings.get("xlabel", "t-SNE 1"), color=INK, fontsize=12)
        ax.set_ylabel(settings.get("ylabel", "t-SNE 2"), color=INK, fontsize=12)
        ax.tick_params(colors=INK, labelsize=10)
        ax.grid(color="white", alpha=.12, linewidth=.6)
    for spine_item in ax.spines.values():
        spine_item.set_color(spine)
    if bounds:
        xmin, xmax, ymin, ymax = bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    if rows.empty:
        ax.text(.5, .5, "Sem imagens nesta faixa", color=ink, fontsize=12,
                ha="center", va="center", transform=ax.transAxes)
        return
    for name, group in rows.groupby("classe", sort=True):
        color = palette.get(name, "#FFFFFF")
        xy = group[["x", "y"]].to_numpy(dtype=float)
        n = len(xy)
        if n >= DENSITY_MIN_POINTS and gaussian_filter is not None and bounds and settings.get("density", True):
            smooth, xedge, yedge = class_density(xy, bounds)
            peak = float(smooth.max())
            bands = density_levels(peak, settings, name)
            if bands is not None:
                levels = bands + [max(peak, bands[-1]) + 1e-9]
                xx = (xedge[:-1] + xedge[1:]) * .5
                yy = (yedge[:-1] + yedge[1:]) * .5
                ax.contourf(xx, yy, smooth.T, levels=levels,
                            colors=[to_rgba(color, alpha) for alpha in DENSITY_ALPHAS])
                if settings.get("contours", False):
                    ax.contour(xx, yy, smooth.T, levels=levels[:-1],
                               colors=[color], alpha=.92, linewidths=.9)
        if settings.get("points", True) or n < DENSITY_MIN_POINTS:
            if n > 4000:
                xy = xy[np.linspace(0, n - 1, 4000, dtype=int)]
            ax.scatter(xy[:, 0], xy[:, 1], s=13, color=color, alpha=.83,
                       rasterized=n > 1000, edgecolors="none")


def _counts(panels):
    return [rows["classe"].value_counts() for _, rows in panels]


def legend_lines(panels, palette, aliases=None):
    """As contagens são sempre do recorte exibido, inclusive no terceiro painel."""
    if not panels:
        return []
    aliases = aliases or {}
    counts = _counts(panels)
    labels = [name for name in palette if any(int(c.get(name, 0)) > 0 for c in counts)]
    result = []
    for name in labels:
        display = str(aliases.get(name) or name)
        if len(panels) > 1:
            parts = ["{}: {:,}".format("ABC"[i], int(c.get(name, 0)))
                     for i, c in enumerate(counts)]
            result.append((name, "{} ({})".format(display, " | ".join(parts))))
        else:
            result.append((name, "{} ({:,})".format(display, int(counts[0].get(name, 0)))))
    return result


def _wrap_legend(lines, n, names, font_pt, usable_width_in):
    """Prepara linhas com espaço real para nome e contagens A/B/C."""
    max_chars = max(8, int(usable_width_in * 72 / (float(font_pt) * .61)))
    formatted=[]
    for name,label in lines:
        display = str(names.get(name) or name)
        if n>=2:
            suffix = label[len(display):].strip()
            pieces = textwrap.wrap(display, width=max_chars,
                                   break_long_words=True,break_on_hyphens=False)
            pieces.extend(textwrap.wrap(suffix, width=max_chars,
                                        break_long_words=True,break_on_hyphens=False))
        else:
            pieces = textwrap.wrap(label, width=max_chars,
                                   break_long_words=True,break_on_hyphens=False)
        formatted.append((name, pieces or [label]))
    return formatted


def _export_geometry(fig, panels, lines, style):
    """Expande a tela do Matplotlib antes de renderizar. Nunca comprime letras.

    Os usuários informam uma dimensão desejada; se os textos não couberem,
    a opção padrão 'auto_expand' cresce a figura exportável em vez de
    sobrepor rótulos ou encolher a fonte escolhida.
    """
    n=len(panels)
    title_size=float(style["title_size"])
    legend_size=float(style["legend_size"])
    legend_title_size=float(style["legend_title_size"])
    titles=[_panel_title(style["title_template"],label,i,style)
            for i,(label,_) in enumerate(panels)]
    # Cada gráfico mantém uma largura própria, mesmo em comparação tripla.
    preferred_panel=max(3.6, min(7.4, (max(map(len,titles),default=10)
                         * title_size * .57 / 72 / 2.4) + .8))
    preferred_leg=max(4.2, min(8.0, 3.4 + .066 * legend_size))
    # Para fontes grandes, a legenda recebe largura suficiente para
    # não desperdiçar altura quebrando cada contagem em muitas linhas.
    preferred_leg=max(preferred_leg, 4.0 + .095*max(0,legend_size-12))
    necessary_width=(n*preferred_panel + preferred_leg + .48*n + 1.25)
    desired_w=max(6.,float(style.get("figure_width",16.5)))
    desired_h=max(4.,float(style.get("figure_height",8.5)))
    if desired_w < necessary_width and not style.get("auto_expand",True):
        raise ValueError(
            f"Largura insuficiente para títulos e legenda ({necessary_width:.1f} pol). "
            "Aumente a largura ou ative Expansão automática.")
    width=max(desired_w, necessary_width) if style.get("auto_expand",True) else desired_w
    # Os painéis usam toda a largura adicional, sem diminuir a área da legenda.
    panel_width=(width-1.25-.48*n-preferred_leg)/n
    title_chars=max(10,int((panel_width-.32)*72/(title_size*.58)))
    title_lines=[len(textwrap.wrap(t,width=title_chars,
                                   break_long_words=True,break_on_hyphens=False)) or 1
                 for t in titles]
    title_space=max(1.0,max(title_lines)*title_size/72*1.38+.58)
    leg_lines=_wrap_legend(lines,n,style.get("legend_names") or {},
                           legend_size,preferred_leg-.75)
    legend_height=(legend_title_size/72*1.7+.35 +
                   sum(len(part)*legend_size/72*1.27 + .115
                       for _,part in leg_lines))
    necessary_height=max(5.0, legend_height + 2.65,
                         title_space+3.8+1.05)
    if desired_h < necessary_height and not style.get("auto_expand",True):
        raise ValueError(
            f"Altura insuficiente para a legenda ({necessary_height:.1f} pol). "
            "Aumente a altura ou ative Expansão automática.")
    height=max(desired_h,necessary_height) if style.get("auto_expand",True) else desired_h
    if width>54 or height>55:
        raise ValueError("A fonte/texto exige uma figura muito grande. "
                         "Reduza a fonte ou a quantidade de classes.")
    fig.set_size_inches(width,height,forward=False)
    return dict(panel_width=panel_width,leg_width=preferred_leg,
                top=1.0-title_space/height,
                bottom=.85/height,
                title_chars=title_chars,
                legend_parts=leg_lines,
                legend_title_size=legend_title_size)


def render_figure(fig, panels, bounds, settings, palette, style=None):
    """Renderiza gráficos e legenda na mesma Figure, sem sobreposição.

    Para exportação, amplia a geometria quando necessário e preserva
    exatamente o tamanho de fonte solicitado (sem compactação vertical).
    """
    fig.clear()
    fig.patch.set_facecolor(style["background"] if style else BG)
    panels=list(panels)
    if not panels:
        panels=[("Nenhum intervalo", __import__("pandas").DataFrame(
            columns=["classe","x","y"]))]
    n=len(panels)
    lines=legend_lines(panels,palette,
                       aliases=style.get("legend_names") if style else None)
    if style:
        geo=_export_geometry(fig,panels,lines,style)
        style={**style,"_title_chars":geo["title_chars"]}
        widths=[geo["panel_width"]]*n+[geo["leg_width"]]
        gs=fig.add_gridspec(1,n+1,width_ratios=widths,
                            left=.8/fig.get_figwidth(),
                            right=1-.26/fig.get_figwidth(),
                            bottom=geo["bottom"],top=geo["top"],
                            wspace=.18)
    else:
        gs=fig.add_gridspec(
            1,n+1,
            width_ratios=[3.3,2.4] if n==1 else ([2.7,2.7,2.85] if n==2
                                                 else [2.7,2.7,2.7,3.25]),
            left=.084 if n==1 else .060,right=.985,
            bottom=.12,top=.92,wspace=.24 if n==1 else .26)
    axes=[]
    for k,(title,rows) in enumerate(panels):
        ax=fig.add_subplot(gs[0,k])
        _plot(ax,rows,title,bounds,settings,palette,style,k)
        axes.append(ax)
    leg=fig.add_subplot(gs[0,-1])
    leg.set_facecolor(style["background"] if style else BG)
    leg.set_axis_off()
    title_font=float(style["legend_title_size"]) if style else 11.
    legend_font=float(style["legend_size"]) if style else 11.
    leg.text(.02,1.008,style["legend_title"] if style else "CLASSES",
             transform=leg.transAxes,ha="left",va="bottom",
             color=style["legend_title_color"] if style else INK,
             family=style["legend_family"] if style else "sans-serif",
             weight="bold",fontsize=title_font)
    if not lines:
        leg.text(.02,.94,"Sem classes neste intervalo.",transform=leg.transAxes,
                 color=style["legend_color"] if style else INK,va="top",fontsize=10)
        return axes

    if style:
        formatted=geo["legend_parts"]
        # Não altera a fonte nem encolhe a distância entre itens:
        # o tamanho mínimo da figura já foi calculado acima.
        height_in=leg.get_position().height * fig.get_figheight()
        row_in=legend_font/72 * 1.27
        gap_in=.115
        y=.985
        for name,parts in formatted:
            leg.plot([.026],[y-row_in/height_in*.5],"o",color=palette[name],
                     markersize=8,transform=leg.transAxes,clip_on=False)
            leg.text(.091,y,"\n".join(parts),color=style["legend_color"],
                     fontsize=legend_font,family=style["legend_family"],
                     linespacing=1.14,transform=leg.transAxes,
                     ha="left",va="top",clip_on=False)
            y-= (len(parts)*row_in+gap_in)/height_in
        return axes

    # Apenas para o painel escuro responsivo: a fonte é reduzida quando
    # a janela fica pequena. As exportações não usam essa compactação.
    width_px=fig.get_figwidth()*fig.dpi
    height_px=fig.get_figheight()*fig.dpi
    usable_px=leg.get_position().width*width_px*.78
    available_px=leg.get_position().height*height_px
    def fmt(font_pt):
        max_chars=max(12,int(usable_px/(font_pt*fig.dpi/72*.57)))
        wrapped=[]
        for name,label in lines:
            pieces=textwrap.wrap(label,width=max_chars,break_long_words=True,
                                 break_on_hyphens=False)
            wrapped.append((name,"\n".join(pieces or [label])))
        required=sum((1+lab.count("\n")+.32)*font_pt*fig.dpi/72*1.18
                     for _,lab in wrapped)
        return wrapped,required
    font=11
    for candidate in (11,10,9,8,7):
        formatted,required=fmt(candidate)
        font=candidate
        if required <= available_px*.95:
            break
    line_height=font*fig.dpi/72*1.18/max(available_px,1)
    if required > available_px*.95:
        line_height*=available_px*.95/required
    y=.97
    for name,label in formatted:
        leg.plot([.034],[y-line_height*.53],"o",color=palette[name],
                 markersize=7.5 if font>=9 else 6,
                 transform=leg.transAxes,clip_on=False)
        leg.text(.12,y,label,color=INK,fontsize=font,
                 transform=leg.transAxes,ha="left",va="top",clip_on=True)
        y-=line_height*(1+label.count("\n")+.32)
    return axes


def _savefig_kwargs(path, style):
    """savefig options for *path*: dpi, transparency, margin trimming and the
    lossless/high-quality settings of raster formats (TIFF LZW, JPEG q95)."""
    import os
    ext = os.path.splitext(str(path))[1].lower().lstrip(".")
    style = style or {}
    transparent = bool(style.get("transparent", False)) and ext not in ("jpg", "jpeg")
    kw = dict(dpi=float(style.get("dpi", 300)), transparent=transparent,
              facecolor="none" if transparent else style.get("background", "#FFFFFF"))
    if style.get("trim", True):
        kw.update(bbox_inches="tight", pad_inches=.18)
    if ext in ("tif", "tiff"):
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    elif ext in ("jpg", "jpeg"):
        kw["pil_kwargs"] = {"quality": 95}
    return kw


def save_figure(path, panels, bounds, settings, palette, style):
    """Render *panels* with the export *style* and write *path* (format from the
    extension: png, svg, pdf, tif/tiff, jpg).  ``style['dpi']`` is the output
    resolution (also used for the rasterized point clouds of SVG/PDF)."""
    fig = Figure(figsize=(float(style["figure_width"]), float(style["figure_height"])),
                 dpi=100, facecolor=style["background"])
    render_figure(fig, panels, bounds, settings, palette, style=style)
    fig.savefig(path, **_savefig_kwargs(path, style))
    return path


def _fixed_frame_style(view, frames, bounds, settings, palette, style):
    """Same size for every interval: measured once with the longest title and
    every class present (the GIF rule), then auto-expansion is frozen."""
    fig = Figure(figsize=(float(style["figure_width"]), float(style["figure_height"])),
                 dpi=100, facecolor=style["background"])
    longest = max((title for title, _ in frames), key=lambda t: len(str(t)))
    distinct = view.drop_duplicates(subset=["classe"]) if not view.empty else view
    render_figure(fig, [(longest, distinct)], bounds, settings, palette, style=style)
    w, h = fig.get_size_inches()
    return {**style, "figure_width": float(w), "figure_height": float(h), "auto_expand": False}


def save_frame_images(folder, stem, fmt, view, frames, bounds, settings, palette, style, progress=None):
    """One image per interval (``<stem>_001_<interval>.<fmt>``), all the same size."""
    import re
    from pathlib import Path
    if not frames:
        raise ValueError("Não há intervalos para exportar.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    style = {**style, "panel_title_a": "", "panel_title_b": "", "panel_title_c": ""}
    style = _fixed_frame_style(view, frames, bounds, settings, palette, style)
    paths = []
    for i, (title, mask) in enumerate(frames, 1):
        rows = view.loc[mask.fillna(False)]
        safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(title)).strip("_")[:60] or "intervalo"
        path = folder / f"{stem}_{i:03d}_{safe}.{fmt}"
        save_figure(path, [(title, rows)], bounds, settings, palette, style)
        paths.append(path)
        if progress:
            progress(i, len(frames))
    return paths


def save_animated_gif(path, view, frames, bounds, settings, palette, duration_ms=1000,
                      progress=None, style=None):
    """Exportar quadros reais, sem interpolar posições ou alterar resultados."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    if not frames:
        raise ValueError("Não há intervalos para exportar.")
    if len(frames) > 240:
        raise ValueError("Limite de 240 quadros por GIF. Aumente o passo dos intervalos.")
    # Resolution of every GIF frame: style['dpi'] when given (web editor),
    # 100 dpi otherwise (classic editor, unchanged).
    gif_dpi = float(style.get("dpi", 100)) if style else 100
    fig = Figure(figsize=(float(style.get("figure_width",15.5)),
                          float(style.get("figure_height",8.2))) if style else (15.5,8.2),
                 dpi=gif_dpi,facecolor=style["background"] if style else BG)
    if style:
        # Todos os quadros do GIF têm exatamente as mesmas dimensões.
        # Dimensionar primeiro com o conjunto de classes e o maior título
        # evita cortar legendas que cresçam durante a animação.
        longest=max((title for title,_ in frames),key=lambda t:len(str(t)))
        distinct=view.drop_duplicates(subset=["classe"]) if not view.empty else view
        render_figure(fig,[(longest,distinct)],bounds,settings,palette,style=style)
        final_w,final_h=fig.get_size_inches()
        style={**style,"figure_width":float(final_w),
               "figure_height":float(final_h),"auto_expand":False}
    canvas = FigureCanvasAgg(fig)
    snapshots = []
    for idx, (title, mask) in enumerate(frames):
        rows = view.loc[mask.fillna(False)]
        render_figure(fig, [(title, rows)], bounds, settings, palette, style=style)
        canvas.draw()
        image = Image.frombuffer("RGBA", canvas.get_width_height(),
                                 bytes(canvas.buffer_rgba()), "raw", "RGBA", 0, 1)
        snapshots.append(image.quantize(colors=192, method=Image.Quantize.FASTOCTREE))
        if progress:
            progress(idx + 1, len(frames))
    snapshots[0].save(path, format="GIF", save_all=True, append_images=snapshots[1:],
                      duration=max(100, int(duration_ms)), loop=0, disposal=2,
                      optimize=False)
    return len(snapshots)
