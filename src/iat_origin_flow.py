"""IAT › Análise Temporal: fluxo da classe de origem (pasta) para a previsão do modelo.

Somente leitura; nada é gravado nos CSVs do projeto.  Colunas usadas, todas já
existentes:

* ``source_label`` — pasta de onde a imagem veio, gravada em toda
  classificação (com ou sem o modo auditoria); em projetos antigos é derivada
  da última pasta de ``klass``;
* ``preds1`` — previsão top-1 do modelo;
* ``manual_label`` — rótulo atual do IAT (coluna ``classe`` da análise
  temporal), usado na terceira coluna opcional "Rótulo revisado".

Previsões de versões anteriores do modelo são lidas das cópias que o ciclo
integrado guarda antes de aplicar uma versão
(``model_history/<versão>/dataframes_before/*.csv``).

A concordância origem = previsão só equivale a acurácia quando as pastas foram
conferidas por especialistas (Congalton 1991; Northcutt, Jiang & Chuang 2021).
Diagrama: Parallel Sets (Kosara, Bendix & Hauser 2006), alluvial (Rosvall &
Bergstrom 2010), Sankey (Schmidt 2008).  Referências completas em
``docs/METRICAS_E_REFERENCIAS.md``, seção 10.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NO_SOURCE = 'Sem pasta'
NO_PRED = 'Sem previsão'
NO_LABEL = 'Sem classe'
SPECIAL = (NO_SOURCE, NO_PRED, NO_LABEL)
NEUTRAL = '#8A949C'
COLUMN_KEYS = ('origem', 'previsao', 'revisado')
DEFAULT_TITLES = ('Origem (pasta)', 'Previsão', 'Rótulo revisado')
SNAPSHOT_DIR = 'dataframes_before'


# ---------------------------------------------------------------- data
def origin_labels(df):
    """Folder class of each row: ``source_label`` or, when blank, the leaf of ``klass``
    (same rule as ``iat_backend`` and ``batch._source_label_from_klass``)."""
    derived = None
    if 'klass' in df.columns:
        derived = (df['klass'].fillna('').astype(str).str.rstrip('/\\')
                   .str.replace(r'^.*[\\/]', '', regex=True).str.strip())
    if 'source_label' in df.columns:
        src = df['source_label'].fillna('').astype(str).str.strip()
        if derived is not None:
            blank = src.eq('')
            if bool(blank.any()):
                src = src.where(~blank, derived)
        return src
    if derived is not None:
        return derived
    return pd.Series('', index=df.index, dtype=object)


def name_key(name):
    """Comparison key that ignores letter case and repeated/edge spaces."""
    return ' '.join(str(name).split()).casefold()


def _spelling_map(table, cols):
    """One display name per key.  Preference: the model's spelling (predictions),
    then the current labels, then the folder; inside each, the most frequent."""
    chosen = {}
    for col in ('previsao', 'revisado', 'origem'):
        if col not in cols:
            continue
        freq = table.groupby(col, sort=False)['imagens'].sum().sort_values(ascending=False, kind='stable')
        for name in freq.index:
            if name in SPECIAL:
                continue
            chosen.setdefault(name_key(name), name)
    names = set()
    for col in cols:
        names.update(table[col].tolist())
    return {n: (n if n in SPECIAL else chosen.get(name_key(n), n)) for n in names}


def flow_table(origin=None, prediction=None, label=None, merge=True, aliases=None):
    """Images per path through the chosen columns, largest first.

    Any two or all three of *origin* (``origem``), *prediction* (``previsao``)
    and *label* (``revisado``) may be given, as Series with the same index
    (category or text); the columns keep that fixed order.  Counting is done on
    the raw values (fast on categorical columns) and the clean-up — blank
    values and the optional merge of spellings that differ only in letter case
    or spaces — runs on the small aggregated table.  *aliases* ({column:
    {name: display name}}, see ``iat_display_groups``) renames or groups the
    original names of each column before that merge.
    """
    cols = {key: series for key, series in (('origem', origin), ('previsao', prediction), ('revisado', label))
            if series is not None}
    keys = list(cols)
    if len(keys) < 2 or len(next(iter(cols.values()))) == 0:
        return pd.DataFrame(columns=keys + ['imagens'])
    frame = pd.DataFrame(cols)
    counts = frame.groupby(keys, observed=True, sort=False, dropna=False).size()
    out = counts.reset_index(name='imagens')
    out = out.loc[out['imagens'] > 0]
    blanks = {'origem': NO_SOURCE, 'previsao': NO_PRED, 'revisado': NO_LABEL}
    for col in keys:
        text = out[col].astype(object).where(out[col].notna(), '').astype(str).str.strip()
        out[col] = text.where(text.ne(''), blanks[col])
    for col in keys:
        mapping = (aliases or {}).get(col) or {}
        if mapping:
            out[col] = out[col].map(lambda name, m=mapping: m.get(name, name))
    if merge:
        mapping = _spelling_map(out, keys)
        for col in keys:
            out[col] = out[col].map(mapping)
    out = out.groupby(keys, sort=False, as_index=False)['imagens'].sum()
    out['imagens'] = out['imagens'].astype(np.int64)
    return out.sort_values('imagens', ascending=False, kind='stable').reset_index(drop=True)


def visible_flows(table, min_share):
    """Flows drawn on screen: those with at least *min_share* of the images.
    Returns ``(kept, hidden_images)``; the summary always uses every flow."""
    if table is None or table.empty:
        return table, 0
    total = float(table['imagens'].sum())
    if not min_share or min_share <= 0 or total <= 0:
        return table, 0
    keep = table['imagens'] >= float(min_share) * total
    if not bool(keep.any()):
        keep.iloc[0] = True  # never an empty chart: keep the largest flow
    return table.loc[keep].reset_index(drop=True), int(table.loc[~keep, 'imagens'].sum())


def flow_summary(table, class_names=None, merge=True):
    """Agreement, largest divergence, per-category rows and review counts for
    the first two columns (a, b) of *table* (origem/previsao, origem/revisado
    or previsao/revisado).

    agreement = Σ n(c→c) / N over all images in the scope (overall agreement of
    the a × b contingency table); per category of a: share kept as the same
    class in b and the main other destination.  ``review`` (three columns
    only): among images whose folder differs from the prediction, how many
    currently carry the folder's class, the prediction or another class.
    """
    cols = [c for c in COLUMN_KEYS if table is not None and c in table.columns]
    empty = {'total': 0, 'agree': 0, 'rows': [], 'worst': None, 'review': None, 'class_share': None,
             'pair': tuple(cols[:2]) if len(cols) >= 2 else None}
    if table is None or table.empty or len(cols) < 2:
        return empty
    a, b = cols[0], cols[1]
    two = table.groupby([a, b], sort=False)['imagens'].sum()
    total = int(two.sum())
    if total <= 0:
        return empty
    o = two.index.get_level_values(0)
    p = two.index.get_level_values(1)
    agree = int(two[np.asarray(o == p)].sum())
    rows, worst = [], None
    by_first = two.groupby(level=0, sort=False).sum().sort_values(ascending=False, kind='stable')
    for name, n in by_first.items():
        sub = two.xs(name, level=0)
        same = int(sub.get(name, 0))
        others = sub.drop(labels=[name], errors='ignore')
        best = None
        if len(others) and int(others.max()) > 0:
            best = (str(others.idxmax()), int(others.max()))
        rows.append({'name': str(name), 'n': int(n), 'same': same, 'best': best})
        if best and name not in SPECIAL and (worst is None or best[1] > worst['n']):
            worst = {'a': str(name), 'b': best[0], 'n': best[1], 'of': int(n)}
    review = None
    if len(cols) == 3:
        dis = table.loc[table['origem'] != table['previsao']]
        to_origin = int(dis.loc[dis['revisado'] == dis['origem'], 'imagens'].sum())
        to_pred = int(dis.loc[dis['revisado'] == dis['previsao'], 'imagens'].sum())
        review = {'disagree': int(dis['imagens'].sum()), 'origin': to_origin, 'prediction': to_pred,
                  'other': int(dis['imagens'].sum()) - to_origin - to_pred}
    class_share = None
    if class_names and a == 'origem':
        known = {name_key(c) for c in class_names} if merge else {str(c) for c in class_names}
        match = sum(int(v) for k, v in by_first.items()
                    if (name_key(k) if merge else str(k)) in known)
        class_share = match / total
    return {'total': total, 'agree': agree, 'rows': rows, 'worst': worst, 'review': review,
            'class_share': class_share, 'pair': (a, b)}


def flow_palette(names, base=None):
    """Colours of every category: the temporal panel's colour when the class is
    there; other names get colours not used by that palette (same generator,
    ``make_class_palette``); blank categories are grey."""
    from iat_temporal_figure import make_class_palette
    base = {str(k): v for k, v in (base or {}).items()}
    ordered = sorted({str(n) for n in names}, key=lambda n: (n.casefold(), n))
    out = {n: base[n] for n in ordered if n in base}
    missing = [n for n in ordered if n not in out and n not in SPECIAL]
    if missing:
        used = {str(c).upper() for c in base.values()}
        seq = list(make_class_palette([f'{i:05d}' for i in range(len(used) + len(missing) + 4)]).values())
        free = [c for c in seq if str(c).upper() not in used]
        for name, colour in zip(missing, free):
            out[name] = colour
    for name in ordered:
        out.setdefault(name, NEUTRAL)
    return out


def category_order(table, cols):
    """Same order in every column (alphabetical; blank categories last) so a
    class faces itself across the diagram."""
    names = set()
    for col in cols:
        names.update(table[col].tolist())
    return sorted(names, key=lambda n: (n in SPECIAL, n.casefold(), n))


# ------------------------------------------------- previous model versions
def prediction_snapshots(project_dir):
    """Model versions whose application left a copy of the previous CSVs."""
    root = Path(project_dir) / 'model_history'
    out = []
    try:
        for folder in sorted(root.iterdir(), key=lambda p: p.name):
            snap = folder / SNAPSHOT_DIR
            if folder.is_dir() and snap.is_dir() and any(snap.glob('*.csv')):
                out.append(folder.name)
    except OSError:
        return []
    return out


def snapshot_predictions(project_dir, version, keys, progress=None):
    """``preds1`` saved before *version* was applied, aligned to *keys*
    (columns ``batch_source`` = CSV name and ``names``).  Rows without a copy get
    a blank prediction.  Returns ``(series, versions_found)``."""
    snap = Path(project_dir) / 'model_history' / str(version) / SNAPSHOT_DIR
    out = np.full(len(keys), '', dtype=object)
    found = set()
    positions = pd.Series(np.arange(len(keys)), index=keys.index)
    groups = list(keys.groupby('batch_source', sort=False).groups.items())
    for i, (csv_name, idx) in enumerate(groups, 1):
        path = snap / str(csv_name)
        if path.is_file():
            cols = [c for c in pd.read_csv(path, nrows=0).columns if c in ('names', 'preds1', 'model_version')]
            if {'names', 'preds1'} <= set(cols):
                old = pd.read_csv(path, usecols=cols, low_memory=False)
                if 'model_version' in old.columns:
                    found.update(str(v) for v in old['model_version'].dropna().unique() if str(v).strip())
                lookup = pd.Series(old['preds1'].fillna('').astype(str).str.strip().to_numpy(),
                                   index=old['names'].astype(str))
                lookup = lookup[~lookup.index.duplicated()]
                vals = keys.loc[idx, 'names'].astype(str).map(lookup).fillna('').to_numpy(object)
                out[positions.loc[idx].to_numpy()] = vals
        if progress:
            progress(i, len(groups), f'Previsões anteriores a {version}: {i}/{len(groups)}')
    return pd.Series(out, index=keys.index).astype('category'), sorted(found)


# --------------------------------------------------------- static figure
EXPORT_DEFAULTS = {
    'title': '', 'col_a': DEFAULT_TITLES[0], 'col_b': DEFAULT_TITLES[1], 'col_c': DEFAULT_TITLES[2],
    'font': 'DejaVu Sans', 'title_size': 13.0, 'header_size': 11.0, 'label_size': 9.0,
    'background': '#FFFFFF', 'ink': '#131B2A', 'ribbon_alpha': 0.55, 'show_counts': True,
    'figure_width': 25.0 / 2.54, 'figure_height': 16.0 / 2.54, 'dpi': 300.0, 'transparent': False, 'trim': True,
}


def _fmt_int(n):
    return f'{int(n):,}'.replace(',', '.')


def render_alluvial(fig, table, palette, color_by='origem', style=None):
    """Static alluvial (Parallel Sets) of *table* on the matplotlib *fig*.

    Each column stacks its categories in the same order; block height and
    ribbon width are proportional to the number of images.  Ribbons are drawn
    per pair of neighbouring columns, coloured by the origin or the prediction.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MPath

    st = dict(EXPORT_DEFAULTS, **(style or {}))
    font, ink, bg = st['font'], st['ink'], st['background']
    fig.set_facecolor(bg)
    canvas = FigureCanvasAgg(fig)
    width_in, height_in = fig.get_size_inches()
    ax = fig.add_axes([0.2, 0.06, 0.6, 0.8])
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    title = str(st.get('title') or '').strip()
    cols = [c for c in COLUMN_KEYS if table is not None and c in table.columns]
    if table is None or table.empty or len(cols) < 2:
        ax.text(.5, .5, 'Sem imagens neste recorte', ha='center', va='center', color=ink, fontfamily=font,
                fontsize=st['label_size'] + 2, transform=ax.transAxes)
        if title:
            fig.text(.5, 1 - .12 / height_in, title, ha='center', va='top', color=ink, fontfamily=font,
                     fontsize=st['title_size'], fontweight='bold')
        return fig
    k = len(cols)
    total = float(table['imagens'].sum())
    order = category_order(table, cols)
    n_max = max(int(table[c].nunique()) for c in cols)
    gap = total * min(0.02, 0.25 / max(n_max - 1, 1))
    bw = 0.045 * (k - 1)

    def left(j):
        return j / (k - 1) - bw * j / (k - 1)

    sizes = [table.groupby(c, sort=False)['imagens'].sum() for c in cols]
    heights = [float(s.sum()) + gap * (len(s) - 1) for s in sizes]
    h_max = max(heights)
    tops = []
    for j, s in enumerate(sizes):
        y = (h_max - heights[j]) / 2.0
        pos = {}
        for name in order:
            if name in s.index:
                pos[name] = y
                y += float(s[name]) + gap
        tops.append(pos)

    color_col = color_by if color_by in cols else cols[0]
    rank = {n: i for i, n in enumerate(order)}
    alpha = float(st['ribbon_alpha'])
    patches, faces = [], []
    for j in range(k - 1):
        a, b = cols[j], cols[j + 1]
        keys = list(dict.fromkeys([a, b, color_col]))
        seg = table.groupby(keys, sort=False, as_index=False)['imagens'].sum()
        src = seg[a].astype(str).tolist()
        dst = seg[b].astype(str).tolist()
        tint = seg[color_col].astype(str).tolist()
        vals = seg['imagens'].astype(float).tolist()
        ra, rb, rc = [rank[s] for s in src], [rank[d] for d in dst], [rank[c] for c in tint]
        n = len(vals)
        # Ribbons leave a block ordered by destination and enter ordered by source.
        src_y, dst_y = [0.0] * n, [0.0] * n
        out_off, in_off = dict(tops[j]), dict(tops[j + 1])
        for i in sorted(range(n), key=lambda i: (ra[i], rb[i], rc[i])):
            src_y[i] = out_off[src[i]]
            out_off[src[i]] += vals[i]
        for i in sorted(range(n), key=lambda i: (rb[i], ra[i], rc[i])):
            dst_y[i] = in_off[dst[i]]
            in_off[dst[i]] += vals[i]
        x0, x1 = left(j) + bw, left(j + 1)
        xm = (x0 + x1) / 2.0
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
                 MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        for i in range(n):
            h, a0, b0 = vals[i], src_y[i], dst_y[i]
            verts = [(x0, a0), (xm, a0), (xm, b0), (x1, b0), (x1, b0 + h), (xm, b0 + h), (xm, a0 + h),
                     (x0, a0 + h), (x0, a0)]
            patches.append(PathPatch(MPath(verts, codes)))
            faces.append(to_rgba(palette.get(tint[i], NEUTRAL), alpha))
    if patches:
        ax.add_collection(PatchCollection(patches, facecolors=faces, edgecolors='none', linewidths=0,
                                          match_original=False))
    pad = h_max * 0.01
    ax.set_xlim(0, 1)
    ax.set_ylim(h_max + pad, -pad)

    # Blocks and labels.
    label_texts = {'left': [], 'right': []}
    small = 0
    axes_h_pt = 0.8 * height_in * 72.0
    for j, c in enumerate(cols):
        for name, top in tops[j].items():
            size = float(sizes[j][name])
            ax.add_patch(Rectangle((left(j), top), bw, size, facecolor=palette.get(name, NEUTRAL),
                                   edgecolor='none'))
            if size / (h_max + 2 * pad) * axes_h_pt < st['label_size'] * 0.85:
                small += 1
                continue
            text = name + (f'  {_fmt_int(size)}' if st['show_counts'] else '')
            mid = top + size / 2.0
            if j == 0:
                t = ax.annotate(text, xy=(left(j), mid), xytext=(-4, 0), textcoords='offset points', ha='right',
                                va='center', color=ink, fontfamily=font, fontsize=st['label_size'])
                label_texts['left'].append(t)
            elif j == k - 1:
                t = ax.annotate(text, xy=(left(j) + bw, mid), xytext=(4, 0), textcoords='offset points',
                                ha='left', va='center', color=ink, fontfamily=font, fontsize=st['label_size'])
                label_texts['right'].append(t)
            else:
                ax.annotate(text, xy=(left(j) + bw, mid), xytext=(3, 0), textcoords='offset points', ha='left',
                            va='center', color=ink, fontfamily=font, fontsize=st['label_size'],
                            bbox=dict(boxstyle='round,pad=0.15', fc=bg, ec='none', alpha=0.75))
    headers = [st['col_a'], st['col_b'], st['col_c']][:k]
    for j, text in enumerate(headers):
        if not str(text or '').strip():
            continue
        x = left(j) if j == 0 else (left(j) + bw if j == k - 1 else left(j) + bw / 2)
        ha = 'left' if j == 0 else ('right' if j == k - 1 else 'center')
        ax.annotate(str(text), xy=(x, -pad), xytext=(0, 6), textcoords='offset points', ha=ha, va='bottom',
                    color=ink, fontfamily=font, fontsize=st['header_size'], fontweight='bold')
    if small:
        note = ('1 categoria pequena sem rótulo' if small == 1 else f'{small} categorias pequenas sem rótulo')
        fig.text(.5, .1 / height_in, note + ' (os nomes aparecem ao passar o mouse no ILT).',
                 ha='center', va='bottom', color=ink, alpha=.7, fontfamily=font, fontsize=st['label_size'] * .85)
    if title:
        fig.text(.5, 1 - .12 / height_in, title, ha='center', va='top', color=ink, fontfamily=font,
                 fontsize=st['title_size'], fontweight='bold')

    # Fit the axes between the outer labels (sizes in points do not depend on it).
    renderer = canvas.get_renderer()
    dpi = fig.dpi
    w_px = width_in * dpi

    def widest(texts):
        return max((t.get_window_extent(renderer).width for t in texts), default=0.0)

    l_frac = (widest(label_texts['left']) + 10 * dpi / 72.0) / w_px
    r_frac = (widest(label_texts['right']) + 10 * dpi / 72.0) / w_px
    l_frac = max(0.02, l_frac)
    r_frac = max(0.02, r_frac)
    if 1 - l_frac - r_frac < 0.3:
        extra = (0.3 - (1 - l_frac - r_frac)) / 2
        l_frac, r_frac = max(0.02, l_frac - extra), max(0.02, r_frac - extra)
    top_in = (st['title_size'] * 1.9 / 72.0 if title else 0.08) + st['header_size'] * 2.0 / 72.0
    bottom_in = 0.12 + (st['label_size'] * 1.6 / 72.0 if small else 0.0)
    b_frac = min(0.3, bottom_in / height_in)
    t_frac = min(0.4, top_in / height_in)
    ax.set_position([l_frac, b_frac, max(0.05, 1 - l_frac - r_frac), max(0.05, 1 - b_frac - t_frac)])
    return fig


def save_alluvial(path, table, palette, color_by, style):
    """Write the static alluvial to *path* (png, tif/tiff, jpg, svg or pdf)."""
    from matplotlib.figure import Figure
    from iat_temporal_figure import _savefig_kwargs
    st = dict(EXPORT_DEFAULTS, **(style or {}))
    fig = Figure(figsize=(float(st['figure_width']), float(st['figure_height'])), dpi=100,
                 facecolor=st['background'])
    render_alluvial(fig, table, palette, color_by, st)
    fig.savefig(path, **_savefig_kwargs(path, st))
    return path
