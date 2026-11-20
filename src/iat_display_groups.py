"""Análise Temporal: grupos de exibição de classes (nomes alternativos).

Os nomes alternativos valem só para gráficos, resumos e exportações da
Análise Temporal; os rótulos das imagens, os CSVs e os modelos não mudam.
Cada coluna tem o próprio mapeamento ``nome original -> nome no gráfico``:

* ``revisado`` — rótulo atual (``manual_label``): classes da projeção, da
  evolução entre intervalos, do resumo e da terceira coluna do alluvial;
* ``origem`` — pasta de origem (``source_label``), coluna do alluvial;
* ``previsao`` — previsão do modelo (``preds1``), coluna do alluvial.

Classes com o mesmo nome no gráfico formam um grupo (somam imagens).  O
editor mostra, para cada nome, a distribuição das suas imagens nas classes
previstas e sugere a classe dominante (``match_matrix``).  Os mapeamentos
ficam em ``<projeto>/exports/temporal/grupos_de_exibicao.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

COLUMNS = ('revisado', 'origem', 'previsao')
FILENAME = 'grupos_de_exibicao.json'


def empty_groups():
    return {col: {} for col in COLUMNS}


def clean_groups(groups):
    """Only str -> non-empty, different str entries, for the known columns."""
    out = empty_groups()
    for col in COLUMNS:
        for name, shown in dict((groups or {}).get(col) or {}).items():
            name, shown = str(name), str(shown or '').strip()
            if shown and shown != name:
                out[col][name] = shown
    return out


def groups_path(project_dir):
    return Path(project_dir) / 'exports' / 'temporal' / FILENAME


def load_groups(project_dir):
    path = groups_path(project_dir)
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return empty_groups()
    return clean_groups(payload.get('nomes') if isinstance(payload, dict) else None)


def save_groups(project_dir, groups):
    from ilt_contracts import atomic_write_json
    path = groups_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'nota': 'Nomes usados só nos gráficos e exportações da Análise Temporal; '
                'os rótulos das imagens não mudam.',
        'colunas': {'revisado': 'rótulo atual (manual_label)', 'origem': 'pasta de origem (source_label)',
                    'previsao': 'previsão do modelo (preds1)'},
        'nomes': {col: dict(sorted(v.items())) for col, v in clean_groups(groups).items()},
    }
    atomic_write_json(payload, path, ensure_ascii=False, indent=2)
    return path


def count_renamed(groups):
    return sum(len(v) for v in (groups or {}).values())


def map_names(series, mapping):
    """*series* with the display names (unchanged when *mapping* is empty)."""
    if not mapping or series is None or len(series) == 0:
        return series
    text = series.astype(object)
    return text.map(mapping).fillna(text)


def match_matrix(row_values, ref_values):
    """Image counts for every (row name, reference name) pair.

    Returns ``rows`` (names sorted), ``refs`` (reference names sorted),
    ``counts`` {row: {ref: n}}, ``totals`` {row: n} and ``dominant``
    {row: (ref, share)} — the reference class holding most images of the row.
    Blank names are left out.
    """
    frame = pd.DataFrame({'a': row_values, 'b': ref_values})
    tab = frame.groupby(['a', 'b'], observed=True, sort=False, dropna=False).size()
    counts, totals = {}, {}
    refs = set()
    for (a, b), n in tab.items():
        a = '' if pd.isna(a) else str(a).strip()
        b = '' if pd.isna(b) else str(b).strip()
        if not a or not n:
            continue
        totals[a] = totals.get(a, 0) + int(n)
        if b:
            row = counts.setdefault(a, {})
            row[b] = row.get(b, 0) + int(n)
            refs.add(b)
    order = lambda n: (n.casefold(), n)
    rows = sorted(totals, key=order)
    dominant = {}
    for a in rows:
        row = counts.get(a) or {}
        if row:
            best = max(row.items(), key=lambda kv: (kv[1], -len(kv[0])))
            dominant[a] = (best[0], best[1] / totals[a])
    return rows, sorted(refs, key=order), counts, totals, dominant
