#!/usr/bin/env python3
"""IAT › Clustering (generate / update dirty classes / relabel by group).

Web additions (items 81 and 83): clicking a merge node of the dendrogram sets
the cut just below it, and the groups of the current cut are listed with their
size, label composition and representative thumbnails; a group can be viewed
in full, labelled at once or opened in the Gallery.
"""
from __future__ import annotations

import os
import threading

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from iat_cluster_groups import cluster_group_summary, embedding_index

from .. import components as C
from ..headless import NOTIFY
from ..iat_plots import dendrogram_figure
from ..theme import GRAPH_CONFIG
from .execution import act, dyn


def _p(lang, pt, en):
    return pt if lang == 'pt' else en


def on_show(iat):
    iat.prepare_clustering()
    iat.cluster_view = {'status': _p(iat.lang(), 'Preparando seleção…', 'Preparing selection…')}
    iat.load_cluster_view()


def _bar(iat, key):
    pct = float(iat.cluster_progress_vars[key].get() or 0)
    return html.Div([
        html.Div(html.I(style={'width': f'{pct}%'}), className='bar'),
        html.Div(iat.cluster_status_vars[key].get(), className='hint small'),
    ])


# ---------------- groups of the current cut (computed off the UI lock) ----------------
_GROUPS = {'key': None, 'data': None, 'error': None, 'busy_key': None}
_EMB = {'key': None, 'value': None}
_LOCK = threading.Lock()


def _groups_key(iat):
    view = iat.cluster_view or {}
    path = view.get('path')
    if not path or 'icoord' not in view:
        return None
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return None
    gids = tuple(sorted(str(g) for g in iat._active_cluster_gids()))
    cut = round(float(view.get('cut', iat.vars['cluster_cut'].get())), 4)
    # Any relabel/undo pushes or pops the project's undo stack: the label
    # composition of the groups must then be recomputed.
    stack = getattr(iat.project, 'undo_stack', None) or []
    labels_state = (len(stack), id(stack[-1]) if stack else 0)
    return (str(path), mtime, cut, hash(gids), len(gids), view.get('id_mode') or 'names', labels_state), gids


def ensure_groups(iat, ctl):
    """Return ``(data, error)``; ``data == 'busy'`` while a worker computes it."""
    got = _groups_key(iat)
    if got is None:
        return None, None
    key, gids = got
    with _LOCK:
        if _GROUPS['key'] == key:
            return _GROUPS['data'], _GROUPS['error']
        if _GROUPS['busy_key'] == key:
            return 'busy', None
        _GROUPS['busy_key'] = key
    project = iat.project
    path, mtime, cut, _h, _n, id_mode, _labels = key

    def work():
        data, err = None, None
        try:
            from iat_workspace import IATWorkspace
            rows = iat._cluster_selection_rows(gids=gids, project=project)
            art = IATWorkspace._cached_cluster_artifact(path)
            ekey = (path, mtime, key[3], key[4])
            with _LOCK:
                emb = _EMB['value'] if _EMB['key'] == ekey else None
            if emb is None:
                emb = embedding_index(project, rows)
                with _LOCK:
                    _EMB.update(key=ekey, value=emb)
            data = cluster_group_summary(rows, art, cut, id_mode, embeddings=emb)
        except Exception as exc:  # shown in the panel
            err = f'{type(exc).__name__}: {exc}'
        with _LOCK:
            if _GROUPS['busy_key'] == key:
                _GROUPS.update(key=key, data=data, error=err, busy_key=None)
        with ctl.lock:
            iat.web_version = getattr(iat, 'web_version', 0) + 1

    threading.Thread(target=work, daemon=True, name='iat-cluster-groups').start()
    return 'busy', None


def invalidate_groups():
    with _LOCK:
        _GROUPS.update(key=None, data=None, error=None, busy_key=None)


def _find_group(gid):
    with _LOCK:
        data = _GROUPS['data']
    if not isinstance(data, dict):
        return None
    return next((g for g in data.get('all_groups', []) if int(g['id']) == int(gid)), None)


def _group_card(g, lang):
    from .iat_gallery import LAZY_PLACEHOLDER, thumb_url
    comp = ' · '.join(f'{lab} {100 * f:.0f}%' for lab, f in g['composition'])
    thumbs = [html.Img(src=LAZY_PLACEHOLDER, alt=name, title=name, **{'data-src': thumb_url(bn, name)})
              for bn, name in g['repr']]
    return html.Div([
        html.Div([html.B(_p(lang, f"Grupo {g['id']}", f"Group {g['id']}")),
                  html.Span(f"{g['n']:,} {_p(lang, 'imagens', 'images')}".replace(',', '.'),
                            className='push accent-text small')], className='row nowrap'),
        html.Div(comp, className='hint small'),
        html.Div(html.I(style={'width': f"{100 * g['dominant_frac']:.0f}%"}), className='bar',
                 title=_p(lang, 'Fração do rótulo dominante', 'Dominant label fraction')),
        html.Div(thumbs, className='thumbs'),
        html.Div([C.btn(_p(lang, 'Ver todas', 'View all'), {'type': 'clg', 'act': 'view', 'g': g['id']}, 'sm'),
                  C.btn(_p(lang, 'Abrir na Galeria', 'Open in Gallery'), {'type': 'clg', 'act': 'open', 'g': g['id']},
                        'sm')], className='row'),
    ], className='cl-group')


def groups_panel(iat, ctl, lang):
    data, err = ensure_groups(iat, ctl)
    head = _p(lang, 'Grupos no corte atual', 'Groups at the current cut')
    if data is None and err is None:
        return C.hud(head, C.hint(_p(lang, 'Gere ou carregue um clustering para ver os grupos.',
                                     'Generate or load a clustering to see its groups.')), variant='alt')
    if err:
        return C.hud(head, C.note(err), variant='alt')
    if data == 'busy':
        return C.hud(head, C.hint(_p(lang, 'Calculando grupos e imagens representativas…',
                                     'Computing groups and representative images…')), variant='alt')
    cards = [_group_card(g, lang) for g in data['groups']]
    extra = data['n_groups'] - len(data['groups'])
    notes = [C.hint(_p(lang,
                       'Clique em um nó do dendrograma para cortar logo abaixo dele. As imagens mostradas são as mais '
                       'típicas de cada grupo (maior similaridade de cosseno com a direção média dos embeddings).',
                       'Click a dendrogram node to cut just below it. The images shown are the most typical of each '
                       'group (highest cosine similarity to the mean embedding direction).'))]
    if extra > 0:
        notes.append(C.hint(_p(lang, f'+ {extra} grupo(s) menores não exibidos.', f'+ {extra} smaller group(s) not shown.')))
    return C.hud(head, [*notes, html.Div(cards, className='cl-groups')],
                 meta=f"{data['n_groups']} {_p(lang, 'grupos', 'groups')} • corte {data['cut']:.2f}", variant='alt')


def group_modal(g, lang, limit=300):
    from .iat_gallery import LAZY_PLACEHOLDER, thumb_url
    gids = g['gids']
    shown = gids[:limit]
    thumbs = []
    for gid in shown:
        bn, _sep, cid = gid.partition('::')
        thumbs.append(html.Img(src=LAZY_PLACEHOLDER, alt=cid, title=cid, **{'data-src': thumb_url(bn, cid)}))
    comp = ' · '.join(f'{lab} {100 * f:.1f}%' for lab, f in g['composition'])
    body = html.Div([
        html.Div(comp, className='hint'),
        html.Div(thumbs, className='cl-modal-grid'),
        C.hint(_p(lang, f'Mostrando {len(shown)} de {len(gids)} imagens.', f'Showing {len(shown)} of {len(gids)} images.'))
        if len(gids) > len(shown) else None,
        html.Div([html.Label(_p(lang, 'Rotular todas as imagens do grupo como', 'Label every image of the group as'),
                             className='hint'),
                  dcc.Input(id={'type': 'clg-label', 'k': 'v'}, value='', debounce=False, className='inp',
                            placeholder=g['dominant'])], className='xf-row'),
        C.hint(_p(lang, 'Usa o mesmo procedimento da Galeria (desfazer, salvar automaticamente e marcar classes '
                        'para atualizar o clustering).',
                  'Uses the Gallery procedure (undo, auto-save and dirty-class tracking).')),
    ], className='stack')
    title = _p(lang, f"Grupo {g['id']} • {g['n']} imagens", f"Group {g['id']} • {g['n']} images")
    return C.modal_card(title, body, act('modal_close'), wide=True, actions=[
        C.btn(_p(lang, 'Abrir na Galeria', 'Open in Gallery'), {'type': 'clg', 'act': 'open', 'g': g['id']}),
        C.btn(_p(lang, 'Aplicar rótulo ao grupo', 'Label the group'), {'type': 'clg', 'act': 'apply', 'g': g['id']}, 'pri'),
        C.btn(_p(lang, 'Fechar', 'Close'), act('modal_close'))])


def render(iat, lang, theme):
    from .iat import iact, iat_select
    iat.prepare_clustering()
    view = iat.cluster_view or {}
    n = view.get('n')
    classes = view.get('classes')
    if view.get('missing'):
        msg = (_p(lang, 'Nenhum clustering salvo para esta seleção.\nClique em “Gerar / atualizar cluster”.',
                  'No clustering saved for this selection.\nClick “Generate / update cluster”.')
               if view.get('has_selection') else iat.tr('no_selection'))
        view = dict(view, message=msg)
    elif view.get('status') and 'icoord' not in view:
        view = dict(view, missing=True, message=view['status'])
    cut = float(iat.vars['cluster_cut'].get())
    stats = ''
    if 'icoord' in view:
        stats = (f"{_p(lang, 'Escopo: seleção', 'Scope: selection')}\nN: {view['n']:,}\n"
                 f"{_p(lang, 'Classes', 'Classes')}: {view.get('classes')}\ndist_max: {view['dist_max']:.6g}\n"
                 f"{iat.tr('groups')}: {view['groups']}\n{_p(lang, 'Modo', 'Mode')}: {view.get('mode', '')}"
                 + (_p(lang, ' • visual reduzido', ' • truncated view') if view.get('truncated') else '')).replace(',', '.')
    controls = C.hud(iat.tr('cluster_config'), [
        html.Label(iat.tr('cluster_sel_count'), className='lbl'),
        html.Div(f'{n:,}'.replace(',', '.') if isinstance(n, int) else '—', className='big-count accent'),
        html.Label(iat.tr('cluster_sel_classes'), className='hint'),
        html.Div(f'{classes} classe(s)' if classes is not None else (view.get('status') or '—'), className='hint'),
        html.Div([
            html.Div([html.Label(iat.tr('metric'), className='hint'),
                      iat_select(iat, 'cluster_metric', [('cosine', 'cosine'), ('euclidean', 'euclidean')])], className='grow'),
            html.Div([html.Label(iat.tr('linkage'), className='hint'),
                      iat_select(iat, 'cluster_linkage', [(m, m) for m in ('average', 'complete', 'single', 'ward')])],
                     className='grow'),
        ], className='row nowrap'),
        html.Div([html.Label(iat.tr('cut'), className='lbl'), html.B(f'{cut:.2f}', className='push accent-text')],
                 className='row'),
        dcc.Input(id={'type': 'iv', 'key': 'cluster_cut'}, type='range', min=0, max=1, step=0.01, value=cut,
                  debounce=False),
        C.hint(_p(lang, 'Também é possível clicar em um nó do dendrograma para cortar logo abaixo dele.',
                  'You can also click a dendrogram node to cut just below it.')),
        C.hint(iat.tr('use_selected_scope')),
        C.btn(iat.tr('generate_cluster'), iact('generate_cluster'), 'pri'), _bar(iat, 'generate'),
        C.btn(iat.tr('update_dirty'), iact('update_dirty_clusters')), _bar(iat, 'dirty'),
        C.btn(iat.tr('relabel_group'), iact('relabel_by_group'), 'warnb'), _bar(iat, 'relabel'),
        html.Label(iat.tr('cluster_desc_title'), className='lbl'),
        *[C.hint(iat.tr(k)) for k in ('cluster_desc_generate', 'cluster_desc_dirty', 'cluster_desc_relabel')],
    ], class_name='ctrl-col')
    plot = C.hud(iat.tr('cluster_view'), [
        dcc.Graph(id='cl-dendro', figure=dendrogram_figure(view, theme, lang), config=dict(GRAPH_CONFIG, scrollZoom=True),
                  style={'height': '60vh', 'minHeight': '400px'}),
        html.Pre(stats, className='pre-block'),
    ])
    groups = groups_panel(iat, iat._ctl, lang) if 'icoord' in view else None
    return html.Div([controls, html.Div([plot, groups], className='stack')], className='g iat-2col')


ACTIONS = {
    'generate_cluster': lambda iat, lang: (iat.prepare_clustering(), iat.generate_cluster()),
    'update_dirty_clusters': lambda iat, lang: (iat.prepare_clustering(), iat.update_dirty_clusters()),
    'relabel_by_group': lambda iat, lang: (iat.prepare_clustering(), iat.relabel_by_group()),
}


def _label_group(iat, gids, label):
    """Relabel through the Gallery's own ``apply_label`` (undo, autosave,
    dirty classes, live dataframes), keeping the user's checked images."""
    prev_checked = set(iat.checked_gids)
    prev_label = iat.vars['new_label'].get()
    try:
        iat.checked_gids = set(gids)
        iat.vars['new_label'].set(label)
        iat.apply_label()
    finally:
        iat.checked_gids = prev_checked - set(gids)
        iat.vars['new_label'].set(prev_label)
    invalidate_groups()


def register(app, ctl, get_iat, respond):
    @app.callback(Output(dyn(ALL), 'children', allow_duplicate=True),
                  Input('cl-dendro', 'clickData'), State('lang-store', 'data'), State('theme-store', 'data'),
                  prevent_initial_call=True)
    def dendro_click(click, lang, theme):
        from .iat import versions
        pts = (click or {}).get('points') or []
        if not pts or pts[0].get('y') is None:
            raise PreventUpdate
        y = float(pts[0]['y'])
        iat = get_iat()
        with ctl.lock:
            view = iat.cluster_view or {}
            dmax = float(view.get('dist_max') or 0)
            if dmax <= 0:
                raise PreventUpdate
            before = versions(iat)
            iat.vars['cluster_cut'].set(round(max(0.0, min(1.0, y * 0.999 / dmax)), 4))
            iat._schedule_cluster_view()
            iat.web_version = getattr(iat, 'web_version', 0) + 1
            return respond(iat, before, lang, theme)

    @app.callback(Output(dyn(ALL), 'children', allow_duplicate=True),
                  Output('modal-host', 'children', allow_duplicate=True),
                  Input({'type': 'clg', 'act': ALL, 'g': ALL}, 'n_clicks'),
                  State({'type': 'clg-label', 'k': ALL}, 'value'),
                  State('lang-store', 'data'), State('theme-store', 'data'), prevent_initial_call=True)
    def group_action(_clicks, label_values, lang, theme):
        from .iat import versions
        trig = ctx.triggered_id
        if not isinstance(trig, dict) or not ctx.triggered[0]['value']:
            raise PreventUpdate
        lang = lang or 'pt'
        g = _find_group(trig['g'])
        if g is None:
            NOTIFY.push('info', 'Clustering', _p(lang, 'O grupo mudou; atualize a página.', 'The group changed; refresh.'))
            raise PreventUpdate
        iat = get_iat()
        with ctl.lock:
            before = versions(iat)
            modal = no_update
            if trig['act'] == 'view':
                modal = group_modal(g, lang)
            elif trig['act'] == 'open':
                iat.selected_gids = set(g['gids'])
                iat._selection_outline = None
                iat.build_selection()
                modal = []
            elif trig['act'] == 'apply':
                label = str((label_values or [''])[0] or '').strip()
                if not label:
                    NOTIFY.push('warning', 'Clustering', _p(lang, 'Digite o rótulo do grupo.', 'Type the group label.'))
                    raise PreventUpdate
                _label_group(iat, g['gids'], label)
                modal = []
            iat.web_version = getattr(iat, 'web_version', 0) + 1
            return respond(iat, before, lang, theme), modal
