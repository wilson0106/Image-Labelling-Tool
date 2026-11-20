#!/usr/bin/env python3
"""IAT › Gallery & labelling (same membership, order, similarity and relabel rules)."""
from __future__ import annotations

import re
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from dash import ALL, ClientsideFunction, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

from iat_backend import SEGMENTATION_METHODS

from .. import components as C
from ..iat_plots import histogram_figure
from .execution import dyn


def _p(lang, pt, en):
    return pt if lang == 'pt' else en


def thumb_url(bn, name, outline=False, method='otsu', size='s'):
    return '/iat-thumb?' + urlencode({'bn': bn, 'name': name, 'o': int(bool(outline)), 'm': method, 's': size})


# Transparent 1×1 GIF shown until the card scrolls near the viewport; the real
# URL sits in data-src and assets/hud.js (IntersectionObserver) swaps it in, so
# a 500-card page requests only the thumbnails that are visible.
LAZY_PLACEHOLDER = 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=='


def _order_value(iat, varname):
    cur = str(iat.vars[varname].get() or '')
    cur = re.sub(r'_(asc|desc)$', '', cur)
    return {'az': 'manual_label', 'var': 'var_norm'}.get(cur, cur)


def render(iat, lang, theme):
    from .iat import iact, iat_input, iat_select, iat_switch
    src = iat._gallery_source()
    page, pages, size = iat._gallery_page_state(src)
    outline = bool(iat.vars['gallery_outline'].get())
    method = iat.vars['seg_method'].get()
    generated = iat.project.available_segmentation_methods() if iat.project else []
    if generated and method not in generated:
        iat.vars['seg_method'].set(generated[0])
        method = generated[0]
    hist = iat._gallery_histogram_source()
    conf_vals = pd.to_numeric(hist['confs1'], errors='coerce').to_numpy(float) if 'confs1' in hist.columns else np.asarray([])
    var_vals = pd.to_numeric(hist['var_norm'], errors='coerce').to_numpy(float) if 'var_norm' in hist.columns else np.asarray([])
    c, v = iat._ranges()
    choices = iat._gallery_available_orders(src)
    ref = iat._similarity_reference_row() if iat.project else None
    sim_status = ''
    if getattr(iat, '_gallery_similarity_active', False) and ref is not None:
        crit = iat.tr('similarity_pca') if iat.vars['similarity_mode'].get() == 'pca' else iat.tr('similarity_tsne')
        sim_status = f"{crit} • {ref.get('names', '')}"

    def order_row(label, varname, dirvar, first):
        opts = ([] if first else [('', '—')]) + [(k, lbl) for lbl, k in choices]
        d = iat.vars[dirvar].get()
        return html.Div([
            html.Span(label, className='hint', style={'width': '28px'}),
            dcc.Dropdown(id={'type': 'ivs', 'key': varname}, options=[{'label': l, 'value': k} for k, l in opts],
                         value=_order_value(iat, varname), clearable=False, className='dd grow'),
            C.btn('↑' if d == 'asc' else '↓', iact(f'toggle_{dirvar}'), 'sm icon'),
        ], className='row nowrap')

    controls = C.hud(iat.tr('allb_title'), [
        html.Div([C.btn(iat.tr('prev'), iact('gal_prev'), 'sm'), C.btn(iat.tr('next'), iact('gal_next'), 'sm')],
                 className='row'),
        html.Div(f"{iat.tr('page')} {iat.page_idx + 1} / {pages}  •  {len(src):,} imagens".replace(',', '.'), className='hint'),
        html.Div([html.Span(_p(lang, 'Imagens por página (máx. 500)', 'Images per page (max 500)'), className='hint'),
                  iat_input(iat, 'page_size', class_name='inp sm', style={'width': '70px'})], className='row nowrap'),
        html.Label(iat.tr('new_label'), className='lbl'),
        iat_input(iat, 'new_label', debounce=False),
        iat_switch(iat, 'save_auto', iat.tr('save_auto')),
        iat_switch(iat, 'discard_after', iat.tr('discard_after')),
        iat_switch(iat, 'hide_relabel', iat.tr('hide_labeled')),
        iat_switch(iat, 'gallery_outline', iat.tr('show_outline')),
        html.Label(iat.tr('gallery_segmentation'), className='lbl'),
        iat_select(iat, 'seg_method', [(m, SEGMENTATION_METHODS.get(m, m)) for m in generated]) if generated
        else C.hint(iat.tr('no_segmentation')),
        C.btn(iat.tr('apply_label'), iact('apply_label'), 'pri'),
        html.Div([C.btn(iat.tr('invert'), iact('invert_checks'), 'sm'), C.btn(iat.tr('undo'), iact('undo_label'), 'sm')],
                 className='row'),
        html.Div([
            html.Div(iat.tr('similarity_title'), className='kicker'),
            html.Div([
                html.Img(src=thumb_url(ref.get('csv_basename', ''), ref.get('names', '')), className='ref-thumb')
                if ref is not None else html.Div('—', className='ref-thumb empty'),
                html.Div([html.Div(iat.tr('similarity_reference'), className='hint'),
                          html.Div(str(ref.get('names', '')) if ref is not None else '—', className='small')]),
            ], className='row nowrap'),
            C.btn(iat.tr('similarity_use_checked'), iact('sim_use_checked'), 'sm'),
            html.Label(iat.tr('similarity_metric'), className='hint'),
            iat_select(iat, 'similarity_mode', [('pca', iat.tr('similarity_pca')), ('tsne', iat.tr('similarity_tsne'))]),
            html.Div([C.btn(iat.tr('similarity_apply'), iact('sim_apply'), 'sm pri'),
                      C.btn(iat.tr('similarity_clear'), iact('sim_clear'), 'sm')], className='row'),
            html.Div(sim_status, className='accent-text small'),
        ], className='subcard'),
        html.Label(iat.tr('filters_order'), className='lbl'),
        html.Span(iat.tr('confidence'), className='hint'),
        dcc.Graph(figure=histogram_figure(conf_vals, c[0], c[1], theme, '#22d3ee'), config={'displayModeBar': False},
                  style={'height': '90px'}),
        html.Div([iat_input(iat, 'conf_min', class_name='inp sm'), iat_input(iat, 'conf_max', class_name='inp sm')],
                 className='row nowrap'),
        html.Span(iat.tr('variance'), className='hint'),
        dcc.Graph(figure=histogram_figure(var_vals, v[0], v[1], theme, '#a78bfa'), config={'displayModeBar': False},
                  style={'height': '90px'}),
        html.Div([iat_input(iat, 'var_min', class_name='inp sm'), iat_input(iat, 'var_max', class_name='inp sm')],
                 className='row nowrap'),
        html.Label(iat.tr('order_priority'), className='lbl'),
        order_row(iat.tr('primary'), 'order', 'order_dir', True),
        order_row(iat.tr('secondary'), 'order2', 'order2_dir', False),
        order_row(iat.tr('tertiary'), 'order3', 'order3_dir', False),
    ], class_name='ctrl-col')

    cards = []
    for rd in page.to_dict('records'):
        gid = str(rd.get('gid'))
        bn, name = str(rd.get('csv_basename', '')), str(rd.get('names', ''))
        on = gid in iat.checked_gids
        conf = rd.get('confs1')
        try:
            conf_txt = f'{float(conf):.2f}'
        except Exception:
            conf_txt = ''
        # No per-card Dash ids: one delegated click handler (assets/hud.js)
        # keeps 500-card pages responsive.
        cards.append(html.Div([
            html.Div([
                html.Img(src=LAZY_PLACEHOLDER, alt=name, **{'data-src': thumb_url(bn, name, outline, method)}),
                html.Div([html.Span(name[:26], className='nm'), html.Code(conf_txt)], className='lb'),
                html.Div(f"manual: {rd.get('manual_label', '')}"[:40], className='ml'),
            ], className='gcard' + (' sel' if on else ''), **{'data-gid': gid, 'data-sel': '1' if on else '0'}),
            html.Button('⤢', className='open', title=iat.tr('open_image'), **{'data-gid': gid}),
        ], className='gwrap'))
    grid = C.hud(iat.tr('gallery'), [
        C.hint(_p(lang, 'Clique na imagem para marcar/desmarcar • Shift+clique marca um intervalo • ⤢ abre em tamanho real',
                  'Click an image to check/uncheck • Shift+click checks a range • ⤢ opens full size')),
        # data-render changes on every redraw so the browser re-applies the
        # server's checked state (data-sel) to the cards it marked locally.
        html.Div(cards, className='gal', **{'data-render': f'{iat.gallery_version}-{time.monotonic_ns()}'})
        if cards else C.hint(iat.tr('no_selection')),
    ], meta=html.Span(f"{len(iat.checked_gids)} {iat.tr('checked')} / {len(page)}", id=dyn('gal-meta')))
    return html.Div([controls, grid], className='g iat-2col')


ACTIONS = {
    'gal_prev': lambda iat, lang: iat.change_page(-1),
    'gal_next': lambda iat, lang: iat.change_page(1),
    'apply_label': lambda iat, lang: iat.apply_label(),
    'invert_checks': lambda iat, lang: iat.invert_checks(),
    'undo_label': lambda iat, lang: iat.undo_label(),
    'sim_use_checked': lambda iat, lang: (iat._set_similarity_reference_from_checked(), _bump(iat)),
    'sim_apply': lambda iat, lang: iat._apply_similarity_sort(),
    'sim_clear': lambda iat, lang: iat._clear_similarity_sort(),
    'toggle_order_dir': lambda iat, lang: iat._toggle_order_direction('order_dir'),
    'toggle_order2_dir': lambda iat, lang: iat._toggle_order_direction('order2_dir'),
    'toggle_order3_dir': lambda iat, lang: iat._toggle_order_direction('order3_dir'),
}


def _bump(iat):
    iat.gallery_version += 1


def full_image_modal(iat, lang, gid):
    from .execution import act
    from .full_image import full_image_payload
    data = full_image_payload(iat, gid)
    if data is None:
        iat._log(iat.tr('no_image'), 'warn')
        return no_update
    from project_settings import format_pixel_size
    um = float(data['um'])
    pt = lambda v: f'{v:,.2f}'.replace(',', ' ').replace('.', ',').replace(' ', '.').rstrip('0').rstrip(',')

    def size_text(px):
        v = px * um
        return f'{pt(v / 1000)} mm' if v >= 1000 else f'{pt(v)} µm'

    tool = lambda label, title, **attrs: html.Button(label, type='button', className='btn sm', title=title, **attrs)
    viewer = html.Div([
        html.Div(html.Img(src=data['url'], className='fi-img', alt=data['name']), className='fi-scroll'),
        html.Div([
            tool('Ajustar', 'Ajustar a imagem à janela', **{'data-fz': 'fit'}),
            tool('1×', 'Tamanho real (1 pixel da imagem = 1 pixel da tela)', **{'data-fz': '1'}),
            tool('2×', 'Ampliar 2 vezes', **{'data-fz': '2'}),
            tool('4×', 'Ampliar 4 vezes', **{'data-fz': '4'}),
            tool('−', 'Diminuir o zoom', **{'data-fzs': '0.8'}),
            html.Span('—', className='fi-zoom'),
            tool('+', 'Aumentar o zoom', **{'data-fzs': '1.25'}),
            html.Button('Barra', type='button', className='btn sm on', title='Mostrar ou ocultar a barra de escala',
                        **{'data-fbar': '1'}),
        ], className='fi-tools'),
        html.Div([html.I(), html.Span('—'), html.Small('')], className='fi-bar'),
    ], className='fi-box', **{'data-um': repr(um), 'data-z': 'fit'})
    body = html.Div([
        viewer,
        html.Div([
            html.H3(iat.tr('metrics')),
            html.Div(f"{iat.tr('class')}: {data['class_name']}", className='ok-text'),
            html.Div(f"{iat.tr('method')}: {data['method_label']}", className='accent-text small'),
            html.Div(f"Pixel size: {format_pixel_size(um).replace('.', ',')} µm/pixel  •  {data['w']} × {data['h']} px"
                     f"  •  {size_text(data['w'])} × {size_text(data['h'])}", className='hint'),
            C.hint('A barra de escala usa este pixel size e acompanha o zoom. Ctrl + roda do mouse também muda o '
                   'zoom. Para mudar o pixel size, edite o campo Pixel size no topo do IAT.'),
            html.Pre('\n'.join(data['metrics']), className='pre-block'),
        ], className='full-side'),
    ], className='full-grid')
    return C.modal_card(f"{iat.tr('full_image')} — {data['name']}", body, act('modal_close'), wide=True)


def register(app, ctl, get_iat, respond):
    # One click from the delegated handler → update checked_gids; the browser
    # then repaints the frames of the visible cards (shift ranges included).
    @app.callback(Output('gsel', 'data'), Input('gclick', 'data'), prevent_initial_call=True)
    def card_click(click):
        if not click or not click.get('gid'):
            raise PreventUpdate
        iat = get_iat()
        with ctl.lock:
            gid = str(click['gid'])
            new_state = gid not in iat.checked_gids
            ordered = iat._gallery_ordered_gids
            if click.get('shift') and iat._gallery_anchor_gid in ordered and gid in ordered:
                a, b = ordered.index(iat._gallery_anchor_gid), ordered.index(gid)
                lo, hi = sorted((a, b))
                if new_state:
                    iat.checked_gids.update(ordered[lo:hi + 1])
                else:
                    iat.checked_gids.difference_update(ordered[lo:hi + 1])
            elif new_state:
                iat.checked_gids.add(gid)
            else:
                iat.checked_gids.discard(gid)
            iat._gallery_anchor_gid = gid
            page = set(map(str, click.get('page') or []))
            checked = [g for g in page if g in iat.checked_gids]
            return {'checked': checked, 'meta': f"{len(iat.checked_gids)} {iat.tr('checked')} / {len(page)}",
                    't': click.get('t')}

    app.clientside_callback(ClientsideFunction('ilt', 'gallerySel'), Output('sink-g', 'data'),
                            Input('gsel', 'data'), prevent_initial_call=True)

    @app.callback(Output('modal-host', 'children', allow_duplicate=True),
                  Input('gopen', 'data'), State('lang-store', 'data'), prevent_initial_call=True)
    def open_image(data, lang):
        if not data or not data.get('gid'):
            raise PreventUpdate
        iat = get_iat()
        with ctl.lock:
            return full_image_modal(iat, lang or 'pt', str(data['gid']))
