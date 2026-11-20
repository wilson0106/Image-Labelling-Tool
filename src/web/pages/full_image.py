#!/usr/bin/env python3
"""Image routes for the IAT web pages: thumbnails and full image with scale bar.

Thumbnails reuse ``IATGalleryMixin._prepare_thumbnail`` (same decode, JPEG
draft and outline drawing).  The full image follows the classic viewer:
outline, conventional scale bar in a footer, metrics from the segmentation
table converted with the authoritative pixel size.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
import uuid
from collections import OrderedDict

import numpy as np
from flask import Response, abort, request

from iat_backend import SEGMENTATION_METHODS

_THUMBS = OrderedDict()
_FULL = OrderedDict()
_LOCK = threading.Lock()
# ~3 KB per JPEG thumbnail: 6000 entries ≈ 18 MB of RAM.
_THUMB_LIMIT = int(os.environ.get('ILT_IAT_THUMB_CACHE', '6000'))


def _put(cache, key, value, limit):
    with _LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)


def _get(cache, key):
    with _LOCK:
        val = cache.get(key)
        if val is not None:
            cache.move_to_end(key)
        return val


def _thumb_etag(iat, bn, name, outline, method):
    try:
        project = iat.project
        if project is None:
            return None
        path = project.resolve_image(bn, name)
        if not path:
            return None
        st = path.stat()
        parts = [str(path), str(st.st_mtime_ns), str(st.st_size), '1' if outline else '0']
        if outline:
            cache = project.segmentation_cache_path(bn, method)
            parts.append(method)
            parts.append(str(cache.stat().st_mtime_ns) if cache.exists() else '-')
        return '"' + hashlib.md5('|'.join(parts).encode('utf-8')).hexdigest() + '"'
    except Exception:
        return None


def register_routes(server, get_iat):
    @server.route('/iat-thumb')
    def _thumb():
        bn, name = request.args.get('bn', ''), request.args.get('name', '')
        outline = request.args.get('o') == '1'
        method = request.args.get('m', 'otsu')
        iat = get_iat()
        # Validator: image file (mtime/size) plus, with an outline, the
        # segmentation cache file.  The browser keeps the thumbnail for a week
        # and revalidates with If-None-Match → 304, so paging back through the
        # Gallery costs no decoding and no transfer.
        etag = _thumb_etag(iat, bn, name, outline, method)
        headers = {'Cache-Control': 'private, max-age=604800'}
        if etag:
            headers['ETag'] = etag
            if request.if_none_match and etag in request.if_none_match:
                return Response(status=304, headers=headers)
        key = (bn, name, outline, method if outline else None, etag)
        data = _get(_THUMBS, key)
        if data is None:
            # Decoding runs in Flask's worker threads (parallel), never under the UI lock.
            im = iat._prepare_thumbnail(bn, name, show_outline=outline, method=method)
            if im is None:
                abort(404)
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=88)
            data = buf.getvalue()
            _put(_THUMBS, key, data, _THUMB_LIMIT)
        return Response(data, mimetype='image/jpeg', headers=headers)

    @server.route('/iat-prev')
    def _preview():
        """Segmentation preview (330×240, optional outline) like the classic tab."""
        from PIL import Image, ImageDraw
        bn, name = request.args.get('bn', ''), request.args.get('name', '')
        outline = request.args.get('o') == '1'
        method = request.args.get('m', 'otsu')
        iat = get_iat()
        if not iat.project:
            abort(404)
        p = iat.project.resolve_image(bn, name)
        if not p:
            abort(404)
        try:
            im = Image.open(p).convert('RGB')
        except Exception:
            abort(404)
        ow, oh = im.size
        im.thumbnail((330, 240), Image.Resampling.LANCZOS)
        if outline:
            ol = iat.project.load_outline_cache(bn, method).get(name)
            if ol and ol.get('points'):
                sx = im.width / max(float(ol.get('width') or ow), 1)
                sy = im.height / max(float(ol.get('height') or oh), 1)
                pts = [(x * sx, y * sy) for x, y in ol['points']]
                if len(pts) > 1:
                    ImageDraw.Draw(im).line(pts + [pts[0]], fill='#ff4f66', width=1)
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        return Response(buf.getvalue(), mimetype='image/png', headers={'Cache-Control': 'no-store'})

    @server.route('/iat-segview')
    def _segview():
        """Live segmentation preview: original | mask | overlay, computed with the
        method and radius chosen now (same ``_threshold_mask`` as the batch run)."""
        bn, name = request.args.get('bn', ''), request.args.get('name', '')
        method = request.args.get('m', 'otsu')
        try:
            radius = int(float(request.args.get('r', '25') or 25))
        except ValueError:
            radius = 25
        iat = get_iat()
        if not iat.project:
            abort(404)
        p = iat.project.resolve_image(bn, name)
        if not p:
            abort(404)
        data = segmentation_view_png(p, method, radius)
        if data is None:
            abort(404)
        return Response(data, mimetype='image/png', headers={'Cache-Control': 'no-store'})

    @server.route('/iat-full/<token>.png')
    def _full(token):
        data = _get(_FULL, token)
        if data is None:
            abort(404)
        return Response(data, mimetype='image/png', headers={'Cache-Control': 'private, max-age=600'})


_SEG_CACHE = OrderedDict()


def segmentation_live(path, method, radius):
    """``(gray, mask)`` for one image with the current parameters (small LRU cache)."""
    import cv2
    from iat_backend import SEGMENTATION_METHODS, _threshold_mask
    method = method if method in SEGMENTATION_METHODS else 'otsu'
    try:
        st = os.stat(path)
    except OSError:
        return None, None
    key = (str(path), st.st_mtime_ns, method, int(radius))
    hit = _get(_SEG_CACHE, key)
    if hit is not None:
        return hit
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, None
    try:
        mask = _threshold_mask(gray, method, int(radius)).astype(bool)
    except Exception:
        mask = np.zeros_like(gray, dtype=bool)
    _put(_SEG_CACHE, key, (gray, mask), 8)
    return gray, mask


def segmentation_view_png(path, method, radius, height=240):
    import cv2
    from PIL import Image
    gray, mask = segmentation_live(path, method, radius)
    if gray is None:
        return None
    color = cv2.imread(str(path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) if color is not None else np.dstack([gray] * 3)
    over = rgb.astype(np.float32)
    red = np.array([255, 48, 72], dtype=np.float32)
    over[mask] = over[mask] * 0.55 + red * 0.45
    over = over.clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thick = max(1, int(round(max(gray.shape) / 250)))
    cv2.drawContours(over, contours, -1, (255, 230, 0), thick)
    mask_img = np.where(mask, 255, 0).astype(np.uint8)
    h, w = gray.shape
    scale = height / max(h, 1)
    size = (max(1, int(round(w * scale))), height)
    panels = [Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS),
              Image.fromarray(mask_img).convert('RGB').resize(size, Image.Resampling.NEAREST),
              Image.fromarray(over).resize(size, Image.Resampling.LANCZOS)]
    gap = 8
    out = Image.new('RGB', (size[0] * 3 + gap * 2, height), (6, 9, 11))
    for i, im in enumerate(panels):
        out.paste(im, (i * (size[0] + gap), 0))
    buf = io.BytesIO()
    out.save(buf, format='PNG')
    return buf.getvalue()


def full_image_payload(iat, gid):
    """Compose the full image exactly like the classic viewer and register it."""
    from PIL import Image, ImageDraw
    df = iat.current_df
    row = None
    if iat._gallery_selection_df is not None and not iat._gallery_selection_df.empty:
        hit = iat._gallery_selection_df[iat._gallery_selection_df['gid'].astype(str) == gid]
        row = hit.iloc[0] if not hit.empty else None
    if row is None and df is not None and not df.empty:
        hit = df[df['gid'].astype(str) == gid]
        row = hit.iloc[0] if not hit.empty else None
    if row is None or not iat.project:
        return None
    bn, name = str(row.get('csv_basename', '')), str(row.get('names', ''))
    p = iat.project.resolve_image(bn, name)
    if not p or not p.exists():
        return None
    im = Image.open(p).convert('RGB')
    ow, oh = im.size
    method = iat.vars['seg_method'].get()
    if iat.vars['gallery_outline'].get():
        ol = iat.project.load_outline_cache(bn, method).get(name)
        if ol and ol.get('points'):
            pts = [(float(x), float(y)) for x, y in ol['points']]
            if len(pts) > 1:
                ImageDraw.Draw(im).line(pts + [pts[0]], fill='#ff3048', width=1)
    fallback = iat._segmentation_pixel_size(None, iat.vars['pixel_size'].get())
    seg_row = None
    try:
        sdf = iat.project.get_segmentation_table(bn, method)
        if not sdf.empty and 'Image Name' in sdf.columns:
            sm = sdf[sdf['Image Name'].astype(str) == name]
            if not sm.empty:
                seg_row = sm.iloc[0]
    except Exception:
        seg_row = None
    um = iat._segmentation_pixel_size(seg_row, fallback)
    # The web viewer draws the scale bar itself (it follows the zoom and the
    # pixel size); the PNG carries only the image and the optional outline.
    combined = im
    class_name = str(row.get('manual_label', '—'))
    try:
        m = df[(df['csv_basename'].astype(str) == bn) & (df['names'].astype(str) == name)]
        if not m.empty:
            class_name = str(m.iloc[0].get('manual_label', '—'))
    except Exception:
        pass
    buf = io.BytesIO()
    combined.save(buf, format='PNG')
    token = uuid.uuid4().hex
    _put(_FULL, token, buf.getvalue(), 12)
    metrics = iat._format_segmentation_metrics(seg_row, um) if seg_row is not None else []
    return {'url': f'/iat-full/{token}.png', 'name': name, 'class_name': class_name,
            'method_label': SEGMENTATION_METHODS.get(method, method), 'um': um, 'w': ow, 'h': oh,
            'metrics': metrics or [iat.tr('no_segmentation')]}
