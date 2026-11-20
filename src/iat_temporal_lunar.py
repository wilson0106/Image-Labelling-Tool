"""Ciclos astronômicos aparentes no local da coleta (somente leitura).

Usa Swiss Ephemeris (pyswisseph, Moshier embutido) para conjunções aparentes
Sol-Lua vistas da latitude/longitude escolhida. O horário dos nomes das imagens
é interpretado no deslocamento UTC informado; não se deduz fuso por longitude.
Não calcula marés ou iluminação local, nem garante visibilidade acima do horizonte.
"""
from __future__ import annotations

from datetime import timedelta
import math
import pandas as pd

PERIOD = "Lunar (astronômico)"


def location_settings(latitude, longitude, utc_offset):
    """Valida coordenadas geográficas e fuso UTC fixo, sem suposições de DST."""
    try:
        lat = float(str(latitude).strip().replace(",", "."))
        lon = float(str(longitude).strip().replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise ValueError("Informe latitude e longitude do local das amostras.") from exc
    if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90
            and -180 <= lon <= 180):
        raise ValueError("Latitude deve estar entre -90 e 90; longitude entre -180 e 180.")
    raw = str(utc_offset).strip().replace(",", ".")
    if not raw:
        raise ValueError("Informe o fuso UTC do horário das imagens (ex.: -3 ou +5.5).")
    try:
        if ":" in raw:
            from datetime import time
            sign = -1 if raw.startswith("-") else 1
            value = raw.lstrip("+-")
            hour, minute = map(int, value.split(":"))
            offset = sign * (hour + minute / 60)
        else:
            offset = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fuso UTC inválido. Ex.: -3, -03:00, +5.5.") from exc
    if not math.isfinite(offset) or abs(offset) > 14:
        raise ValueError("O deslocamento UTC deve estar entre -14 e +14 horas.")
    return lat, lon, offset


def _swisseph():
    try:
        import swisseph as swe
    except ImportError as exc:
        raise ValueError(
            "Para o calendário lunar astronômico, instale pyswisseph no ambiente ILT "
            "(python -m pip install pyswisseph)."
        ) from exc
    return swe


def _jd(ts, swe):
    ts = pd.Timestamp(ts)
    hours = ts.hour + ts.minute / 60 + ts.second / 3600 + ts.microsecond / 3.6e9
    return swe.julday(ts.year, ts.month, ts.day, hours)


def _local_time(jd, swe, offset):
    year, month, day, hour = swe.revjul(float(jd), swe.GREG_CAL)
    return pd.Timestamp(year=year, month=month, day=day) + pd.Timedelta(hours=hour + offset)


def _angle(jd, swe):
    flags = swe.FLG_MOSEPH | swe.FLG_TOPOCTR
    moon = swe.calc_ut(jd, swe.MOON, flags)[0][0]
    sun = swe.calc_ut(jd, swe.SUN, flags)[0][0]
    return (moon - sun) % 360


def _conjunctions(low_jd, high_jd, lat, lon, swe):
    """Retorna instantes em que a elongação topocêntrica cruza 360 -> 0."""
    swe.set_topo(lon, lat, 0)
    crosses = []
    a = low_jd
    previous = _angle(a, swe)
    # Passos menores que um dia evitam saltar uma conjunção.
    while a < high_jd:
        b = min(a + .75, high_jd)
        current = _angle(b, swe)
        if previous > 270 and current < 90:
            left, right = a, b
            for _ in range(35):
                mid = (left + right) / 2
                if _angle(mid, swe) > 180:
                    left = mid
                else:
                    right = mid
            crosses.append((left + right) / 2)
        a, previous = b, current
    return crosses


def lunar_intervals(df, lower, upper, step, latitude, longitude, utc_offset, cap=240):
    """Intervalos Lua Nova -> Lua Nova (topocêntrica), nos horários locais de coleta.

    Os limites são astronômicos, sem a aproximação fixa de 29,53 dias.
    Cada quadro recebe as imagens do intervalo real sem duplicação.
    """
    if df.empty:
        return []
    lat, lon, offset = location_settings(latitude, longitude, utc_offset)
    try:
        cycles = int(step)
    except (TypeError, ValueError) as exc:
        raise ValueError("O passo lunar deve ser um número inteiro positivo de ciclos.") from exc
    if not 1 <= cycles <= 24:
        raise ValueError("O passo lunar deve estar entre 1 e 24 ciclos.")
    timestamps = df.get("timestamp")
    if timestamps is None:
        timestamps = df["day"] + pd.to_timedelta(df["minute"], unit="m")
    valid = timestamps.dropna()
    if valid.empty:
        return []
    try:
        lo = pd.Timestamp(lower).normalize() if str(lower).strip() else valid.min().normalize()
        hi = pd.Timestamp(upper).normalize() if str(upper).strip() else valid.max().normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("Data inválida. Use AAAA-MM-DD.") from exc
    if hi < lo:
        raise ValueError("Data final deve ser igual ou posterior à inicial.")
    end = hi + pd.Timedelta(days=1)
    swe = _swisseph()
    utc_lo = lo - pd.Timedelta(hours=offset)
    utc_hi = end - pd.Timedelta(hours=offset)
    moon_jds = _conjunctions(_jd(utc_lo, swe) - 40, _jd(utc_hi, swe) + 40 + 29.7*cycles,
                             lat, lon, swe)
    if len(moon_jds) < 2:
        raise ValueError("A efeméride lunar não encontrou conjunções para o período.")
    moons = [_local_time(jd, swe, offset) for jd in moon_jds]
    result = []
    i = max(0, next((k for k in range(len(moons)-1)
                     if moons[k] <= lo < moons[k+1]), 0))
    while i + cycles < len(moons) and moons[i] < end:
        if len(result) >= cap:
            raise ValueError("Mais de 240 quadros; aumente o passo lunar.")
        a, b = moons[i], moons[i+cycles]
        low, high = max(a, lo), min(b, end)
        mask = (timestamps >= low) & (timestamps < high)
        label = f"Lua nova {a:%d/%m/%Y %H:%M} → {b:%d/%m/%Y %H:%M}"
        result.append((label, mask.fillna(False)))
        i += cycles
    return result
