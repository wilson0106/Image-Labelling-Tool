"""Filtros visuais por dados presentes no nome da imagem (sem editar CSVs)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, time

import pandas as pd


@dataclass(frozen=True)
class ImageNameFilter:
    date_from: str = ""
    date_to: str = ""
    time_from: str = ""
    time_to: str = ""
    attribute_regex: str = ""
    attribute_value: str = ""
    depth_min: str = ""
    depth_max: str = ""

    @property
    def enabled(self) -> bool:
        return any((self.date_from, self.date_to, self.time_from, self.time_to,
                    self.attribute_regex, self.attribute_value, self.depth_min, self.depth_max))

    def validate(self) -> None:
        for value in (self.date_from, self.date_to):
            if value:
                date.fromisoformat(value)
        for value in (self.time_from, self.time_to):
            if value:
                time.fromisoformat(value)
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("A data inicial deve ser anterior à data final.")
        # Profundidade é um valor numérico em mm, independente do atributo regex.
        def _mm(value: str) -> float:
            try:
                number = float(value.strip().replace(",", "."))
            except (ValueError, AttributeError) as exc:
                raise ValueError("A profundidade deve ser um número em milímetros.") from exc
            if not math.isfinite(number):
                raise ValueError("A profundidade deve ser um número finito em milímetros.")
            return number
        minimum = _mm(self.depth_min) if self.depth_min.strip() else None
        maximum = _mm(self.depth_max) if self.depth_max.strip() else None
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("A profundidade mínima deve ser menor ou igual à máxima.")
        if self.attribute_regex:
            pattern = re.compile(self.attribute_regex, flags=re.IGNORECASE)
            if self.attribute_value and not pattern.groups:
                raise ValueError("Para filtrar por valor, a expressão deve conter um grupo entre parênteses.")


# Reconhece YYYYMMDD, YYYY-MM-DD, YYYY_MM_DD, YYYY.MM.DD;
# também DD-MM-YYYY, DD_MM_YYYY e DD.MM.YYYY.
_STAMP = re.compile(r"(?<!\d)(20\d{2})([01]\d)([0-3]\d)([01]\d|2[0-3])([0-5]\d)([0-5]\d)(?!\d)")
_ISO_DATE = re.compile(r"(?<!\d)(20\d{2})[-_.]?([01]\d)[-_.]?([0-3]\d)(?!\d)")
_BR_DATE = re.compile(r"(?<!\d)([0-3]\d)[-_.]([01]\d)[-_.](20\d{2})(?!\d)")
# HHMMSS ou HH:MM:SS, HH-MM-SS, HH_MM_SS, HH.MM.SS.
_TIME_MIN = re.compile(r"(?:[Tt_ -])([01]\d|2[0-3])[:h._-]?([0-5]\d)(?!\d)")
_TIME = re.compile(r"(?<!\d)([01]\d|2[0-3])[:h_.-]?([0-5]\d)[:m_.-]?([0-5]\d)(?!\d)")
# Profundidade em mm: sinal + entre underscores, ex.: _+0001537_.
# Exige underscore após o número para não confundir com coordenadas x/y.
_DEPTH_MM = re.compile(r"_(\+\d+)(?=_)")



def filter_image_names(names: pd.Series, spec: ImageNameFilter) -> pd.Series:
    """Máscara alinhada ao índice de names. Falta de data/hora não gera falso positivo."""
    spec.validate()
    selected = pd.Series(True, index=names.index)
    if not spec.enabled:
        return selected

    # Colunas 'names' podem trazer caminhos; avalie somente o nome do arquivo.
    stems = names.fillna("").astype(str).str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    if spec.date_from or spec.date_to:
        iso = stems.str.extract(_ISO_DATE, expand=True)
        stamp = stems.str.extract(_STAMP, expand=True)
        dmy = stems.str.extract(_BR_DATE, expand=True)
        # Monta ISO sem converter cada entrada em objeto Python datetime.
        iso_text = iso[0] + iso[1] + iso[2]
        stamp_text = stamp[0] + stamp[1] + stamp[2]
        br_text = dmy[2] + dmy[1] + dmy[0]
        date_text = iso_text.combine_first(stamp_text).combine_first(br_text)
        days = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
        if spec.date_from:
            selected &= days >= pd.Timestamp(spec.date_from)
        if spec.date_to:
            selected &= days <= pd.Timestamp(spec.date_to)
    if spec.time_from or spec.time_to:
        stamp = stems.str.extract(_STAMP, expand=True)
        # Remove the date first: otherwise 2026-11-03 may resemble 20:26:11.
        time_part = stems.str.replace(_ISO_DATE, "", n=1, regex=True)
        time_part = time_part.str.replace(_BR_DATE, "", n=1, regex=True)
        hm = time_part.str.extract(_TIME, expand=True)
        minutes = time_part.str.extract(_TIME_MIN, expand=True)
        hours = hm[0].combine_first(stamp[3]).combine_first(minutes[0])
        mins = hm[1].combine_first(stamp[4]).combine_first(minutes[1])
        secs_column = hm[2].combine_first(stamp[5]).fillna("00")
        seconds = (pd.to_numeric(hours, errors="coerce") * 3600
                   + pd.to_numeric(mins, errors="coerce") * 60
                   + pd.to_numeric(secs_column, errors="coerce"))
        def secs(value: str) -> int:
            t = time.fromisoformat(value)
            return t.hour * 3600 + t.minute * 60 + t.second
        start = secs(spec.time_from) if spec.time_from else None
        end = secs(spec.time_to) if spec.time_to else None
        if start is not None and end is not None and start > end:
            # Intervalo noturno: por exemplo, 22:00 até 03:00.
            selected &= (seconds >= start) | (seconds <= end)
        else:
            if start is not None:
                selected &= seconds >= start
            if end is not None:
                selected &= seconds <= end
    if spec.depth_min.strip() or spec.depth_max.strip():
        depth_text = stems.str.extract(_DEPTH_MM, expand=False)
        depth_mm = pd.to_numeric(depth_text, errors="coerce")
        if spec.depth_min.strip():
            selected &= depth_mm >= float(spec.depth_min.strip().replace(",", "."))
        if spec.depth_max.strip():
            selected &= depth_mm <= float(spec.depth_max.strip().replace(",", "."))
    if spec.attribute_regex:
        pattern = re.compile(spec.attribute_regex, re.IGNORECASE)
        if spec.attribute_value:
            extracted = stems.str.extract(pattern, expand=True)
            if isinstance(extracted, pd.DataFrame):
                group = "valor" if "valor" in extracted.columns else extracted.columns[0]
                extracted = extracted[group]
            selected &= extracted.fillna("").astype(str).str.casefold() == spec.attribute_value.casefold()
        else:
            selected &= stems.str.contains(pattern, na=False, regex=True)
    elif spec.attribute_value:
        selected &= stems.str.contains(re.escape(spec.attribute_value), case=False, na=False, regex=True)
    return selected.fillna(False).astype(bool)
