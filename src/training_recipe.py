#!/usr/bin/env python3
"""Training recipe shared by ``train.py``, the dashboards and the IAT cycle.

A *recipe* groups the choices that change how the network is trained but not
what it is: input pre-processing, augmentation, optimizer, weight decay,
learning-rate schedule, label smoothing and stochastic depth.  The module is
stdlib-only so the interfaces can import it without PyTorch.

Two named presets exist:

* ``DEFAULT_RECIPE`` ("recomendada") — the fine-tuning practice of the ConvNeXt
  paper adapted to small, early-stopped plankton datasets: ImageNet
  normalisation, pad-to-square, rotation/flip/photometric augmentation, AdamW
  (weight decay 0.05, none on norms/biases), linear warmup, discriminative
  learning rate (classifier head 10×), label smoothing 0.1 for cross-entropy,
  stochastic depth 0.1.
* ``LEGACY_RECIPE`` ("legada 1.0") — exactly what ILT ≤ 1.0 did: stretch to
  224×224, no normalisation, no augmentation, Adam, constant learning rate.
  Keep it for controlled comparisons and to resume old checkpoints.

References: Liu et al. 2022 (ConvNeXt, CVPR); Loshchilov & Hutter 2019 (AdamW,
ICLR) and 2017 (SGDR/cosine, ICLR); Goyal et al. 2017 (warmup, arXiv:1706.02677);
Müller, Kornblith & Hinton 2019 (label smoothing, NeurIPS); Huang et al. 2016
(stochastic depth, ECCV); Howard & Ruder 2018 (discriminative fine-tuning, ACL);
Orenstein & Beijbom 2017 (rotation/flip augmentation for plankton, WACV);
Lumini & Nanni 2019 (aspect-preserving input for plankton, Ecol. Inform.).
"""
from __future__ import annotations

RECIPE_KEYS = (
    'inputNormalize', 'inputResize', 'augment', 'optimizer', 'weightDecay',
    'scheduler', 'warmupEpochs', 'headLrMult', 'labelSmoothing', 'stochasticDepth',
)

CHOICES = {
    'inputNormalize': ('imagenet', 'none'),
    'inputResize': ('pad', 'stretch'),
    'optimizer': ('adamw', 'adam'),
    'scheduler': ('plateau', 'cosine', 'none'),
}

DEFAULT_RECIPE = {
    'inputNormalize': 'imagenet',
    'inputResize': 'pad',
    'augment': True,
    'optimizer': 'adamw',
    'weightDecay': 0.05,
    'scheduler': 'plateau',
    'warmupEpochs': 3,
    'headLrMult': 10.0,
    'labelSmoothing': 0.1,
    'stochasticDepth': 0.1,
}

LEGACY_RECIPE = {
    'inputNormalize': 'none',
    'inputResize': 'stretch',
    'augment': False,
    'optimizer': 'adam',
    'weightDecay': 0.0,
    'scheduler': 'none',
    'warmupEpochs': 0,
    'headLrMult': 1.0,
    'labelSmoothing': 0.0,
    'stochasticDepth': 0.1,
}

PRESETS = {'recommended': DEFAULT_RECIPE, 'legacy': LEGACY_RECIPE}

INPUT_SIZE = 224


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 't', 'yes', 'y', 'sim', 's', 'on'}


def _as_float(value, name, lo=None, hi=None) -> float:
    try:
        out = float(str(value).strip().replace(',', '.'))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}: invalid number {value!r}.') from exc
    if lo is not None and out < lo:
        raise ValueError(f'{name}: must be >= {lo}.')
    if hi is not None and out > hi:
        raise ValueError(f'{name}: must be <= {hi}.')
    return out


def normalize_recipe(raw) -> dict:
    """Return a complete, validated recipe (missing keys take the default)."""
    raw = dict(raw or {})
    out = dict(DEFAULT_RECIPE)
    for key, options in CHOICES.items():
        if key in raw and raw[key] is not None:
            value = str(raw[key]).strip().lower()
            if value not in options:
                raise ValueError(f'{key}: expected one of {options}, received {raw[key]!r}.')
            out[key] = value
    if 'augment' in raw and raw['augment'] is not None:
        out['augment'] = _as_bool(raw['augment'])
    if 'weightDecay' in raw and raw['weightDecay'] not in (None, ''):
        out['weightDecay'] = _as_float(raw['weightDecay'], 'weightDecay', 0.0, 1.0)
    if 'warmupEpochs' in raw and raw['warmupEpochs'] not in (None, ''):
        out['warmupEpochs'] = int(_as_float(raw['warmupEpochs'], 'warmupEpochs', 0, 1000))
    if 'headLrMult' in raw and raw['headLrMult'] not in (None, ''):
        out['headLrMult'] = _as_float(raw['headLrMult'], 'headLrMult', 0.01, 1000.0)
    if 'labelSmoothing' in raw and raw['labelSmoothing'] not in (None, ''):
        out['labelSmoothing'] = _as_float(raw['labelSmoothing'], 'labelSmoothing', 0.0, 0.5)
    if 'stochasticDepth' in raw and raw['stochasticDepth'] not in (None, ''):
        out['stochasticDepth'] = _as_float(raw['stochasticDepth'], 'stochasticDepth', 0.0, 0.9)
    return out


def recipe_from_config(cfg) -> dict:
    """Recipe part of a training configuration (``train_params.json``)."""
    return normalize_recipe({k: cfg[k] for k in RECIPE_KEYS if k in (cfg or {})})


def preprocess_of(recipe) -> dict:
    """Input pre-processing that classification must replicate for this model."""
    r = normalize_recipe(recipe)
    return {'normalize': r['inputNormalize'], 'resize': r['inputResize'], 'input_size': INPUT_SIZE}


def same_recipe(a, b) -> bool:
    ra, rb = normalize_recipe(a), normalize_recipe(b)
    for key in RECIPE_KEYS:
        va, vb = ra[key], rb[key]
        if isinstance(va, float) or isinstance(vb, float):
            if abs(float(va) - float(vb)) > 1e-9:
                return False
        elif va != vb:
            return False
    return True


def recipe_preset_name(recipe) -> str:
    """'recommended', 'legacy' or 'custom'."""
    for name, preset in PRESETS.items():
        if same_recipe(recipe, preset):
            return name
    return 'custom'


def describe_recipe(recipe, lang='pt') -> str:
    r = normalize_recipe(recipe)
    name = recipe_preset_name(r)
    label = {'recommended': ('recomendada', 'recommended'), 'legacy': ('legada 1.0', 'legacy 1.0'),
             'custom': ('personalizada', 'custom')}[name][0 if lang == 'pt' else 1]
    return (f"receita {label}: normalize={r['inputNormalize']}, resize={r['inputResize']}, augment={r['augment']}, "
            f"optimizer={r['optimizer']} (wd={r['weightDecay']}), scheduler={r['scheduler']} "
            f"(warmup={r['warmupEpochs']}), head_lr×{r['headLrMult']:g}, label_smoothing={r['labelSmoothing']}, "
            f"stochastic_depth={r['stochasticDepth']}")
