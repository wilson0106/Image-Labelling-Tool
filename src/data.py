import json

from ilt_contracts import validate_labels


def read_labels(label_path):
    with open(label_path, encoding='utf-8') as json_file:
        return validate_labels(json.load(json_file), source=label_path)


def label_predictions(batch_df, label_source, audit_source_labels=False):
    """Attach model labels while preserving the original folder-derived label.

    Normal classification keeps the historic behavior: the model prediction is
    the initial ``correct_label``/``manual_label``. In training-audit mode the
    original ``source_label`` becomes the initial human-review label, while
    ``pred``/``preds1`` remain independent model predictions.
    """
    labels_raw = label_source if isinstance(label_source, dict) else read_labels(label_source)
    labels_d = {v: k for k, v in labels_raw.items()}
    batch_df['colors'] = batch_df['pred'] + 1
    batch_df['pred'] = batch_df['pred'].replace(labels_d)

    if 'source_label' not in batch_df.columns and 'klass' in batch_df.columns:
        batch_df['source_label'] = (
            batch_df['klass'].astype(str)
            .str.rstrip('/\\')
            .str.replace(r'^.*[\\/]', '', regex=True)
        )
    if 'source_label' in batch_df.columns:
        batch_df['source_label'] = batch_df['source_label'].fillna('').astype(str)

    if audit_source_labels and 'source_label' in batch_df.columns:
        source = batch_df['source_label'].astype(str)
        valid = source.str.len() > 0
        initial = batch_df['pred'].astype(str).copy()
        initial.loc[valid] = source.loc[valid]
        batch_df['correct_label'] = initial
        batch_df['manual_label'] = initial
        # In audit mode the initial human-review label is the source label, so
        # its color id must follow that label as well (not the model prediction).
        audit_classes = sorted(initial.astype(str).unique().tolist())
        audit_colors = {label: i + 1 for i, label in enumerate(audit_classes)}
        batch_df['colors'] = initial.astype(str).map(audit_colors).astype(int)
        batch_df['audit_source_labels'] = True
    else:
        batch_df['correct_label'] = batch_df['pred']
        batch_df['manual_label'] = batch_df['pred']
        batch_df['audit_source_labels'] = False
