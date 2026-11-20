import os

import numpy as np
import openTSNE
import pandas as pd
from PIL import ImageFile

from aux import defaults
from ilt_contracts import atomic_write_csv, safe_perplexity
from hw_resources import cpu_jobs


def margin_uncertainty(conf_top1, conf_top2):
    """Prediction uncertainty from the top-1/top-2 softmax margin, in [0, 1].

    ``u = 1 - (p1 - p2)``: 0 when the model is certain (p1 = 1), 1 when the two
    best classes are tied.  This is the classic *margin sampling* criterion
    (Settles 2009, *Active Learning Literature Survey*, Univ. Wisconsin-Madison
    CS Tech. Rep. 1648; Scheffer, Decomain & Wrobel 2001, IDA).  It is stored
    in the ``var_norm`` column, which the IAT uses as "uncertainty" for
    filters, histograms and the review priority.  Models with a single class
    repeat top-1 as top-2, giving u = 1 (no decision possible).
    """
    p1=np.asarray(conf_top1,dtype=np.float32)
    p2=np.asarray(conf_top2,dtype=np.float32)
    return np.clip(1.0-(p1-p2),0.0,1.0).astype(np.float32)


def opentsne_fit(features, n=2):
    tsne=openTSNE.TSNE(
        n_components=n,
        perplexity=safe_perplexity(len(features)),
        initialization='pca',
        metric='cosine',
        random_state=0,
        # openTSNE defaults to a single thread; kNN search and gradient
        # evaluation scale almost linearly with CPU cores.
        n_jobs=cpu_jobs(),
    )
    ImageFile.LOAD_TRUNCATED_IMAGES=True
    return tsne.fit(features)


def opentsne_transform(features, base_tsne):
    return base_tsne.transform(features)


def compute_projections(
    output_path, project_name, batch_id, features, path_images, df_batches,
    predictions, confs, base_tsne=None, compute_base=False, save=True
):
    """Project one batch and optionally write its IAT dataframe.

    Numeric columns are built directly as numeric arrays instead of first
    combining them with filenames into one string/object ndarray. This keeps the
    per-batch temporary footprint smaller and avoids needless dtype conversions.
    """
    if compute_base:
        base_tsne=opentsne_fit(features)
        projection=np.asarray(base_tsne, dtype=np.float64)
    else:
        projection=np.asarray(opentsne_transform(features,base_tsne),dtype=np.float64)

    names=np.asarray(path_images,dtype=object)
    preds=np.asarray(predictions,dtype=np.int32)
    if projection.ndim!=2 or projection.shape[1]<2:
        raise ValueError(f'Projection must be Nx2; got shape={projection.shape}')
    if not (len(names)==len(projection)==len(preds)==len(confs)):
        raise ValueError(
            'Projection inputs have inconsistent lengths: '
            f'names={len(names)}, projection={len(projection)}, '
            f'predictions={len(preds)}, confidences={len(confs)}'
        )

    confs1=np.asarray([c['top1'][1] for c in confs],dtype=np.float32)
    confs2=np.asarray([c['top2'][1] for c in confs],dtype=np.float32)
    df_preds=pd.DataFrame({
        'names':names,
        'x':projection[:,0],
        'y':projection[:,1],
        'pred':preds,
        'preds1':[c['top1'][0] for c in confs],
        'confs1':confs1,
        'preds2':[c['top2'][0] for c in confs],
        'confs2':confs2,
        'preds3':[c['top3'][0] for c in confs],
        'confs3':np.asarray([c['top3'][1] for c in confs],dtype=np.float32),
        # Uncertainty (top-1 − top-2 margin); see margin_uncertainty().
        'var_norm':margin_uncertainty(confs1,confs2),
    })

    manifest=df_batches
    if 'batch' in manifest.columns:
        # main.py already supplies a single-batch manifest; keep compatibility
        # with callers that still pass the complete batches.csv dataframe.
        batch_values=manifest['batch'].astype(str)
        if batch_values.nunique(dropna=True)>1 or (len(manifest) and batch_values.iloc[0]!=str(batch_id)):
            manifest=manifest.loc[batch_values==str(batch_id)]
    df=pd.merge(df_preds,manifest,on='names',how='inner',validate='one_to_one')

    if len(df)!=len(df_preds):
        missing=set(df_preds['names'].astype(str))-set(df['names'].astype(str))
        sample=', '.join(sorted(missing)[:5])
        raise RuntimeError(
            f'{batch_id}: projection/manifest mismatch; {len(missing)} image(s) '
            f'were not matched. Examples: {sample}'
        )

    if save:
        dataframes_folder=os.path.join(output_path,defaults['dataframes'])
        os.makedirs(dataframes_folder,mode=0o755,exist_ok=True)
        atomic_write_csv(
            df,
            os.path.join(dataframes_folder,f'{batch_id}_{project_name}.csv'),
            index=False,
        )

    return base_tsne
