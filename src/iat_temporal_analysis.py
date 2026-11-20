"""Análise exploratória, somente leitura, de variáveis extraídas de nomes de imagens."""
from __future__ import annotations
import math, re
import numpy as np
import pandas as pd
from iat_filename_filters import (ImageNameFilter, filter_image_names,
    _ISO_DATE, _STAMP, _BR_DATE, _TIME, _TIME_MIN, _DEPTH_MM)

FIELDS={'names','x','y','manual_label','correct_label','preds1','feature_version','embeddings_version','model_version',
        'feature_model_version','embed_file','embed_idx'}
POINT_COLUMNS=['names','x','y','classe','versao','batch_source','embed_file','embed_idx']
# Extra columns of load_points(with_origin=True): folder class and top-1 prediction.
ORIGIN_FIELDS={'source_label','klass'}
ORIGIN_COLUMNS=['origem','previsto']

# Coordinate systems the temporal view can display.  'original' = the x/y of
# the project CSVs (openTSNE frame of the classification); the others are
# computed from the stored embeddings of every loaded image (see
# temporal_projection) and cached in <project>/projections/.
PROJECTION_METHODS={
    'original':'Original (t-SNE do projeto)',
    'pca':'PCA 2D',
    'tsne':'t-SNE recalculado (openTSNE)',
    'umap':'UMAP',
    'umap_supervised':'UMAP supervisionado (rótulos manuais)',
    'pacmap':'PaCMAP',
}
PROJECTION_AXES={
    'original':('t-SNE 1','t-SNE 2'),'pca':('PC 1','PC 2'),'tsne':('t-SNE 1','t-SNE 2'),
    'umap':('UMAP 1','UMAP 2'),'umap_supervised':('UMAP sup. 1','UMAP sup. 2'),'pacmap':('PaCMAP 1','PaCMAP 2'),
}

def load_points(project, progress=None, with_origin=False):
    """Points of every batch.  ``with_origin=True`` (web panel "Origem → Previsão")
    also returns ``origem`` (folder class, ``source_label``/``klass``) and
    ``previsto`` (``preds1``) as categorical columns; the default output is
    unchanged."""
    fields=FIELDS|ORIGIN_FIELDS if with_origin else FIELDS
    out_cols=POINT_COLUMNS+(ORIGIN_COLUMNS if with_origin else [])
    parts=[]
    names=list(project.csv_names())
    for i,bn in enumerate(names,1):
        cached=getattr(project,'batch_cache',{}).get(bn)
        if cached is None:
            path=project.dataframes_dir/bn
            cols=[c for c in pd.read_csv(path,nrows=0).columns if c in fields]
            if not {'names','x','y'}<=set(cols):
                continue
            df=pd.read_csv(path,usecols=cols,low_memory=False)
        else:
            df=cached[[c for c in cached.columns if c in fields]].copy()
        if not {'names','x','y'}<=set(df.columns): continue
        df['x']=pd.to_numeric(df['x'],errors='coerce')
        df['y']=pd.to_numeric(df['y'],errors='coerce')
        df=df.loc[np.isfinite(df['x']) & np.isfinite(df['y'])].copy()
        label=next((s for s in ('manual_label','correct_label','preds1') if s in df),None)
        version=next((s for s in ('feature_model_version','feature_version','embeddings_version','model_version') if s in df),None)
        df['classe']=df[label].fillna('').astype(str).str.strip() if label else 'Sem classe'
        df['classe']=df['classe'].replace('','Sem classe')
        df['versao']=df[version].fillna('').astype(str).str.strip() if version else 'Sem versão'
        df['versao']=df['versao'].replace('','Sem versão')
        df['batch_source']=bn
        for col in ('embed_file','embed_idx'):
            if col not in df.columns:
                df[col]=np.nan
        if with_origin:
            from iat_origin_flow import origin_labels
            df['origem']=origin_labels(df)
            df['previsto']=(df['preds1'].fillna('').astype(str).str.strip() if 'preds1' in df.columns
                            else pd.Series('',index=df.index,dtype=object))
        parts.append(df[out_cols])
        if progress: progress(i,len(names),bn)
    if not parts:
        return pd.DataFrame(columns=out_cols+['day','minute','depth','timestamp'])
    joined=pd.concat(parts,ignore_index=True,copy=False)
    if with_origin:
        # Few distinct names: categorical keeps memory and grouping cheap.
        for col in ORIGIN_COLUMNS:
            joined[col]=joined[col].astype('category')
    return parse_metadata(joined)


def frame_class_counts(view, frames):
    """Images per class (columns) in each interval (rows, in frame order)."""
    if view is None or view.empty or not frames:
        return pd.DataFrame()
    classes=sorted(view['classe'].astype(str).unique().tolist())
    data={}
    for label,mask in frames:
        sel=np.asarray(mask.fillna(False),dtype=bool) if hasattr(mask,'fillna') else np.asarray(mask,dtype=bool)
        counts=view.loc[sel,'classe'].astype(str).value_counts()
        data[str(label)]=counts.reindex(classes,fill_value=0).astype(int)
    out=pd.DataFrame(data,index=classes).T
    out.index.name='intervalo'
    return out


def diversity_indices(counts):
    """Per-interval diversity of the image counts.

    N = images; S = richness (classes present); H' = Shannon entropy (nats);
    N1 = exp(H') and N2 = 1/Σp² = Hill numbers of order 1 and 2 (effective
    number of classes); J = H'/ln S = Pielou evenness.  Counts of images are
    not abundance normalised by sampling effort (see docs).
    Shannon 1948; Simpson 1949; Pielou 1966; Hill 1973; Jost 2006 (Ecology
    87:2427-2439).
    """
    if counts is None or counts.empty:
        return pd.DataFrame(columns=['N','S','H','N1','N2','J'])
    c=counts.astype(float)
    n=c.sum(axis=1)
    p=c.div(n.replace(0,np.nan),axis=0)
    with np.errstate(divide='ignore',invalid='ignore'):
        plogp=p*np.log(p)
    h=-(plogp.fillna(0.0)).sum(axis=1)
    s=(c>0).sum(axis=1)
    d=(p**2).sum(axis=1)
    out=pd.DataFrame({'N':n.astype(int),'S':s.astype(int),'H':h,'N1':np.exp(h),
                      'N2':1.0/d.replace(0,np.nan)},index=counts.index)
    out['J']=np.where(s>1,h/np.log(s.where(s>1,2)),np.nan)
    out.loc[n<=0,['H','N1','N2','J']]=np.nan
    return out


def _projection_cache_key(method, keys, labels):
    import hashlib
    h=hashlib.sha1()
    h.update(str(method).encode('utf-8'))
    for k in keys:
        h.update(str(k).encode('utf-8')); h.update(b'\n')
    if labels is not None:
        h.update(b'--labels--')
        for lab in labels:
            h.update(str(lab).encode('utf-8')); h.update(b'\n')
    return h.hexdigest()


def temporal_projection(project, df, method, progress=None, pca_dim=50, use_cache=True):
    """2-D coordinates of every loaded image by *method*, from the stored embeddings.

    Returns ``(coords, note)`` with ``coords`` an (N, 2) array aligned to
    ``df`` (NaN where the image has no embedding).  Pipeline: L2-normalised
    768-D embeddings → PCA to *pca_dim* (incremental over the embedding files
    when there are more than 60 000 images) → projection: PCA 2-D; openTSNE
    (cosine, PCA init, all cores); UMAP (n_neighbors 30, min_dist 0.1);
    UMAP supervised with the current manual labels (target_weight 0.35) — the
    labels pull same-class images together, so it shows the structure *given*
    the annotation and must not be used to look for annotation errors; PaCMAP.
    Results are cached in ``<project>/projections/`` keyed by method, image
    set and (for the supervised variant) labels.
    McInnes, Healy & Melville 2018 (UMAP, arXiv:1802.03426); Poličar, Stražar
    & Zupan 2019 (openTSNE); Wang et al. 2021 (PaCMAP, JMLR 22:1-73).
    """
    from iat_backend import _project_validation_space, _pca_for_validation, _l2, load_feats, resolve_embed
    method=str(method or 'original')
    if method not in PROJECTION_METHODS or method=='original':
        raise ValueError('Escolha um método de projeção diferente do original.')
    if df is None or df.empty:
        raise ValueError('Carregue a projeção do projeto primeiro.')
    if 'embed_file' not in df.columns or 'embed_idx' not in df.columns:
        raise ValueError('Os CSVs deste projeto não têm referência aos embeddings (embed_file/embed_idx).')
    idx_num=pd.to_numeric(df['embed_idx'],errors='coerce')
    valid=(idx_num>=0)&df['embed_file'].notna()
    n_valid=int(valid.sum())
    if n_valid<10:
        raise ValueError('Menos de 10 imagens com embeddings; a projeção não pode ser calculada.')
    keys=(df.loc[valid,'batch_source'].astype(str)+'::'+df.loc[valid,'names'].astype(str)).to_numpy(object)
    labels=df.loc[valid,'classe'].astype(str).to_numpy(object) if method=='umap_supervised' else None
    cache_dir=project.project_dir/'projections'
    key=_projection_cache_key(method,keys,labels)
    cache=cache_dir/f'temporal_{method}_{key[:16]}.npz'
    coords=np.full((len(df),2),np.nan,dtype=float)
    pos=np.flatnonzero(valid.to_numpy())
    if use_cache and cache.exists():
        try:
            with np.load(cache,allow_pickle=False) as z:
                if len(z['x'])==n_valid and str(z['key'])==key:
                    coords[pos,0]=z['x']; coords[pos,1]=z['y']
                    if progress: progress(1,1,'Projeção carregada do cache')
                    return coords,f"{PROJECTION_METHODS[method]} • {n_valid:,} imagens • cache".replace(',','.')
        except Exception:
            pass
    sub=df.loc[valid,['embed_file','embed_idx']].copy()
    sub['embed_idx']=idx_num[valid].astype(int).to_numpy()
    files=sub['embed_file'].astype(str).to_numpy(object)
    idxs=sub['embed_idx'].to_numpy()
    groups={}
    for i,(ef,ix) in enumerate(zip(files,idxs)):
        groups.setdefault(ef,[]).append((i,int(ix)))
    n_files=len(groups)
    big=n_valid>60000
    pca=None
    if big:
        from sklearn.decomposition import IncrementalPCA
        pca=IncrementalPCA(n_components=int(pca_dim))
        for j,(ef,items) in enumerate(groups.items(),1):
            feats=load_feats(resolve_embed(project,ef))
            rows=np.array([ix for _,ix in items if 0<=ix<feats.shape[0]],dtype=int)
            if len(rows)>=int(pca_dim):
                pca.partial_fit(_l2(feats[rows].astype(np.float32)))
            if progress: progress(j,2*n_files,f'PCA incremental {j}/{n_files}')
    xstat=np.full((n_valid,int(pca_dim) if big else 768),np.nan,dtype=np.float32)
    dims=None
    for j,(ef,items) in enumerate(groups.items(),1):
        feats=load_feats(resolve_embed(project,ef))
        for i,ix in items:
            if 0<=ix<feats.shape[0]:
                vec=_l2(feats[ix:ix+1].astype(np.float32))
                if big:
                    xstat[i]=pca.transform(vec)[0]
                else:
                    if dims is None:
                        dims=vec.shape[1]
                        if dims!=xstat.shape[1]:
                            xstat=np.full((n_valid,dims),np.nan,dtype=np.float32)
                    xstat[i]=vec[0]
        if progress: progress((n_files if big else 0)+j,(2*n_files if big else n_files)+1,f'Embeddings {j}/{n_files}')
    ok=np.isfinite(xstat).all(axis=1)
    if int(ok.sum())<10:
        raise ValueError('Embeddings insuficientes ou inválidos para a projeção.')
    x_in=xstat[ok]
    if not big:
        x_in=_pca_for_validation(x_in,int(pca_dim))
    if progress: progress(1,1,f'{PROJECTION_METHODS[method]}: {int(ok.sum()):,} pontos… (pode levar minutos)'.replace(',','.'))
    if method=='tsne':
        from projections import opentsne_fit
        emb=np.asarray(opentsne_fit(x_in),dtype=float)
    else:
        emb=np.asarray(_project_validation_space(x_in,method,labels[ok] if labels is not None else None),dtype=float)
    coords[pos[ok],0]=emb[:,0]; coords[pos[ok],1]=emb[:,1]
    try:
        cache_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache,x=coords[pos,0],y=coords[pos,1],key=np.array(key),method=np.array(method),
                            n=np.int64(n_valid),created=np.array(pd.Timestamp.now().isoformat()))
    except Exception:
        pass
    skipped=n_valid-int(ok.sum())
    note=f"{PROJECTION_METHODS[method]} • {int(ok.sum()):,} imagens".replace(',','.')
    if skipped: note+=f' • {skipped} sem embedding'
    if method=='umap_supervised': note+=' • usa os rótulos manuais atuais (não serve para procurar erros de rótulo)'
    return coords,note

def parse_metadata(df):
    out=df.copy()
    stems=out['names'].fillna('').astype(str).str.replace('\\','/',regex=False).str.rsplit('/',n=1).str[-1]
    iso=stems.str.extract(_ISO_DATE)
    stamp=stems.str.extract(_STAMP)
    br=stems.str.extract(_BR_DATE)
    days=(iso[0]+iso[1]+iso[2]).combine_first(stamp[0]+stamp[1]+stamp[2]).combine_first(br[2]+br[1]+br[0])
    out['day']=pd.to_datetime(days,format='%Y%m%d',errors='coerce')
    trimmed=stems.str.replace(_ISO_DATE,'',n=1,regex=True).str.replace(_BR_DATE,'',n=1,regex=True)
    full=trimmed.str.extract(_TIME)
    half=trimmed.str.extract(_TIME_MIN)
    h=pd.to_numeric(full[0].combine_first(stamp[3]).combine_first(half[0]),errors='coerce')
    m=pd.to_numeric(full[1].combine_first(stamp[4]).combine_first(half[1]),errors='coerce')
    out['minute']=h*60+m
    out['depth']=pd.to_numeric(stems.str.extract(_DEPTH_MM,expand=False),errors='coerce')
    # Instante de coleta: preservar segundos/milissegundos do nome da imagem.
    # Essencial quando a conjunção lunar cai no meio de um minuto.
    secs=pd.to_numeric(full[2].combine_first(stamp[5]),errors='coerce').fillna(0)
    fractions=stems.str.extract(
        r"(?<!\d)(?:[01]\d|2[0-3])[_:.\-][0-5]\d[_:.\-][0-5]\d\.(\d{1,6})(?=_|\.|$)",
        expand=False)
    fractional=pd.to_numeric(fractions.fillna("0").str.ljust(6,"0"),
                             errors="coerce").fillna(0)/1e6
    out['timestamp']=(out['day']+pd.to_timedelta(out['minute'],unit='m')
                       +pd.to_timedelta(secs+fractional,unit='s'))
    return out

def captured_attribute(names, pattern):
    """Read a named token from the filename; plain prefixes require no regex.

    A plain prefix such as 'estacao' finds '_estacaoA_' and '_estacaoB_'.
    The legacy capturing-regex calling convention is kept for compatibility.
    """
    prefix=str(pattern or '').strip()
    if not prefix:
        raise ValueError('Informe o prefixo do atributo no nome da imagem (ex.: estacao).')
    if re.fullmatch(r'[\w-]+',prefix,re.UNICODE):
        compiled=re.compile(r'(?:^|_)(' + re.escape(prefix) + r'[^_.]*)(?=_|\.|$)',
                            re.IGNORECASE)
    else:
        try:compiled=re.compile(prefix,re.IGNORECASE)
        except re.error as exc:raise ValueError('Padrão de atributo inválido: '+str(exc)) from exc
        if not compiled.groups:
            raise ValueError('O padrão antigo deve conter um grupo de captura.')
    stems=names.fillna('').astype(str).str.replace('\\','/',regex=False).str.rsplit('/',n=1).str[-1]
    found=stems.str.extract(compiled)
    key='valor' if 'valor' in found else found.columns[0]
    return found[key].fillna('').astype(str).str.strip()

def _calendar_floor(date, period):
    date = pd.Timestamp(date)
    if period == "Semanal":
        return date.normalize()-pd.Timedelta(days=date.weekday())
    if period == "Mensal":
        return pd.Timestamp(date.year,date.month,1)
    if period == "Trimestral":
        return pd.Timestamp(date.year,((date.month-1)//3)*3+1,1)
    if period == "Anual":
        return pd.Timestamp(date.year,1,1)
    return date.normalize()

def _calendar_next(start, period, unit_count):
    if period == "Horário":
        return start+pd.Timedelta(hours=unit_count)
    if period == "Diário":
        return start+pd.Timedelta(days=unit_count)
    if period == "Semanal":
        return start+pd.Timedelta(weeks=unit_count)
    if period == "Mensal":
        return start+pd.DateOffset(months=unit_count)
    if period == "Trimestral":
        return start+pd.DateOffset(months=3*unit_count)
    if period == "Anual":
        return start+pd.DateOffset(years=unit_count)
    raise ValueError("Período não reconhecido.")

def calendar_intervals(df, lower, upper, step, period, cap=240, lunar_site=None):
    """Recortes sem sobreposição; meses/anos obedecem o calendário real.

    O ciclo lunar astronômico utiliza efemérides e a localização informada.
    """
    if df.empty:
        return []
    valid={"Horário","Diário","Semanal","Mensal","Trimestral","Anual",
}
    if period == "Lunar (astronômico)":
        if not lunar_site:
            raise ValueError("Para analisar ciclos lunares, informe latitude, longitude e fuso UTC.")
        from iat_temporal_lunar import lunar_intervals
        return lunar_intervals(df, lower, upper, step, *lunar_site, cap=cap)
    if period not in valid:
        raise ValueError("Selecione um período válido.")
    try:
        unit_count = int(step)
    except (TypeError,ValueError) as exc:
        raise ValueError("O passo deve ser um número inteiro positivo.") from exc
    if unit_count <= 0:
        raise ValueError("O passo deve ser um número inteiro positivo.")
    dates=df["day"].dropna()
    if dates.empty:
        return []
    try:
        lo=pd.Timestamp(lower).normalize() if str(lower).strip() else dates.min().normalize()
        hi=pd.Timestamp(upper).normalize() if str(upper).strip() else dates.max().normalize()
    except (ValueError,TypeError) as exc:
        raise ValueError("Data inválida. Use AAAA-MM-DD.") from exc
    if hi < lo:
        raise ValueError("Data final deve ser igual ou posterior à inicial.")
    # Um limite de data inclui até 23:59 do último dia.
    end=hi+pd.Timedelta(days=1)
    start=_calendar_floor(lo,period)
    if period=="Horário":
        # Agrupar a cada N horas, não recalcular a posição de cada imagem.
        timestamps=df["timestamp"] if "timestamp" in df else (
            df["day"]+pd.to_timedelta(df["minute"],unit="m"))
    else:
        timestamps=df["day"]
    result=[]
    a=start
    while a < end:
        if len(result)>=cap:
            raise ValueError("Mais de 240 quadros. Aumente o passo ou reduza o período.")
        b=_calendar_next(a,period,unit_count)
        if b<=a:
            raise ValueError("Passo inválido.")
        from_time=max(a,lo)
        until_time=min(b,end)
        mask=(timestamps>=from_time)&(timestamps<until_time)
        if period=="Horário":
            label=f"{from_time:%d/%m/%Y %H:%M}–{until_time:%d/%m/%Y %H:%M}"
        elif period=="Mensal" and unit_count==1:
            label=f"{a:%m/%Y}"
        elif period=="Anual" and unit_count==1:
            label=str(a.year)
        else:
            label=f"{from_time:%d/%m/%Y} a {(until_time-pd.Timedelta(microseconds=1)):%d/%m/%Y}"
            if period=="Lunar (~29,53 dias)":
                label+=" (ciclo aprox.)"
        result.append((label,mask))
        a=b
    return result


def intervals(df, axis, lower='', upper='', step='1', regex='', cap=240, period=None, lunar_site=None):
    if df.empty: return []
    result=[]
    if axis=='Atributo':
        values=captured_attribute(df['names'],regex)
        labels=sorted(set(values)-{''})
        if len(labels)>cap: raise ValueError('Muitos valores do atributo; escolha um prefixo mais específico.')
        return [(v,values.eq(v)) for v in labels]
    if axis=='Data':
        return calendar_intervals(df, lower, upper, step, period or 'Diário', cap, lunar_site=lunar_site)
    elif axis=='Hora':
        from datetime import time
        def minutes(s,default):
            if not s.strip():return default
            tt=time.fromisoformat(s.strip())
            return tt.hour*60+tt.minute
        lo,hi=minutes(lower,0),minutes(upper,1440)
        width=int(step)
        if width<=0 or hi<=lo:raise ValueError('Hora inicial/final ou passo inválidos.')
        if math.ceil((hi-lo)/width)>cap:raise ValueError('Muitos intervalos horários.')
        def fmt(v):return '24:00' if v==1440 else f'{v//60:02d}:{v%60:02d}'
        for a in range(lo,hi,width):
            b=min(a+width,hi)
            result.append((f'{fmt(a)}–{fmt(b)}',(df['minute']>=a)&(df['minute']<b)))
    elif axis=='Profundidade':
        v=df['depth'].dropna()
        if v.empty:return []
        def mm(s,default): return float(s.strip().replace(',','.')) if s.strip() else default
        width=mm(step,500)
        if width<=0 or not math.isfinite(width):raise ValueError('Passo deve ser positivo.')
        lo=mm(lower,math.floor(v.min()/width)*width)
        hi=mm(upper,math.ceil(v.max()/width)*width)
        if not all(math.isfinite(x) for x in (lo,hi)) or hi<lo:raise ValueError('Profundidade inválida.')
        if hi==lo:hi=lo+width
        n=math.ceil((hi-lo)/width)
        if n>cap:raise ValueError('Muitos intervalos de profundidade.')
        for k in range(n):
            a=lo+width*k;b=min(lo+width*(k+1),hi)
            mask=(df['depth']>=a)&((df['depth']<=b) if k==n-1 else (df['depth']<b))
            result.append((f'{a:g}–{b:g} mm',mask))
    else:
        raise ValueError('Eixo desconhecido')
    return result

def filter_fixed(df,axis,params,classes,version=None):
    if df.empty or not classes:return df.iloc[:0].copy()
    mask=df['classe'].isin(classes)
    if version:mask &= df['versao'].eq(version)
    fields=dict(date_from=params.get('date_from','') if axis!='Data' else '',
                date_to=params.get('date_to','') if axis!='Data' else '',
                time_from=params.get('time_from','') if axis!='Hora' else '',
                time_to=params.get('time_to','') if axis!='Hora' else '',
                depth_min=params.get('depth_min','') if axis!='Profundidade' else '',
                depth_max=params.get('depth_max','') if axis!='Profundidade' else '',
                attribute_regex=params.get('attribute_regex',''),
                attribute_value=params.get('attribute_value','') if axis!='Atributo' else '')
    spec=ImageNameFilter(**fields)
    if spec.enabled:mask &= filter_image_names(df['names'],spec)
    return df.loc[mask].copy()
