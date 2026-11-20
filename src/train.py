import contextlib
import json
import math
import os
import sys
from os.path import exists, join
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from features import INPUT_SIZE, get_device, get_model, get_transforms, seed_everything
from ilt_contracts import ILT_VERSION, atomic_write_csv, atomic_write_json, is_image_filename, validate_labels
from ilt_losses import FocalLoss
from project_settings import DEFAULT_PIXEL_SIZE_UM_PER_PX, parse_pixel_size
from torch_runtime import (
    amp_dtype, auto_num_workers, configure_backends, describe_runtime, input_memory_format,
    loader_kwargs, maybe_compile, resolve_batch_size, use_channels_last,
)
from training_dataset import list_split_image_files
from training_evaluation import save_evaluation_artifacts
from training_recipe import LEGACY_RECIPE, describe_recipe, preprocess_of, recipe_from_config, same_recipe

CHECKPOINT_FORMAT_VERSION=3  # 3 = carries recipe/pre-processing/class names (1.1.0)

ImageFile.LOAD_TRUNCATED_IMAGES=True


def _split_image_files(split_dir):
    """Return supported image files located exactly one class folder below a split."""
    return list_split_image_files(split_dir)


def _validate_split_classes(train_dir,valid_dir,test_dir,map_label):
    """Reject semantic class mismatches before a DataLoader starts training."""
    expected=set(map_label)
    for split_name,split_dir in (('valid',valid_dir),('test',test_dir)):
        actual={d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir,d))}
        extra=sorted(actual-expected)
        if extra:
            raise ValueError(
                f'{split_name}: class folders not present in train: {extra}. '
                'Train/valid/test must use the same class names.'
            )
    empty=[]
    for cls in sorted(expected):
        cls_dir=os.path.join(train_dir,cls)
        if not any(
            os.path.isfile(os.path.join(cls_dir,n)) and is_image_filename(n)
            for n in os.listdir(cls_dir)
        ):
            empty.append(cls)
    if empty:
        raise ValueError(f'train: class folders without supported images: {empty}')


def _atomic_torch_save(payload,path):
    """Write a checkpoint fully, then replace its destination atomically."""
    path=os.path.abspath(path)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    tmp=path+'.tmp'
    try:
        torch.save(payload,tmp)
        os.replace(tmp,path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _load_checkpoint(path, device):
    """Load a torch checkpoint compatibly across supported PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_published_model_for_evaluation(model, model_path, device):
    """Load exactly the published *best* checkpoint before final test evaluation."""
    checkpoint = _load_checkpoint(model_path, device)
    state = checkpoint.get('model') if isinstance(checkpoint, dict) else checkpoint
    if state is None:
        raise RuntimeError(f'Checkpoint {model_path} does not contain model weights.')
    model.load_state_dict(state)
    meta = checkpoint.get('ilt_state', {}) if isinstance(checkpoint, dict) else {}
    return meta



class CustomDataset(Dataset):
    def __init__(self,file_list,map_label,transform=None):
        self.file_list=file_list
        self.map_label=map_label
        self.transform=transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
        img_path=self.file_list[idx]
        with Image.open(img_path) as src:
            img=src.convert('RGB')
        if self.transform is not None:
            img=self.transform(img)
        label=os.path.basename(os.path.dirname(img_path))
        return img,self.map_label[label],img_path


class EarlyStopper:
    def __init__(self,patience=20,counter=0,min_val_loss=float('inf'),max_val_acc=0):
        self.patience=patience
        self.counter=counter
        self.min_val_loss=min_val_loss
        self.max_val_acc=max_val_acc

    def early_stop(self,model,val_metric,val,optimizer,scaler,save_path,save_name,epoch=None,elapsed=0.0,
                   extra_state=None,scheduler_state=None):
        improved=False
        if val_metric=='valid_loss' and val<self.min_val_loss:
            self.min_val_loss=val
            improved=True
        elif val_metric in ('valid_acc','valid_macro') and val>self.max_val_acc:
            self.max_val_acc=val
            improved=True

        best_path=join(save_path,'model_'+save_name)
        # The first completed epoch must always publish a usable best model,
        # even when valid_acc/valid_macro is exactly zero.
        if not exists(best_path):
            improved=True

        if improved:
            self.counter=0
        else:
            self.counter+=1
            print('No improvement -',self.counter)

        ilt_state={
            'format_version':CHECKPOINT_FORMAT_VERSION,
            'ilt_version':ILT_VERSION,
            'epoch':int(epoch) if epoch is not None else None,
            'elapsed':float(elapsed),
            'counter':int(self.counter),
            'min_val_loss':float(self.min_val_loss),
            'max_val_acc':float(self.max_val_acc),
            'valid_metric':str(val_metric),
        }
        # Self-describing checkpoint: recipe, pre-processing, architecture and
        # class names travel with the weights (classification reads them).
        ilt_state.update(dict(extra_state or {}))
        checkpoint={
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scaler':scaler.state_dict(),
            'scheduler':scheduler_state,
            'ilt_state':ilt_state,
        }
        _atomic_torch_save(checkpoint,join(save_path,'model_'+save_name+'_last_epoch'))
        if improved:
            _atomic_torch_save(checkpoint,join(save_path,'model_'+save_name))
        return self.counter>=self.patience


def compute_macro(matrix,div=-1):
    """Macro-averaged recall, precision and F1 over the classes of *matrix*.

    *div* is kept for backward compatibility only: ``run()`` always passes the
    real number of classes (``len(map_label)``).  Dividing by any other value
    silently rescales the metrics (a user-typed "macro divisor" used to be
    forwarded here).  Macro-F1 = unweighted mean of the per-class F1 (Sokolova &
    Lapalme 2009, *A systematic analysis of performance measures for
    classification tasks*, Inf. Process. Manag. 45(4):427-437).
    """
    avg_recall=avg_precision=macro=0.0
    n=matrix.shape[0]
    for x in range(n):
        tp=matrix[x,x]
        row_sum=matrix[x].sum()
        col_sum=matrix[:,x].sum()
        recall=tp/row_sum if row_sum>0 else 0.0
        precision=tp/col_sum if col_sum>0 else 0.0
        f1=2*recall*precision/(recall+precision) if recall+precision>0 else 0.0
        avg_recall+=recall
        avg_precision+=precision
        macro+=f1
    if div<0:
        div=n
    div=max(float(div),1.0)
    return avg_recall/div,avg_precision/div,macro/div


def _run_epoch(model,loader,criterion,softmax,device,use_amp,num_classes,
               optimizer=None,scaler=None,memory_format=torch.preserve_format,dtype=torch.float16):
    """Run one pass over *loader*; trains when *optimizer* is given.

    Accuracy, loss and the confusion matrix are accumulated on the device and
    copied to the host once per pass.  The previous per-batch ``.item()`` and
    ``.cpu()`` calls forced a GPU synchronisation on every batch, stalling the
    CUDA queue.  Integer counts and float64 loss sums are numerically identical
    to the previous host-side accumulation.

    Returns ``(correct, loss_sum, n_batches, confusion_matrix)``.
    """
    training=optimizer is not None
    nn2=num_classes*num_classes
    correct=torch.zeros((),dtype=torch.float64,device=device)
    loss_sum=torch.zeros((),dtype=torch.float64,device=device)
    confusion=torch.zeros(nn2,dtype=torch.int64,device=device)
    n_batches=0
    grad_ctx=contextlib.nullcontext() if training else torch.inference_mode()
    with grad_ctx:
        for data,label,_ in tqdm(loader):
            data=data.to(device,non_blocking=use_amp,memory_format=memory_format)
            label=label.to(device,non_blocking=use_amp)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,dtype=dtype,enabled=use_amp):
                output=model(data)
                loss_input=softmax(output) if softmax is not None else output
                loss=criterion(loss_input,label)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            pred=output.argmax(dim=1)
            correct+=(pred==label).sum()
            loss_sum+=loss.detach().double()
            confusion+=torch.bincount(label.long()*num_classes+pred.long(),minlength=nn2)
            n_batches+=1
    matrix=confusion.view(num_classes,num_classes).cpu().numpy().astype(np.int64,copy=False)
    return float(correct.item()),float(loss_sum.item()),n_batches,matrix


def build_optimizer(model,lr,recipe,single_group=False):
    """Adam/AdamW with the recipe's weight decay and head learning rate.

    * weight decay is not applied to 1-D parameters (LayerNorm weights, biases,
      ConvNeXt layer-scale) — the usual decoupled-decay practice (Loshchilov &
      Hutter 2019; Liu et al. 2022 fine-tuning setup);
    * the freshly initialised classifier (``classifier[2]``) uses
      ``lr × headLrMult`` (discriminative fine-tuning, Howard & Ruder 2018);
    * ``single_group=True`` rebuilds the one-group Adam of ILT <= 1.0 so that
      legacy ``_last_epoch`` checkpoints can still be resumed.
    """
    lr=float(lr)
    if single_group:
        return torch.optim.Adam(model.parameters(),lr=lr)
    head=list(model.classifier[2].parameters())
    head_ids={id(p) for p in head}
    decay,no_decay=[],[]
    for name,param in model.named_parameters():
        if not param.requires_grad or id(param) in head_ids:
            continue
        (no_decay if (param.ndim<=1 or name.endswith('.bias')) else decay).append(param)
    wd=float(recipe['weightDecay'])
    groups=[
        {'params':decay,'lr':lr,'weight_decay':wd,'name':'backbone'},
        {'params':no_decay,'lr':lr,'weight_decay':0.0,'name':'backbone_no_decay'},
        {'params':head,'lr':lr*float(recipe['headLrMult']),'weight_decay':wd,'name':'head'},
    ]
    if recipe['optimizer']=='adam':
        return torch.optim.Adam(groups,lr=lr)
    return torch.optim.AdamW(groups,lr=lr)


class LrSchedule:
    """Per-epoch learning-rate control (1-based epochs).

    * linear warmup over ``warmupEpochs`` (Goyal et al. 2017);
    * ``cosine``: cosine annealing from the end of warmup to ``numEpochs``,
      floor 1 % of the base rate (Loshchilov & Hutter 2017);
    * ``plateau``: halve the rate when the validation metric stops improving
      for ``patience // 2`` epochs (``ReduceLROnPlateau``), the companion of
      early stopping used by the default configuration (500 epochs, patience
      10), where a cosine over 500 epochs would barely decay;
    * ``none``: constant after warmup (ILT <= 1.0 behaviour with warmup 0).
    """

    def __init__(self,optimizer,recipe,total_epochs,valid_metric,patience):
        self.opt=optimizer
        self.kind=str(recipe['scheduler'])
        self.warmup=max(0,int(recipe['warmupEpochs']))
        self.total=max(1,int(total_epochs))
        self.base=[float(g['lr']) for g in optimizer.param_groups]
        self.min_factor=0.01
        self.plateau=None
        if self.kind=='plateau':
            self.plateau=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,mode='min' if valid_metric=='valid_loss' else 'max',factor=0.5,
                patience=max(1,int(patience)//2),min_lr=[b*self.min_factor for b in self.base],
            )

    def _set(self,factor):
        for group,base in zip(self.opt.param_groups,self.base):
            group['lr']=base*float(factor)

    def begin_epoch(self,epoch):
        epoch=int(epoch)
        if self.warmup>0 and epoch<=self.warmup:
            self._set(epoch/float(self.warmup))
            return
        if self.kind=='cosine':
            span=max(1,self.total-self.warmup)
            t=min(1.0,max(0.0,(epoch-self.warmup)/float(span)))
            self._set(self.min_factor+(1.0-self.min_factor)*0.5*(1.0+math.cos(math.pi*t)))
        elif self.warmup>0 and epoch==self.warmup+1:
            self._set(1.0)  # warmup finished: full rate (plateau/none take over)

    def after_validation(self,metric_value,epoch):
        if self.plateau is not None and int(epoch)>self.warmup:
            self.plateau.step(float(metric_value))

    def current(self):
        return [float(g['lr']) for g in self.opt.param_groups]

    def state_dict(self):
        return {'kind':self.kind,'plateau':self.plateau.state_dict() if self.plateau is not None else None}

    def load_state_dict(self,state):
        if self.plateau is not None and isinstance(state,dict) and state.get('plateau'):
            try:
                self.plateau.load_state_dict(state['plateau'])
            except Exception as exc:
                print('Warning: could not restore the plateau scheduler state:',exc)


def _vram_mb(device):
    if device.type!='cuda':
        return 0.0
    try:
        free,total=torch.cuda.mem_get_info()
        return (total-free)/(1024*1024)
    except Exception:
        return 0.0


def _non_interactive():
    """True when nobody can answer a console prompt (GUI worker, no TTY)."""
    if str(os.environ.get('ILT_NONINTERACTIVE','')).strip().lower() in {'1','true','t','yes','y'}:
        return True
    try:
        return not sys.stdin or not sys.stdin.isatty()
    except Exception:
        return True


def _resolve_existing_model(model_path,last_path,read_model,existing_model):
    """Decide what to do when ``model_<name>`` already exists.

    Returns ``(read_model, action)`` where *action* is ``'resume'``,
    ``'overwrite'`` or ``'stop'``.  The decision comes from the configuration
    (``existingModel`` written by the interface) or, only when a terminal is
    attached, from the console prompt.  The worker never blocks on ``input()``
    without a TTY: the dashboard launches it with stdin closed and
    ``ILT_NONINTERACTIVE=1``, so a silent hang/EOFError is replaced by a clear
    error message.
    """
    partial_model=exists(last_path)
    decision=str(existing_model or '').strip().lower()
    if read_model=='T' and partial_model and decision in ('','resume'):
        return 'T','resume'
    if decision=='resume':
        if partial_model:
            return 'T','resume'
        raise RuntimeError(
            f'Resume requested but {last_path} does not exist (the previous run finished). '
            'Choose overwrite or a new project name.'
        )
    if decision=='overwrite':
        return 'F','overwrite'
    if decision in ('stop','abort','cancel'):
        return read_model,'stop'
    if _non_interactive():
        raise RuntimeError(
            f'Model {model_path} already exists. Choose "overwrite" or "resume" in the interface '
            '(or set existingModel in the training configuration).'
        )
    print('Warning! Model with name',os.path.basename(model_path),'already exists!')
    options=['0','1']+(['2'] if partial_model else [])
    print('Choose an option below:')
    option=''
    while option not in options:
        print('  1: Start training from scratch and overwrite the existing model')
        if partial_model:
            print('  2: Resume training of existing model')
        print('  0: Stop running this script.')
        option=input('Enter your option: ')
    if option=='2':
        return 'T','resume'
    if option=='1':
        return 'F','overwrite'
    return read_model,'stop'


def run(images_path,save_path,save_name,train_name='train',valid_name='valid',test_name='test',
        lr=0.00001,valid_metric='valid_acc',num_epochs=200,patience=20,loss_name='ce',
        fgamma=2,seed=-1,div=-1,num_workers=0,read_model='F',batch_size=30,existing_model='',recipe=None):

    recipe=recipe_from_config(recipe or {})
    if seed!=-1:
        seed_everything(seed)
    device=get_device()
    use_amp=device.type=='cuda'
    dtype=amp_dtype(device)
    # Deterministic runs (fixed seed) keep cuDNN autotuning off, exactly as before.
    configure_backends(device,deterministic=(seed!=-1))
    channels_last=use_channels_last(device)
    memory_format=input_memory_format(channels_last)
    num_workers=auto_num_workers(num_workers)
    auto_batch=str(batch_size).strip().lower() in ('auto','')
    batch_size=resolve_batch_size(batch_size,device,training=True)
    print('Device:',device)
    print(describe_runtime(device,num_workers,channels_last))
    if use_amp:
        print(f'Mixed precision: {str(dtype).replace("torch.","")} | batch size: {batch_size}'+(' (auto)' if auto_batch else ''))
    print('File name:',save_name)

    train_dir=os.path.join(images_path,train_name)
    valid_dir=os.path.join(images_path,valid_name)
    test_dir=os.path.join(images_path,test_name)

    labels=sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,d)))
    map_label=validate_labels({c:i for i,c in enumerate(labels)}, source='training classes')
    _validate_split_classes(train_dir,valid_dir,test_dir,map_label)

    train_list=_split_image_files(train_dir)
    valid_list=_split_image_files(valid_dir)
    test_list=_split_image_files(test_dir)
    if not train_list or not valid_list or not test_list:
        raise RuntimeError('Train/valid/test folders must each contain supported images inside class subfolders.')
    num_classes=len(map_label)
    # Macro metrics are always averaged over the real classes.  A different
    # configured divisor (old "macro divisor" field) is reported and ignored.
    try:
        configured_div=int(div)
    except (TypeError,ValueError):
        configured_div=-1
    if configured_div>0 and configured_div!=num_classes:
        print(f'Warning: configured macro divisor {configured_div} differs from the number of classes '
              f'({num_classes}); metrics are averaged over {num_classes} classes.')
    div=num_classes

    model_path=join(save_path,f'model_{save_name}')
    last_path=join(save_path,f'model_{save_name}_last_epoch')

    # A resumed run must keep the recipe of its checkpoint: legacy files
    # (format < 3) were trained with the 1.0 pipeline and a one-group Adam.
    resume_ckpt=None
    legacy_resume=False
    if read_model=='T' and exists(last_path):
        resume_ckpt=_load_checkpoint(last_path,device)
        rstate=(resume_ckpt.get('ilt_state') or {}) if isinstance(resume_ckpt,dict) else {}
        if int(rstate.get('format_version',0) or 0)>=3:
            saved_recipe=rstate.get('recipe') or {}
            if not same_recipe(saved_recipe,recipe):
                raise ValueError(
                    'Resume blocked: the checkpoint was trained with a different recipe '
                    f'({describe_recipe(saved_recipe)}); current: {describe_recipe(recipe)}. '
                    'Use the same recipe or start a new model.'
                )
        else:
            legacy_resume=True
            if not same_recipe(recipe,LEGACY_RECIPE):
                print('Warning: legacy checkpoint (ILT <= 1.0); resuming with the legacy recipe.')
            recipe=dict(LEGACY_RECIPE)
    preprocess=preprocess_of(recipe)
    print('Recipe:',describe_recipe(recipe,'en'))

    train_tf=get_transforms(train=True,preprocess=preprocess,augment=recipe['augment'])
    eval_tf=get_transforms(preprocess=preprocess)
    train_data=CustomDataset(train_list,map_label,transform=train_tf)
    valid_data=CustomDataset(valid_list,map_label,transform=eval_tf)
    test_data=CustomDataset(test_list,map_label,transform=eval_tf)

    def make_loaders(bs):
        kw=loader_kwargs(bs,num_workers,pin_memory=use_amp)
        return (DataLoader(train_data,shuffle=True,**kw),DataLoader(valid_data,shuffle=False,**kw),
                DataLoader(test_data,shuffle=False,**kw))
    train_loader,valid_loader,test_loader=make_loaders(batch_size)

    model=get_model(load=True,num_classes=num_classes,pretrained=True,
                    stochastic_depth_prob=recipe['stochasticDepth'])
    if loss_name=='focal':
        criterion=FocalLoss(gamma=fgamma)
    else:
        criterion=nn.CrossEntropyLoss(label_smoothing=float(recipe['labelSmoothing']))
    softmax=None

    optimizer=build_optimizer(model,lr,recipe,single_group=legacy_resume)
    # Loss scaling is only needed for float16; bfloat16 has the float32 range.
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp and dtype==torch.float16)
    lr_schedule=LrSchedule(optimizer,recipe,num_epochs,valid_metric,patience)
    checkpoint_extra={
        'recipe':dict(recipe),
        'preprocess':dict(preprocess),
        'arch':'convnext_tiny',
        'input_size':INPUT_SIZE,
        'class_names':list(labels),
        'torchvision':str(torchvision.__version__),
    }

    counter=0
    if exists(model_path):
        read_model,action=_resolve_existing_model(model_path,last_path,read_model,existing_model)
        if action=='stop':
            print('Stopped: existing model kept.')
            return
        if action=='overwrite':
            # A fresh run must not inherit the old best/partial checkpoints:
            # EarlyStopper publishes the first epoch only when no best exists.
            for stale in (model_path,last_path):
                try:
                    if exists(stale):
                        os.remove(stale)
                except OSError as exc:
                    raise RuntimeError(f'Could not remove existing checkpoint {stale}: {exc}') from exc
            print('Existing model removed; training from scratch.')
        else:
            print('Resuming the existing partial model.')

    os.makedirs(save_path,exist_ok=True)
    labels_out=join(save_path,'labels_'+save_name+'.json')
    if read_model=='T' and exists(last_path) and exists(labels_out):
        with open(labels_out,encoding='utf-8') as infile:
            prior_labels=validate_labels(json.load(infile), source=labels_out)
        if prior_labels != map_label:
            raise ValueError(
                'Resume blocked: current training classes do not match the saved labels map. '
                'Use the same dataset class mapping or start a new model.'
            )
    atomic_write_json(map_label,labels_out)

    if channels_last:
        model.to(device,memory_format=torch.channels_last)
    else:
        model.to(device)
    # ``runner`` may be a torch.compile wrapper (opt-in); checkpoints always
    # come from ``model`` so their keys stay unchanged.
    runner=maybe_compile(model)
    min_val_loss=float('inf')
    max_val_acc=0.0
    stats_path=join(save_path,f'stats_{save_name}.csv')

    stats_columns=[
        'Epoch','Time','VRAM','RAM','Train Acc','Train Loss','Train Recall',
        'Train Precision','Train Macro-F1','Valid Acc','Valid Loss',
        'Valid Recall','Valid Precision','Valid Macro-F1','LR'
    ]

    if read_model=='T' and exists(last_path):
        checkpoint=resume_ckpt if resume_ckpt is not None else _load_checkpoint(last_path,device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler_state=checkpoint.get('scaler')
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        lr_schedule.load_state_dict(checkpoint.get('scheduler'))

        df_stats=pd.read_csv(stats_path,index_col=None) if exists(stats_path) else pd.DataFrame(columns=stats_columns)
        if 'LR' not in df_stats.columns:
            df_stats['LR']=np.nan  # stats written before 1.1.0
        state=checkpoint.get('ilt_state') or {}
        if int(state.get('format_version',0) or 0)>=2:
            saved_metric=str(state.get('valid_metric',''))
            if saved_metric and saved_metric!=valid_metric:
                raise ValueError(
                    f'Resume blocked: checkpoint used valid_metric={saved_metric!r}, '
                    f'current configuration requests {valid_metric!r}.'
                )
            saved_epoch=state.get('epoch')
            if saved_epoch is None:
                raise ValueError('Resume checkpoint is missing its saved epoch.')
            saved_epoch=int(saved_epoch)
            epoch=saved_epoch+1
            prev_time=float(state.get('elapsed',0.0))
            counter=int(state.get('counter',0))
            min_val_loss=float(state.get('min_val_loss',float('inf')))
            max_val_acc=float(state.get('max_val_acc',0.0))
            if not df_stats.empty and 'Epoch' in df_stats.columns:
                epoch_num=pd.to_numeric(df_stats['Epoch'],errors='coerce')
                keep=(epoch_num.isna()) | (epoch_num<=saved_epoch)
                df_stats=df_stats.loc[keep].copy().reset_index(drop=True)
                atomic_write_csv(df_stats,stats_path,index=False)
        else:
            # Backward-compatible resume for checkpoints created before v6.7.
            if df_stats.empty:
                raise ValueError(
                    'Legacy checkpoint found without usable stats CSV. '
                    'Cannot infer resume epoch safely; start a new run or restore its stats file.'
                )
            numeric_epoch=pd.to_numeric(df_stats['Epoch'],errors='coerce')
            train_rows=df_stats.loc[numeric_epoch.notna()].copy()
            if train_rows.empty:
                raise ValueError('Legacy stats CSV has no training epochs; cannot resume safely.')
            epoch=int(pd.to_numeric(train_rows['Epoch']).max())+1
            if valid_metric=='valid_loss':
                vals=pd.to_numeric(train_rows['Valid Loss'],errors='coerce')
                best_pos=vals.idxmin()
                min_val_loss=float(vals.min())
            elif valid_metric=='valid_acc':
                vals=pd.to_numeric(train_rows['Valid Acc'],errors='coerce')
                best_pos=vals.idxmax()
                max_val_acc=float(vals.max())
            else:
                vals=pd.to_numeric(train_rows['Valid Macro-F1'],errors='coerce')
                best_pos=vals.idxmax()
                max_val_acc=float(vals.max())
            best_epoch=int(pd.to_numeric(df_stats.loc[best_pos,'Epoch']))
            counter=max(0,(epoch-1)-best_epoch)
            prev_time=float(pd.to_numeric(train_rows['Time'],errors='coerce').iloc[-1])
    else:
        epoch=1
        prev_time=0.0
        df_stats=pd.DataFrame(columns=stats_columns)

    early_stopper=EarlyStopper(
        patience=patience,counter=counter,min_val_loss=min_val_loss,max_val_acc=max_val_acc
    )
    start=timer()

    epoch_kw=dict(
        criterion=criterion,softmax=softmax,device=device,use_amp=use_amp,
        num_classes=num_classes,memory_format=memory_format,dtype=dtype,
    )
    first_pass=True
    while epoch<=num_epochs and early_stopper.counter<patience:
        lr_schedule.begin_epoch(epoch)
        model.train()
        try:
            train_acc,train_loss,_,train_matrix=_run_epoch(
                runner,train_loader,optimizer=optimizer,scaler=scaler,**epoch_kw
            )
        except torch.cuda.OutOfMemoryError:
            # Automatic batch size: halve it and redo this epoch (only the
            # first pass, before any checkpoint was written).
            if not (auto_batch and first_pass and batch_size>8):
                raise
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            batch_size=max(8,batch_size//2)
            train_loader,valid_loader,test_loader=make_loaders(batch_size)
            print(f'GPU out of memory; batch size reduced to {batch_size}, restarting epoch {epoch}.')
            continue
        first_pass=False
        train_acc/=len(train_data)
        train_loss/=max(len(train_loader),1)
        train_rec,train_prec,train_macro=compute_macro(train_matrix,div)

        model.eval()
        valid_acc,valid_loss,_,valid_matrix=_run_epoch(runner,valid_loader,**epoch_kw)

        valid_acc/=len(valid_data)
        valid_loss/=max(len(valid_loader),1)
        valid_rec,valid_prec,valid_macro=compute_macro(valid_matrix,div)

        gpu_mem_usage=_vram_mb(device)
        elapsed=timer()-start
        ram_usage=psutil.Process(os.getpid()).memory_info().rss/(1024*1024)
        lr_now=lr_schedule.current()[0]  # backbone rate (head = × headLrMult)
        row=pd.DataFrame([[
            epoch,elapsed+prev_time,gpu_mem_usage,ram_usage,
            train_acc,train_loss,train_rec,train_prec,train_macro,
            valid_acc,valid_loss,valid_rec,valid_prec,valid_macro,lr_now
        ]],columns=df_stats.columns)
        df_stats=pd.concat([df_stats,row],ignore_index=True)
        atomic_write_csv(df_stats,stats_path,index=False)

        print(f'Epoch {epoch}, time: {(elapsed+prev_time)/60:.2f} min, VRAM: {gpu_mem_usage:.1f}MB, RAM: {ram_usage:.1f}MB')
        print(f'\ttrain acc: {train_acc:.5f}, loss: {train_loss:.5f}, rec: {train_rec:.5f}, prec: {train_prec:.5f}, macro: {train_macro:.5f}')
        print(f'\tvalid acc: {valid_acc:.5f}, loss: {valid_loss:.5f}, rec: {valid_rec:.5f}, prec: {valid_prec:.5f}, macro: {valid_macro:.5f}')
        print(f'\tlr: {lr_now:.3e}')

        metric_value={'valid_loss':valid_loss,'valid_acc':valid_acc,'valid_macro':valid_macro}[valid_metric]
        lr_schedule.after_validation(metric_value,epoch)
        early_stopper.early_stop(
            model,valid_metric,metric_value,optimizer,scaler,save_path,save_name,
            epoch=epoch,elapsed=elapsed+prev_time,
            extra_state=checkpoint_extra,scheduler_state=lr_schedule.state_dict(),
        )
        epoch+=1

    if not exists(model_path):
        raise RuntimeError(
            f'No published best checkpoint was produced at {model_path}. '
            'Final test evaluation cannot proceed safely.'
        )
    best_meta=_load_published_model_for_evaluation(model,model_path,device)
    best_epoch=best_meta.get('epoch')
    if best_epoch is not None:
        print(f'Evaluating published best checkpoint from epoch {best_epoch}.')
    else:
        print('Evaluating published best checkpoint.')

    model.eval()
    test_acc,test_loss,_,test_matrix=_run_epoch(runner,test_loader,**epoch_kw)

    test_acc/=len(test_data)
    test_loss/=max(len(test_loader),1)
    test_rec,test_prec,test_macro=compute_macro(test_matrix,div)
    gpu_mem_usage=_vram_mb(device)
    ram_usage=psutil.Process(os.getpid()).memory_info().rss/(1024*1024)
    elapsed=timer()-start

    print(f'test acc: {test_acc:.5f}, loss: {test_loss:.5f}, rec: {test_rec:.5f}, prec: {test_prec:.5f}, macro: {test_macro:.5f}')
    row=pd.DataFrame([[
        'Test',elapsed,gpu_mem_usage,ram_usage,'','','','','',
        test_acc,test_loss,test_rec,test_prec,test_macro,''
    ]],columns=df_stats.columns)
    df_stats=pd.concat([df_stats,row],ignore_index=True)
    atomic_write_csv(df_stats,stats_path,index=False)
    np.savetxt(join(save_path,f'matrix_{save_name}.csv'),test_matrix,delimiter=', ')
    save_evaluation_artifacts(
        test_matrix, map_label, save_path, save_name,
        test_loss=test_loss, configured_macro_divisor=div,
    )
    if exists(last_path):
        os.remove(last_path)
    print('=============== Finished ===============')
    print()


def main(argv=None):
    argv=list(sys.argv if argv is None else argv)
    if len(argv)<5:
        raise SystemExit(
            'Missing parameters. Usage: train.py <images> <output> <project> <params.json>'
        )
    with open(argv[4],encoding='utf-8') as read_file:
        dc=json.load(read_file)

    images_path=argv[1]
    save_path=argv[2]
    save_name=argv[3]
    if not images_path:
        raise ValueError('Images path is empty.')
    if not save_path:
        raise ValueError('Output path is empty.')

    # Pixel size is project metadata in the current training pipeline; the
    # network/loss calculations do not use physical calibration.  Persist it
    # beside the model so classification can reuse the calibration later.
    dc['pixelSize']=parse_pixel_size(dc.get('pixelSize', DEFAULT_PIXEL_SIZE_UM_PER_PX), DEFAULT_PIXEL_SIZE_UM_PER_PX)
    # Record the complete, effective recipe beside the model (missing keys
    # take the recommended defaults; see training_recipe.py).
    recipe=recipe_from_config(dc)
    dc.update(recipe)
    os.makedirs(save_path,exist_ok=True)
    atomic_write_json(
        dc,
        join(save_path,f'training_config_{save_name}.json'),
        ensure_ascii=False,
        indent=2,
    )

    run(
        images_path,save_path,save_name,
        dc['trainFolder'],dc['validFolder'],dc['testFolder'],
        dc['lr'],dc['validMetric'],dc['numEpochs'],dc['patience'],
        dc['lossName'],dc['focalGamma'],dc['seed'],dc.get('numClasses',-1),
        dc['numWorkers'],dc['readModel'],dc.get('batchSize',30),
        existing_model=dc.get('existingModel',''),recipe=recipe,
    )


if __name__=='__main__':
    main()
