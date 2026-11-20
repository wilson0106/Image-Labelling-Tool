import os
import random

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from aux import defaults
from ilt_contracts import IMAGE_EXTENSIONS, validate_labels
from torch_runtime import (
    amp_dtype, env_num_workers, get_device, input_memory_format, loader_kwargs, maybe_compile,
    resolve_batch_size, use_channels_last,
)

activation={}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True


# ImageNet statistics of the ConvNeXt_Tiny_Weights.IMAGENET1K_V1 pre-training
# (torchvision weight metadata; Liu et al. 2022).
IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD=(0.229,0.224,0.225)
INPUT_SIZE=224
# Pre-processing of models trained by this version (recorded in the checkpoint).
DEFAULT_PREPROCESS={'normalize':'imagenet','resize':'pad','input_size':INPUT_SIZE}
# Pre-processing of ILT <= 1.0 checkpoints (no record in the file).
LEGACY_PREPROCESS={'normalize':'none','resize':'stretch','input_size':INPUT_SIZE}


def border_colour(img):
    """Median colour of the 1-pixel frame of a PIL image (the vignette background)."""
    arr=np.asarray(img)
    if arr.ndim==2:
        arr=arr[:,:,None]
    h,w=arr.shape[:2]
    if h<3 or w<3:
        med=np.median(arr.reshape(-1,arr.shape[2]),axis=0)
    else:
        border=np.concatenate([arr[0,:,:],arr[-1,:,:],arr[1:-1,0,:],arr[1:-1,-1,:]],axis=0)
        med=np.median(border,axis=0)
    vals=tuple(int(round(float(v))) for v in med)
    return vals[0] if len(vals)==1 else vals


class SquareCanvas:
    """Aspect-preserving canvas (and the geometric/photometric augmentation).

    * ``pad=True``: the image is pasted on a square canvas of side max(w, h)
      filled with its own border colour, so organisms keep their shape instead
      of being stretched to 224×224 (Lumini & Nanni 2019; Kyathanahally et
      al. 2021).  Elongation is a taxonomic character.
    * ``augment=True`` (training only): random horizontal/vertical flips, a
      random rotation in [0, 360) with the frame expanded (nothing is cut; the
      organism scale varies by up to 1/√2, which acts as scale jitter) filled
      with the border colour, and brightness/contrast jitter of ±10 %.
      Plankton images have no canonical orientation (Orenstein & Beijbom 2017;
      Luo et al. 2018).  Randomness comes from ``random`` (seeded per
      DataLoader worker by PyTorch).
    """

    def __init__(self,pad=True,augment=False,brightness=0.10,contrast=0.10):
        self.pad=bool(pad); self.augment=bool(augment)
        self.brightness=float(brightness); self.contrast=float(contrast)

    def __call__(self,img):
        if img.mode!='RGB':
            img=img.convert('RGB')
        bg=border_colour(img)
        if self.augment:
            if random.random()<0.5:
                img=img.transpose(PIL.Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random()<0.5:
                img=img.transpose(PIL.Image.Transpose.FLIP_TOP_BOTTOM)
            angle=random.uniform(0.0,360.0)
            img=img.rotate(angle,resample=PIL.Image.Resampling.BILINEAR,expand=True,fillcolor=bg)
        if self.pad:
            w,h=img.size
            side=max(w,h)
            if side!=w or side!=h:
                canvas=PIL.Image.new('RGB',(side,side),bg)
                canvas.paste(img,((side-w)//2,(side-h)//2))
                img=canvas
        if self.augment:
            from PIL import ImageEnhance
            if self.brightness>0:
                img=ImageEnhance.Brightness(img).enhance(random.uniform(1-self.brightness,1+self.brightness))
            if self.contrast>0:
                img=ImageEnhance.Contrast(img).enhance(random.uniform(1-self.contrast,1+self.contrast))
        return img


def get_transforms(train=False,preprocess=None,augment=False):
    """Input pipeline for one model.

    *preprocess* is the dict stored in the checkpoint (``normalize``/``resize``);
    ``None`` means the current default.  Training with ``augment=True`` adds
    the random flips/rotation/photometric jitter of :class:`SquareCanvas`.
    """
    pp=dict(DEFAULT_PREPROCESS,**(preprocess or {}))
    size=int(pp.get('input_size',INPUT_SIZE))
    pad=str(pp.get('resize','pad'))=='pad'
    use_aug=bool(train and augment)
    ops=[]
    if pad or use_aug:
        ops.append(SquareCanvas(pad=pad,augment=use_aug))
    ops.append(transforms.Resize((size,size)))
    ops.append(transforms.ToTensor())
    if str(pp.get('normalize','imagenet'))=='imagenet':
        ops.append(transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD))
    return transforms.Compose(ops)


def preprocess_from_checkpoint(checkpoint):
    """Pre-processing recorded in a checkpoint; ILT <= 1.0 files mean legacy."""
    if isinstance(checkpoint,dict):
        state=checkpoint.get('ilt_state')
        if isinstance(state,dict) and isinstance(state.get('preprocess'),dict):
            return dict(LEGACY_PREPROCESS,**state['preprocess'])
    return dict(LEGACY_PREPROCESS)


class ILTDataset(Dataset):
    def __init__(self,file_list,transform=None):
        self.file_list=file_list
        self.transform=transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
        img_path=self.file_list[idx]
        with PIL.Image.open(img_path) as src:
            img=src.convert('RGB')
        if self.transform is not None:
            img=self.transform(img)
        return img,img_path


def get_activation(name):
    def hook(model,input,output):
        activation[name]=output.detach()
    return hook


def get_model(load=False,num_classes=1000,pretrained=True,stochastic_depth_prob=0.1):
    """ConvNeXt-Tiny; *stochastic_depth_prob* is torchvision's default 0.1
    (Huang et al. 2016), active only in ``model.train()``."""
    weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model=convnext_tiny(weights=weights,stochastic_depth_prob=float(stochastic_depth_prob))
    if load:
        model.classifier[2]=nn.Linear(768,num_classes)
    return model


def register_hooks(model):
    """Register the single activation hook used for exported embeddings once."""
    if getattr(model,'_ilt_embedding_hook_registered',False):
        return
    model.classifier[0].register_forward_hook(get_activation('layer4'))
    model._ilt_embedding_hook_registered=True


def _load_inference_weights_once(model,weights_path,device):
    """Load the checkpoint once and remember the pre-processing it was trained with."""
    if not weights_path:
        # ImageNet weights only (no ILT checkpoint): their own pre-processing.
        model._ilt_preprocess=dict(DEFAULT_PREPROCESS)
        return model._ilt_preprocess
    path=os.path.abspath(weights_path)
    if getattr(model,'_ilt_loaded_checkpoint',None)==path:
        return getattr(model,'_ilt_preprocess',dict(LEGACY_PREPROCESS))
    try:
        checkpoint=torch.load(path,map_location=device,weights_only=True)
    except TypeError:
        checkpoint=torch.load(path,map_location=device)
    state=checkpoint.get('model',checkpoint) if isinstance(checkpoint,dict) else checkpoint
    model.load_state_dict(state)
    model._ilt_loaded_checkpoint=path
    model._ilt_preprocess=preprocess_from_checkpoint(checkpoint)
    print(f"Model pre-processing: normalize={model._ilt_preprocess['normalize']}, "
          f"resize={model._ilt_preprocess['resize']}")
    return model._ilt_preprocess


def _image_files(folder):
    allowed=IMAGE_EXTENSIONS
    with os.scandir(folder) as it:
        files=[
            e.path for e in it
            if e.is_file(follow_symlinks=True)
            and os.path.splitext(e.name)[1].lower() in allowed
        ]
    files.sort()
    return files


def compute_features(images_folder,batch_id,model,weights_path,labels_dict=None):
    """Compute ConvNeXt embeddings/predictions for one batch.

    The model/checkpoint and hook are prepared once and reused across batches.
    Feature chunks are moved to CPU immediately instead of repeatedly vstacking
    growing CUDA tensors.
    """
    global activation

    device=get_device()
    use_amp=device.type=='cuda'
    dtype=amp_dtype(device)
    if use_amp:
        torch.backends.cudnn.benchmark=True
    batch_size=resolve_batch_size(os.environ.get('ILT_INFER_BATCH_SIZE','auto'),device,training=False)

    channels_last=use_channels_last(device)
    if channels_last:
        model.to(device,memory_format=torch.channels_last)
    else:
        model.to(device)
    # The checkpoint records how its images were pre-processed (normalisation,
    # pad/stretch); classification must use exactly the same pipeline.
    preprocess=_load_inference_weights_once(model,weights_path,device)
    transform=get_transforms(preprocess=preprocess)
    register_hooks(model)
    model.eval()
    activation={}
    # Kept in the instance __dict__, NOT as a module attribute: assigning an
    # nn.Module attribute registers it as a child, and without torch.compile
    # the runner is the model itself (a module containing itself makes
    # model.to() recurse forever on the next batch).
    runner=model.__dict__.get('_ilt_runner')
    if runner is None:
        runner=maybe_compile(model)
        model.__dict__['_ilt_runner']=runner

    inner_folder=os.path.join(images_folder,batch_id,defaults['inner_folder'])
    file_list=_image_files(inner_folder)
    if not file_list:
        raise RuntimeError(f'No images found in batch folder: {inner_folder}')

    # Image decoding/resizing is the bottleneck when the GPU is fast: use
    # several CPU workers automatically (ILT_INFER_WORKERS still overrides).
    workers=env_num_workers('ILT_INFER_WORKERS')
    dataset=ILTDataset(file_list,transform=transform)
    memory_format=input_memory_format(channels_last)

    if labels_dict:
        labels_dict=validate_labels(labels_dict, source='classification labels')
    index_to_class={int(v):k for k,v in (labels_dict or {}).items()}

    def run_batches(bs):
        # One DataLoader per batch folder, iterated once: no persistent workers
        # (see torch_runtime.loader_kwargs; they leaked descriptors per batch).
        loader=DataLoader(dataset=dataset,shuffle=False,
                          **loader_kwargs(bs,workers,pin_memory=use_amp,persistent=False))
        path_images=[];predictions=[];confs=[];feature_chunks=[]
        with torch.inference_mode():
            for data,paths in loader:
                data=data.to(device,non_blocking=use_amp,memory_format=memory_format)
                with torch.amp.autocast(device_type=device.type,dtype=dtype,enabled=use_amp):
                    output=runner(data)
                    probs=F.softmax(output.float(),dim=1)
                    aux=torch.amax(activation['layer4'],(2,3))
                feature_chunks.append(aux.float().cpu())

                k=min(3,int(probs.shape[1]))
                top_probs,top_classes=torch.topk(probs,k=k,dim=1)
                top_probs=top_probs.float().cpu()
                top_classes=top_classes.cpu()
                pred_idx=top_classes[:,0].tolist()

                clean_paths=[os.path.basename(str(path)) for path in paths]
                for i,pred in enumerate(pred_idx):
                    predictions.append(int(pred))
                    conf={}
                    for rank in range(k):
                        cls_idx=int(top_classes[i,rank])
                        cls=index_to_class.get(cls_idx,cls_idx)
                        conf[f'top{rank+1}']=(cls,float(top_probs[i,rank]))
                    # Keep downstream schema stable even for classifiers with <3 classes.
                    for rank in range(k,3):
                        conf[f'top{rank+1}']=conf[f'top{k}']
                    confs.append(conf)
                path_images.extend(clean_paths)
        return torch.cat(feature_chunks,dim=0).numpy(),path_images,predictions,confs

    # The batch is halved and the batch folder restarted when the GPU runs out
    # of memory (the chosen size is remembered for the following batches).
    while True:
        try:
            result=run_batches(batch_size)
            break
        except torch.cuda.OutOfMemoryError:
            if not use_amp or batch_size<=8:
                raise
            torch.cuda.empty_cache()
            batch_size=max(8,batch_size//2)
            os.environ['ILT_INFER_BATCH_SIZE']=str(batch_size)
            print(f'GPU out of memory; inference batch reduced to {batch_size}.')
    return result
