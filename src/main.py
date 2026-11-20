import datetime
import os
import sys
import timeit

import pandas as pd
import tqdm

from aux import defaults
from batch import create_batches, symlink_images
from data import label_predictions, read_labels
from features import compute_features, get_model
from projections import compute_projections
from ilt_contracts import atomic_savez_compressed, atomic_write_csv, validate_labels
from project_settings import DEFAULT_PIXEL_SIZE_UM_PER_PX, parse_pixel_size, save_project_settings
from hw_resources import raise_open_file_limit


def apply_runtime_overrides():
    batch_max_size=os.environ.get('ILT_BATCH_MAX_SIZE')
    img_mode=os.environ.get('ILT_IMG_MODE')
    pixel_size=os.environ.get('ILT_PIXEL_SIZE_UM_PER_PX')
    if batch_max_size:
        defaults['BATCH_MAX_SIZE']=max(1,int(batch_max_size))
    if img_mode:
        defaults['img_mode']=img_mode
    if pixel_size:
        defaults['pixel_size']=parse_pixel_size(pixel_size, DEFAULT_PIXEL_SIZE_UM_PER_PX)


def read_paths(argv):
    input_path=os.path.abspath(argv[1])
    output_path=os.path.abspath(argv[2])
    parent=os.path.dirname(output_path)
    if not (os.path.isdir(input_path) or os.path.isfile(input_path)):
        raise FileNotFoundError(f'Input folder/list does not exist: {input_path}')
    if not os.path.isdir(parent):
        raise FileNotFoundError(f'Output parent folder does not exist: {parent}')
    try:
        os.mkdir(output_path,mode=0o755)
    except FileExistsError as exc:
        raise FileExistsError(f'Output folder already exists: {output_path}') from exc
    return input_path,output_path


def read_network(argv):
    weights_path=os.path.abspath(argv[3])
    labels_path=os.path.abspath(argv[4])
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f'Model file not found: {weights_path}')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f'Label file not found: {labels_path}')
    return weights_path,labels_path


def log(file,string):
    file.write(string)
    file.flush()


def _batch_group(grouped,batch_id):
    """Return one small batch manifest without repeatedly scanning all rows."""
    try:
        return grouped.get_group(batch_id).copy()
    except KeyError as exc:
        raise RuntimeError(f'Batch {batch_id} missing from batches manifest') from exc


def main(argv=None):
    raise_open_file_limit()  # safety margin for very long runs (Linux only)
    audit_source_labels=str(os.environ.get('ILT_AUDIT_SOURCE_LABELS','0')).strip().lower() in {'1','true','t','yes','y'}
    argv=list(sys.argv if argv is None else argv)
    apply_runtime_overrides()
    if len(argv) not in (3,5):
        raise SystemExit(
            'Usage: main.py <input_folder_or_list> <output_folder> '
            '[<model_path> <label_path>]'
        )

    # Validate the model/labels contract before creating the output directory.
    # This avoids leaving a dead project folder for common configuration errors.
    labels_dict=None
    if len(argv)==3:
        weights_path=labels_path=''
        num_classes=int(defaults['num_classes'])
    else:
        weights_path,labels_path=read_network(argv)
        labels_dict=validate_labels(read_labels(labels_path), source=labels_path)
        num_classes=len(labels_dict)
        print(f'Num Classes: {num_classes}')
        print(f'Weights: {weights_path}')
        print(f'Labels: {labels_path}')

    input_path,output_path=read_paths(argv)
    pixel_size_um=parse_pixel_size(defaults.get('pixel_size', DEFAULT_PIXEL_SIZE_UM_PER_PX), DEFAULT_PIXEL_SIZE_UM_PER_PX)
    save_project_settings(
        output_path,
        pixel_size_um_per_px=pixel_size_um,
        source='classification',
        extra={'project_name': os.path.basename(output_path)},
    )
    project_name=os.path.basename(output_path)
    log_path=os.path.join(output_path,f'{project_name}.log')

    with open(log_path,'a',encoding='utf-8',buffering=1) as log_file:
        log(log_file,f'{datetime.datetime.now()}: START {project_name}\n')

        # Step 1: batch manifest + image references.
        df_batches=create_batches(input_path,output_path)
        log(log_file,f'{datetime.datetime.now()}: Image references\n')
        symlink_images(input_path,df_batches,output_path)

        batch_ids=sorted(df_batches['batch'].astype(str).unique().tolist())
        if not batch_ids:
            raise RuntimeError('No batches found in batches.csv')
        print(f'Classification batches: {len(batch_ids)}')

        base_id=str(defaults['base_tsne_id'])
        if base_id not in batch_ids:
            base_id=batch_ids[0]

        # Group once; get_group() avoids a full 8M-row boolean scan per batch.
        grouped=df_batches.groupby('batch',sort=False,observed=True)

        # A trained checkpoint completely replaces model weights, so avoid an
        # unnecessary ImageNet download in classification mode.
        model=get_model(load=True,num_classes=num_classes,pretrained=not bool(weights_path))
        images_folder=os.path.join(output_path,defaults['images'])

        log(log_file,f'{datetime.datetime.now()}: Base features\n')
        print('Computing base features...')
        start=timeit.default_timer()
        base_features,base_paths,base_predictions,base_confs=compute_features(
            images_folder,base_id,model,weights_path,labels_dict
        )
        print('Total time:',datetime.timedelta(seconds=timeit.default_timer()-start),'\n')

        log(log_file,f'{datetime.datetime.now()}: Base projections\n')
        print('Computing base projections...')
        start=timeit.default_timer()
        base_manifest=_batch_group(grouped,base_id)
        base_tsne=compute_projections(
            output_path,project_name,base_id,
            base_features,base_paths,base_manifest,base_predictions,base_confs,
            compute_base=True,save=False
        )
        print('Total time:',datetime.timedelta(seconds=timeit.default_timer()-start),'\n')

        df_folder=os.path.join(output_path,defaults['dataframes'])
        embeddings_folder=os.path.join(output_path,'embeddings')
        os.makedirs(embeddings_folder,mode=0o755,exist_ok=True)

        # Step 2: features/projections. Reuse the already-computed base features.
        log(log_file,f'{datetime.datetime.now()}: All feat/proj\n')
        print('Computing all features/projections...')
        for batch_id in tqdm.tqdm(batch_ids,unit='batch'):
            if batch_id==base_id:
                features,path_images,predictions,confs=(
                    base_features,base_paths,base_predictions,base_confs
                )
            else:
                features,path_images,predictions,confs=compute_features(
                    images_folder,batch_id,model,weights_path,labels_dict
                )

            embed_file=f'{batch_id}.npz'
            atomic_savez_compressed(os.path.join(embeddings_folder,embed_file),feats=features)

            manifest=_batch_group(grouped,batch_id)
            compute_projections(
                output_path,project_name,batch_id,
                features,path_images,manifest,predictions,confs,
                base_tsne=base_tsne
            )

            df_path=os.path.join(df_folder,f'{batch_id}_{project_name}.csv')
            batch_df=pd.read_csv(df_path,low_memory=False)
            embed_index={name:idx for idx,name in enumerate(path_images)}
            batch_df['embed_file']=os.path.join('embeddings',embed_file)
            batch_df['embed_idx']=batch_df['names'].map(embed_index)
            atomic_write_csv(batch_df,df_path,index=False)

        # Free the largest base tensors before labeling/writing CSVs.
        del base_features,base_paths,base_predictions,base_confs
        print()

        # Step 3: label predictions + stream complete.csv batch by batch.
        if labels_path:
            log(log_file,f'{datetime.datetime.now()}: Labeling\n')
            print('Labeling predictions...')
            start=timeit.default_timer()
            complete_path=os.path.join(output_path,'complete.csv')
            complete_tmp=complete_path+'.part'
            try:
                if os.path.exists(complete_tmp):
                    os.remove(complete_tmp)
            except OSError:
                pass
            first=True
            for batch_id in tqdm.tqdm(batch_ids,unit='batch'):
                df_path=os.path.join(df_folder,f'{batch_id}_{project_name}.csv')
                batch_df=pd.read_csv(df_path,low_memory=False)
                label_predictions(batch_df,labels_dict,audit_source_labels=audit_source_labels)
                atomic_write_csv(batch_df,df_path,index=False)
                batch_df[['names','pred']].to_csv(
                    complete_tmp,
                    mode='w' if first else 'a',
                    header=first,
                    index=False,
                )
                first=False
            os.replace(complete_tmp,complete_path)
            print('Total time:',datetime.timedelta(seconds=timeit.default_timer()-start))

        log(log_file,f'{datetime.datetime.now()}: END {project_name}\n')


if __name__=='__main__':
    main()
