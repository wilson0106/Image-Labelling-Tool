import math
import os
import shutil
import re
import timeit
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from aux import defaults
from ilt_contracts import atomic_write_csv, is_image_filename


def _discover_images(input_path):
    """Recursively discover supported image files below *input_path*.

    Returns ``(filename, relative_parent)`` tuples. Directory symlinks are
    followed, while visited directories are tracked by device/inode to avoid
    cycles.
    """
    root=os.path.abspath(input_path)
    imgs=[]
    stack=[root]
    seen_dirs=set()
    dirs_scanned=0
    errors=0

    print('Scanning image files recursively...')
    while stack:
        current=stack.pop()
        try:
            st=os.stat(current,follow_symlinks=True)
            key=(st.st_dev,st.st_ino)
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=True):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=True) and is_image_filename(entry.name):
                            parent=os.path.dirname(entry.path)
                            rel_parent=os.path.relpath(parent,root)
                            imgs.append((entry.name,'' if rel_parent=='.' else rel_parent))
                    except OSError as exc:
                        errors+=1
                        print(f'[WARN] Could not inspect: {entry.path} ({exc})')
        except OSError as exc:
            errors+=1
            print(f'[WARN] Could not scan folder: {current} ({exc})')
            continue
        dirs_scanned+=1
        if dirs_scanned%50==0:
            print(f'  scanned folders: {dirs_scanned} | images found: {len(imgs)}')

    print(f'Image scan complete: {len(imgs)} images in {dirs_scanned} folders.')
    if errors:
        print(f'[WARN] Scan completed with {errors} filesystem read error(s).')
    print()
    return imgs


def _discover_from_list(list_path):
    """Read a text file containing image paths.

    Absolute paths remain absolute in ``klass``. This makes the list portable
    across platforms without any hard-coded dataset root.
    """
    imgs=[]
    list_dir=os.path.dirname(os.path.abspath(list_path))
    with open(list_path,'r',encoding='utf-8-sig') as fh:
        for raw in fh:
            raw=raw.strip().strip('"')
            if not raw or not is_image_filename(raw):
                continue
            p=Path(raw).expanduser()
            if not p.is_absolute():
                p=Path(list_dir)/p
            p=p.resolve()
            imgs.append((p.name,str(p.parent)))
    return imgs


def _source_label_from_klass(value):
    """Return the original class/source folder label without altering ``klass``.

    For the normal folder-based datasets used by training/classification this is
    the leaf parent directory (the same convention used by train.py). For list
    inputs, ``klass`` can be an absolute parent path; the leaf directory remains
    a stable human-readable source label.
    """
    raw=str(value or '').strip().rstrip('/\\')
    if not raw:
        return ''
    parts=[x for x in re.split(r'[\\/]+',raw) if x]
    return parts[-1] if parts else raw


def create_batches(input_path, output_path):
    """Create ``batches.csv`` and return its dataframe.

    Random batch assignment is performed without shuffling/copying the complete
    dataframe. On multi-million-image datasets this saves a substantial temporary
    memory spike compared with ``df.sample(frac=1)``.
    """
    print('Creating batches...')
    if os.path.isdir(input_path):
        imgs=_discover_images(input_path)
    else:
        imgs=_discover_from_list(input_path)

    n=len(imgs)
    if n==0:
        raise RuntimeError(f'No supported images found in input: {input_path}')

    max_size=max(1,int(defaults['BATCH_MAX_SIZE']))
    num_batches=math.ceil(n/max_size)
    images_per_batch=math.ceil(n/num_batches)

    df=pd.DataFrame(imgs,columns=['names','klass'])
    # Preserve the folder-derived label explicitly. ``klass`` remains the
    # path locator used to find the original image; source_label is never
    # replaced by a model prediction and is therefore safe for training-set audits.
    df['source_label']=df['klass'].map(_source_label_from_klass).astype(str)

    # Downstream CSV/projection joins use the image basename as the key.
    # Duplicate basenames would silently collide inside batch folders and make
    # image-to-embedding association ambiguous, so fail early with examples.
    dup_mask=df['names'].duplicated(keep=False)
    if bool(dup_mask.any()):
        examples=df.loc[dup_mask,['names','klass']].head(12)
        details='\n'.join(
            f"  {r.names}  <-  {r.klass}" for r in examples.itertuples(index=False)
        )
        raise RuntimeError(
            'Duplicate image filenames were found in different source folders. '
            'ILT requires unique basenames so images, CSV rows and embeddings stay '
            f'unambiguous. First occurrences:\n{details}'
        )

    # Assign each source row to a random balanced batch without reordering the
    # large object/string columns. int32 is sufficient for realistic batch counts.
    rng=np.random.default_rng(0)
    perm=rng.permutation(n)
    batch_codes=np.empty(n,dtype=np.int32)
    batch_codes[perm]=np.arange(n,dtype=np.int64)//images_per_batch
    batch_codes=np.minimum(batch_codes,num_batches-1).astype(np.int32,copy=False)
    categories=[f'batch_{i:04d}' for i in range(1,num_batches+1)]
    df['batch']=pd.Categorical.from_codes(batch_codes,categories=categories)

    atomic_write_csv(df,os.path.join(output_path,'batches.csv'),index=False)
    print(f'Done creating batches: {n} images in {num_batches} batches.\n')
    return df


def _original_path(input_path, klass, name):
    parent=Path(str(klass))
    if not parent.is_absolute():
        parent=Path(input_path)/parent
    return parent/str(name)


def _same_volume(a: Path, b: Path) -> bool:
    """Best-effort same-volume check for hard-link fallback."""
    try:
        if os.name == 'nt':
            return os.path.splitdrive(str(a.resolve()))[0].lower() == os.path.splitdrive(str(b.resolve()))[0].lower()
        return os.stat(a).st_dev == os.stat(b.parent).st_dev
    except Exception:
        return False


def _make_symbolic_link(original_path: Path, dest_path: Path) -> None:
    """Create a relative symlink when possible; use absolute target across Windows drives."""
    try:
        target=os.path.relpath(str(original_path), start=str(dest_path.parent))
    except ValueError:
        # Windows relpath cannot cross drive letters. An absolute symlink still
        # avoids duplicating the image, although moving the project then requires
        # rebuilding links.
        target=str(original_path.resolve())
    os.symlink(target, str(dest_path), target_is_directory=False)


def _link_or_copy(original_path, dest_path, mode):
    original_path=Path(original_path)
    dest_path=Path(dest_path)
    mode=str(mode or 'auto').lower()

    if mode=='auto':
        # Prefer a portable symbolic link. On Windows without Developer Mode /
        # symlink privilege, fall back to a hard link when both files live on
        # the same volume. Never silently copy a huge dataset.
        try:
            _make_symbolic_link(original_path,dest_path)
            return 'ln'
        except OSError as link_exc:
            if _same_volume(original_path,dest_path):
                try:
                    os.link(str(original_path),str(dest_path))
                    return 'hardlink'
                except OSError:
                    pass
            raise RuntimeError(
                'Could not create a zero-copy image reference. On Windows, enable '
                'Developer Mode to allow symbolic links, or keep the dataset and '
                'output on the same drive so AUTO can use hard links. '
                'Choose CP only if you intentionally want to duplicate images.'
            ) from link_exc
    if mode=='ln':
        _make_symbolic_link(original_path,dest_path)
        return 'ln'
    if mode=='hardlink':
        if not _same_volume(original_path,dest_path):
            raise RuntimeError('Hard links require source and output on the same filesystem/drive.')
        os.link(str(original_path),str(dest_path))
        return 'hardlink'
    if mode=='cp':
        shutil.copy2(str(original_path),str(dest_path))
        return 'cp'
    if mode=='mv':
        raise ValueError(
            "Destructive img_mode='mv' is disabled. Use auto, hardlink, ln or cp."
        )
    raise ValueError(f'Unsupported image mode: {mode!r}')


def _preflight_image_mode(input_path, images_folder, df):
    """Fail early if the chosen filesystem operation cannot be created."""
    if df.empty:
        return
    mode=str(defaults.get('img_mode','auto')).lower()
    if mode not in {'auto','ln','hardlink','cp'}:
        raise ValueError(
            f'Wrong img_mode {mode!r}; expected auto, ln, hardlink or cp. '
            "Destructive mode 'mv' is disabled."
        )

    row=next(df.itertuples(index=False))
    src=_original_path(input_path,row.klass,row.names)
    if not src.exists():
        raise FileNotFoundError(f'Input image not found: {src}')
    test_dir=Path(images_folder)/'.ilt_preflight'
    test_dir.mkdir(parents=True,exist_ok=True)
    dest=test_dir/str(row.names)
    try:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        _link_or_copy(src,dest,mode)
    except Exception as exc:
        if mode in {'auto','ln','hardlink'}:
            raise RuntimeError(
                f'Image reference preflight failed in mode {mode!r}: {exc}'
            ) from exc
        raise
    finally:
        try:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            test_dir.rmdir()
        except OSError:
            pass


def symlink_batch_images(input_path, images_folder, df):
    """Materialize one batch and return ``(created, errors)``."""
    if df.empty:
        return 0,[]
    batch_id=str(df.iloc[0]['batch'])
    batch_folder=Path(images_folder)/batch_id/defaults['inner_folder']
    batch_folder.mkdir(parents=True,exist_ok=True)
    mode=str(defaults.get('img_mode','auto')).lower()
    created=0
    errors=[]

    for row in df.itertuples(index=False):
        src=_original_path(input_path,row.klass,row.names)
        dst=batch_folder/str(row.names)
        try:
            if dst.exists() or dst.is_symlink():
                # Output folders are expected to be new. Treat an identical
                # existing destination as already materialized.
                created+=1
                continue
            _link_or_copy(src,dst,mode)
            created+=1
        except Exception as exc:
            errors.append((str(src),str(dst),repr(exc)))
            if len(errors)>=20:
                # Keep diagnostics bounded while still counting failures below.
                break
    return created,errors


def symlink_images(input_path, df, dataset_path):
    """Create batch image folders with bounded cross-platform concurrency."""
    mode=str(defaults.get('img_mode','auto')).strip().lower()
    if mode=='mv':
        raise ValueError(
            "Destructive img_mode='mv' is disabled. Use auto, hardlink, ln or cp."
        )
    images_folder=os.path.join(dataset_path,defaults['images'])
    os.makedirs(images_folder,mode=0o755,exist_ok=False)
    print(f'Creating image references in {dataset_path}')
    start=timeit.default_timer()

    _preflight_image_mode(input_path,images_folder,df)

    workers=max(1,min(int(os.environ.get('ILT_FILE_WORKERS','6')),os.cpu_count() or 1))
    pending=set()
    total_created=0
    failures=[]

    # At most 2*workers batch dataframes are outstanding, preventing a second
    # copy of the complete multi-million-row manifest in memory.
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='ilt-files') as ex:
        for _,group in df.groupby('batch',sort=False,observed=True):
            pending.add(ex.submit(symlink_batch_images,input_path,images_folder,group.copy()))
            if len(pending)>=workers*2:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for fut in done:
                    created,errs=fut.result()
                    total_created+=created
                    failures.extend(errs)
                    if failures:
                        break
            if failures:
                break
        if not failures:
            for fut in pending:
                created,errs=fut.result()
                total_created+=created
                failures.extend(errs)
                if failures:
                    break

    if failures:
        first='\n'.join(f'  {src} -> {dst}: {err}' for src,dst,err in failures[:5])
        raise RuntimeError(
            f'Failed to materialize one or more batch images. First errors:\n{first}'
        )

    end=timeit.default_timer()
    print(f'Image references ready: {total_created} files.')
    print(f'Total time: {timedelta(seconds=(end-start))}\n')
