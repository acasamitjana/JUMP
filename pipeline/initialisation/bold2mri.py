import os
from os.path import exists, dirname, join, basename
from os import makedirs
from argparse import ArgumentParser
import subprocess
import json
import nibabel as nib
import bids
import numpy as np
from joblib import delayed, Parallel

from setup import *
from utils.io_utils import write_json_derivatives



def process_subject(subject, it_subject):
    print('Subject: ' + subject + ' (' + str(it_subject) + '/' + str(len(subject_list)) + ')')
    image_files_list = list(filter(lambda x: 'acq' not in x.filename and 'rec' not in x.filename, bids_loader.get(subject=subject, extension='nii.gz', suffix='bold')))
    for image_file in image_files_list:
        # print(t1w_i.filename)
        # if int(t1w_i.entities['session'][1:]) < 60: continue
        if image_file.entities['suffix'] not in ACCEPTED_MODALITIES: continue

        entities = {k: str(v) for k, v in image_file.entities.items() if k in filename_entities}
        bids_dirname = image_file.dirname

        proxy = nib.load(image_file.path)

        rec_slice = proxy.shape[-1] // 2
        entities['reconstruction'] = rec_slice
        anat_input = basename(bids_loader.build_path(entities, path_patterns=BIDS_PATH_PATTERN, strict=False))
        if not exists(join(bids_dirname, anat_input)):
            if len(proxy.shape) <= 3: continue

            aff = proxy.affine
            pixdim = np.sqrt(np.sum(aff * aff, 0))
            if any(pixdim == 0): continue  # aff[np.where(pixdim == 0)[0][0], np.where(pixdim == 0)[0][0]] = pixdim[0]

            img = nib.Nifti1Image(np.array(proxy.dataobj[..., rec_slice]), aff)
            nib.save(img, join(bids_dirname, anat_input))

            write_json_derivatives(pixdim, proxy.shape,
                                   join(bids_dirname, anat_input).replace('nii.gz', 'json'),
                                   extra_kwargs={"SelectedSlice": str(rec_slice)})

        synthsr_dirname = join(DIR_PIPELINES['bold-mri'], 'sub-' + subject, 'ses-' + image_file.entities['session'],
                               image_file.entities['datatype'])

        if not exists(join(synthsr_dirname, anat_input)):
            if basename(DIR_PIPELINES['bold-mri']) == 'synthsr':
                synthsr_out = subprocess.run(
                    ['mri_synthsr', '--i', join(bids_dirname, anat_input), '--o', join(synthsr_dirname, anat_input),
                     '--lowfield', '--cpu'], stderr=subprocess.PIPE)

                if synthsr_out.returncode != 0:
                    print('BOLD2MRI Error:')
                    print(synthsr_out.stderr)
                else:
                    print('BOLD2MRI Error: pipeline not implemented. Exiting...')
                    exit()

    return



if __name__ == 'main':

    print('\n\n\n\n\n')
    print('# --------------------- #')
    print('# Bold to MRI pipeline  #')
    print('# --------------------- #')
    print('\n\n')

    parser = ArgumentParser(description="PET-MRI synthesis", epilog='\n')
    parser.add_argument("--bids", default=BIDS_DIR, help="Bids root directory, including rawdata")
    parser.add_argument("--force", action='store_true', help="Force the script to overwriting existing segmentations in the derivatives/synthseg directory.")

    args = parser.parse_args()
    bids_dir = args.bids
    force_flag = args.force

    print('\nReading dataset.')
    db_file = join(dirname(bids_dir), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False, database_path=db_file)

    bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False)
    subject_list = bids_loader.get_subjects()

    print('\n\n########')
    if force_flag is True:
        print('Running PET-MRI synthesis over the dataset in ' + bids_dir + ', OVERWRITING existing files.')
    else:
        print('Running PET-MRI synthesis over the dataset in ' + bids_dir + ', only on files where segmentation is missing.')
    print('########')


    results = Parallel(n_jobs=4, backend="threading")(
        delayed(process_subject)(subject, it_subject) for it_subject, subject in enumerate(subject_list))

    print('\n')
    print('# --------- FI (Bold to MRI pipeline) --------- #')
    print('\n')

