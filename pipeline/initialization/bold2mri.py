from setup import *

from os.path import exists, dirname, join, basename
from argparse import ArgumentParser
import subprocess
import nibabel as nib
import bids
import numpy as np
from joblib import delayed, Parallel

from utils.io_utils import write_json_derivatives



def process_subject(subject, bids_loader, force_flag=False):
    image_files_list = list(filter(lambda x: 'acq' not in x.filename and 'rec' not in x.filename, bids_loader.get(subject=subject, extension='nii.gz', suffix='bold')))
    for image_file in image_files_list:
        print('  * Running ' + image_file.path)
        if image_file.entities['suffix'] not in VALID_MODALITIES: continue

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

        if not exists(join(synthsr_dirname, anat_input)) or force_flag:
            if basename(DIR_PIPELINES['bold-mri']) == 'synthsr':
                synthsr_out = subprocess.run(
                    ['mri_synthsr', '--i', join(bids_dirname, anat_input), '--o', join(synthsr_dirname, anat_input),
                      '--lowfield', '--cpu'], stderr=subprocess.PIPE)

                if synthsr_out.returncode != 0:
                    print('BOLD2MRI Error:')
                    print(synthsr_out.stderr)
                # else:
                #     print('BOLD2MRI Error: pipeline not implemented. Exiting...')
                #     exit()

    return


if __name__ == '__main__':

    print('\n\n\n\n\n')
    print('# --------------------- #')
    print('# Bold to MRI pipeline  #')
    print('# --------------------- #')
    print('\n\n')

    parser = ArgumentParser(description="PET-MRI synthesis", epilog='\n')
    parser.add_argument("--bids", default=BIDS_DIR, help="Bids root directory, including rawdata")
    parser.add_argument('--subjects', default=None, nargs='+', help="(optional) specify which subjects to process")
    parser.add_argument('--num_cores', default=1, type=int, help="(optional) specify the number of cores used.")
    parser.add_argument("--force", action='store_true', help="Force the script to overwriting existing segmentations in the derivatives/synthseg directory.")

    args = parser.parse_args()
    bids_dir = args.bids
    init_subject_list = args.subjects
    force_flag = args.force

    print('\n\n########################')
    if force_flag is True:
        print('Running PET-MRI synthesis over the dataset in ' + bids_dir + ', OVERWRITING existing files.')
    else:
        print('Running PET-MRI synthesis over the dataset in ' + bids_dir + ', only on files where output is missing.')
        if init_subject_list is not None:
            print('   - Selected subjects: ' + ','.join(init_subject_list) + '.')
    print('########################')

    print('\nReading dataset.\n')
    db_file = join(dirname(bids_dir), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False, database_path=db_file)

    bids_loader.add_derivatives(DIR_PIPELINES['bold-mri'])
    subject_list = bids_loader.get_subjects() if init_subject_list is None else init_subject_list

    if args.num_cores == 1:
        for it_s, subject in enumerate(subject_list):
            print('Subject: ' + subject + ' (' + str(it_s) + '/' + str(len(subject_list)) + ')')
            process_subject(subject, bids_loader, force_flag=force_flag)
    else:
        print('Running N=' + str(len(subject_list)) + ' in parallel using ' + str(args.num_cores) + ' jobs.')
        results = Parallel(n_jobs=args.num_cores, backend="threading")(
            delayed(process_subject)(subject, bids_loader) for it_subject, subject in enumerate(subject_list))

    print('\n')
    print('# --------- FI (Bold to MRI pipeline) --------- #')
    print('\n')

