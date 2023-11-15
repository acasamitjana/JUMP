from os.path import exists, dirname, join, basename
from argparse import ArgumentParser
import subprocess
import nibabel as nib
import bids
import numpy as np
from joblib import delayed, Parallel

from setup import *
from utils.io_utils import write_json_derivatives



def process_subject(subject, it_subject):
    print('Subject: ' + subject + ' (' + str(it_subject) + '/' + str(len(subject_list)) + ')')
    image_files_list = list(filter(lambda x: 'acq' not in x.filename and 'rec' not in x.filename, bids_loader.get(subject=subject, extension='nii.gz', suffix='pet')))
    for image_file in image_files_list:
        # print(t1w_i.filename)
        # if int(t1w_i.entities['session'][1:]) < 60: continue
        if image_file.entities['suffix'] not in VALID_MODALITIES: continue

        entities = {k: str(v) for k, v in image_file.entities.items() if k in filename_entities}
        bids_dirname = image_file.dirname

        proxy = nib.load(image_file.path)
        if len(proxy.shape) <= 3:
            anat_input = image_file.filename

        else:
            rec_slice = proxy.shape[-1] // 2
            entities['reconstruction'] = rec_slice
            anat_input = basename(bids_loader.build_path(entities, path_patterns=BIDS_PATH_PATTERN, strict=False))
            if not exists(join(bids_dirname, anat_input)):
                data = np.array(proxy.dataobj[..., rec_slice])
                aff = proxy.affine
                pixdim = np.sqrt(np.sum(aff * aff, 0))
                if any([p == 0 for p in pixdim]): continue #aff[np.where(pixdim == 0)[0][0], np.where(pixdim == 0)[0][0]] = pixdim[0]

                img = nib.Nifti1Image(data, aff)
                nib.save(img, join(bids_dirname, anat_input))

                write_json_derivatives(pixdim, proxy.shape,
                                       join(bids_dirname, anat_input).replace('nii.gz', 'json'),
                                       extra_kwargs={"SelectedSlice": str(rec_slice)})

        pet2mri_dirname = join(DIR_PIPELINES['pet-mri'], 'sub-' + subject, 'ses-' + image_file.entities['session'],
                               image_file.entities['datatype'])
        if not exists(join(pet2mri_dirname, anat_input)):
            subprocess.call(
                ['mri_synthsr', '--i', join(bids_dirname, anat_input), '--o', join(pet2mri_dirname, anat_input),
                 'lowfield', '--cpu'], stderr=subprocess.PIPE)


    return



if __name__ == '__main__':

    print('\n\n\n\n\n')
    print('# --------------------- #')
    print('# Bold to MRI pipeline  #')
    print('# --------------------- #')
    print('\n\n')

    # parse arguments
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

    bids_loader.add_derivatives(DIR_PIPELINES['pet-mri'])
    subject_list = bids_loader.get_subjects() if init_subject_list is None else init_subject_list

    results = Parallel(n_jobs=4, backend="threading")(
        delayed(process_subject)(subject, it_subject) for it_subject, subject in enumerate(subject_list))

    print('\n')
    print('# --------- FI (PET to MRI pipeline) --------- #')
    print('\n')


