import os
from os.path import exists, dirname, join, basename
from os import makedirs
from argparse import ArgumentParser
import subprocess
import json
import nibabel as nib
import bids
import numpy as np

from setup import *
from utils.io_utils import write_json_derivatives


# parse arguments
parser = ArgumentParser(description="SynthSeg segmentation using freesurfer implementation. It includes  segmentation "
                                    "and parcellation volumes, a summary volfile (synthseg dir) and the resampled image "
                                    "(rawdata dir). No robust or QC flags are used.", epilog='\n')
parser.add_argument("--bids", default=BIDS_DIR, help="Bids root directory, including rawdata")
parser.add_argument("--force", action='store_true', help="Force the script to overwriting existing segmentations in the derivatives/synthseg directory.")

args = parser.parse_args()
bids_dir = args.bids #'/vol/home/student_pd12/ADRIA/PD-BIDS'
force_flag = args.force #'/vol/home/student_pd12/ADRIA/PD-BIDS'

synthseg_dir = join(dirname(BIDS_DIR), 'derivatives', 'synthseg')
if not exists(synthseg_dir): makedirs(synthseg_dir)

data_descr = {}
data_descr['Name'] = 'synthsr'
data_descr['BIDSVersion'] = '1.0.2'
data_descr['GeneratedBy'] = [{'Name': 'synthsr'}]
data_descr['Description'] = 'SynthSR using Freesurfer 7.4. It includes the --lowfield flag. '

data_descr_path = join(SYNTHSR_DIR, 'dataset_description.json')
json_object = json.dumps(data_descr, indent=4)
with open(data_descr_path, 'w') as outfile:
    outfile.write(json_object)

print('\nReading dataset.')
# if exists(bids_dir + '.db'):
#     shutil.rmtree(bids_dir + '.db')
bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False)
bids_loader.add_derivatives(SYNTHSEG_DIR)
bids_loader.add_derivatives(SYNTHSR_DIR)
# bids_loader.save(bids_dir + '.db')
# bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False, database_path=bids_dir + '.db')
subject_list = bids_loader.get_subjects()

print('\n\n########')
if force_flag is True:
    print('Running SynthSeg over the dataset in ' + bids_dir + ', OVERWRITING existing files.')
else:
    print('Running SynthSeg over the dataset in ' + bids_dir + ', only on files where segmentation is missing.')
print('########')

def process_subject(subject, it_subject):
    print('Subject: ' + subject + ' (' + str(it_subject) + '/' + str(len(subject_list)) + ')')
    input_files = []
    res_files = []
    output_files = []
    vol_files = []
    t1w_list = list(filter(lambda x: 'acq' not in x.filename and 'rec' not in x.filename, bids_loader.get(subject=subject, extension='nii.gz')))
    for t1w_i in t1w_list:
        # print(t1w_i.filename)
        # if int(t1w_i.entities['session'][1:]) < 60: continue
        if t1w_i.entities['suffix'] not in ['T1w', 'T2w', 'FLAIR', 'bold', 'pet']: continue

        synthseg_dirname = join(SYNTHSEG_DIR, 'sub-' + subject, 'ses-' + t1w_i.entities['session'],
                                t1w_i.entities['datatype'])
        if not exists(synthseg_dirname): makedirs(synthseg_dirname)
        entities = {k: str(v) for k, v in t1w_i.entities.items() if k in filename_entities}
        bids_dirname = t1w_i.dirname

        if t1w_i.entities['datatype'] == 'func':
            anat_res_full = basename( bids_loader.build_path({**entities, **{'acquisition': '1'}}, path_patterns=BIDS_PATH_PATTERN))
            proxy = nib.load(t1w_i.path)

            entities['reconstruction'] = proxy.shape[-1] // 2
            anat_input = basename(bids_loader.build_path(entities, path_patterns=BIDS_PATH_PATTERN, strict=False))
            if not exists(join(bids_dirname, anat_input)):
                if len(proxy.shape) <= 3: continue

                aff = proxy.affine
                pixdim = np.sqrt(np.sum(aff * aff, 0))
                if any(pixdim == 0): continue  # aff[np.where(pixdim == 0)[0][0], np.where(pixdim == 0)[0][0]] = pixdim[0]

                img = nib.Nifti1Image(np.array(proxy.dataobj[..., proxy.shape[-1] // 2]), aff)
                nib.save(img, join(bids_dirname, anat_input))

                write_json_derivatives(pixdim, proxy.shape,
                                       join(bids_dirname, anat_input).replace('nii.gz', 'json'),
                                       extra_kwargs={"SelectedSlice": str(proxy.shape[-1] // 2)})

            synthsr_dirname = join(SYNTHSR_DIR, 'sub-' + subject, 'ses-' + t1w_i.entities['session'], t1w_i.entities['datatype'])
            if not exists(join(synthsr_dirname, anat_input)):
                synthsr_out = subprocess.run(['mri_synthsr', '--i', join(bids_dirname, anat_input), '--o',
                                              join(synthsr_dirname, anat_input), '--lowfield', '--cpu'],
                                             stderr=subprocess.PIPE)
                if synthsr_out.returncode != 0:
                    print()
                    print(synthsr_out.stderr)

        elif t1w_i.entities['datatype'] == 'pet':
            proxy = nib.load(t1w_i.path)
            if len(proxy.shape) <= 3:
                anat_input = t1w_i.filename

            else:
                entities['reconstruction'] = proxy.shape[-1] // 2
                anat_input = basename(
                    bids_loader.build_path(entities, path_patterns=BIDS_PATH_PATTERN, strict=False))
                if not exists(join(bids_dirname, anat_input)):

                    data = np.array(proxy.dataobj[..., proxy.shape[-1] // 2])
                    aff = proxy.affine
                    pixdim = np.sqrt(np.sum(aff * aff, 0))
                    if any(pixdim == 0):
                        aff[np.where(pixdim == 0)[0][0], np.where(pixdim == 0)[0][0]] = pixdim[0]
                    img = nib.Nifti1Image(data, aff)
                    nib.save(img, join(bids_dirname, anat_input))

                    write_json_derivatives(pixdim, proxy.shape,
                                           join(bids_dirname, anat_input).replace('nii.gz', 'json'),
                                           extra_kwargs={"SelectedSlice": str(proxy.shape[-1] // 2)})

            synthsr_dirname = join(SYNTHSR_DIR, 'sub-' + subject, 'ses-' + t1w_i.entities['session'],
                                   t1w_i.entities['datatype'])
            if not exists(join(synthsr_dirname, anat_input)):
                synthsr_out = subprocess.run(['mri_synthsr', '--i', join(bids_dirname, anat_input), '--o',
                                              join(synthsr_dirname, anat_input), '--lowfield', '--cpu'],
                                             stderr=subprocess.PIPE)

            bids_dirname = synthsr_dirname

        elif t1w_i.entities['suffix'] != 'T1w':
            proxy = nib.load(t1w_i.path)
            aff = proxy.affine
            pixdim = np.sqrt(np.sum(aff * aff, 0))
            anat_input = t1w_i.filename
            if any(pixdim > 2):
                synthsr_dirname = join(SYNTHSR_DIR, 'sub-' + subject, 'ses-' + t1w_i.entities['session'],
                                       t1w_i.entities['datatype'])
                if not exists(join(synthsr_dirname, anat_input)):
                    synthsr_out = subprocess.run(['mri_synthsr', '--i', join(bids_dirname, anat_input), '--o',
                                                  join(synthsr_dirname, anat_input), '--cpu'], stderr=subprocess.PIPE)
                bids_dirname = synthsr_dirname

        else:
            anat_input = t1w_i.filename

        anat_res = basename(bids_loader.build_path({**entities, **{'acquisition': '1'}},
                                                   path_patterns=BIDS_PATH_PATTERN, validate=False))
        anat_seg = anat_res.replace(entities['suffix'], entities['suffix'] + 'dseg')
        anat_vols = anat_seg.replace('nii.gz', 'tsv')
        if not exists(join(synthseg_dirname, anat_seg)) or force_flag:
            input_files += [join(bids_dirname, anat_input)]
            res_files += [join(bids_dirname, anat_res)]
            output_files += [join(synthseg_dirname, anat_seg)]
            vol_files += [join(synthseg_dirname, anat_vols)]

    return input_files, res_files, output_files, vol_files

from joblib import delayed, Parallel

results = Parallel(n_jobs=4, backend="threading")(delayed(process_subject)(subject, it_subject)
                                                  for it_subject, subject in enumerate(subject_list))

input_files = []
res_files = []
output_files = []
vol_files = []

for i in results[0]: input_files.extend(i)
for i in results[1]: res_files.extend(i)
for i in results[2]: output_files.extend(i)
for i in results[3]: vol_files.extend(i)

print('[mri_synthsr] DONE.')
print('\nDone\n')

