import os
import pdb
import json

filename_entities = ['subject', 'session', 'run', 'acquisition', 'suffix', 'extension', 'task', 'tracer', 'reconstruction']
anat_modalities = ['T1w', 'T2w', 'T2star', 'T2starw', 'FLAIR', 'FLASH', 'PD', 'PDw', 'PDT2']
ACCEPTED_MODALITIES = ['T1w', 'T2w', 'FLAIR', 'bold', 'pet']
BIDS_PATH_PATTERN = [
    "sub-{subject}[/ses-{session}]/{datatype<anat>|anat}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<T1w|T2w|T2star|T2starw|FLAIR|FLASH|PD|PDw|PDT2|inplaneT[12]|angio|dseg|posteriors|svf|T1wmask|T1wdseg|T2wmask|T2wdseg|FLAIRmask|FLAIRdseg>}{extension<.nii|.nii.gz|.json|.txt|.npy>|.nii.gz}",
    "sub-{subject}[/ses-{session}]/{datatype<func>|func}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<bold|cbv|sbref>}{extension<.nii|.nii.gz|.json|.txt|.npy>|.nii.gz}",
    "sub-{subject}[/ses-{session}]/{datatype<pet>|pet}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_trc-{tracer}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<pet>}{extension<.nii|.nii.gz|.json||.txt|.npy>|.nii.gz}",

    # "sub-{subject}[/ses-{session}]/{datatype<func>|func}/sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_dir-{direction}][_run-{run}][_echo-{echo}][_part-{part}]_{suffix<bold|cbv|sbref>}{extension<.nii|.nii.gz|.json>|.nii.gz}",
    # "sub-{subject}[/ses-{session}]/{datatype<pet>|pet}/sub-{subject}[_ses-{session}][_task-{task}][_acq-{acquisition}][_trc-{tracer}][_rec-{reconstruction}][_run-{run}]_{suffix<pet>}{extension<.nii|.nii.gz|.json>|.nii.gz}",
    # "sub-{subject}[/ses-{session}]/{datatype<anat>|anat}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_trc-{tracer}][_run-{run}][_desc-{desc}]_{suffix<T1w|T2w|T2star|T2starw|FLAIR|FLASH|PD|PDw|PDT2|inplaneT[12]|angio|dseg>}{extension<.nii|.nii.gz|.json|.npy>|.nii.gz}",

]

# MRI Templates
repo_home = os.environ.get('PYTHONPATH')

MNI_TEMPLATE = os.path.join(repo_home, 'data', 'atlas', 'mni_icbm152_t1norm_tal_nlin_sym_09a.nii.gz')
MNI_to_ATLAS = os.path.join(repo_home, 'data', 'atlas', 'mni_to_synthmorph_atlas.aff.npy')

MNI_ATLAS_TEMPLATE = os.path.join(repo_home, 'data', 'atlas', 'mni_reg_to_synthmorph_atlas.nii.gz')
MNI_ATLAS_TEMPLATE_SEG = os.path.join(repo_home, 'data', 'atlas', 'mni_reg_to_synthmorph_atlas.seg.nii.gz')
MNI_ATLAS_TEMPLATE_MASK = os.path.join('data', 'atlas', 'mni_reg_to_synthmorph_atlas.mask.nii.gz')

MNI_TEMPLATE_SEG = os.path.join(repo_home, 'data', 'atlas', 'mni_icbm152_synthseg_tal_nlin_sym_09a.nii.gz')
MNI_TEMPLATE_MASK = os.path.join(repo_home, 'data', 'atlas', 'mni_icbm152_mask_tal_nlin_sym_09a.nii.gz')

# NiftyReg
NIFTY_REG_DIR = os.environ['NIFTY_REG'] if 'NIFTY_REG' in os.environ else False # '/mnt/HDD/Software/niftyreg/build/' #
if not NIFTY_REG_DIR:
    print("NiftyReg not available.")

# DEBUG
DEBUG = os.environ['DEBUG']
if DEBUG == 'True':
    DEBUG = True
else:
    DEBUG = False

# BIDS directories ---- Environment variables.
BIDS_DIR = os.environ['BIDS_DIR']
if BIDS_DIR[-1] == '/': BIDS_DIR = BIDS_DIR[:-1]
JUMP_DIR = os.path.dirname(BIDS_DIR)
DERIVATIVES_DIR = os.path.join(JUMP_DIR, 'derivatives')
RESULTS_DIR = os.path.join(JUMP_DIR, 'results')

DIR_PIPELINES = {'pet-mri': os.path.join(JUMP_DIR, 'synthsr'),
                 'bold-mri': os.path.join(JUMP_DIR, 'synthsr'),
                 'seg': os.path.join(JUMP_DIR, 'synthseg'),
                 'jump-reg': os.path.join(JUMP_DIR, 'jump-reg')}
DESC_PIPELINES = {'pet-mri': 'PET-MRI synthesis using a super resolution using Freesurfer 7.3.2',
                  'bold-mri': 'BOLD-MRI synthesis using a super resolution using Freesurfer 7.3.2',
                  'seg': 'SynthSeg segmentation using Freesurfer 7.3.2',
                  'jump-reg': 'JUMP multimodal registration'}
if not BIDS_DIR: raise ValueError("Please, specify environment variable DB")

for d, d_str in DESC_PIPELINES:
    if not os.path.exists(DIR_PIPELINES[d]): os.makedirs(DIR_PIPELINES[d])
    data_descr_path = os.path.join(DIR_PIPELINES[d], 'dataset_description.json')
    if not os.path.exists(data_descr_path):
        data_descr = {}
        data_descr['Name'] = d
        data_descr['BIDSVersion'] = '1.0.2'
        data_descr['GeneratedBy'] = [{'Name': d}]
        data_descr['Description'] = d_str
        data_descr_path = os.path.join(DIR_PIPELINES[d], 'dataset_description.json')
        json_object = json.dumps(data_descr, indent=4)
        with open(data_descr_path, 'w') as outfile:
            outfile.write(json_object)

VERBOSE = os.environ['VERBOSE'] if 'VERBOSE' in os.environ.keys() else False
if VERBOSE:
    print('     ')
    print('     ')
    print('DEBUG: ' + str(DEBUG is True))
    print('DATASET USED ($BIDS_DIR): ' + BIDS_DIR)
    print('DERIVATIVES_DIR: ' + DERIVATIVES_DIR)
    print('RESULTS_DIR: ' + RESULTS_DIR)
    print('using CPU, hiding all CUDA_VISIBLE_DEVICES')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = 'cpu'