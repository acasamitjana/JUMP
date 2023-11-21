# Author: Adria Casamitjana
# Date creation: 13/11/2021
# Historial of modification:
#    - Initial commit: Adria -  13/11/2021
from setup import *

from os.path import join, exists, dirname
from os import makedirs
import time
from argparse import ArgumentParser
import bids
import subprocess

# third party imports
import numpy as np
import nibabel as nib
import itertools

# project imports
from utils.jump_utils import initialize_graph_linear
from utils.fn_utils import compute_centroids_ras, compute_center_RAS
from utils.synthmorph_utils import labels_registration
from utils.io_utils import print_title_script


def register(subject, force_flag=False):
    sub_str = 'sub-' + subject
    timepoints = bids_loader.get_session(subject=subject)

    seg_dict = {'subject': subject, 'scope': 'synthseg', 'extension': 'nii.gz'}
    suffix_seg_list = bids_loader.get(**{**seg_dict, 'return_type': 'id', 'target': 'suffix'})
    suffix_seg_list = [s for s in suffix_seg_list if 'dseg' in s]
    for tp_id in timepoints:
        print('* Registering T=' + str(tp_id), end=': ', flush=True)
        sess_str = 'ses-' + tp_id
        seg_files = bids_loader.get(**{'suffix': suffix_seg_list, 'session': tp_id, **seg_dict})
        if len(bids_loader.get(subject=subject, session=tp_id, extension='npy', desc='aff', scope='jump-reg')) == len(seg_files):
            print('[done] Latent transforms already available.')
            continue

        deformations_dir = join(DIR_PIPELINES['jump-reg'], sub_str , sess_str, 'deformations')
        if force_flag: subprocess.call(['rm', '-rf', deformations_dir])
        if not exists(deformations_dir): makedirs(deformations_dir)


        if len(seg_files) <= 1:
            print('[done] It has only ' + str(len(seg_files)) + ' modalities. No registration is made.')
            continue

        print('centering images in RAS;', end=' ', flush=True)
        modality_proxies = {}
        for seg_file in seg_files:
            modality = seg_file.entities['suffix'].split('dseg')[0]

            outproxy, Tc = compute_center_RAS(nib.load(seg_file.path))

            cog_entities = {k: v for k, v in seg_file.entities.items() if k in filename_entities}
            cog_entities['suffix'] = modality
            if 'acquisition' in cog_entities.keys():
                cog_entities.pop('acquisition')
            im_cog_filename = bids_loader.build_path({**cog_entities, **{'desc': 'cog'}}, validate=False,
                                                     absolute_paths=False, path_patterns=BIDS_PATH_PATTERN)
            np.save(join(DIR_PIPELINES['seg'], im_cog_filename.replace('nii.gz', 'npy')), Tc)

            if 'run' in seg_file.entities.keys():
                modality += '.' + str(seg_file.entities['run'])
            modality_proxies[modality] = outproxy

        if all([exists(join(deformations_dir, str(mod_ref) + '_to_' + str(mod_flo) + '.npy')) for mod_ref, mod_flo in itertools.combinations(modality_proxies.keys(), 2)]):
            print('[done] Registrations have been already been computed.')
            continue

        print('computing centroids.')
        centroid_dict = {}
        ok = {}
        for mod, proxy in modality_proxies.items():
            centroid_dict[mod], ok[mod] = compute_centroids_ras(proxy, labels_registration)

        for ref_t, flo_t in itertools.combinations(modality_proxies.items(), 2):
            modality_ref = ref_t[0]
            modality_flo = flo_t[0]

            filename = str(modality_ref) + '_to_' + str(modality_flo)

            if exists(join(deformations_dir, filename + '.npy')):
                continue

            print('   > registering ' + str(modality_ref) + ' and ' + modality_flo + ';', end=' ', flush=True)
            initialize_graph_linear([centroid_dict[modality_ref], centroid_dict[modality_flo]],
                                    join(deformations_dir, filename + '.npy'))
            print('done.')

if __name__ == '__main__':

    print('\n\n\n\n\n')
    print('# ------------------------------------ #')
    print('# JUMP registration: initialize graph  #')
    print('# ------------------------------------ #')
    print('\n\n')

    parser = ArgumentParser(description="JUMP-registration: initialize graph", epilog='\n')
    parser.add_argument("--bids", default=BIDS_DIR, help="Bids root directory, including rawdata")
    parser.add_argument('--subjects', default=None, nargs='+')
    parser.add_argument("--force", action='store_true', help="Force the overwriting of existing files.")

    args = parser.parse_args()
    bids_dir = args.bids
    init_subject_list = args.subjects
    force_flag = args.force

    title = 'Running JUMP registration over the dataset in'
    print_title_script(title, args)

    print('\nReading dataset.\n')
    db_file = join(dirname(bids_dir), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False, database_path=db_file)

    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    bids_loader.add_derivatives(DIR_PIPELINES['jump-reg'])
    subject_list = bids_loader.get_subjects() if init_subject_list is None else init_subject_list

    ####################
    # Run registration #
    ####################
    failed_subjects = []
    for it_subject, subject in enumerate(subject_list):
        print('Subject: ' + subject)
        t_init = time.time()
        try:
            ms = register(subject, force_flag=force_flag)
        except:
            ms = subject

        if ms is not None:
            failed_subjects.append(subject)

        print('\nTotal registration time: ' + str(np.round(time.time() - t_init, 2)) + '\n')


    f = open(join(LOGS_DIR, 'initialize_graph.txt'), 'w')
    f.write('Total unprocessed subjects: ' + str(len(failed_subjects)))
    f.write(','.join(['\'' + s + '\'' for s in failed_subjects]))

    print('\n')
    print('Total failed subjects ' + str(len(failed_subjects)) + '. See ' + join(LOGS_DIR, 'initialize_graph.txt') + ' for more information.' )
    print('\n')
    print('# --------- FI (JUMP-reg: graph initialization) --------- #')
    print('\n')
