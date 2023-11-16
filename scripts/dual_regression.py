import os
import pdb
from os import listdir, makedirs
from os.path import exists, join, isdir, dirname
from argparse import ArgumentParser

import bids
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils import def_utils, fn_utils, jump_utils
from src.preprocessing import runICA, registerMNI2Image, register2MNI
from setup import *

# def compute_MNI_field(mni_dir, subject, session):
#     # Registrar to MNI2mm: func --> T1w --> MNI 2mm.
#     def_MNI_files = list(filter(lambda f: 'desc-field_nonlinear' in f and 'space-IMAGE' in f, listdir(join(mni_dir, subject, session, 'anat'))))
#     if len(def_MNI_files) != 1:
#         print('None / Too many T1w image available N=' + str(len(def_MNI_files)) + '. Please, refine search.')
#         return None, None
#
#     def_MNI_file = def_MNI_files[0]
#     def_MNI_proxy = nib.load(join(mni_dir, subject, session, 'anat', def_MNI_file))
#     mni_2mm_proxy = nib.load(MNI_TEMPLATE_2mm)
#     def_MNI_2mm_proxy = def_utils.vol_resample(mni_2mm_proxy, def_MNI_proxy)
#
#     t1w_entities = {}
#     if 'run' in def_MNI_file:
#         bold_entities['run'] = def_MNI_file.split('run-')[1].split('_')[0]
#
#     return def_MNI_2mm_proxy, t1w_entities

# def get_reg2t1w(bold_entities, t1w_entities):
#     subject = bold_entities['subject']
#     sess = bold_entities['session']
#
#     aff_entities = {'subject': subject, 'session': sess, 'scope': 'smr-lin', 'desc': 'aff', 'extension': 'npy'}
#
#     aff_bold_entities = {'suffix': 'bold'}
#     if 'run' in bold_entities.keys():
#         aff_bold_entities['run'] = bold_entities['run']
#
#     aff_t1w_entities = {'suffix': 'T1w'}
#     if 'run' in t1w_entities.keys():
#         aff_t1w_entities['run'] = t1w_entities['run']
#
#     affine_bold_file = list(filter(lambda f:
#                                    all([v in f for k, v in {**aff_entities, **aff_bold_entities}.items()
#                                         if k in ['subject', 'session', 'desc', 'extension', 'suffix', 'run']]),
#                                    listdir(join(smr_lin_dir, subject, sess, 'func'))))
#     # affine_bold_file = bids_loader.get(**{**aff_entities, **aff_bold_entities})
#     if len(affine_bold_file) != 1:
#         print('No aff-bold available. It is required for T1w-MNI registration.')
#         return
#     affine_bold_matrix = np.load(join(smr_lin_dir, subject, sess, 'func', affine_bold_file[0]))
#
#     affine_t1w_file = list(filter(lambda f:
#                                   all([v in f for k, v in {**aff_entities, **aff_t1w_entities}.items()
#                                        if k in ['subject', 'session', 'desc', 'extension', 'suffix', 'run']]),
#                                   listdir(join(smr_lin_dir,  subject,  sess, 'anat'))))
#     # affine_t1w_file = bids_loader.get(**{**aff_entities, **aff_t1w_entities})
#     if len(affine_t1w_file) != 1:
#         print('No aff-T1w available. It is required for T1w-MNI registration.')
#         return
#
#     affine_t1w_matrix = np.load(join(smr_lin_dir,  subject,  sess, 'anat', affine_t1w_file[0]))
#     new_aff = affine_bold_matrix @ np.linalg.inv(affine_t1w_matrix)  # @ t1w_seg_proxy.affine
#
#     return new_aff



# Need to downscale the field.

if __name__ == '__main__':


    print('\n\n\n\n\n')
    print('# --------------------------------- #')
    print('# JUMP registration: compute graph  #')
    print('# --------------------------------- #')
    print('\n\n')

    parser = ArgumentParser(description='Computes the prediction of certain models')
    parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
    parser.add_argument('--subjects', default=None, nargs='+', help="(optional) specify which subjects to process")
    parser.add_argument('--ica', default=join(RESULTS_DIR, 'melodic.ica.group', 'melodic_IC.nii.gz'),
                        help="ICA maps to run the algorithm; by default, it uses the group-ICA from bold_preproc.py")
    parser.add_argument('--subject_space', action='store_true', help="run dual regression also un subject space.")
    parser.add_argument('--force', action='store_true', help="force the preprocessing even if it already exists.")
    args = parser.parse_args()

    proxyica = nib.load(args.ica)
    n_ica = proxyica.shape[-1]
    ica_maps = np.array(proxyica.dataobj)
    tmp_dir = '/tmp/BOLD_dual_regression'
    if not exists(tmp_dir): makedirs(tmp_dir)

    print('\n\n########################')
    if args.force is True:
        print('Running BOLD dual-regression over the dataset in ' + args.bids + ', OVERWRITING existing files.')
    else:
        print(
            'Running BOLD dual-regression over the dataset in ' + args.bids + ', only on files where output is missing.')
        if args.subjects is not None:
            print('   - Selected subjects: ' + ','.join(args.subjects) + '.')
    print('########################')

    print('\nReading dataset.\n')
    db_file = join(dirname(args.bids), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False, database_path=db_file)

    bids_loader.add_derivatives(DIR_PIPELINES['jump-reg'])
    bids_loader.add_derivatives(DIR_PIPELINES['bold-prep'])
    bids_loader.add_derivatives(DIR_PIPELINES['session-mni'])
    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    subject_list = bids_loader.get_subjects() if args.subjects is None else args.subjects

    lin_reg = LinearRegression()
    mni_2mm_proxy = nib.load(MNI_TEMPLATE_2mm)

    bold_files = []
    for subject in subject_list:
        print('Subject: ' + subject)
        if not exists(join(tmp_dir, subject)): makedirs(join(tmp_dir, subject))
        ica_results_dict = {'session': [], 'statistic': [], 'space': [], **{'ica' + str(i): [] for i in range(n_ica)}}
        ica_tsv_file = bids_loader.get(scope='bold-prep', subject=subject, desc='icamaps', extension='tsv')
        if len(ica_tsv_file) == 1:
            ica_tsv_df = pd.read_csv(ica_tsv_file[0].path, sep='\t')
            ica_results_dict = {col: ica_tsv_df[col].to_list() for col in ica_tsv_df.columns if 'Unnamed' not in col}

        bold_im_list = bids_loader.get(subject=subject, suffix='bold', scope='bold-prep', extension='nii.gz', desc='nuisance')
        for bold_im in bold_im_list:
            session = bold_im.entities['session']
            print(' * Session: ' + session)

            # if not isdir(join(sess_data_dir, 'func')): continue
            # if exists(join(sess_data_dir, 'func', sbj + '_' + sess + '_task-rest_space-MNI_desc-icamaps_bold.nii.gz')) and not SUBJECT_SPACE: continue
            # if exists(join(sess_data_dir, 'func', sbj + '_' + sess + '_task-rest_space-cog_desc-icamaps_bold.nii.gz')) and SUBJECT_SPACE: continue
            ica_mni = bids_loader.get(subject=subject, session=session, suffix='bold', scope='bold-prep',
                                      extension='nii.gz', desc='icamaps', space='MNI')
            if len(ica_mni) == 1 or not args.force:
                print('[done] It has already been computed.')
                continue

            def_MNI_file = bids_loader.get(subject=subject, session=session, space='MNI', desc='field', scope='session-mni', extension='nii.gz')
            if len(def_MNI_file) != 1:
                print('[error] N=' + str(len(def_MNI_file)) + ' MNI deformation fields found.' \
                                                              'Please, refine search. Skipping')
                continue

            def_MNI_file = def_MNI_file[0]
            def_MNI_proxy = nib.load(def_MNI_file.path)
            def_MNI_2mm_proxy = def_utils.vol_resample(mni_2mm_proxy, def_MNI_proxy)
            nib.save(def_MNI_2mm_proxy, join(tmp_dir, subject, 'warp.nii.gz'))

            t1w_entities = {'subject': subject, 'session': session, 'suffix': 'T1w'}
            if 'run' in def_MNI_file.entities.keys():
                t1w_entities['run'] = def_MNI_file.entities['run']

            bold_entities = {'subject': subject, 'session': session, 'suffix': 'bold'}
            if 'run' in bold_im.entities.keys():
                bold_entities['run'] = bold_im.entities['run']

            aff_bold = jump_utils.get_aff(bids_loader, bold_entities)
            # aff_t1 = jump_utils.get_aff(bids_loader, t1w_entities)
            if aff_bold is None:
                # mf.append(bold_image.path)
                print('[error] processing of failed due to no valid JUMP-reg matrices. Skipping')
                continue

            new_aff = aff_bold
            np.save(join(tmp_dir, subject, 'aff.npy'), new_aff)
            np.save(join(tmp_dir, subject, 'aff_inv.npy'), np.linalg.inv(new_aff))

            register2MNI(MNI_TEMPLATE_2mm,
                         bold_im.path,
                         join(tmp_dir, subject, 'output.nii.gz'),
                         join(tmp_dir, subject, 'aff.npy'),
                         join(tmp_dir, subject, 'warp.nii.gz'))

            bold_proxy = nib.load(join(tmp_dir, subject, 'output.nii.gz'))
            bold_data = np.array(bold_proxy.dataobj)


            X = ica_maps.reshape((-1, ica_maps.shape[-1]))
            y = bold_data.reshape((-1, bold_proxy.shape[-1]))
            lin_reg.fit(X, y)
            mean = np.mean(lin_reg.coef_, axis=0)
            std = np.std(lin_reg.coef_, axis=0)
            ica_results_dict['session'].append(session)
            ica_results_dict['statistic'].append('mean')
            ica_results_dict['space'].append('MNI')
            for it_m, m in enumerate(mean): ica_results_dict['ica' + str(it_m)].append(m)

            ica_results_dict['session'].append(session)
            ica_results_dict['statistic'].append('std')
            ica_results_dict['space'].append('MNI')
            for it_m, m in enumerate(std): ica_results_dict['ica' + str(it_m)].append(m)

            ica_time_series = lin_reg.coef_ / std

            lin_reg.fit(ica_time_series, y.T)
            subject_specific_maps = lin_reg.coef_

            img = nib.Nifti1Image(subject_specific_maps.reshape(bold_proxy.shape[:3] + (ica_maps.shape[-1],)), bold_proxy.affine)
            nib.save(img, join(DIR_PIPELINES['bold-prep'], 'sub-' + subject, 'ses-' + session, 'func',
                               'sub-' + subject + '_ses-' + session + '_task-rest_space-MNI_desc-icamaps_bold.nii.gz'))

            if args.subject_space:# and not exists(join(sess_data_dir, 'func', sbj + '_' + sess + '_task-rest_space-cog_desc-icamaps_bold.nii.gz')):
                ica_session = bids_loader.get(subject=subject, session=session, suffix='bold', scope='bold-prep',
                                              extension='nii.gz', desc='icamaps', space='SESSION')
                if len(ica_mni) == 1 or not args.force:
                    print('[done] It has already been computed.')
                    continue

                def_IMAGE_file = bids_loader.get(subject=subject, session=session, space='SESSION', desc='field',
                                               scope='session-mni', extension='nii.gz')
                if len(def_IMAGE_file) != 1:
                    print('[error] N=' + str(len(def_IMAGE_file)) + ' MNI to image deformation fields found. '
                                                                    'Please, refine search. Skipping')
                    continue

                def_IMAGE_file = def_IMAGE_file[0]

                registerMNI2Image(bold_im.path,
                                  args.ica,
                                  join(tmp_dir, subject, 'output_session.nii.gz'),
                                  join(tmp_dir, subject, 'aff_inv.npy'),
                                  def_IMAGE_file.path)

                proxy = nib.load(join(tmp_dir, subject, 'output_session.nii.gz'))
                ica_maps = np.array(proxy.dataobj)

                bold_proxy = nib.load(bold_im.path)
                bold_data = np.array(bold_proxy.dataobj)

                X = ica_maps.reshape((-1, ica_maps.shape[-1]))
                y = bold_data.reshape((-1, bold_proxy.shape[-1]))
                lin_reg.fit(X, y)
                mean = np.mean(lin_reg.coef_, axis=0)
                std = np.std(lin_reg.coef_, axis=0)

                ica_results_dict['session'].append(session)
                ica_results_dict['statistic'].append('mean')
                ica_results_dict['space'].append('cog')
                for it_m, m in enumerate(mean): ica_results_dict['ica' + str(it_m)].append(m)

                ica_results_dict['session'].append(session)
                ica_results_dict['statistic'].append('std')
                ica_results_dict['space'].append('cog')
                for it_m, m in enumerate(std): ica_results_dict['ica' + str(it_m)].append(m)

                ica_time_series = lin_reg.coef_ / std

                lin_reg.fit(ica_time_series, y.T)
                subject_specific_maps = lin_reg.coef_

                img = nib.Nifti1Image(subject_specific_maps.reshape(bold_proxy.shape[:3] + (ica_maps.shape[-1],)),
                                      np.linalg.inv(aff_bold) @ bold_proxy.affine)
                nib.save(img, join(DIR_PIPELINES['bold-prep'], 'sub-' + subject, 'ses-' + session, 'func', 'sub-' +
                                   subject + '_sess-' + session + '_task-rest_space-SESSION_desc-icamaps_bold.nii.gz'))


        ica_df = pd.DataFrame(ica_results_dict)
        ica_df.to_csv(join(DIR_PIPELINES['bold-prep'], 'sub-' + subject,
                           'sub-' + subject + '_desc-icamaps_bold.tsv'), sep='\t')
