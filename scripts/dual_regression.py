import os
import time
from os import listdir, makedirs
from os.path import exists, join, dirname
from argparse import ArgumentParser
import subprocess

import bids
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils import def_utils, io_utils, jump_utils
from src.preprocessing import runICA, registerMNI2Image, register2MNI
from setup import *

class EmptyImage():
    def __init__(self, path):
        self.path = path

def process_subject(subject, bids_loader, ica_maps, tmp_dir):
    if not exists(join(tmp_dir, subject)): makedirs(join(tmp_dir, subject))
    ica_results_dict = {'session': [], 'statistic': [], 'space': [], **{'ica' + str(i): [] for i in range(n_ica)}}
    ica_tsv_file = bids_loader.get(scope='bold-prep', subject=subject, desc='icamaps', extension='tsv')
    if len(ica_tsv_file) == 1:
        ica_tsv_df = pd.read_csv(ica_tsv_file[0].path, sep='\t')
        ica_results_dict = {col: ica_tsv_df[col].to_list() for col in ica_tsv_df.columns if 'Unnamed' not in col}

    bold_im_list = bids_loader.get(subject=subject, suffix='bold', scope='bold-prep', extension='nii.gz',
                                   desc='nuisance')
    missing_sessions = []
    for bold_im in bold_im_list:
        session = bold_im.entities['session']
        print('   - Session: ' + session)

        ica_mni = bids_loader.get(subject=subject, session=session, suffix='bold', scope='bold-prep',
                                  extension='nii.gz', desc='icamaps', space='MNI')
        ica_session = bids_loader.get(subject=subject, session=session, suffix='bold', scope='bold-prep',
                                  extension='nii.gz', desc='icamaps', space='SESSION')


        # if len(ica_mni) == 1 and len(ica_session) == 1 and not args.force:
        #     print('[done] It has already been computed.')
        #     continue

        if len(ica_mni) > 1 or len(ica_session) > 1:
            print('[error] More than 1 file found. Probably has been computed using different names..')
            missing_sessions.append(subject + '_' + session)
            continue

        bold_entities = {'subject': subject, 'session': session, 'suffix': 'bold'}
        if 'run' in bold_im.entities.keys():
            bold_entities['run'] = bold_im.entities['run']

        aff_bold = jump_utils.get_aff(bids_loader, bold_entities)
        if aff_bold is None:
            print('[error] processing of failed due to no valid JUMP-reg matrices. Skipping')
            missing_sessions.append(subject + '_' + session)
            continue

        np.save(join(tmp_dir, subject, 'aff.npy'), aff_bold)
        np.save(join(tmp_dir, subject, 'aff_inv.npy'), np.linalg.inv(aff_bold))


        if len(ica_mni) == 0 or args.force:
            print('     Processing @ MNI space: ', end='', flush=True)
            def_MNI_file = bids_loader.get(subject=subject, session=session, space='MNI', desc='field', scope='session-mni',
                                           extension='nii.gz')
            if len(def_MNI_file) != 1:
                print('[error] N=' + str(len(def_MNI_file)) + ' MNI def.fields found. Please, refine search. Skipping')
                missing_sessions.append(subject + '_' + session)
                continue

            def_MNI_file = def_MNI_file[0]
            def_MNI_proxy = nib.load(def_MNI_file.path)
            def_MNI_2mm_proxy = def_utils.vol_resample(mni_2mm_proxy, def_MNI_proxy)
            nib.save(def_MNI_2mm_proxy, join(tmp_dir, subject, 'warp.nii.gz'))

            print('(1) register to MNI space; ', end='', flush=True)
            register2MNI(MNI_TEMPLATE_2mm,
                         bold_im.path,
                         join(tmp_dir, subject, 'output.nii.gz'),
                         join(tmp_dir, subject, 'aff.npy'),
                         join(tmp_dir, subject, 'warp.nii.gz'))

            bold_proxy = nib.load(join(tmp_dir, subject, 'output.nii.gz'))
            bold_data = np.array(bold_proxy.dataobj)

            print('(2) first stage regression; ', end='', flush=True)
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

            print('(3) second stage regression; ', end='', flush=True)
            ica_time_series = lin_reg.coef_ / std
            lin_reg.fit(ica_time_series, y.T)
            subject_specific_maps = lin_reg.coef_

            img = nib.Nifti1Image(subject_specific_maps.reshape(bold_proxy.shape[:3] + (ica_maps.shape[-1],)),
                                  bold_proxy.affine)
            nib.save(img, join(DIR_PIPELINES['bold-prep'], 'sub-' + subject, 'ses-' + session, 'func',
                               'sub-' + subject + '_ses-' + session + '_task-rest_space-MNI_desc-icamaps_bold.nii.gz'))

            subprocess.call(['rm', '-rf', join(tmp_dir, subject, 'output.nii.gz')])
            subprocess.call(['rm', '-rf', join(tmp_dir, subject, 'warp.nii.gz')])

        print('')
        if True:#(len(ica_session) == 0 or args.force) and args.subject_space:
            print('     Processing @ SESSION space: ', end='', flush=True)
            def_IMAGE_file = bids_loader.get(subject=subject, session=session, space='IMAGE', desc='field',
                                             scope='session-mni', extension='nii.gz')
            if len(def_IMAGE_file) == 1:
                subprocess.call(['mv', def_IMAGE_file[0].path, def_IMAGE_file[0].path.replace('IMAGE', 'SESSION')])
                def_IMAGE_file = [EmptyImage(path=def_IMAGE_file[0].path.replace('IMAGE', 'SESSION'))]

            else:
                def_IMAGE_file = bids_loader.get(subject=subject, session=session, space='SESSION', desc='field',
                                                 scope='session-mni', extension='nii.gz')

            if len(def_IMAGE_file) != 1:
                print('[error] N=' + str(len(def_IMAGE_file)) + ' MNI to image deformation fields found. '
                                                                'Please, refine search. Skipping')
                missing_sessions.append(subject + '_' + session + '_subject_space')
                continue

            def_IMAGE_file = def_IMAGE_file[0]

            print('(1) register to session space; ', end='', flush=True)
            proxytmp = nib.load('/tmp/smr_to_template/' + 'sub-' + subject + '_ses-' + session + session + '_space-SESSION_T1w.nii.gz')
            proxy = nib.load(def_IMAGE_file.path)
            if np.sum(proxy.affine == proxytmp.affine) != 16:
                data = np.array(proxy.dataobj)
                img = nib.Nifti1Image(data, proxytmp.affine)
                nib.save(img, def_IMAGE_file.path)

            registerMNI2Image(bold_im.path, args.ica, join(tmp_dir, subject, 'output_session.nii.gz'),
                              join(tmp_dir, subject, 'aff_inv.npy'), def_IMAGE_file.path)

            proxy = nib.load(join(tmp_dir, subject, 'output_session.nii.gz'))
            ica_maps_sbj = np.array(proxy.dataobj)

            bold_proxy = nib.load(bold_im.path)
            bold_data = np.array(bold_proxy.dataobj)

            print('(2) first stage regression; ', end='', flush=True)
            X = ica_maps_sbj.reshape((-1, ica_maps_sbj.shape[-1]))
            y = bold_data.reshape((-1, bold_proxy.shape[-1]))
            lin_reg.fit(X, y)
            mean = np.mean(lin_reg.coef_, axis=0)
            std = np.std(lin_reg.coef_, axis=0)

            ica_results_dict['session'].append(session)
            ica_results_dict['statistic'].append('mean')
            ica_results_dict['space'].append('SESSION')
            for it_m, m in enumerate(mean): ica_results_dict['ica' + str(it_m)].append(m)

            ica_results_dict['session'].append(session)
            ica_results_dict['statistic'].append('std')
            ica_results_dict['space'].append('SESSION')
            for it_m, m in enumerate(std): ica_results_dict['ica' + str(it_m)].append(m)

            print('(3) second stage regression; ', end='', flush=True)
            ica_time_series = lin_reg.coef_ / std
            lin_reg.fit(ica_time_series, y.T)
            subject_specific_maps = lin_reg.coef_

            img = nib.Nifti1Image(subject_specific_maps.reshape(bold_proxy.shape[:3] + (ica_maps_sbj.shape[-1],)),
                                  np.linalg.inv(aff_bold) @ bold_proxy.affine)
            nib.save(img, join(DIR_PIPELINES['bold-prep'], 'sub-' + subject, 'ses-' + session, 'func', 'sub-' +
                               subject + '_ses-' + session + '_task-rest_space-SESSION_desc-icamaps_bold.nii.gz'))

            del ica_maps_sbj, X, y, ica_time_series, bold_data

            subprocess.call(['rm', '-rf', join(tmp_dir, subject, 'output_session.nii.gz')])

        print('')

    ica_df = pd.DataFrame(ica_results_dict)
    ica_df.to_csv(join(DIR_PIPELINES['bold-prep'], 'sub-' + subject,
                       'sub-' + subject + '_desc-icamaps_bold.tsv'), sep='\t')

    return missing_sessions

if __name__ == '__main__':


    print('\n\n\n\n\n')
    print('# ------------------------------ #')
    print('# BOLD scripts: dual regression  #')
    print('# ------------------------------ #')
    print('\n\n')

    parser = ArgumentParser(description='Computes the prediction of certain models')
    parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
    parser.add_argument('--subjects', default=None, nargs='+', help="(optional) specify which subjects to process")
    parser.add_argument('--ica', default=join(RESULTS_DIR, 'melodic.ica.group', 'melodic.ica', 'melodic_IC.nii.gz'),
                        help="ICA maps to run the algorithm; by default, it uses the group-ICA from bold_preproc.py")
    parser.add_argument('--subject_space', action='store_true', help="run dual regression also un subject space.")
    parser.add_argument('--force', action='store_true', help="force the preprocessing even if it already exists.")
    args = parser.parse_args()

    proxyica = nib.load(args.ica)
    n_ica = proxyica.shape[-1]
    ica_maps = np.array(proxyica.dataobj)
    tmp_dir = '/tmp/BOLD_dual_regression'
    if not exists(tmp_dir): makedirs(tmp_dir)

    title = 'Running BOLD dual-regression over the dataset in'
    io_utils.print_title_script(title, args)


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

    missing_files = []
    for it_subject, subject in enumerate(subject_list):
        print(' * Subject: ' + str(subject) + '  -  ' + str(it_subject) + '/' + str(len(subject_list)))
        t_init = time.time()
        mf = process_subject(subject, bids_loader, ica_maps, tmp_dir)
        missing_files.extend(mf)
        # try:
        #     mf = process_subject(subject, bids_loader, ica_maps, tmp_dir)
        #     missing_files.extend(mf)
        # except:
        #     mf = subject
        #     missing_files.append(mf)

        print('   Total Elapsed time: ' + str(np.round(time.time() - t_init, 2)) + ' seconds.\n')

    f = open(join(LOGS_DIR, 'dual_regression.txt'), 'w')
    f.write('Total unprocessed subjects: ' + str(len(missing_files)))
    f.write(','.join(['\'' + s + '\'' for s in missing_files]))

    print('\n')
    print('Total failed subjects ' + str(len(missing_files)) + '. See ' +
          join(LOGS_DIR, 'dual_regression.txt') + ' for more information.')
    print('\n')
    print('# --------- FI (BOLD scripts: dual regression) --------- #')
    print('\n')
