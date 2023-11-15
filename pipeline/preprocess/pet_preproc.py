import os
import pdb
from os import listdir, makedirs
from os.path import join, exists, dirname
import json
from argparse import ArgumentParser
import copy

import subprocess
import bids
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from nipype.interfaces import fsl

from setup import *
from utils import fn_utils, jump_utils
from src.preprocessing import *



def process_subject(subject, tmp_proc_dir, tmp_res_dir):
    print('Subject: ' + subject)

    if not exists(join(tmp_proc_dir, subject)): makedirs(join(tmp_proc_dir, subject))
    if not exists(join(tmp_res_dir, subject)): makedirs(join(tmp_res_dir, subject))

    pet_image_list = bids_loader.get(subject=subject, suffix='pet', scope='raw', extension='nii.gz')
    missing_files = []
    for pet_image in pet_image_list:
        if 'reconstruction' in pet_image.entities.keys(): continue
        if 'acquisition' in pet_image.entities.keys(): continue

        print(' * File: ' + pet_image.filename, end=': ', flush=True)

        sess = pet_image.entities['session']
        im_entities = pet_image.entities

        # Filenames
        final_pet_entities = {k: v for k, v in im_entities.items() if k in filename_entities}
        final_pet_entities['acquisition'] = None
        filepath_pet_pvc = bids_loader.build_path(final_pet_entities, scope='prep-pet', path_patterns=BIDS_PATH_PATTERN,
                                                  absolute_paths=False, validate=False)

        final_pet_entities['desc'] = 'cllGM'
        filepath_pet_suvr_cllGM = bids_loader.build_path(final_pet_entities, scope='prep-pet',
                                                         path_patterns=BIDS_PATH_PATTERN, absolute_paths=False,
                                                         validate=False)

        final_pet_entities['desc'] = 'crWM'
        filepath_pet_suvr_crWM = bids_loader.build_path(final_pet_entities, scope='prep-pet',
                                                        path_patterns=BIDS_PATH_PATTERN, absolute_paths=False,
                                                        validate=False)

        final_pet_entities['space'] = 'pet'
        final_pet_entities['suffix'] = 'T1wdseg'
        final_pet_entities.pop('tracer')
        final_pet_entities.pop('desc')
        filepath_t1w_seg = bids_loader.build_path(final_pet_entities, scope='synthseg', path_patterns=BIDS_PATH_PATTERN,
                                                  absolute_paths=False, validate=False)
        if not exists(dirname(join(DIR_PIPELINES['prep-pet'], filepath_pet_pvc))):
            os.makedirs(dirname(join(DIR_PIPELINES['prep-pet'], filepath_pet_pvc)))

        if not exists(dirname(join(DIR_PIPELINES['prep-pet'], filepath_t1w_seg))):
            os.makedirs(dirname(join(DIR_PIPELINES['prep-pet'], filepath_t1w_seg)))

        image_res_tmp = join(tmp_res_dir, subject, pet_image.filename)
        image_proc_tmp = join(tmp_proc_dir, subject, pet_image.filename)

        if not force_flag and exists(join(DIR_PIPELINES['prep-pet'], filepath_pet_suvr_cllGM)) \
                and exists(join(DIR_PIPELINES['prep-pet'], filepath_pet_suvr_crWM)) \
                and exists(join(DIR_PIPELINES['prep-pet'], filepath_pet_pvc)):
            print('already processed.')
            continue
        # try:

        # INPUT
        print('reading inputs; ', end='', flush=True)
        pet_proxy = nib.load(pet_image.path)
        pet_array = np.array(pet_proxy.dataobj)
        pixdim = np.sqrt(np.sum(pet_proxy.affine * pet_proxy.affine, axis=0))[:-1]
        if any([np.isnan(p) or p == 0 for p in pixdim]):
            missing_files.append(pet_image.path)
            print('[error] wrong PET image v2r matrix. Skipping.')
            continue

        # INPUT SEG
        ent_pet_synthseg = {k: v for k, v in pet_image.entities.items() if k in filename_entities}
        ent_pet_synthseg['suffix'] = 'petdseg'
        ent_pet_synthseg['scope'] = 'synthseg'
        if len(pet_proxy.shape) == 4:
            ent_pet_synthseg['reconstruction'] = pet_proxy.shape[-1] // 2

        pet_seg_image = bids_loader.get(**ent_pet_synthseg)
        if len(pet_seg_image) != 1:
            missing_files.append(pet_image.path)
            print('[error] no PET segmentation available. Skipping.')
            continue
        pet_seg_proxy = nib.load(pet_seg_image[0])

        # UPSAMPLE PET
        if upsample_flag:
            print('upsample PET image to 1mm3; ', end='', flush=True)
            if any([np.abs(p - 1) > 0.01 for p in pixdim]):
                if len(pet_proxy.shape) == 4:
                    output_upscale = [fn_utils.rescale_voxel_size(pet_array[..., it_dim], pet_proxy.affine, [1, 1, 1])
                                      for it_dim in range(pet_proxy.shape[-1])]
                    pet_array = np.stack([outs[0] for outs in output_upscale], axis=-1)
                else:
                    pet_array, _ = fn_utils.rescale_voxel_size(pet_array, pet_proxy.affine, [1, 1, 1])

            pet_proxy = nib.Nifti1Image(pet_array, pet_seg_proxy.affine)
            nib.save(pet_proxy, image_res_tmp)
            pet_image = bids.layout.BIDSImageFile(image_res_tmp)
            vx_size = (1, 1, 1)
        else:
            vx_size = tuple(np.sqrt(np.sum(pet_proxy.affine * pet_proxy.affine, axis=0))[:-1])

        # RESAMPLE T1w LABELS TO PET
        ## read T1w segmentation
        print('reading T1w image and grouping in VOIs; ', end='', flush=True)
        ent_t1w_synthseg = copy.copy(ent_pet_synthseg)
        ent_t1w_synthseg['suffix'] = 'T1wdseg'
        if 'reconstruction' in ent_t1w_synthseg.keys():
            ent_t1w_synthseg.pop('reconstruction')
        if 'tracer' in ent_t1w_synthseg.keys():
            ent_t1w_synthseg.pop('tracer')

        t1w_seg_image = bids_loader.get(**ent_t1w_synthseg)
        if len(t1w_seg_image) == 0:
            missing_files.append(pet_image.path)
            print('[error] no T1w segmentation available. It is required for T1w-MNI registration. Skipping.')
            continue

        elif len(t1w_seg_image) > 1:
            ent_t1w_synthseg['run'] = '01'
            t1w_seg_image = bids_loader.get(**ent_t1w_synthseg)
            if len(t1w_seg_image) != 1:
                missing_files.append(pet_image.path)
                print('[error] too many T1w image available. Please, refine search. Skipping')
                continue
            else:
                t1w_seg_image = t1w_seg_image[0]
        else:
            t1w_seg_image = t1w_seg_image[0]

        ## group regions
        t1w_seg_proxy = nib.load(t1w_seg_image.path)
        t1w_seg_array = np.array(t1w_seg_proxy.dataobj)
        new_t1w_seg_array = group_regions_seg(t1w_seg_array)

        # affine matrix
        t1w_entities = t1w_seg_image.entities
        t1w_entities['suffix'] = 'T1w'
        aff_bold = jump_utils.get_aff(bids_loader, im_entities)
        aff_t1 = jump_utils.get_aff(bids_loader, t1w_entities)
        new_aff = aff_bold @ np.linalg.inv(aff_t1)
        if new_aff is None:
            missing_files.append(pet_image.path)
            print('[error] processing of subject ' + subject + ' no valid JUMP-REG matrices. Skipping')
            continue
        new_v2r = new_aff @ t1w_seg_proxy.affine

        ## resample T1w to PET
        print('resampling T1w VOIs to PET space; ', end='', flush=True)
        if resampling == 'nearest':
            t1w_seg_proxy = nib.Nifti1Image(new_t1w_seg_array, new_v2r)
            t1w_seg_proxy = def_utils.vol_resample(pet_proxy, t1w_seg_proxy, mode='nearest')
            t1w_seg_array = np.array(t1w_seg_proxy.dataobj)

        else:
            unique_labels = {lab: it_lab for it_lab, lab in enumerate(np.unique(new_t1w_seg_array))}
            t1w_onehot_array = fn_utils.one_hot_encoding(new_t1w_seg_array.astype('float'), categories=unique_labels)
            t1w_seg_proxy = nib.Nifti1Image(t1w_onehot_array.astype('float'), new_v2r)
            t1w_onehot_array = def_utils.vol_resample(pet_proxy, t1w_seg_proxy, return_np=True)

            t1w_seg_resampled_array = np.argmax(t1w_onehot_array, axis=-1)
            t1w_seg_array = np.zeros_like(t1w_seg_resampled_array)
            for lab, it_lab in unique_labels.items():
                t1w_seg_array[t1w_seg_resampled_array == it_lab] = lab

            t1w_seg_proxy = nib.Nifti1Image(t1w_seg_array.astype('uint16'), pet_proxy.affine)

        nib.save(t1w_seg_proxy, join(DIR_PIPELINES['prep-pet'], filepath_t1w_seg))

        # MOTION CORRECTION
        if len(pet_proxy.shape) > 3:
            print('motion correction; ', end='', flush=True)
            mcflirt = fsl.MCFLIRT(in_file=pet_image.path, out_file=image_proc_tmp, cost='mutualinfo')
            output_mc = mcflirt.run()
            if output_mc.runtime.returncode != 0:
                missing_files.append(pet_image.path)
                print('[error] Could not run motion correction. Skipping.')
                continue
            else:
                proxy = nib.load(image_proc_tmp)
                data = np.array(proxy.dataobj)
                pet_run_array = np.mean(data, axis=-1)


        else:
            proxy = nib.load(pet_image.path)
            pet_run_array = np.array(proxy.dataobj)

        # PVC USING YANG'S METHOD
        if pvc_flag:
            print('partial volume correction using iterative yang method; ', end='', flush=True)
            fwhm = get_fwhm(pet_image)
            kernel = psf_gaussian(vx_size=vx_size, fwhm=fwhm)
            pet_run_array = pvc_yang(pet_run_array, t1w_seg_array, kernel, iter=5)[0]

        # SMOOTHING PET IMAGE
        if smoothing_flag:
            fwhm_target = [8, 8, 8]
            print('smoothing to FWHM=' + str(fwhm_target), end='; ', flush=True)
            kernel = psf_gaussian(vx_size=get_fwhm(pet_image), fwhm=fwhm_target)
            pet_run_array = conv_separable(pet_run_array, kernel)

        # SUVR CALCULATION
        print('normalising to cerebral WM and cerebellar GM; ', end='', flush=True)
        norm_cllGM = 0.5 * (np.mean(pet_run_array[t1w_seg_array == 8]))
        norm_crWM = 0.5 * (np.mean(pet_run_array[t1w_seg_array == 2]))

        pet_suvr_cllGM_proxy = nib.Nifti1Image(pet_run_array / norm_cllGM, pet_seg_proxy.affine)
        nib.save(pet_suvr_cllGM_proxy, join(DIR_PIPELINES['prep-pet'], filepath_pet_suvr_cllGM))

        pet_suvr_crWM_proxy = nib.Nifti1Image(pet_run_array / norm_crWM, pet_seg_proxy.affine)
        nib.save(pet_suvr_crWM_proxy, join(DIR_PIPELINES['prep-pet'], filepath_pet_suvr_crWM))

        print('done. ')

    if exists(join(tmp_res_dir, subject)):
        subprocess.call(['rm', '-rf', join(tmp_res_dir, subject)])

    if exists(join(tmp_proc_dir, subject)):
        subprocess.call(['rm', '-rf', join(tmp_proc_dir, subject)])

    return missing_files

if __name__ == '__main__':

    print('\n\n\n\n\n')
    print('# --------------------------------- #')
    print('# JUMP registration: compute graph  #')
    print('# --------------------------------- #')
    print('\n\n')

    parser = ArgumentParser(description='Computes the prediction of certain models')
    parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
    parser.add_argument('--subjects', default=None, nargs='+', help="(optional) specify which subjects to process")
    parser.add_argument('--resampling', default='linear', choices=['nearest', 'linear'],
                            help="(optional) specify nearest/linear (one-hot) resampling of the T1w labels")
    parser.add_argument('--force', action='store_true', help="force the preprocessing even if it already exists.")
    parser.add_argument('--upsample_pet', action='store_true', help="upsample PET image to 1x1x1 m3.")
    parser.add_argument('--pvc', action='store_true', help="perform partial volume correction using iterative Yang's.")
    parser.add_argument('--smooth', action='store_true', help="smooth to fwhm=[8,8,8]")
    parser.add_argument('--accept_no_mri', action='store_true',
                            help="(optional) if jump-reg registration to MRI is not "
                                 "found, accept synthsr segmetation")

    args = parser.parse_args()
    bids_dir = args.bids
    resampling = args.resampling
    init_subject_list = args.subjects
    force_flag = args.force
    accept_no_mri = args.accept_no_mri
    upsample_flag = args.upsample_pet
    pvc_flag = args.pvc
    smoothing_flag = args.smooth

    tmp_proc_dir = '/tmp/PET_proc'
    tmp_res_dir = '/tmp/PET_resample'

    print('\n\n########')
    if force_flag is True:
        print('Running PET pre-processing over the dataset in ' + bids_dir + ', OVERWRITING existing files.')
    else:
        print(
            'Running PET pre-processing over the dataset in ' + bids_dir + ', only on files where segmentation is missing.')
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

    bids_loader.add_derivatives(DIR_PIPELINES['jump-reg'])
    bids_loader.add_derivatives(DIR_PIPELINES['pet-prep'])
    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    subject_list = bids_loader.get_subjects() if init_subject_list is None else init_subject_list

    missing_files = []
    for it_s, subject in enumerate(subject_list):
        mf = process_subject(subject, tmp_proc_dir, tmp_res_dir)
        missing_files.extend(missing_files)

    if exists(tmp_proc_dir):
        subprocess.call(['rm', '-rf', tmp_proc_dir])
    if exists(tmp_res_dir):
        subprocess.call(['rm', '-rf', tmp_res_dir])

    f = open(join(LOGS_DIR, 'pet_preproc.txt'), 'w')
    f.write('Total unprocessed files: ' + str(len(missing_files)))
    f.write(','.join(['\'' + s + '\'' for s in missing_files]))

    print('  Total failed subjects ' + str(len(missing_files)) + '. See ' + join(LOGS_DIR, 'pet_preproc.txt') + ' for more information.')
    print('\n')
    print('# --------- FI (PET Preprocessing) --------- #')
    print('\n')
