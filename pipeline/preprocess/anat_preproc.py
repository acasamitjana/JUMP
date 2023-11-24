from setup import *

import pdb
from os import makedirs
from os.path import exists, join, basename, dirname
from argparse import ArgumentParser
import copy
from joblib import delayed, Parallel
import subprocess

import numpy as np
import nibabel as nib
from skimage.transform import resize
from skimage.morphology import ball, binary_dilation
import bids

from utils.fn_utils import one_hot_encoding, rescale_voxel_size
from utils.labels import SYNTHSEG_LUT, CSF_LABELS
from utils.io_utils import write_json_derivatives, print_title_script
from utils.bf_utils import convert_posteriors_to_unified, bias_field_corr

def process_subject(subject, bids_loader, args, force_flag=False):
    print('\nSubject: ' + subject)
    seg_dict = {'subject': subject, 'scope': args.scope, 'extension': 'nii.gz'}
    suffix_seg_list = [s for s in bids_loader.get(**{**seg_dict, 'return_type': 'id', 'target': 'suffix'}) if 'dseg' in s]
    timepoints = bids_loader.get_session(subject=subject)

    missed_images = []
    for tp_id in timepoints:
        print('\n* Session: ' + tp_id)

        seg_files = bids_loader.get(**{'suffix': suffix_seg_list, 'session': tp_id, **seg_dict})
        for seg_file in seg_files:
            if seg_file.entities['datatype'] != 'anat': continue

            proxyseg = nib.load(seg_file.path)
            sess_seg_dir = dirname(seg_file.path)
            sess_raw_dir = join(BIDS_DIR, 'sub-' + subject, 'ses-' + tp_id, 'anat')

            # Modality
            modality = seg_file.entities['suffix'].split('dseg')[0]
            print('  o Modality: ' + modality + ': ', end='', flush=True)  # , end=' ', flush=True)


            # build output paths
            ent_res = {k: str(v) for k, v in seg_file.entities.items() if k in filename_entities}
            ent_res['suffix'] = modality
            ent_raw = copy.copy(ent_res)
            ent_raw['acquisition'] = None

            im_raw_fname = basename(bids_loader.build_path(ent_raw, path_patterns=BIDS_PATH_PATTERN, validate=False))
            json_raw_fename = im_raw_fname.replace('nii.gz', 'json')

            inimagef = join(sess_raw_dir, seg_file.filename.replace('dseg', ''))
            inrawimagef = join(sess_raw_dir, im_raw_fname)

            outimagef = join(sess_seg_dir, seg_file.filename.replace('dseg', ''))
            outjsonf = join(sess_seg_dir, seg_file.filename.replace('dseg', '').replace('nii.gz', 'json'))
            outrawimagef = join(sess_seg_dir, im_raw_fname)
            outrawjsonf = join(sess_seg_dir, json_raw_fename)
            outmaskf = seg_file.filename.replace('dseg', 'mask')

            if exists(join(sess_seg_dir, im_raw_fname)) and not force_flag:
                print('image already processed.')
                continue

            # ------------------------ #
            #      Computing masks     #
            # ------------------------ #
            print('computing masks from dseg files; ', end='', flush=True)
            if not exists(join(sess_seg_dir, outmaskf)):
                seg = np.array(proxyseg.dataobj)
                mask = seg > 0
                for lab in CSF_LABELS:
                    mask[seg == lab] = 0

                img = nib.Nifti1Image(mask.astype('uint8'), proxyseg.affine)
                nib.save(img, join(sess_seg_dir, outmaskf))

            if not exists(inimagef):
                proxy_im_raw = nib.load(inrawimagef)
                pixdim = np.sqrt(np.sum(proxy_im_raw.affine * proxy_im_raw.affine, axis=0))[:-1]
                if all([np.abs(p-1) < 0.01 for p in pixdim]):
                    rf = subprocess.call(['ln', '-s', inrawimagef, join(tmp_dir, basename(outimagef))], stderr=subprocess.PIPE)
                    if rf != 0:
                        subprocess.call(['cp', inrawimagef, join(tmp_dir, basename(outimagef))])

                else:
                    # some dimension may be wrong
                    if any([p < 0.01 for p in pixdim]):
                        continue

                    v, aff = rescale_voxel_size(np.array(proxy_im_raw.dataobj), proxy_im_raw.affine, [1, 1, 1])
                    img = nib.Nifti1Image(v, aff)
                    nib.save(img, join(tmp_dir, basename(outimagef)))

                inimagef = join(tmp_dir, basename(outimagef))


            proxy_im_raw = nib.load(inrawimagef)
            proxy_im_res = nib.load(inimagef)

            # ------------------------ #
            # Bias field correction    #
            # ------------------------ #
            print('correcting for inhomogeneities and normalisation (min/max); ', end='', flush=True)
            if not exists(outrawimagef) or args.force:
                vox2ras0 = proxy_im_res.affine
                mri_acq = np.asarray(proxy_im_res.dataobj)
                seg = np.array(proxyseg.dataobj)

                soft_seg = one_hot_encoding(seg, categories=SYNTHSEG_LUT)
                soft_seg = convert_posteriors_to_unified(soft_seg, lut=SYNTHSEG_LUT)
                try:
                    mri_acq_corr, bias_field = bias_field_corr(mri_acq, soft_seg, penalty=1, VERBOSE=False, filter_exceptions=True)
                except:
                    print("[error] bias field cannot be computed.", end='', flush=True)
                    missed_images.append(seg_file.path)
                    continue

                if mri_acq_corr is None:
                    if not args.keep_wrong:
                        print("[error] bias field cannot be computed -- removing segmentation related files and "
                              "exiting: " + seg_file.path, end='', flush=True)
                        os.remove(join(sess_seg_dir, outmaskf))
                        os.remove(seg_file.path)
                        os.remove(seg_file.path.replace('nii.gz', 'tsv'))
                    continue

                del soft_seg

                mask = seg > 0

                del seg

                M = np.percentile(mri_acq_corr[mask], 99)
                m = np.percentile(mri_acq_corr[mask], 1)
                mri_acq_corr = 255 * (mri_acq_corr - m) / (M-m)

                mask_dilated = binary_dilation(mask, ball(3))
                mri_acq_corr *= mask_dilated

                img = nib.Nifti1Image(np.clip(mri_acq_corr, 0, 255).astype('uint8'), proxy_im_res.affine)
                nib.save(img, outimagef)

                vox2ras0_orig = proxy_im_raw.affine
                mri_acq_orig = np.asarray(proxy_im_raw.dataobj)
                if len(mri_acq_orig.shape)>3:
                    mri_acq_orig = mri_acq_orig[..., 0]

                new_vox_size = np.linalg.norm(vox2ras0_orig, 2, 0)[:3]
                vox_size = np.linalg.norm(vox2ras0, 2, 0)[:3]

                #JSON
                write_json_derivatives(vox_size, mri_acq_corr.shape, outjsonf,
                                       extra_kwargs={"Description": "Bias field corrected image."})

                del mri_acq, mri_acq_corr



                if all([v1==v2 for v1, v2 in zip(vox_size, new_vox_size)]):
                    rf = subprocess.call(['ln', '-s', outimagef, outrawimagef], stderr=subprocess.PIPE)
                    rf_j = subprocess.call(['ln', '-s', outjsonf, outrawjsonf], stderr=subprocess.PIPE)
                    if rf != 0:
                        subprocess.call(['cp', outimagef, outrawimagef])
                    if rf_j != 0:
                        subprocess.call(['cp', outjsonf, outrawjsonf])


                else:
                    bias_field_resize, _ = rescale_voxel_size(bias_field, vox2ras0, new_vox_size)
                    if bias_field_resize.shape != mri_acq_orig.shape:
                        bias_field_resize = resize(bias_field_resize, mri_acq_orig.shape)

                    mask_resize, _ = rescale_voxel_size(mask.astype('float'), vox2ras0, new_vox_size)
                    if mask_resize.shape != mri_acq_orig.shape:
                        mask_resize = resize(mask_resize, mri_acq_orig.shape, order=1)
                    mask_resize = mask_resize > 0

                    mri_acq_orig_corr = copy.copy(mri_acq_orig.astype('float32'))
                    mri_acq_orig_corr[mask_resize] = mri_acq_orig_corr[mask_resize] / bias_field_resize[mask_resize]
                    M = np.percentile(mri_acq_orig_corr[mask_resize], 99)
                    m = np.percentile(mri_acq_orig_corr[mask_resize], 1)
                    mri_acq_orig_corr = 255 * (mri_acq_orig_corr - m) / (M - m)

                    mask_dilated = binary_dilation(mask_resize, ball(3))
                    mri_acq_orig_corr *= mask_dilated

                    img = nib.Nifti1Image(np.clip(mri_acq_orig_corr, 0, 255).astype('uint8'), proxy_im_raw.affine)
                    nib.save(img, outrawimagef)

                    write_json_derivatives(new_vox_size, mri_acq_orig_corr.shape, outrawjsonf,
                                           extra_kwargs={"Description": "Bias field corrected image."})

                    del bias_field, bias_field_resize, mri_acq_orig, mri_acq_orig_corr, mask_dilated

            print('done.')
    return missed_images




if __name__ == '__main__':

    parser = ArgumentParser(description='Computes the prediction of certain models')
    parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
    parser.add_argument('--subjects', default=None, nargs='+', help="(optional) specify which subjects to process")
    parser.add_argument('--num_cores', default=1, type=int, help="(optional) specify the number of cores used.")
    parser.add_argument('--scope', default='synthseg', nargs='+',
                            help="where to find the segmentations (derivative name)")
    parser.add_argument('--force', action='store_true', help="force the bias field correction.")
    parser.add_argument('--keep_wrong', action='store_true',
                            help="keep the segmentation files with errors in the related"
                                 " image intensity corrections. Typically, a wrong BF "
                                 "is related with a wrong segmentation.")

    args = parser.parse_args()
    bids_dir = args.bids
    init_subject_list = args.subjects
    num_cores = args.num_cores
    scope = args.scope
    force_flag = args.force
    keep_wrong = args.keep_wrong

    title = 'Running anatomical pre-processing over the dataset in '
    print_title_script(title, args)

    print('\nReading dataset.\n')
    db_file = join(dirname(bids_dir), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False, database_path=db_file)

    bids_loader.add_derivatives(DIR_PIPELINES['anat-prep'])
    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    subject_list = bids_loader.get_subjects() if init_subject_list is None else init_subject_list


    tmp_dir = '/tmp/JUMP-anat-prep/'
    if not exists(tmp_dir): makedirs(tmp_dir)
    failed_subjects = []
    if args.num_cores == 1:
        for it_s, subject in enumerate(subject_list):
            missing_subjects = process_subject(subject, bids_loader, args, force_flag=force_flag)
            failed_subjects.extend(missing_subjects)
    else:
        results = Parallel(n_jobs=args.num_cores)(
            delayed(process_subject)(subject, bids_loader, args, force_flag=force_flag) for subject in subject_list)

    subprocess.call(['rm', '-rf', tmp_dir])
    f = open(join(LOGS_DIR, 'anat_preproc.txt'), 'w')
    f.write('Total unprocessed files: ' + str(len(failed_subjects)) + '\n')
    f.write(','.join(['\'' + s + '\'' for s in failed_subjects]))

    print('  Total failed subjects ' + str(len(failed_subjects)) + '. See ' + join(LOGS_DIR, 'anat_preproc.txt') + ' for more information.' )
    print('\n')
    print('# --------- FI (JUMP anatomical preprocessing) --------- #')
    print('\n')
