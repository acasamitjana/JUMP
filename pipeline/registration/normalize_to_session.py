from setup import *

import subprocess
from argparse import ArgumentParser
from joblib import delayed, Parallel

import bids

from src.jump_reg import *
from utils.io_utils import print_title_script


def compute_subject_template(subject, verbose=True):
    aff_dict = {'subject': subject, 'desc': 'aff', 'extension': 'npy', 'scope': 'jump-reg'}
    seg_dict = {'subject': subject, 'scope': 'synthseg', 'extension': 'nii.gz'}
    suffix_seg_list = [s for s in bids_loader.get(**{**seg_dict, 'return_type': 'id', 'target': 'suffix'}) if 'dseg' in s]

    sbj_str = 'sub-' + subject
    timepoints = bids_loader.get_session(subject=subject)
    for tp_id in timepoints:
        print(' * Timepoint: ' + tp_id)
        sess_str = 'ses-' + tp_id

        seg_files = bids_loader.get(**{'suffix': suffix_seg_list, 'session': tp_id, **seg_dict})
        dir_results_sess = join(DIR_PIPELINES['jump-reg'], sbj_str, sess_str, 'anat')

        filename_template = 'sub-' + subject + '_desc-linTemplate_anat'
        linear_template = join(dir_results_sess, filename_template + '.nii.gz')
        linear_template_mask = join(dir_results_sess, filename_template + 'mask.nii.gz')
        linear_template_seg = join(dir_results_sess, filename_template + 'dseg.nii.gz')

        if len(seg_files) == 0:
            print('  Skipping. No modalities found.')
            continue

        elif len(seg_files) == 1 and seg_files[0].entities['datatype'] == 'anat':
            modality = seg_files[0].entities['suffix'].split('dseg')[0]
            ent_im_res = copy.deepcopy(seg_files[0].entities)
            ent_mask_res = copy.deepcopy(seg_files[0].entities)
            ent_im_res['suffix'] = modality
            ent_mask_res['suffix'] = modality + 'mask'
            ent_res = {k: str(v) for k, v in seg_files[0].entities.items() if k in filename_entities}
            if 'run' in modality:
                suffix, run = modality.split('.')
                ent_res['suffix'] = suffix
                ent_res['run'] = run

            else:
                ent_res['run'] = modality

            im_res = bids_loader.get(scope='synthseg', **ent_im_res)[0]
            im_mask = bids_loader.get(scope='synthseg', **ent_mask_res)[0]

            if not exists(join(dir_results_sess, 'anat', im_res.filename)):
                try:
                    subprocess.call(['ln', '-s', im_res.path, join(dir_results_sess, 'anat', im_res.filename)])
                except:
                    subprocess.call(['cp', im_res.path, join(dir_results_sess, 'anat', im_res.filename)])

            if not exists(join(dir_results_sess, 'anat', im_mask.filename)):
                try:
                    subprocess.call(['ln', '-s', im_mask.path, join(dir_results_sess, 'anat', im_mask.filename)])
                except:
                    subprocess.call(['cp', im_mask.path, join(dir_results_sess, 'anat', im_mask.filename)])

            if not exists(linear_template):
                try:
                    subprocess.call(['ln', '-s', join(dir_results_sess, 'anat', im_res.filename), linear_template])
                except:
                    subprocess.call(['cp', join(dir_results_sess, 'anat', im_res.filename), linear_template])

            print('   It has only 1 modality. No registration is made.')
            continue

        if not exists(linear_template) or force_flag:
            proxytemplate = nib.load(join(dir_results_sess, 'subject_space.nii.gz'))
            template_size = proxytemplate.shape

            if verbose: print('   - [Deforming] Computing linear template ... ')
            mri_list = []
            mask_list = []
            aparc_aseg = np.concatenate((APARC_ARR, ASEG_ARR), axis=0)
            seg_list = np.zeros(template_size + (len(aparc_aseg),))
            for seg_file in seg_files:
                modality = seg_file.entities['suffix'].split('dseg')[0]
                if 'run' in seg_file.entities.keys():
                    modality_run = modality + '.' + str(seg_file.entities['run'])
                else:
                    modality_run = modality

                if modality not in anat_modalities: continue

                ent_im_res = {k: v for k, v in seg_file.entities.items()if k in filename_entities}
                ent_im_res['acquisition'] = 'orig'
                ent_im_res['suffix'] = modality
                im_raw_file = bids_loader.get(**ent_im_res, scope='synthseg')
                if len(im_raw_file) != 1:
                    print('     !! WARNING: More than one image is found in the synthseg directory ' +
                          str([m.filename for m in im_raw_file]) + ' for modality: ' + str(modality))
                    continue
                else:
                    im_raw_file = im_raw_file[0]

                ent_im_res_sess = {k: str(v) for k, v in im_raw_file.entities.items() if k in filename_entities}
                ent_im_res_sess['space'] = 'SESSION'
                ent_im_res_sess['acquisition'] = '1'
                im_res_sess_filepath = bids_loader.build_path(ent_im_res_sess, path_patterns=BIDS_PATH_PATTERN, validate=False)

                proxyim = nib.load(im_raw_file.path)
                pixdim = np.sqrt(np.sum(proxyim.affine * proxyim.affine, axis=0))[:-1]
                new_vox_size = np.array([1, 1, 1])
                factor = pixdim / new_vox_size
                sigmas = 0.25 / factor
                sigmas[factor > 1] = 0  # don't blur if upsampling

                im_orig_array = np.array(proxyim.dataobj)
                if len(im_orig_array.shape) > 3:
                    im_orig_array = im_orig_array[..., 0]
                volume_filt = gaussian_filter(im_orig_array, sigmas)

                im_orig_mri = nib.Nifti1Image(volume_filt, headers_orig[modality_run])
                im_mri = vol_resample(proxytemplate, im_orig_mri)
                nib.save(im_mri, im_res_sess_filepath)
                mri_list.append(np.array(im_mri.dataobj))

                ent_mask_res = {k: v for k, v in seg_file.entities.items()if k in filename_entities}
                ent_mask_res['suffix'] = modality + 'mask'
                mask_file = bids_loader.get(**ent_mask_res, scope='synthseg')
                if len(mask_file) != 1:
                    print('     !! WARNING: More than one mask is found in the synthseg directory ' + str([m.filename for m in mask_file]))
                    continue
                else:
                    mask_file = mask_file[0]

                ent_mask_res_sess = {k: str(v) for k, v in mask_file.entities.items() if k in filename_entities}
                ent_mask_res_sess['space'] = 'SESSION'
                mask_res_sess_filepath = bids_loader.build_path(ent_mask_res_sess, path_patterns=BIDS_PATH_PATTERN, validate=False)

                mask = np.array(nib.load(mask_file.path).dataobj)
                proxyflo = nib.Nifti1Image(mask, headers[modality_run])
                im_mri = vol_resample(proxytemplate, proxyflo)
                nib.save(im_mri, mask_res_sess_filepath)
                mask_list.append(np.array(im_mri.dataobj))

                seg = np.array(nib.load(seg_file.path).dataobj)
                proxyflo = nib.Nifti1Image(seg, headers[modality_run])
                nib.save(proxyflo, join(dir_results_sess, seg_file.filename)) #save orig, to be used in longitudinal
                im_mri = vol_resample(proxytemplate, proxyflo, mode='nearest')
                seg_list += one_hot_encoding(np.array(im_mri.dataobj), categories=aparc_aseg)

            template = np.median(mri_list, axis=0)
            template = template.astype('uint8')
            img = nib.Nifti1Image(template, template_vox2ras0)
            nib.save(img, linear_template)

            template = np.sum(mask_list, axis=0)/len(mask_list) > 0.5
            template = template.astype('uint8')
            img = nib.Nifti1Image(template, template_vox2ras0)
            nib.save(img, linear_template_mask)

            seg_hard = np.argmax(seg_list, axis=-1)
            template = np.zeros(template_size, dtype='int16')
            for it_l, l in enumerate(aparc_aseg): template[seg_hard == it_l] = l
            img = nib.Nifti1Image(template, template_vox2ras0)
            nib.save(img, linear_template_seg)

            if verbose: print('   - [Deforming] Total Elapsed time: ' + str(time.time() - t_init) + '\n')


if __name__ == 'main':

    print('\n\n\n\n\n')
    print('# ------------------------------------ #')
    print('# JUMP registration: compute template  #')
    print('# ------------------------------------ #')
    print('\n\n')

    parser = ArgumentParser(description="JUMP-registration: compute template", epilog='\n')
    parser.add_argument("--bids", default=BIDS_DIR, help="Bids root directory, including rawdata")
    parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
    parser.add_argument('--num_cores', default=1, type=int, help='Run the algorithm in parallel using nu_cores.')
    parser.add_argument('--subjects', default=None, nargs='+', help='Specify subjects to process. By default, '
                                                                    'it runs over the entire dataset.')
    parser.add_argument("--force", action='store_true', help="Force the overwriting of existing files.")

    args = parser.parse_args()
    bids_dir = args.bids
    cost = args.cost
    num_cores = args.num_cores
    init_subject_list = args.subjects
    force_flag = args.force

    title = 'Running JUMP registration over the dataset in'
    print_title_script(title, args)

    print('\nReading dataset.')
    db_file = join(dirname(bids_dir), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False, database_path=db_file)

    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    bids_loader.add_derivatives(DIR_PIPELINES['jump-reg'])
    subject_list = bids_loader.get_subjects() if init_subject_list is None else init_subject_list

    if num_cores > 1:
        VERBOSE = False
        results = Parallel(n_jobs=num_cores)(delayed(compute_subject_template)(subject, verbose=True) for subject in subject_list)
    else:
        VERBOSE = True
        for it_subject, subject in enumerate(subject_list):
            print('Subject: ' + subject)
            compute_subject_template(subject, verbose=True)




