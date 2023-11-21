import pdb

from setup import *

from os import listdir, makedirs
from os.path import join, exists, dirname, basename
from argparse import ArgumentParser
import warnings
import subprocess

import bids
from nipype.interfaces import fsl
from nipype.interfaces.fsl import ICA_AROMA as FSL_ICA_AROMA
from nilearn.image import clean_img

from src.preprocessing import *
from utils.jump_utils import get_aff
from utils.io_utils import print_title_script

warnings.filterwarnings("ignore")

def get_bold_im_mask(bold_image):
    bold_proxy = nib.load(bold_image.path)
    bold_array = np.array(bold_proxy.dataobj[..., 5:])

    ent_bold_synthseg = {k: v for k, v in bold_image.entities.items() if k in filename_entities}
    ent_bold_synthseg['suffix'] = 'bolddseg'
    ent_bold_synthseg['scope'] = basename(DIR_PIPELINES['seg'])
    bold_seg_image = bids_loader.get(**ent_bold_synthseg)
    if len(bold_seg_image) == 0:
        # print('no BOLD image available. It is required for BOLD masking. Skipping. ')
        return None, None
    elif len(bold_seg_image) > 1:
        # print('too many T1w image available. Please, refine search. Skipping. ')
        return None, None
    else:
        bold_seg_image = bold_seg_image[0]

    bold_seg_proxy = nib.load(bold_seg_image.path)

    bold_seg_proxy = def_utils.vol_resample(bold_proxy, bold_seg_proxy)

    bold_seg_array = np.array(bold_seg_proxy.dataobj)
    bold_array[bold_seg_array == 0] = 0
    bold_proxy = nib.Nifti1Image(bold_array, bold_proxy.affine)
    bold_mask_proxy = nib.Nifti1Image((bold_seg_array > 0).astype('uint8'), bold_proxy.affine)

    return bold_proxy, bold_mask_proxy

def process_subject(subject, bold_params, tmp_proc_dir, tmp_field_dir):
    mf = []
    file4ICA = {'aff': None, 'im': None, 'sess': None}

    bold_image_list = bids_loader.get(subject=subject, acquisition=None, reconstruction=None, suffix='bold',
                                      scope='raw', extension='nii.gz')
    for bold_image in bold_image_list:
        bold_fname = bold_image.filename
        sess = bold_image.entities['session']
        im_entities = bold_image.entities

        print(' * File: ' + bold_fname, end=': ', flush=True)
        try:
            _ = nib.load(bold_image.path)
        except:
            mf.append(bold_image.path)
            print('[error] not valid bold image. Skipping')
            continue


        # Filenames
        final_bold_entities = {k: v for k, v in im_entities.items() if k in filename_entities}
        if not aroma_flag:
            final_bold_entities['desc'] = 'nuisance'
        else:
            final_bold_entities['desc'] = 'aroma'

        # TEMPORAL FILES
        if not exists(join(tmp_proc_dir, subject)): makedirs(join(tmp_proc_dir, subject))
        if not exists(join(tmp_field_dir, subject)): makedirs(join(tmp_field_dir, subject))
        filepath_proc_tmp = join(tmp_proc_dir, subject, bold_fname)
        filepath_mc_proc_tmp = join(tmp_proc_dir, subject, 'mc_' + bold_fname)
        filepath_nuisance_proc_tmp = join(tmp_proc_dir, subject, 'nuisance_' + bold_fname)
        filepath_mask_tmp = join(tmp_proc_dir, subject, bold_fname.replace('bold', 'mask'))
        filepath_tissue_mask_tmp = join(tmp_proc_dir, subject, bold_fname.replace('bold', 'tissuemask'))
        filepath_aff_tmp = join(tmp_proc_dir, subject, bold_fname.replace('bold', 'aff').replace('nii.gz', 'npy'))

        filepath_bold = bids_loader.build_path(final_bold_entities, scope='synthseg', path_patterns=BIDS_PATH_PATTERN,
                                               absolute_paths=False, validate=False)

        # CHEKING IF ALREADY EXISTS
        if exists(join(DIR_PIPELINES['bold-prep'], filepath_bold)) and not force_flag:
            if file4ICA['im'] is None:
                file4ICA['sess'] = sess
                file4ICA['aff'] = get_aff(bids_loader, im_entities)
                file4ICA['im'] = join(DIR_PIPELINES['bold-prep'], filepath_bold)

            if not args.indep_melodic or (args.indep_melodic and
                                          exists(join(DIR_PIPELINES['bold-prep'], dirname(filepath_bold), 'melodic'))):
                print('already processed. Skipping.')
                continue
            else:
                print('nuisance already done. Running independent MELODIC; ', end='', flush=True)
                bold_proxy, bold_mask_proxy = get_bold_im_mask(bold_image)
                if bold_proxy is None:
                    print('[error]. Failed trying to apply the mask. Skipping.')
                    continue
                nib.save(bold_mask_proxy, filepath_mask_tmp)

                fslDir = os.path.join(os.environ["FSLDIR"], 'bin', '')
                try:
                    runICA(fslDir, inFile=join(DIR_PIPELINES['bold-prep'], filepath_bold),
                           outDir=join(DIR_PIPELINES['bold-prep'], dirname(filepath_bold)),
                           melDirIn='', mask=filepath_mask_tmp, dim=15, TR=bold_params['tr'])
                except:
                    pass
                print('done.')
                continue

        if not exists(dirname(join(DIR_PIPELINES['bold-prep'], filepath_bold))):
            os.makedirs(dirname(join(DIR_PIPELINES['bold-prep'], filepath_bold)))

        # INPUT BOLD
        print('reading inputs; ', end='', flush=True)
        bold_proxy = nib.load(bold_image.path)
        num_ref = bold_proxy.shape[-1] // 2
        bold_proxy, bold_mask_proxy = get_bold_im_mask(bold_image)
        if bold_proxy is None:
            mf.append(bold_image.path)
            print('[error] not a valid bold image/mask. Skipping')
            continue
        bold_ref_proxy = nib.Nifti1Image(np.zeros(bold_proxy.shape[:3]), bold_proxy.affine)
        nib.save(bold_proxy, filepath_proc_tmp)
        nib.save(bold_mask_proxy, filepath_mask_tmp)

        ent_t1w_synthseg = {k: v for k, v in bold_image.entities.items() if k in filename_entities}
        ent_t1w_synthseg['suffix'] = 'T1wdseg'
        ent_t1w_synthseg['scope'] = 'synthseg'
        ent_t1w_synthseg.pop('task')
        t1w_seg_image = bids_loader.get(**ent_t1w_synthseg)
        if len(t1w_seg_image) == 0:
            mf.append(bold_image.path)
            print('[error] no T1w segmentation available. It is required for T1w-MNI registration. Skipping')
            continue
        elif len(t1w_seg_image) > 1:
            ent_t1w_synthseg['run'] = '01'
            t1w_seg_image = bids_loader.get(**ent_t1w_synthseg)
            if len(t1w_seg_image) != 1:
                mf.append(bold_image.path)
                print('[error] too many T1w image available. Please, refine search. Skpping')
                continue
            else:
                t1w_seg_image = t1w_seg_image[0]
        else:
            t1w_seg_image = t1w_seg_image[0]

        try:
            t1w_seg_proxy = nib.load(t1w_seg_image.path)
            t1w_seg_array = np.array(t1w_seg_proxy.dataobj)
        except:
            mf.append(bold_image.path)
            print('[error] could not read synthseg (image/seg) files. Skipping')
            continue

        t1w_entities = t1w_seg_image.entities
        t1w_entities['suffix'] = 'T1w'
        aff_bold = get_aff(bids_loader, im_entities)
        aff_t1 = get_aff(bids_loader, t1w_entities)
        if aff_bold is None or aff_t1 is None:
            mf.append(bold_image.path)
            print('[error] processing of subject ' + subject + ' no valid JUMP-reg matrices. Skipping')
            continue
        new_aff = aff_bold @ np.linalg.inv(aff_t1)
        np.save(filepath_aff_tmp, new_aff)

        # MOTION CORRECTION: need to write '*.mat' file and not only the temporal image
        print('motion correction; ', end='', flush=True)
        if len(bold_proxy.shape) > 3:
            mcflirt = fsl.MCFLIRT(in_file=filepath_proc_tmp, out_file=filepath_mc_proc_tmp, save_plots=True,
                                  cost='mutualinfo', ref_vol=num_ref)
            output_mc = mcflirt.run()
            if output_mc.runtime.returncode != 0:
                print()
                mf.append(bold_image.path)
                continue

        # NUISANCE REGRESSION
        print('nuisance regression; ', end='', flush=True)
        if not aroma_flag:
            try:
                tr = int(bold_image.entities['RepetitionTime'])
            except:
                tr = bold_params['tr']

            nuisance_kwargs = {'t_r': tr, 'low_pass': bold_params['lp'], 'high_pass': bold_params['hp'],
                               'ensure_finite': True, 'standardize': True}

            data_in = np.array(nib.load(filepath_mc_proc_tmp).dataobj)
            if args.no_glob_conf and args.no_csf_wm_conf:
                tissue_masks = np.stack([
                    (t1w_seg_array > 0).astype('uint8'),
                    ((t1w_seg_array == 41) + (t1w_seg_array == 2)).astype('uint8'),
                    ((t1w_seg_array == 24) + (t1w_seg_array == 4) + (t1w_seg_array == 5) + (t1w_seg_array == 43) + (
                            t1w_seg_array == 44)).astype('uint8'),
                ], -1)

            elif args.no_csf_wm_conf:
                tissue_masks = np.stack([
                    ((t1w_seg_array == 41) + (t1w_seg_array == 2)).astype('uint8'),
                    (t1w_seg_array == 24).astype('uint8'),
                ], -1)

            else:
                tissue_masks = (t1w_seg_array[..., np.newaxis] > 0).astype('uint8')


            proxymask_brain = nib.Nifti1Image(tissue_masks, new_aff @ t1w_seg_proxy.affine)
            tissue_np = def_utils.vol_resample(bold_ref_proxy, proxymask_brain, return_np=True)
            proxymasks = nib.Nifti1Image(
                ((tissue_np > 0) * (np.sum(data_in, axis=-1, keepdims=True) > 0)).astype('uint8'),
                bold_ref_proxy.affine)
            nib.save(proxymasks, filepath_tissue_mask_tmp)
            #
            signals = extract_average_signals(data_in, np.array(proxymasks.dataobj))
            confounds_matrix = create_confounds_matrix(signals, np.loadtxt(filepath_mc_proc_tmp + '.par'))
            #
            nuisance_kwargs = {'detrend': True, 'confounds': confounds_matrix.T, **nuisance_kwargs}
            try:
                data_out = clean_img(filepath_mc_proc_tmp, **nuisance_kwargs)
                nib.save(data_out, join(DIR_PIPELINES['bold-prep'], filepath_bold))
                if file4ICA['im'] is None:
                    file4ICA['sess'] = sess
                    file4ICA['aff'] = aff_bold
                    file4ICA['im'] = join(DIR_PIPELINES['bold-prep'], filepath_bold)

                if args.indep_melodic:
                    print('running independent MELODIC; ', end='', flush=True)
                    fslDir = os.path.join(os.environ["FSLDIR"], 'bin', '')
                    runICA(fslDir, inFile=filepath_nuisance_proc_tmp,
                           outDir=join(DIR_PIPELINES['bold-prep'], dirname(filepath_bold)), melDirIn='',
                           mask=filepath_mask_tmp, dim=25, TR=nuisance_kwargs['tr'])

            except:
                print('[error] processing of subject ' + subject + ' has ended prematurely. Skipping')
                mf.append(bold_image.path)
                continue

        else:
            print('running ica-aroma; ', end='', flush = True)
            def_dict = {'subject': subject, 'session': sess, 'suffix': 'nonlinear', 'extension': 'nii.gz',
                        'scope': basename(DIR_PIPELINES['session-mni'])}
            def_MNI_file = bids_loader.get(**def_dict)
            if len(def_MNI_file) != 1:
                print('[error] N=' + str(len(def_MNI_file)) + ' deformation fields to MNI found. Please, for ICA-AROMA'
                                                              ' correction, run first the register_template.py script'
                                                              ' or clean your directory for better search. Skipping')
                mf.append(bold_image.path)
                continue
            else:
                def_MNI_file = def_MNI_file[0]

            def_MNI_proxy = nib.load(def_MNI_file.path)
            mni_2mm_proxy = nib.load(join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz'))
            def_MNI_2mm_proxy = def_utils.vol_resample(mni_2mm_proxy, def_MNI_proxy)
            nib.save(def_MNI_2mm_proxy, join(tmp_field_dir, def_MNI_file.filename))

            # ICA-AROMA
            AROMA_obj = FSL_ICA_AROMA()
            AROMA_obj.inputs.in_file = filepath_nuisance_proc_tmp
            AROMA_obj.inputs.mat_file = filepath_aff_tmp
            AROMA_obj.inputs.fnirt_warp_file = join(tmp_field_dir, subject, def_MNI_file.filename)
            AROMA_obj.inputs.motion_parameters = filepath_mc_proc_tmp + '.par'
            AROMA_obj.inputs.denoise_type = 'nonaggr'  # 'both'
            AROMA_obj.inputs.out_dir = join(tmp_proc_dir, subject, 'ICA_testout')

            subprocess.call(['python'] + AROMA_obj.cmdline.split(' ') + ['-overwrite'])
            subprocess.call(['cp', '-r', join(tmp_proc_dir, subject, 'ICA_testout'), join(DIR_PIPELINES['bold-prep'], dirname(filepath_bold))])

        # SAVE FILES

        print('done. ')
    return mf, file4ICA

if __name__ == '__main__':


    print('\n\n\n\n\n')
    print('# --------------------------------- #')
    print('# JUMP registration: compute graph  #')
    print('# --------------------------------- #')
    print('\n\n')

    parser = ArgumentParser(description='Computes the prediction of certain models')
    parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
    parser.add_argument('--subjects', default=None, nargs='+', help="(optional) specify which subjects to process")
    parser.add_argument('--no_glob_conf', action='store_false', help="(optional) do not include global confounds")
    parser.add_argument('--no_csf_wm_conf', action='store_false', help="(optional) do not include wm/csf confounds")
    parser.add_argument('--resampling', default='linear', choices=['nearest', 'linear'],
                            help="(optional) specify nearest/linear (one-hot) resampling of the T1w labels")
    parser.add_argument('--force', action='store_true', help="force the preprocessing even if it already exists.")
    parser.add_argument('--aroma', action='store_true', help="run ica-aroma for nuisance regressions.")
    parser.add_argument('--indep_melodic', action='store_true', help="run melodic in each image.")
    parser.add_argument('--group_melodic', action='store_true', help="run melodic in the whole sample.")
    parser.add_argument('--accept_no_mri', action='store_true',
                            help="(optional) if jump-reg registration to MRI is not "
                                 "found, accept synthsr segmetation")

    args = parser.parse_args()
    bids_dir = args.bids
    resampling = args.resampling
    init_subject_list = args.subjects
    force_flag = args.force
    aroma_flag = args.aroma
    single_melodic_flag = args.indep_melodic
    group_melodic_flag = args.group_melodic
    global_confound_flag, csf_vwm_confound_flag = args.no_glob_conf, args.no_csf_wm_conf
    accept_no_mri = args.accept_no_mri

    # nuisance regression parameters
    bold_params = {'tr': 2, 'hp': 0.01, 'lp': 0.1}

    # temporal dirs
    tmp_proc_dir = '/tmp/BOLD_proc'
    tmp_field_dir = '/tmp/BOLD_field'
    tmp_gICA_dir = '/tmp/BOLD_gICA'
    if not exists(tmp_proc_dir): os.makedirs(tmp_proc_dir)
    if not exists(tmp_field_dir): os.makedirs(tmp_field_dir)
    if not exists(tmp_gICA_dir): os.makedirs(tmp_gICA_dir)

    title = 'Running BOLD pre-processing over the dataset in'
    print_title_script(title, args)

    print('\nReading dataset.\n')
    db_file = join(dirname(bids_dir), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False, database_path=db_file)

    bids_loader.add_derivatives(DIR_PIPELINES['jump-reg'])
    bids_loader.add_derivatives(DIR_PIPELINES['bold-prep'])
    bids_loader.add_derivatives(DIR_PIPELINES['session-mni'])
    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    subject_list = bids_loader.get_subjects() if init_subject_list is None else init_subject_list


    missing_files = []
    process_flag = False
    # for it_s, subject in enumerate(subject_list):
    #     print('Subject: ' + subject)
    #
    #     try:
    #         mf, file4ICA = process_subject(subject, bold_params, tmp_proc_dir, tmp_field_dir)
    #         missing_files.extend(mf)
    #     except:
    #         mf = subject
    #         missing_files.extend(mf)
    #         continue
    #
    #
    #     if group_melodic_flag and file4ICA['im'] is not None and not exists(join(tmp_gICA_dir, subject, 'output.nii.gz')):
    #         print(' * Register BOLD to MNI for group ICA: ', end='', flush=True)
    #         if not exists(join(tmp_gICA_dir, subject)):
    #             makedirs(join(tmp_gICA_dir, subject))
    #
    #         def_MNI_proxy, file_entities = resample_to_MNI2mm(DIR_PIPELINES['session-mni'], 'sub-' + subject, 'ses-' + file4ICA['sess'])
    #         if def_MNI_proxy is None:
    #             print('not a valid T1w-MNI deformation field. Skipping.')
    #             continue
    #
    #
    #         np.save(join(tmp_gICA_dir, subject, 'aff.npy'), file4ICA['aff'])
    #         nib.save(def_MNI_proxy, join(tmp_gICA_dir, subject, 'warp.nii.gz'))
    #
    #
    #         register2MNI(MNI_TEMPLATE_2mm,
    #                      file4ICA['im'],
    #                      join(tmp_gICA_dir, subject, 'output.nii.gz'),
    #                      join(tmp_gICA_dir, subject, 'aff.npy'),
    #                      join(tmp_gICA_dir, subject, 'warp.nii.gz'))
    #
    #         subprocess.call(['rm', '-rf', join(tmp_gICA_dir, subject, 'aff.npy')])
    #         subprocess.call(['rm', '-rf', join(tmp_gICA_dir, subject, 'warp.nii.gz')])
    #
    #     print('')
    #     if exists(join(tmp_proc_dir, subject)):
    #         subprocess.call(['rm', '-rf', join(tmp_proc_dir, subject)])
    #     if exists(join(tmp_field_dir, subject)):
    #         subprocess.call(['rm', '-rf', join(tmp_field_dir, subject)])
    #
    print('\n\nRunning group-ICA over the sample.')
    print('  * Reading MNI template: ' + join(MNI_TEMPLATE_2mm))

    mask_filepath_tmp = join(tmp_gICA_dir, 'MNI_mask_2mm.nii.gz')
    # if not exists(mask_filepath_tmp):
    #     ref_mni_proxy = nib.load(MNI_TEMPLATE_2mm)
    #     ref_mni_array = np.array(ref_mni_proxy.dataobj)
    #     ref_mni_mask = ref_mni_array > 0
    #     ref_mni_mask_proxy = nib.Nifti1Image(ref_mni_mask.astype('uint8'), ref_mni_proxy.affine)
    #     nib.save(ref_mni_mask_proxy, mask_filepath_tmp)
    #
    # gica_files = []
    # n_time = 10000000
    # print('  * Reading bold files.')
    # for sbj in listdir(tmp_gICA_dir):
    #     if exists(join(tmp_gICA_dir, sbj, 'output.nii.gz')):
    #         curr_time = nib.load(join(tmp_gICA_dir, sbj, 'output.nii.gz')).shape[-1]
    #         if curr_time < n_time:
    #             n_time = curr_time
    #         gica_files.append(join(tmp_gICA_dir, sbj, 'output.nii.gz'))
    #
    # print('  * Cropping ' + str(len(gica_files)) + ' files to same number of timepoints: ' + str(n_time), end='. ', flush=True)
    # final_gica_files = []
    # it_perc = 10
    # for it_bf, bf in enumerate(gica_files):
    #     if np.abs(100 * it_bf / len(gica_files) - it_perc) <= 1e-6 + 1 / len(gica_files):
    #         print(str(it_perc) + '%', end=' ', flush=True)
    #         it_perc += 10
    #
    #     proxy = nib.load(bf)
    #     data = np.array(proxy.dataobj)
    #     img = nib.Nifti1Image(data[..., :n_time], proxy.affine)
    #     nib.save(img, bf.replace('output', 'output' + str(n_time)))
    #     final_gica_files.append(bf.replace('output', 'output' + str(n_time)))
    #
    #     subprocess.call(['rm', 'rf', bf])
    # print('')
    # f = open(join(tmp_gICA_dir, 'files_to_melodic.txt'), 'w')
    # f.writelines('\n'.join(final_gica_files))
    # f.close()
    print('  * Running ICA')
    pdb.set_trace()
    f = open(join(tmp_gICA_dir, 'files_to_melodic.txt'), 'r')
    final_gica_files = f.readlines()
    runICA(join(os.environ["FSLDIR"], 'bin'), inFile=join(tmp_gICA_dir, 'files_to_melodic.txt'), sep_vn=True,
           outDir=join(RESULTS_DIR, 'melodic.ica.group'), melDirIn='', mask=mask_filepath_tmp, dim=15, TR=3)

    if exists(tmp_proc_dir):
        subprocess.call(['rm', '-rf', tmp_proc_dir])
    if exists(tmp_field_dir):
        subprocess.call(['rm', '-rf', tmp_field_dir])

    f = open(join(LOGS_DIR, 'bold_preproc.txt'), 'w')
    f.write('Total unprocessed files: ' + str(len(missing_files)))
    f.write(','.join(['\'' + s + '\'' for s in missing_files]))
    f.write('Total group ICA sessions: ' + str(len(final_gica_files)))
    f.write(','.join(['\'' + s + '\'' for s in final_gica_files]))

    print('  Total group ICA sessions: ' + str(len(final_gica_files)), end='. ', flush=True)
    print('  Total failed subjects ' + str(len(missing_files)) + '. See ' + join(LOGS_DIR, 'bold_preproc.txt') + ' for more information.')
    print('\n')
    print('# --------- FI (JUMP bold preprocessing) --------- #')
    print('\n')
