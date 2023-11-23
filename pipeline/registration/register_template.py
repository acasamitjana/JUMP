import pdb

from setup import *

import time
from os.path import exists, join, dirname
from argparse import ArgumentParser
import nibabel as nib
import numpy as np
import torch
from tensorflow import keras
import bids


# project imports
from utils import synthmorph_utils, def_utils
from utils.io_utils import print_title_script
from utils.jump_utils import initialize_graph_linear, get_aff
from utils.fn_utils import compute_centroids_ras
from utils.synthmorph_utils import VxmDenseOriginalSynthmorph, instance_register, path_model_registration, \
    RescaleTransform, VecInt, fast_3D_interp_torch, fast_3D_interp_field_torch


def register_subject(subject, args):

    missed_sess = []
    timepoints = bids_loader.get_session(subject=subject)
    for tp in timepoints:
        print('    - Session: ' + tp, end=' ', flush=True)
        seg_mni_dir_sbj = join(DIR_PIPELINES['session-mni'], 'sub-' + subject, 'ses-' + tp, 'anat')
        if not exists(seg_mni_dir_sbj): os.makedirs(seg_mni_dir_sbj)
        fname_prefix = 'sub-' + subject + '_ses-' + tp

        im_entities = {'subject': subject, 'session': tp, 'acquisition': 1, 'suffix': 'T1w', 'extension': 'nii.gz',
                       'scope': 'synthseg'}
        im_file = bids_loader.get(**im_entities)
        if len(im_file) > 1:
            im_entities['run'] = '01'
            im_file = bids_loader.get(**im_entities)

        if len(im_file) != 1:
            print('[error] no T1w image found for subject: ' + subject + ' and timpeoint: ' + tp +
                  '. Either inexistent or refine your search query.')
            missed_sess.append(fname_prefix)
            continue

        im_file = im_file[0]
        ent_seg_res = {k: v for k, v in im_file.entities.items() if k in filename_entities}
        ent_seg_res['suffix'] = 'T1wdseg'
        seg_file = bids_loader.get(**ent_seg_res)
        if len(seg_file) != 1:
            print('[error] no T1w segmentation found for subject: ' + subject + ' and timpeoint: ' + tp +
                  '. Either inexistent or refine your search query.')
            missed_sess.append(fname_prefix)
            continue
        seg_file = seg_file[0]

        filepath_aff = join(seg_mni_dir_sbj, fname_prefix + '_space-' + template_str + '_desc-field_aff.npy')
        filepath_def = join(seg_mni_dir_sbj, fname_prefix + '_space-' + template_str + '_desc-field_nonlinear.nii.gz')
        filepath_def_backward = join(seg_mni_dir_sbj, fname_prefix + '_space-' + 'SESSION_desc-field_nonlinear.nii.gz')
        filepath_reg = join(seg_mni_dir_sbj, fname_prefix + '_space-' + template_str + '_T1w.nii.gz')
        filepath_session_im = join(tmp_dir, fname_prefix  + '_space-SESSION_T1w.nii.gz')
        filepath_session_seg = join(tmp_dir, fname_prefix + '_space-SESSION_dseg.nii.gz')
        filepath_imlinear = join(tmp_dir, fname_prefix + '_imlinear.nii.gz')
        filepath_seglinear = join(tmp_dir, fname_prefix + '_seglinear.nii.gz')

        pdb.set_trace()
        if not exists(filepath_def) or not exists(filepath_def_backward) or force_flag:
            if template_str == 'MNI':
                print('(1) computing linear field; ', end=' ', flush=True)

                # IMAGE TO MNI LINEAR
                affine_matrix = get_aff(bids_loader, im_entities)
                if affine_matrix is None:
                    if len(bids_loader.get(
                            scope='raw', subject=subject, session=tp, extension='nii.gz', acquisition=None)) == 1:
                        affine_matrix = np.eye(4)
                    else:
                        print('[error] no affine matrix found for T1w images')
                        missed_sess.append(fname_prefix)
                        continue

                proxy = nib.load(im_file.path)
                data = np.array(proxy.dataobj)
                img = nib.Nifti1Image(data, np.linalg.inv(affine_matrix) @ proxy.affine)
                nib.save(img, filepath_session_im)

                proxy = nib.load(seg_file.path)
                data = np.array(proxy.dataobj)
                img = nib.Nifti1Image(data, np.linalg.inv(affine_matrix) @ proxy.affine)
                nib.save(img, filepath_session_seg)

                centroid_ref, ok_ref = compute_centroids_ras(MNI_TEMPLATE_SEG, labels_registration)
                centroid_flo, ok_flo = compute_centroids_ras(filepath_session_seg, labels_registration)

                initialize_graph_linear([centroid_ref, centroid_flo], filepath_aff)
                M_sbj = np.load(filepath_aff)

                ref_proxy = nib.load(MNI_TEMPLATE)
                flo_proxy, flo_seg_proxy = nib.load(filepath_session_im), nib.load(filepath_session_seg)

                data = np.array(flo_proxy.dataobj)
                v2r = np.linalg.inv(M_sbj) @ flo_proxy.affine
                proxy_reg = nib.Nifti1Image(data, v2r)
                proxy_reg = def_utils.vol_resample(ref_proxy, proxy_reg, mode='bilinear')
                nib.save(proxy_reg, filepath_imlinear)

                data = np.array(flo_seg_proxy.dataobj)
                v2r = np.linalg.inv(M_sbj) @ flo_proxy.affine
                proxy_reg = nib.Nifti1Image(data, v2r)
                proxy_reg = def_utils.vol_resample(ref_proxy, proxy_reg, mode='nearest')
                nib.save(proxy_reg, filepath_seglinear)

                if not exists(MNI_to_ATLAS):
                    centroid_sbj, ok = compute_centroids_ras(MNI_TEMPLATE_SEG, synthmorph_utils.labels_registration)
                    centroid_atlas = np.load(synthmorph_utils.atlas_cog_file)
                    M_ref = synthmorph_utils.getM(centroid_atlas[:, ok > 0], centroid_sbj[:, ok > 0], use_L1=False)
                    np.save(MNI_to_ATLAS, M_ref)

                else:
                    M_ref = np.load(MNI_to_ATLAS)

                Rlin, R_aff, R_h = synthmorph_utils.compute_atlas_alignment(MNI_TEMPLATE, MNI_TEMPLATE_SEG, proxyatlas,
                                                                            M_ref)
                Flin, F_aff, F_h = synthmorph_utils.compute_atlas_alignment(filepath_imlinear, filepath_seglinear,
                                                                            proxyatlas, M_ref)

            else:
                print('[error] ' + template_str + ' atlas still not implemented. Skipping')
                exit()

            print('(2) computing nonlinear field', end='; ', flush=True)
            cnn = VxmDenseOriginalSynthmorph.load(path_model_registration)
            svf1 = cnn.register(Flin.detach().numpy()[np.newaxis, ..., np.newaxis],
                                Rlin.detach().numpy()[np.newaxis, ..., np.newaxis])
            if args.nosym:
                svf = svf1
            else:
                svf2 = cnn.register(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis],
                                    Flin.detach().numpy()[np.newaxis, ..., np.newaxis])
                svf = 0.5 * svf1 - 0.5 * svf2

            if args.nepochs_refinement > 0:
                instance_model = instance_register(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis],
                                                   Flin.detach().numpy()[np.newaxis, ..., np.newaxis],
                                                   svf, inshape=proxy_template.shape,
                                                   epochs=args.nepochs_refinement)
                svf_refined = instance_model.references.flow_layer(Rlin.detach().numpy()[np.newaxis, ..., np.newaxis])
            else:
                svf_refined = svf

            integrator = keras.Sequential([VecInt(method='ss', int_steps=7)])
            upscaler = keras.Sequential([RescaleTransform(2)])
            warp_pos_small = integrator(svf_refined)
            warp_neg_small = integrator(-svf_refined)
            f2r_field = torch.tensor(np.squeeze(upscaler(warp_pos_small)))
            r2f_field = torch.tensor(np.squeeze(upscaler(warp_neg_small)))

            # Saving forward field
            print('(3) saving forward field', end='; ', flush=True)
            II, JJ, KK = np.meshgrid(np.arange(ref_proxy.shape[0]), np.arange(ref_proxy.shape[1]),
                                     np.arange(ref_proxy.shape[2]), indexing='ij')
            II = torch.tensor(II, device='cpu')
            JJ = torch.tensor(JJ, device='cpu')
            KK = torch.tensor(KK, device='cpu')
            affine = torch.tensor(
                np.matmul(np.linalg.inv(atlas_aff), np.matmul(np.linalg.inv(M_ref), ref_proxy.affine)), device='cpu')
            II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
            JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
            KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

            FIELD = fast_3D_interp_field_torch(f2r_field, II2, JJ2, KK2)
            II3 = II2 + FIELD[:, :, :, 0]
            JJ3 = JJ2 + FIELD[:, :, :, 1]
            KK3 = KK2 + FIELD[:, :, :, 2]

            affine = torch.tensor(M_sbj @ M_ref @ atlas_aff, device='cpu')
            RAS_X = affine[0, 0] * II3 + affine[0, 1] * JJ3 + affine[0, 2] * KK3 + affine[0, 3]
            RAS_Y = affine[1, 0] * II3 + affine[1, 1] * JJ3 + affine[1, 2] * KK3 + affine[1, 3]
            RAS_Z = affine[2, 0] * II3 + affine[2, 1] * JJ3 + affine[2, 2] * KK3 + affine[2, 3]
            img = nib.Nifti1Image(torch.stack([RAS_X, RAS_Y, RAS_Z], axis=-1).numpy().astype('float32'), R_aff)
            nib.save(img, filepath_def)

            print('(4) deforming floating image', end='; ', flush=True)
            affine = torch.tensor(np.linalg.inv(flo_proxy.affine), device='cpu')
            II4 = affine[0, 0] * RAS_X + affine[0, 1] * RAS_Y + affine[0, 2] * RAS_Z + affine[0, 3]
            JJ4 = affine[1, 0] * RAS_X + affine[1, 1] * RAS_Y + affine[1, 2] * RAS_Z + affine[1, 3]
            KK4 = affine[2, 0] * RAS_X + affine[2, 1] * RAS_Y + affine[2, 2] * RAS_Z + affine[2, 3]
            registered = fast_3D_interp_torch(torch.tensor(np.array(flo_proxy.dataobj)), II4, JJ4, KK4, 'linear')
            img = nib.Nifti1Image(registered.numpy(), ref_proxy.affine)
            nib.save(img, filepath_reg)

            # Saving backward field
            print('(5) saving backward field;', end=' ', flush=True)
            II, JJ, KK = np.meshgrid(np.arange(flo_proxy.shape[0]), np.arange(flo_proxy.shape[1]),
                                     np.arange(flo_proxy.shape[2]), indexing='ij')
            II = torch.tensor(II, device='cpu')
            JJ = torch.tensor(JJ, device='cpu')
            KK = torch.tensor(KK, device='cpu')
            affine = torch.tensor(
                np.linalg.inv(atlas_aff) @ np.linalg.inv(M_ref) @ np.linalg.inv(M_sbj) @ flo_proxy.affine, device='cpu')
            II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
            JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
            KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

            FIELD = fast_3D_interp_field_torch(r2f_field, II2, JJ2, KK2)
            II3 = II2 + FIELD[:, :, :, 0]
            JJ3 = JJ2 + FIELD[:, :, :, 1]
            KK3 = KK2 + FIELD[:, :, :, 2]

            affine = torch.tensor(M_ref @ atlas_aff, device='cpu')
            RAS_X = affine[0, 0] * II3 + affine[0, 1] * JJ3 + affine[0, 2] * KK3 + affine[0, 3]
            RAS_Y = affine[1, 0] * II3 + affine[1, 1] * JJ3 + affine[1, 2] * KK3 + affine[1, 3]
            RAS_Z = affine[2, 0] * II3 + affine[2, 1] * JJ3 + affine[2, 2] * KK3 + affine[2, 3]
            img = nib.Nifti1Image(torch.stack([RAS_X, RAS_Y, RAS_Z], axis=-1).numpy().astype('float32'), flo_proxy.affine)
            nib.save(img, filepath_def_backward)
            print('done.')

        else:
            print('[done] session already processed.')
    return missed_sess


if __name__ == '__main__':

    parser = ArgumentParser(description="JUMP-registration: register T1w in session-space to MNI", epilog='\n')
    parser.add_argument('--bids', default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
    parser.add_argument('--subjects', default=None, nargs='+')
    parser.add_argument('--template_str', default='MNI', choices=['MNI'])
    parser.add_argument('--nepochs_refinement', default=0, type=int, help='number of epochs for pairwise refinement.')
    parser.add_argument('--nosym', action='store_true', help='do not run symmetric registration.')
    parser.add_argument("--force", action='store_true', help="Force the overwriting of existing files.")

    args = parser.parse_args()
    bids_dir = args.bids
    init_subject_list = args.subjects
    template_str = args.template_str
    force_flag = args.force

    print('\n\n\n\n\n')
    print('# ------------------------------------------------' + '-' * len(template_str) + ' #')
    print('# Register Nonlinear Template to reference space: ' + template_str + ' #')
    print('# ------------------------------------------------' + '-' * len(template_str) + ' #')
    print('\n\n')

    labels_registration = os.path.join(repo_home, 'data', 'labels_classes_priors', 'label_list_registration.npy')
    tmp_dir = '/tmp/JUMP_to_template/'
    if not exists(tmp_dir): os.makedirs(tmp_dir)

    title = 'Running JUMP registration over the dataset in'
    print_title_script(title, args)

    print('\nReading dataset.\n')
    db_file = join(dirname(bids_dir), 'BIDS-raw.db')
    if not exists(db_file):
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False)
        bids_loader.save(db_file)
    else:
        bids_loader = bids.layout.BIDSLayout(root=bids_dir, validate=False, database_path=db_file)

    bids_loader.add_derivatives(DIR_PIPELINES['jump-reg'])
    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    bids_loader.add_derivatives(DIR_PIPELINES['session-mni'])
    subject_list = bids_loader.get_subjects() if init_subject_list is None else init_subject_list

    if template_str == 'MNI':
        template_filepath = MNI_TEMPLATE
        template_mask_filepath = MNI_TEMPLATE_MASK
        template_seg_filepath = MNI_TEMPLATE_SEG
        proxy_template = nib.load(template_filepath)
        proxy_template_seg = nib.load(template_seg_filepath)

    else:
        raise ValueError('Please, specify a valid template name.')

    proxyatlas = nib.load(synthmorph_utils.atlas_file)
    atlas_aff = proxyatlas.affine


    failed_sessions = []
    for it_subject, subject in enumerate(subject_list):
        print(' * Subject: ' + str(subject) + '  -  ' + str(it_subject) + '/' + str(len(subject_list)))

        t_init = time.time()
        try:
            fs = register_subject(subject, args)
        except:
            fs = [subject]
        failed_sessions.extend(fs)

        print('   Total Elapsed time: ' + str(np.round(time.time() - t_init, 2)) + ' seconds.')

    f = open(join(LOGS_DIR, 'register_template.txt'), 'w')
    f.write('Total unprocessed subjects: ' + str(len(failed_sessions)))
    f.write(','.join(['\'' + s + '\'' for s in failed_sessions]))

    print('\n')
    print('Total failed subjects ' + str(len(failed_sessions)) + '. See ' + join(LOGS_DIR,
                                                                                   'register_template.txt') + ' for more information.')
    print('\n')
    print('# --------- FI (JUMP-reg: register to template) --------- #')
    print('\n')


