import pdb
from os.path import join, exists, dirname
from os import makedirs
import itertools
from datetime import date, datetime
import time

import torch
from torch import nn
import numpy as np

from setup import *
from src.callbacks import History, ModelCheckpoint, PrinterCallback
from src.models import InstanceRigidModelLOG
from utils.io_utils import write_affine_matrix, read_affine_matrix

# Read linear st2 graph
# Formulas extracted from: https://math.stackexchange.com/questions/3031999/proof-of-logarithm-map-formulae-from-so3-to-mathfrak-so3
def solve_ST(bids_loader, subject, cost, lr, max_iter, n_epochs, force_flag=False, verbose=False):

    print('Subject: ' + str(subject))

    timepoints = bids_loader.get_session(subject=subject)

    dir_results_sbj = join(DIR_PIPELINES['jump-reg'], 'sub-' + subject)

    date_start = date.today().strftime("%d/%m/%Y")
    time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    exp_dict = {
        'date': date_start,
        'time': time_start,
        'cost': cost,
        'lr': lr,
        'max_iter': max_iter,
        'n_epochs': n_epochs
    }

    filename_template = 'sub-' + subject + '_desc-linTemplate_anat'
    if not exists(join(dir_results_sbj, filename_template + '_anat.json')):
        json_object = json.dumps(exp_dict, indent=4)
        with open(join(dir_results_sbj, filename_template + '_anat.json'), "w") as outfile:
            outfile.write(json_object)

    seg_dict = {'subject': subject, 'scope': 'synthseg', 'extension': 'nii.gz'}
    suffix_seg_list = [s for s in bids_loader.get(**{**seg_dict, 'return_type': 'id', 'target': 'suffix'}) if 'dseg' in s]
    for tp_id in timepoints:
        print('* Computing T=' + str(tp_id))

        dir_results_sess = join(dir_results_sbj, 'ses-' + tp_id)
        deformations_dir = join(dir_results_sess, 'deformations')
        linear_template = join(dir_results_sess, 'anat', filename_template + '.nii.gz')
        seg_files = bids_loader.get(**{'suffix': suffix_seg_list, 'session': tp_id, **seg_dict})
        if len(seg_files) == 1:
            print('  It has only 1 modality. No registration is made.')
            continue

        modality_list = []
        cog_list = []
        for seg_file in seg_files:
            modality = seg_file.entities['suffix'].split('dseg')[0]
            if 'run' in seg_file.entities.keys():
                modality += '.' + str(seg_file.entities['run'])

            cog_entities = {k: v for k, v in seg_file.entities.items() if k in filename_entities}
            cog_entities['extension'] = 'npy'
            cog_entities['desc'] = 'cog'
            cog_entities['suffix'] = modality
            if 'acquisition' in cog_entities.keys():
                cog_entities.pop('acquisition')
            cog_file = bids_loader.get(**cog_entities)
            if len(cog_file) != 1:
                print('  No COG file found. No registration is made.')
                modality_list = []
                break

            modality_list.append(modality)
            cog_list.append(cog_file[0].path)

        if len(modality_list) == 0:
            print('  Skipping. No modalities found.')
            continue

        elif len(modality_list) == 1 and seg_files[0].entities['dataype'] == 'anat':
            print('   It has only 1 modality. No registration is made.')
            return

        elif not exists(join(deformations_dir, modality_list[-2] + '_to_' + modality_list[-1] + '.npy')):
            print('   !!! WARNING -- No observations found for subject ' + subject + '.')
            continue

        # Check if multiple runs in this dataset.
        aff_dict = {'subject': subject, 'desc': 'aff', 'session': tp_id}
        if not len(bids_loader.get(subject=subject, session=tp_id, extension='npy', desc='aff', scope='jump-reg')) == len(seg_files) or force_flag:

            # -------------------------------------------------------------------#
            # -------------------------------------------------------------------#

            if verbose: print('   - [Building the graph] Reading transforms ...')

            t_init = time.time()

            graph_structure = init_ST(modality_list, deformations_dir)
            R_log = graph_structure

            if verbose: print('   - [Building the graph] Total Elapsed time: ' + str(time.time() - t_init) + '\n')
            if verbose: print('   - [JUMP] Computimg the graph ...')

            t_init = time.time()

            Tres = model_ST_pytorch(R_log, modality_list, n_epochs, cost, lr, dir_results_sbj, max_iter=max_iter, verbose=False)

            if verbose: print('   - [JUMP] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

            # -------------------------------------------------------------------#
            # -------------------------------------------------------------------#

            if verbose: print('   - [Integration] Computing the latent rigid transform ... ')
            t_init = time.time()
            for it_mod, mod in enumerate(modality_list):

                if '.' in mod:
                    suffix, run = mod.split('.')
                    extra_kwargs = {'suffix': suffix, 'run': run}
                else:
                    extra_kwargs = {'suffix': mod}

                fname_npy = bids_loader.build_path({**extra_kwargs, **aff_dict, 'extension': 'npy'}, scope='jump-reg',
                                                   path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False,
                                                   absolute_paths=False)

                fname_aff = bids_loader.build_path({**extra_kwargs, **aff_dict,  'extension': 'txt'}, scope='jump-reg',
                                                   path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False,
                                                   absolute_paths=False)


                cog_T = np.load(cog_list[it_mod])
                affine_matrix = np.linalg.inv(cog_T) @ Tres[..., it_mod]

                if not exists(join(DIR_PIPELINES['jump-reg'], dirname(fname_aff))):
                    makedirs(join(DIR_PIPELINES['jump-reg'], dirname(fname_npy)))

                np.save(join(DIR_PIPELINES['jump-reg'], fname_npy), affine_matrix)
                write_affine_matrix(join(DIR_PIPELINES['jump-reg'], fname_aff), affine_matrix)

            if verbose: print('   - [Integration] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

            # -------------------------------------------------------------------#
            # -------------------------------------------------------------------#


def init_ST(timepoints, input_dir, eps=1e-6, extension='aff'):
    nk = 0

    N = len(timepoints)
    K = int(N * (N - 1) / 2)

    phi_log = np.zeros((6, K))

    for tp_ref, tp_flo in itertools.combinations(timepoints, 2):
        if not isinstance(tp_ref, str):
            tid_ref, tid_flo = tp_ref.id, tp_flo.id
        else:
            tid_ref, tid_flo = tp_ref, tp_flo


        filename = str(tid_ref) + '_to_' + str(tid_flo)

        if exists(join(input_dir, filename + '.aff')):
            rotation_matrix, translation_vector = read_affine_matrix(join(input_dir, filename + '.aff'))
        else:
            rigid_matrix = np.load(join(input_dir, filename + '.npy'))
            rotation_matrix, translation_vector = rigid_matrix[:3, :3], rigid_matrix[:3, 3]

        # Log(R) and Log(T)
        t_norm = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1 + eps, 1 - eps)) + eps
        W = 1 / (2 * np.sin(t_norm)) * (rotation_matrix - rotation_matrix.T) * t_norm
        Vinv = np.eye(3) - 0.5 * W + ((1 - (t_norm * np.cos(t_norm / 2)) / ( 2 * np.sin(t_norm / 2))) / t_norm ** 2) * W * W  # np.matmul(W, W)

        phi_log[0, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * t_norm
        phi_log[1, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * t_norm
        phi_log[2, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * t_norm

        phi_log[3:, nk] = np.matmul(Vinv, translation_vector)

        nk += 1

    return phi_log


def model_ST_pytorch(logR, timepoints, n_epochs, cost, lr, results_dir_sbj, max_iter=5, patience=3,
                     device='cpu', verbose=False):

    if len(timepoints) > 2:
        log_keys = ['loss', 'time_duration (s)']
        logger = History(log_keys)
        model_checkpoint = ModelCheckpoint(join(results_dir_sbj, 'checkpoints'), -1)
        callbacks = [logger, model_checkpoint]
        if verbose: callbacks += [PrinterCallback()]

        model = InstanceRigidModelLOG(timepoints, cost=cost, device=device, reg_weight=0)
        optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')

        min_loss = 1000
        iter_break = 0
        log_dict = {}
        logR = torch.FloatTensor(logR)
        for cb in callbacks:
            cb.on_train_init(model)

        for epoch in range(n_epochs):
            for cb in callbacks:
                cb.on_epoch_init(model, epoch)

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                loss = model(logR, timepoints)
                loss.backward()

                return loss

            optimizer.step(closure=closure)

            loss = model(logR, timepoints)

            if loss < min_loss + 1e-4:
                iter_break = 0
                min_loss = loss.item()

            else:
                iter_break += 1

            if iter_break > patience or loss.item() == 0.:
                break

            log_dict['loss'] = loss.item()

            for cb in callbacks:
                cb.on_step_fi(log_dict, model, epoch, iteration=1, N=1)

        T = model._compute_matrix().cpu().detach().numpy()

    else:
        logR = np.squeeze(logR.astype('float32'))
        model = InstanceRigidModelLOG(timepoints, cost=cost, device=device, reg_weight=0)
        model.angle = nn.Parameter(torch.tensor(np.array([[-logR[0]/2, logR[0]/2],
                                                          [-logR[1]/2, logR[1]/2],
                                                          [-logR[2]/2, logR[2]/2]])).float(), requires_grad=False)

        model.translation = nn.Parameter(torch.tensor(np.array([[-logR[3] / 2, logR[3] / 2],
                                                                [-logR[4] / 2, logR[4] / 2],
                                                                [-logR[5] / 2, logR[5] / 2]])).float(), requires_grad=False)
        T = model._compute_matrix().cpu().detach().numpy()

    return T

