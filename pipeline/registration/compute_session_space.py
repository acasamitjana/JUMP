from setup import *

from argparse import ArgumentParser
from joblib import delayed, Parallel

import bids
import nibabel as nib

# project imports
from src.jump_reg import *
from utils.jump_utils import create_template_space
from utils.io_utils import print_title_script


def compute_subject_template(subject, verbose=True):
    aff_dict = {'subject': subject, 'desc': 'aff', 'extension': 'npy', 'scope': 'jump-reg'}
    seg_dict = {'subject': subject, 'scope': 'synthseg', 'extension': 'nii.gz'}
    suffix_seg_list = [s for s in bids_loader.get(**{**seg_dict, 'return_type': 'id', 'target': 'suffix'}) if 'dseg' in s]

    sbj_str = 'sub-' + subject
    timepoints = bids_loader.get_session(subject=subject)
    failed_s = []
    for tp_id in timepoints:
        print(' * Timepoint: ' + tp_id)
        sess_str = 'ses-' + tp_id

        seg_files = bids_loader.get(**{'suffix': suffix_seg_list, 'session': tp_id, **seg_dict})
        session_template = join(DIR_PIPELINES['jump-reg'], sbj_str, sess_str, 'anat', 'subject_space.nii.gz')

        if len(seg_files) <= 1:
            if verbose: print('     !! [error] It has ' + str(len(seg_files)) + ' modalities. Skipping.')
            failed_s.append(sbj_str + '_' + sess_str)
            continue

        if not exists(session_template) or force_flag:

            t_init = time.time()
            if verbose: print('   - Updating vox2ras0  ... ')

            linear_seg_list = {}
            for seg_file in seg_files:
                modality = seg_file.entities['suffix'].split('dseg')[0]
                mod_kwargs = {'suffix': modality}

                modality_run = modality
                if 'run' in seg_file.entities.keys():
                    mod_kwargs['run'] = seg_file.entities['run']
                    modality_run += '.' + str(seg_file.entities['run'])

                affine_file = bids_loader.get(**aff_dict, **mod_kwargs, session= tp_id)
                if len(affine_file) != 1:
                    if verbose: print('     !! [error] wrong affine file entities.')
                    failed_s.append(sbj_str + '_' + sess_str)
                    continue

                aff_m = np.load(affine_file[0])
                if np.linalg.matrix_rank(aff_m) < 4:
                    linear_seg_list = {}
                    failed_s.append(sbj_str + '_' + sess_str)
                    break

                # Update image header
                proxyseg = nib.load(seg_file.path)
                proxyseg_sess = nib.Nifti1Image(np.array(proxyseg.dataobj), np.linalg.inv(aff_m) @ proxyseg.affine)

                linear_seg_list[modality_run] = proxyseg_sess

            # ------------------------------------------------------------------- #
            # ------------------------------------------------------------------- #

            if len(linear_seg_list.keys()) == 0:
                t = time.time() - t_init
                if verbose: print('     !! [error] No affine matrices found. Total Elapsed time: ' + str(t) + '\n')
                failed_s.append(sbj_str + '_' + sess_str)
                continue
            if verbose: print('   - Creating subject space  ... ')

            try:
                rasMosaic, template_vox2ras0, template_size = create_template_space(list(linear_seg_list.values()))
            except:
                if verbose: print('     !! [error] No session space found. Total Elapsed time: ' + str(time.time() - t_init) + '\n')
                failed_s.append(sbj_str + '_' + sess_str)
                continue

            proxytemplate = nib.Nifti1Image(np.zeros(template_size), template_vox2ras0)
            nib.save(proxytemplate, session_template)

            if verbose: print('   [done] Total Elapsed time: ' + str(time.time() - t_init) + '\n')

    return failed_s

if __name__ == '__main__':

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

    if num_cores > 1:
        VERBOSE = False
        results = Parallel(n_jobs=num_cores)(delayed(compute_subject_template)(subject, verbose=True) for subject in subject_list)
    else:
        VERBOSE = True
        failed_sessions = []
        for it_subject, subject in enumerate(subject_list):
            print('Subject: ' + subject)
            f_s = compute_subject_template(subject, verbose=True)
            failed_sessions.extend(f_s)


        f = open(join(LOGS_DIR, 'compute_session_space.txt'), 'w')
        f.write('Total unprocessed subjects: ' + str(len(failed_sessions)))
        f.write(','.join(['\'' + s + '\'' for s in failed_sessions]))

        print('\n')
        print('Total failed subjects ' + str(len(failed_sessions)) +
              '. See ' + join(LOGS_DIR, 'compute_session_space.txt') + ' for more information.')

    print('\n')
    print('# --------- FI (JUMP-reg: graph initialization) --------- #')
    print('\n')
