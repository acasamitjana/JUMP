# Author: Adria Casamitjana
# Date creation: 13/11/2021
# Historial of modification:
#    - Initial commit: Adria -  13/11/2021
from setup import *

from argparse import ArgumentParser
import bids
from joblib import delayed, Parallel

# project imports
from src.jump_reg import *
from utils.io_utils import print_title_script

if __name__ == '__main__':

    print('\n\n\n\n\n')
    print('# --------------------------------- #')
    print('# JUMP registration: compute graph  #')
    print('# --------------------------------- #')
    print('\n\n')

    parser = ArgumentParser(description="JUMP-registration: compute graph", epilog='\n')
    parser.add_argument("--bids", default=BIDS_DIR, help="Bids root directory, including rawdata")
    parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--max_iter', type=int, default=20, help='LBFGS')
    parser.add_argument('--n_epochs', type=int, default=30, help='Mask dilation factor')
    parser.add_argument('--num_cores', default=1, type=int, help='Run the algorithm in parallel using nu_cores.')
    parser.add_argument('--subjects', default=None, nargs='+', help='Specify subjects to process. By default, '
                                                                    'it runs over the entire dataset.')
    parser.add_argument("--force", action='store_true', help="Force the overwriting of existing files.")

    args = parser.parse_args()
    bids_dir = args.bids
    cost = args.cost
    lr = args.lr
    max_iter = args.max_iter
    n_epochs = args.n_epochs
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

    failed_subjects = []
    if num_cores > 1:
        results = Parallel(n_jobs=num_cores)(delayed(solve_ST)(
            bids_loader, subject, cost, lr, max_iter, n_epochs, force_flag=force_flag) for subject in subject_list)
    else:
        for it_subject, subject in enumerate(subject_list):
            t_init = time.time()
            try:
                ms = solve_ST(bids_loader, subject, cost, lr, max_iter, n_epochs, force_flag=force_flag)
            except:
                ms = [subject]

            if isinstance(ms, list):
                failed_subjects.extend(ms)
            elif ms is not None:
                failed_subjects.append(ms)

            print('Total computation time: ' + str(np.round(time.time() - t_init, 2)) + '\n')

    f = open(join(LOGS_DIR, 'compute_graph.txt'), 'w')
    f.write('Total unprocessed subjects: ' + str(len(failed_subjects)) + '\n')
    f.write(','.join(['\'' + s + '\'' for s in failed_subjects]))

    print('\n')
    print('Total failed subjects ' + str(len(failed_subjects)) +
          '. See ' + join(LOGS_DIR, 'compute_graph.txt') + ' for more information.')
    print('\n')
    print('# --------- FI (JUMP-reg: graph initialization) --------- #')
    print('\n')


