# Author: Adria Casamitjana
# Date creation: 13/11/2021
# Historial of modification:
#    - Initial commit: Adria -  13/11/2021

from argparse import ArgumentParser
import bids
from joblib import delayed, Parallel

# project imports
from src.jump_reg import *

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
    print('\n\n########################')
    if force_flag is True:
        print('Running JUMP registration over the dataset in ' + bids_dir + ', OVERWRITING existing files.')
    else:
        print('Running JUMP registration over the dataset in ' + bids_dir + ', only on files where segmentation is missing.')

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
                solve_ST(bids_loader, subject, cost, lr, max_iter, n_epochs)
            except:
                failed_subjects.append(subject)
            print('Total computation time: ' + str(np.round(time.time() - t_init, 2)) + '\n')

    f = open(join(LOGS_DIR, 'compute_graph.txt'), 'w')
    f.write('Total unprocessed subjects: ' + str(len(failed_subjects)))
    f.write(','.join(['\'' + s + '\'' for s in failed_subjects]))

    print('\n')
    print('Total failed subjects ' + str(len(failed_subjects)) +
          '. See ' + join(LOGS_DIR, 'compute_graph.txt') + ' for more information.')
    print('\n')
    print('# --------- FI (JUMP-reg: graph initialization) --------- #')
    print('\n')


