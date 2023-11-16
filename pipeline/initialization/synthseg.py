import pdb
from os.path import exists, dirname, join, basename
from os import makedirs
from argparse import ArgumentParser
import subprocess
import nibabel as nib
import csv
import numpy as np
import bids


from setup import *
from utils.labels import SYNTHSEG_APARC_LUT
from utils.io_utils import write_json_derivatives


def sbj2seg(subject_list):
    '''
    Get subjects to segment
    :param subject_list: initial list with subjects' id.
    :return: input_files, res_files, output_files, vol_files to be passed to mri_synthseg
    '''
    input_files, res_files, output_files, vol_files = [], [], [], []
    for subject in subject_list:
        image_file_list = bids_loader.get(subject=subject, extension='nii.gz')
        image_file_list = list(filter(lambda x: 'rec' not in x.filename, image_file_list))
        for image_file in image_file_list:
            if image_file.entities['suffix'] not in VALID_MODALITIES: continue

            synthseg_dirname = join(DIR_PIPELINES['seg'], 'sub-' + subject,
                                    'ses-' + image_file.entities['session'], image_file.entities['datatype'])
            if not exists(synthseg_dirname): makedirs(synthseg_dirname)

            entities = {k: str(v) for k, v in image_file.entities.items() if k in filename_entities}
            bids_dirname = image_file.dirname

            if image_file.entities['datatype'] == 'func':
                proxy = nib.load(image_file.path)

                entities['reconstruction'] = proxy.shape[-1] // 2
                anat_input = basename(bids_loader.build_path(entities, path_patterns=BIDS_PATH_PATTERN, strict=False))

                synthsr_dirname = join(DIR_PIPELINES['bold-mri'], 'sub-' + subject, 'ses-' + image_file.entities['session'],
                                       image_file.entities['datatype'])
                bids_dirname = synthsr_dirname


            elif image_file.entities['datatype'] == 'pet':
                proxy = nib.load(image_file.path)
                if len(proxy.shape) <= 3:
                    anat_input = image_file.filename

                else:
                    entities['reconstruction'] = proxy.shape[-1] // 2
                    anat_input = basename(
                        bids_loader.build_path(entities, path_patterns=BIDS_PATH_PATTERN, strict=False, validate=False))

                synthsr_dirname = join(DIR_PIPELINES['pet-mri'], 'sub-' + subject, 'ses-' + image_file.entities['session'],
                                       image_file.entities['datatype'])
                bids_dirname = synthsr_dirname

            else:
                anat_input = image_file.filename

            if not exists(join(bids_dirname, anat_input)): continue

            anat_res = basename(bids_loader.build_path({**entities, **{'acquisition': '1'}},
                                                       path_patterns=BIDS_PATH_PATTERN, validate=False))
            anat_seg = anat_res.replace(entities['suffix'], entities['suffix'] + 'dseg')
            anat_vols = anat_seg.replace('nii.gz', 'tsv')
            if not exists(join(synthseg_dirname, anat_seg)) or force_flag:
                proxy = nib.load(join(bids_dirname, anat_input))
                if len(proxy.shape) != 3: continue
                if any(np.sum(np.abs(proxy.affine * proxy.affine), axis=0) <= 0.01): continue
                input_files += [join(bids_dirname, anat_input)]
                res_files += [join(bids_dirname, anat_res)]
                output_files += [join(synthseg_dirname, anat_seg)]
                vol_files += [join(synthseg_dirname, anat_vols)]

    return input_files, res_files, output_files, vol_files


if __name__ == '__main__':
    print('\n\n\n\n\n')
    print('# ----------------- #')
    print('# SynthSeg pipeline #')
    print('# ----------------- #')
    print('\n\n')

    parser = ArgumentParser(description="PET-MRI synthesis", epilog='\n')
    parser.add_argument("--bids", default=BIDS_DIR, help="Bids root directory, including rawdata")
    parser.add_argument('--subjects', default=None, nargs='+', help="(optional) specify which subjects to process")
    parser.add_argument('--threads', default=8, type=int, help="(optional) specify the number of threads for synthseg.")
    parser.add_argument("--force", action='store_true', help="(optional) force the script to overwriting existing "
                                                             "segmentations in the derivatives/synthseg directory.")

    args = parser.parse_args()
    bids_dir = args.bids
    init_subject_list = args.subjects
    force_flag = args.force

    print('\n\n########################')
    if force_flag is True:
        print('Running SynthSeg over the dataset in ' + bids_dir + ', OVERWRITING existing files.')
    else:
        print('Running SynthSeg over the dataset in ' + bids_dir + ', only on files where output is missing.')
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

    bids_loader.add_derivatives(DIR_PIPELINES['pet-mri'])
    if DIR_PIPELINES['bold-mri'] != DIR_PIPELINES['pet-mri']:
        bids_loader.add_derivatives(DIR_PIPELINES['bold-mri'])
    bids_loader.add_derivatives(DIR_PIPELINES['seg'])
    subject_list = bids_loader.get_subjects() if init_subject_list is None else init_subject_list

    input_files, res_files, output_files, vol_files = sbj2seg(subject_list)

    # Segment image
    with open('/tmp/inputfiles_jump.txt', 'w') as f:
        for i_f in input_files:
            f.write(i_f)
            f.write('\n')

    with open('/tmp/resfiles_jump.txt', 'w') as f:
        for i_f in res_files:
            f.write(i_f)
            f.write('\n')

    with open('/tmp/outputfiles_jump.txt', 'w') as f:
        for i_f in output_files:
            f.write(i_f)
            f.write('\n')

    with open('/tmp/volfiles_jump.txt', 'w') as f:
        for i_f in vol_files:
            f.write(i_f)
            f.write('\n')

    if len(output_files) >= 1:
        subprocess.call(
            ['mri_synthseg', '--i', '/tmp/inputfiles_jump.txt', '--o', '/tmp/outputfiles_jump.txt',
             '--resample', '/tmp/resfiles_jump.txt', '--vol', '/tmp/volfiles_jump.txt', '--threads', str(args.threads),
             '--robust', '--parc'], stderr=subprocess.PIPE)

    input_files, res_files, output_files, vol_files = [], [], [], []

    with open('/tmp/volfiles_jump.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            vol_files.append(l.split('\n')[0])

    with open('/tmp/inputfiles_jump.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            input_files.append(l.split('\n')[0])

    with open('/tmp/resfiles_jump.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            res_files.append(l.split('\n')[0])

    with open('/tmp/outputfiles_jump.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            output_files.append(l.split('\n')[0])

    for file in vol_files:
        if not exists(file): continue
        fr = open(file, "r")
        fw = open('/tmp/vol_mm.tsv', "w")

        reader = csv.reader(fr, delimiter=',')
        writer = csv.writer(fw, delimiter='\t')
        writer.writerows(reader)

        fr.close()
        fw.close()

        subprocess.call(['cp', '/tmp/vol_mm.tsv', file])

    for i, r in zip(input_files, res_files):
        if exists(r.replace('nii.gz', 'json')): continue
        if not exists(r) and exists(i):
            rcode = subprocess.call(['ln', '-s', i, r], stderr=subprocess.PIPE)
            if rcode != 0:
                subprocess.call(['cp', i, r], stderr=subprocess.PIPE)

        proxy = nib.load(r)
        aff = proxy.affine
        pixdim = str(np.sqrt(np.sum(aff * aff, axis=0))[:-1])
        write_json_derivatives(pixdim, proxy.shape, r.replace('nii.gz', 'json'))

    for i_file, seg_file in zip(input_files, output_files):
        if not exists(seg_file): continue

        proxy = nib.load(seg_file)
        aff = proxy.affine
        pixdim = str(np.sqrt(np.sum(aff * aff, axis=0))[:-1])

        if 'rec' in i_file:
            sec = [e.split('-')[-1] for e in basename(i_file).split('_') if 'rec' in e][0]
            write_json_derivatives(pixdim, proxy.shape, seg_file.replace('nii.gz', 'json'),
                                   extra_kwargs={'SelectedSlice': sec})

        else:
            write_json_derivatives(pixdim, proxy.shape, seg_file.replace('nii.gz', 'json'))

    if not exists(join(DIR_PIPELINES['seg'], 'synthseg_lut.txt')):

        labels_abbr = {
            0: 'BG',
            2: 'L-Cerebral-WM',
            3: 'L-Cerebral-GM',
            4: 'L-Lat-Vent',
            5: 'L-Inf-Lat-Vent',
            7: 'L-Cerebell-WM',
            8: 'L-Cerebell-GM',
            10: 'L-TH',
            11: 'L-CAU',
            12: 'L-PU',
            13: 'L-PA',
            14: '3-Vent',
            15: '4-Vent',
            16: 'BS',
            17: 'L-HIPP',
            18: 'L-AM',
            26: 'L-ACC',
            28: 'L-VDC',
            41: 'R-Cerebral-WM',
            42: 'R-Cerebral-GM',
            43: 'R-Lat-Vent',
            44: 'R-Inf-Lat-Vent',
            46: 'R-Cerebell-WM',
            47: 'R-Cerebell-WM',
            49: 'R-TH',
            50: 'R-CAU',
            51: 'R-PU',
            52: 'R-PA',
            53: 'R-HIPP',
            54: 'R-AM',
            58: 'R-ACC',
            60: 'R-VDC',
        }

        fs_lut = {0: {'name': 'Background', 'R': 0, 'G': 0, 'B': 0}}
        with open(join(os.environ['FREESURFER_HOME'], 'FreeSurferColorLUT.txt'), 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                info = [r for r in row[None][0].split(' ') if r != '']
                if len(info) < 5: continue
                try:
                    name = info[1].lower().replace('-', ' ')
                    fs_lut[int(info[0])] = {'name': name, 'R': info[2], 'G': info[3], 'B': info[4]}
                except:
                    continue

        header = ['index', 'name', 'abbreviation', 'R', 'G', 'B', 'mapping']
        label_dict = [
            {'index': label, 'name': fs_lut[label]['name'],
             'abbreviation': labels_abbr[label] if label in labels_abbr else fs_lut[label]['name'],
             'R': fs_lut[label]['R'], 'G': fs_lut[label]['G'], 'B': fs_lut[label]['B'], 'mapping': it_label}
            for it_label, label in SYNTHSEG_APARC_LUT.items()
        ]

        with open(join(DIR_PIPELINES['seg'], 'synthseg_lut.txt'), 'w') as csvfile:
            csvreader = csv.DictWriter(csvfile, fieldnames=header, delimiter='\t')
            csvreader.writeheader()
            csvreader.writerows(label_dict)


    print('\n')
    print('# --------- FI (SynthSeg pipeline) --------- #')
    print('\n')

