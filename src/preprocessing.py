import os
import pdb
from os.path import join

import numpy as np
import scipy.ndimage as ndi
import pandas as pd
import nibabel as nib
import torch

from utils.synthmorph_utils import fast_3D_interp_torch, fast_3D_interp_field_torch
from utils import labels, def_utils



def resample_to_MNI2mm(mni_dir, subject, session):
    # Registrar to MNI2mm: func --> T1w --> MNI 2mm.
    def_MNI_files = list(filter(lambda f: 'desc-field_nonlinear' in f and 'space-MNI' in f, os.listdir(join(mni_dir, subject, session, 'anat'))))
    if len(def_MNI_files) != 1:
        print('None / Too many T1w image available N=' + str(len(def_MNI_files)) + '. Please, refine search.')
        return None, None

    def_MNI_file = def_MNI_files[0]
    def_MNI_proxy = nib.load(join(mni_dir, subject, session, 'anat', def_MNI_file))
    mni_2mm_proxy = nib.load(join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz'))
    def_MNI_2mm_proxy = def_utils.vol_resample(mni_2mm_proxy, def_MNI_proxy)

    t1w_entities = {}
    if 'run' in def_MNI_file:
        t1w_entities['run'] = def_MNI_file.split('run-')[1].split('_')[0]

    return def_MNI_2mm_proxy, t1w_entities

# -------------------------- #
# --- BOLD Preprocessing --- #
# -------------------------- #


def extract_average_signals(rs_in, masks):
    n_time_points = rs_in.shape[3]
    signals = np.zeros((n_time_points, masks.shape[-1]))
    for it_dim in range(masks.shape[-1]):
        masks[..., it_dim] = ndi.binary_erosion(masks[..., it_dim] > 0, iterations=1)

    for ntp in range(n_time_points):
        rs_tp = rs_in[:, :, :, ntp]
        for it_dim in range(masks.shape[-1]):
            signals[ntp, it_dim] = rs_tp[masks[..., it_dim] == 1].mean()

    return signals

def create_confounds_matrix(signals, transf_rp):

    confounds_matrix = np.zeros((6 + signals.shape[-1], transf_rp.shape[0]))
    confounds_matrix[0] = transf_rp[:, 3]
    confounds_matrix[1] = transf_rp[:, 4]
    confounds_matrix[2] = transf_rp[:, 5]
    confounds_matrix[3] = transf_rp[:, 0]
    confounds_matrix[4] = transf_rp[:, 1]
    confounds_matrix[5] = transf_rp[:, 2]

    for it_dim in range(signals.shape[-1]):
        confounds_matrix[6+it_dim] = signals[:, it_dim]

    return confounds_matrix


def runICA(fslDir, inFile, outDir, melDirIn, mask, dim, TR, sep_vn=False):
    """ This function runs MELODIC and merges the mixture modeled thresholded ICs into a single 4D nifti file

    Parameters
    ---------------------------------------------------------------------------------
    fslDir:     Full path of the bin-directory of FSL
    inFile:     Full path to the fMRI data file (nii.gz) on which MELODIC should be run
    outDir:     Full path of the output directory
    melDirIn:   Full path of the MELODIC directory in case it has been run before, otherwise define empty string
    mask:       Full path of the mask to be applied during MELODIC
    dim:        Dimensionality of ICA
    TR:         TR (in seconds) of the fMRI data
    sep_vn:
    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    melodic.ica     MELODIC directory
    melodic_IC_thr.nii.gz   merged file containing the mixture modeling thresholded Z-statistical maps located in melodic.ica/stats/ """

    # Import needed modules
    import os
    import subprocess

    # Define the 'new' MELODIC directory and predefine some associated files
    melDir = os.path.join(outDir, 'melodic.ica')
    melIC = os.path.join(melDir, 'melodic_IC.nii.gz')
    melICmix = os.path.join(melDir, 'melodic_mix')
    melICthr = os.path.join(outDir, 'melodic_IC_thr.nii.gz')

    # When a MELODIC directory is specified,
    # check whether all needed files are present.
    # Otherwise... run MELODIC again
    if len(melDir) != 0 and os.path.isfile(os.path.join(melDirIn, 'melodic_IC.nii.gz')) and os.path.isfile(os.path.join(melDirIn, 'melodic_FTmix')) and os.path.isfile(os.path.join(melDirIn, 'melodic_mix')):

        print('  - The existing/specified MELODIC directory will be used.')

        # If a 'stats' directory is present (contains thresholded spatial maps)
        # create a symbolic link to the MELODIC directory.
        # Otherwise create specific links and
        # run mixture modeling to obtain thresholded maps.
        if os.path.isdir(os.path.join(melDirIn, 'stats')):
            os.symlink(melDirIn, melDir)
        else:
            print('  - The MELODIC directory does not contain the required \'stats\' folder. Mixture modeling on the Z-statistical maps will be run.')

            # Create symbolic links to the items in the specified melodic directory
            os.makedirs(melDir)
            for item in os.listdir(melDirIn):
                os.symlink(os.path.join(melDirIn, item),
                           os.path.join(melDir, item))

            # Run mixture modeling
            os.system(' '.join([os.path.join(fslDir, 'melodic'),
                                '--in=' + melIC,
                                '--ICs=' + melIC,
                                '--mix=' + melICmix,
                                '--outdir=' + melDir,
                                '--Ostats --mmthresh=0.5']))

    else:
        # If a melodic directory was specified, display that it did not contain all files needed for ICA-AROMA (or that the directory does not exist at all)
        if len(melDirIn) != 0:
            if not os.path.isdir(melDirIn):
                print('  - The specified MELODIC directory does not exist. MELODIC will be run seperately.')
            else:
                print('  - The specified MELODIC directory does not contain the required files to run ICA-AROMA. MELODIC will be run seperately.')

        # Run MELODIC
        cmd_melodic = [os.path.join(fslDir, 'melodic'),
                            '--in=' + inFile,
                            '--outdir=' + melDir,
                            '--mask=' + mask,
                            '--dim=' + str(dim),
                            '--Ostats --nobet --mmthresh=0.5 --report',
                            '--tr=' + str(TR)]
        if sep_vn:
            cmd_melodic.append('--sep_vn')
        os.system(' '.join(cmd_melodic))

    # Get number of components
    cmd = ' '.join([os.path.join(fslDir, 'fslinfo'),
                    melIC,
                    '| grep dim4 | head -n1 | awk \'{print $2}\''])
    nrICs = int(float(subprocess.getoutput(cmd)))

    # Merge mixture modeled thresholded spatial maps. Note! In case that mixture modeling did not converge, the file will contain two spatial maps. The latter being the results from a simple null hypothesis test. In that case, this map will have to be used (first one will be empty).
    for i in range(1, nrICs + 1):
        # Define thresholded zstat-map file
        zTemp = os.path.join(melDir, 'stats', 'thresh_zstat' + str(i) + '.nii.gz')
        cmd = ' '.join([os.path.join(fslDir, 'fslinfo'),
                        zTemp,
                        '| grep dim4 | head -n1 | awk \'{print $2}\''])
        lenIC = int(float(subprocess.getoutput(cmd)))

        # Define zeropad for this IC-number and new zstat file
        cmd = ' '.join([os.path.join(fslDir, 'zeropad'),
                        str(i),
                        '4'])
        ICnum = subprocess.getoutput(cmd)
        zstat = os.path.join(outDir, 'thr_zstat' + ICnum)

        # Extract last spatial map within the thresh_zstat file
        os.system(' '.join([os.path.join(fslDir, 'fslroi'),
                            zTemp,      # input
                            zstat,      # output
                            str(lenIC - 1),   # first frame
                            '1']))      # number of frames

    # Merge and subsequently remove all mixture modeled Z-maps within the output directory
    os.system(' '.join([os.path.join(fslDir, 'fslmerge'),
                        '-t',                       # concatenate in time
                        melICthr,                   # output
                        os.path.join(outDir, 'thr_zstat????.nii.gz')]))  # inputs

    os.system('rm ' + os.path.join(outDir, 'thr_zstat????.nii.gz'))

    # Apply the mask to the merged file (in case a melodic-directory was predefined and run with a different mask)
    os.system(' '.join([os.path.join(fslDir, 'fslmaths'),
                        melICthr,
                        '-mas ' + mask,
                        melICthr]))

def register2MNI(refFile, inFile, outFile, affmat, warp):
    """ This function registers an image (or time-series of images) to MNI152 T1 2mm. If no affmat is defined, it only warps (i.e. it assumes that the data has been registerd to the structural scan associated with the warp-file already). If no warp is defined either, it only resamples the data to 2mm isotropic if needed (i.e. it assumes that the data has been registered to a MNI152 template). In case only an affmat file is defined, it assumes that the data has to be linearly registered to MNI152 (i.e. the user has a reason not to use non-linear registration on the data).

    Parameters
    ---------------------------------------------------------------------------------
    refFile:    Full path to reference file(nii.gz)
    inFile:     Full path to the data file (nii.gz) which has to be registerd to MNI152 T1 2mm
    outFile:    Full path of the output file
    affmat:     Full path of the mat file describing the linear registration (if data is still in native space)
    warp:       Full path of the warp file describing the non-linear registration (if data has not been registered to MNI152 space yet)

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    melodic_IC_mm_MNI2mm.nii.gz merged file containing the mixture modeling thresholded Z-statistical maps registered to MNI152 2mm """


    # Import needed modules
    import os

    # Define the MNI152 T1 2mm template
    # fslnobin = fslDir.rsplit('/', 2)[0]
    # ref = os.path.join(fslnobin, 'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz')

    # If the no affmat- or warp-file has been specified, assume that the data is already in MNI152 space. In that case only check if resampling to 2mm is needed
    if (len(affmat) == 0) and (len(warp) == 0):
        # ref and inFile already living on the same space. Just check for resampling
        proxyref = nib.load(refFile)
        proxyflo = nib.load(inFile)
        pixdim_ref = np.sqrt(np.sum(proxyref.affine * proxyref.affine, axis=0))[:-1]
        pixdim_flo = np.sqrt(np.sum(proxyflo.affine * proxyflo.affine, axis=0))[:-1]
        if all([a == b for a, b in zip(pixdim_ref, pixdim_flo)]):
            return
        else:
            II, JJ, KK = np.meshgrid(np.arange(proxyref.shape[0]), np.arange(proxyref.shape[1]), np.arange(proxyref.shape[2]), indexing='ij')
            II, JJ, KK = torch.tensor(II, device='cpu'), torch.tensor(JJ, device='cpu'), torch.tensor(KK, device='cpu')
            affine = torch.tensor(proxyflo.affine @ proxyref.affine, device='cpu')
            II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
            JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
            KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

            print('  Deforming floating image')
            flo_torch = torch.tensor(np.array(proxyflo.dataobj))
            if len(flo_torch.shape) == 3:
                registered = fast_3D_interp_torch(flo_torch, II2, JJ2, KK2)
            else:
                registered = fast_3D_interp_field_torch(flo_torch, II2, JJ2, KK2)
            img = nib.Nifti1Image(registered.numpy(), proxyref.affine)
            nib.save(img, outFile)

    # If only a warp-file has been specified, assume that the data has already been registered to the structural scan. In that case apply the warping without a affmat
    elif (len(affmat) == 0) and (len(warp) != 0):
        proxyref = nib.load(refFile)
        proxyflo = nib.load(inFile)
        proxyfield = nib.load(warp)
        f2r_field = torch.tensor(np.array(proxyfield.dataobj))

        II, JJ, KK = np.meshgrid(np.arange(proxyref.shape[0]), np.arange(proxyref.shape[1]), np.arange(proxyref.shape[2]), indexing='ij')
        II, JJ, KK = torch.tensor(II, device='cpu'), torch.tensor(JJ, device='cpu'), torch.tensor(KK, device='cpu')
        affine = torch.tensor(np.linalg.inv(proxyfield.affine) @ proxyref.affine, device='cpu')
        II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

        FIELD = fast_3D_interp_field_torch(f2r_field, II2, JJ2, KK2)
        RAS_X = FIELD[:, :, :, 0]
        RAS_Y = FIELD[:, :, :, 1]
        RAS_Z = FIELD[:, :, :, 2]

        affine = torch.tensor(np.linalg.inv(proxyflo.affine), device='cpu')
        II4 = affine[0, 0] * RAS_X + affine[0, 1] * RAS_Y + affine[0, 2] * RAS_Z + affine[0, 3]
        JJ4 = affine[1, 0] * RAS_X + affine[1, 1] * RAS_Y + affine[1, 2] * RAS_Z + affine[1, 3]
        KK4 = affine[2, 0] * RAS_X + affine[2, 1] * RAS_Y + affine[2, 2] * RAS_Z + affine[2, 3]

        print('  Deforming floating image')
        flo_torch = torch.tensor(np.array(proxyflo.dataobj))
        if len(flo_torch.shape) == 3:
            registered = fast_3D_interp_torch(flo_torch, II4, JJ4, KK4)
        else:
            registered = fast_3D_interp_field_torch(flo_torch, II4, JJ4, KK4)
        img = nib.Nifti1Image(registered.numpy(), proxyref.affine)
        nib.save(img, outFile)

    # If only a affmat-file has been specified perform affine registration to MNI
    elif (len(affmat) != 0) and (len(warp) == 0):
        proxyref = nib.load(refFile)
        proxyflo = nib.load(inFile)
        M = np.load(affmat)

        II, JJ, KK = np.meshgrid(np.arange(proxyref.shape[0]), np.arange(proxyref.shape[1]), np.arange(proxyref.shape[2]), indexing='ij')
        II, JJ, KK = torch.tensor(II, device='cpu'), torch.tensor(JJ, device='cpu'), torch.tensor(KK, device='cpu')
        affine = torch.tensor(proxyflo.affine @ np.linalg.inv(M) @ proxyref.affine, device='cpu')
        II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

        print('  Deforming floating image')
        flo_torch = torch.tensor(np.array(proxyflo.dataobj))
        if len(flo_torch.shape) == 3:
            registered = fast_3D_interp_torch(flo_torch, II2, JJ2, KK2)
        else:
            registered = fast_3D_interp_field_torch(flo_torch, II2, JJ2, KK2)
        img = nib.Nifti1Image(registered.numpy(), proxyref.affine)
        nib.save(img, outFile)


    # If both a affmat- and warp-file have been defined, apply the warping accordingly
    else:
        proxyref = nib.load(refFile)
        proxyflo = nib.load(inFile)
        proxyfield = nib.load(warp)
        M = np.load(affmat)

        f2r_field = torch.tensor(np.array(proxyfield.dataobj))

        II, JJ, KK = np.meshgrid(np.arange(proxyref.shape[0]), np.arange(proxyref.shape[1]), np.arange(proxyref.shape[2]), indexing='ij')
        II, JJ, KK = torch.tensor(II, device='cpu'), torch.tensor(JJ, device='cpu'), torch.tensor(KK, device='cpu')
        affine = torch.tensor(np.linalg.inv(proxyfield.affine) @ proxyref.affine, device='cpu')
        II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

        FIELD = fast_3D_interp_field_torch(f2r_field, II2, JJ2, KK2)
        RAS_X = FIELD[:, :, :, 0]
        RAS_Y = FIELD[:, :, :, 1]
        RAS_Z = FIELD[:, :, :, 2]

        affine = torch.tensor(np.linalg.inv(proxyflo.affine) @ M, device='cpu')
        II4 = affine[0, 0] * RAS_X + affine[0, 1] * RAS_Y + affine[0, 2] * RAS_Z + affine[0, 3]
        JJ4 = affine[1, 0] * RAS_X + affine[1, 1] * RAS_Y + affine[1, 2] * RAS_Z + affine[1, 3]
        KK4 = affine[2, 0] * RAS_X + affine[2, 1] * RAS_Y + affine[2, 2] * RAS_Z + affine[2, 3]

        flo_torch = torch.tensor(np.array(proxyflo.dataobj))
        if len(flo_torch.shape) == 3:
            registered = fast_3D_interp_torch(flo_torch, II4, JJ4, KK4)
        else:
            registered = fast_3D_interp_field_torch(flo_torch, II4, JJ4, KK4)

        img = nib.Nifti1Image(registered.numpy(), proxyref.affine)
        nib.save(img, outFile)

def registerMNI2Image(refFile, inFile, outFile, affmat, warp, aux_affmat=None):
    """ This function registers an image (or time-series of images) to MNI152 T1 2mm. If no affmat is defined, it only warps (i.e. it assumes that the data has been registerd to the structural scan associated with the warp-file already). If no warp is defined either, it only resamples the data to 2mm isotropic if needed (i.e. it assumes that the data has been registered to a MNI152 template). In case only an affmat file is defined, it assumes that the data has to be linearly registered to MNI152 (i.e. the user has a reason not to use non-linear registration on the data).

    Parameters
    ---------------------------------------------------------------------------------
    fslDir:     Full path of the bin-directory of FSL
    inFile:     Full path to the data file (nii.gz) which has to be registerd to MNI152 T1 2mm
    outFile:    Full path of the output file
    affmat:     Full path of the mat file describing the linear registration (if data is still in native space)
    warp:       Full path of the warp file describing the non-linear registration (if data has not been registered to MNI152 space yet)

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    melodic_IC_mm_MNI2mm.nii.gz merged file containing the mixture modeling thresholded Z-statistical maps registered to MNI152 2mm """


    # Import needed modules
    import os

    # Define the MNI152 T1 2mm template
    proxyref = nib.load(refFile)
    proxyflo = nib.load(inFile)

    # If the no affmat- or warp-file has been specified, assume that the data is already in MNI152 space. In that case only check if resampling to 2mm is needed
    if (len(affmat) == 0) and (len(warp) == 0):
        # ref and inFile already living on the same space. Just check for resampling
        pixdim_ref = np.sqrt(np.sum(proxyref.affine * proxyref.affine, axis=0))[:-1]
        pixdim_flo = np.sqrt(np.sum(proxyflo.affine * proxyflo.affine, axis=0))[:-1]
        if all([a == b for a, b in zip(pixdim_ref, pixdim_flo)]):
            return
        else:
            II, JJ, KK = np.meshgrid(np.arange(proxyref.shape[0]), np.arange(proxyref.shape[1]), np.arange(proxyref.shape[2]), indexing='ij')
            II, JJ, KK = torch.tensor(II, device='cpu'), torch.tensor(JJ, device='cpu'), torch.tensor(KK, device='cpu')
            affine = torch.tensor(proxyflo.affine @ proxyref.affine, device='cpu')
            II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
            JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
            KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

            print('  Deforming floating image')
            flo_torch = torch.tensor(np.array(proxyflo.dataobj))
            if len(flo_torch.shape) == 3:
                registered = fast_3D_interp_torch(flo_torch, II2, JJ2, KK2)
            else:
                registered = fast_3D_interp_field_torch(flo_torch, II2, JJ2, KK2)
            img = nib.Nifti1Image(registered.numpy(), proxyref.affine)
            nib.save(img, outFile)

    # If only a warp-file has been specified, assume that the data has already been registered to the structural scan. In that case apply the warping without a affmat
    elif (len(affmat) == 0) and (len(warp) != 0):
        proxyfield = nib.load(warp)
        f2r_field = torch.tensor(np.array(proxyfield.dataobj))

        II, JJ, KK = np.meshgrid(np.arange(proxyref.shape[0]), np.arange(proxyref.shape[1]), np.arange(proxyref.shape[2]), indexing='ij')
        II, JJ, KK = torch.tensor(II, device='cpu'), torch.tensor(JJ, device='cpu'), torch.tensor(KK, device='cpu')
        affine = torch.tensor(np.linalg.inv(proxyfield.affine) @ proxyref.affine, device='cpu')
        II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

        FIELD = fast_3D_interp_field_torch(f2r_field, II2, JJ2, KK2)
        RAS_X = FIELD[:, :, :, 0]
        RAS_Y = FIELD[:, :, :, 1]
        RAS_Z = FIELD[:, :, :, 2]

        affine = torch.tensor(np.linalg.inv(proxyflo.affine), device='cpu')
        II4 = affine[0, 0] * RAS_X + affine[0, 1] * RAS_Y + affine[0, 2] * RAS_Z + affine[0, 3]
        JJ4 = affine[1, 0] * RAS_X + affine[1, 1] * RAS_Y + affine[1, 2] * RAS_Z + affine[1, 3]
        KK4 = affine[2, 0] * RAS_X + affine[2, 1] * RAS_Y + affine[2, 2] * RAS_Z + affine[2, 3]

        print('  Deforming floating image')
        flo_torch = torch.tensor(np.array(proxyflo.dataobj))
        if len(flo_torch.shape) == 3:
            registered = fast_3D_interp_torch(flo_torch, II4, JJ4, KK4)
        else:
            registered = fast_3D_interp_field_torch(flo_torch, II4, JJ4, KK4)
        img = nib.Nifti1Image(registered.numpy(), proxyref.affine)
        nib.save(img, outFile)

    # If only a affmat-file has been specified perform affine registration to MNI
    elif (len(affmat) != 0) and (len(warp) == 0):
        M = np.load(affmat)

        II, JJ, KK = np.meshgrid(np.arange(proxyref.shape[0]), np.arange(proxyref.shape[1]), np.arange(proxyref.shape[2]), indexing='ij')
        II, JJ, KK = torch.tensor(II, device='cpu'), torch.tensor(JJ, device='cpu'), torch.tensor(KK, device='cpu')
        affine = torch.tensor(proxyflo.affine @ np.linalg.inv(M) @ proxyref.affine, device='cpu')
        II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

        print('  Deforming floating image')
        flo_torch = torch.tensor(np.array(proxyflo.dataobj))
        if len(flo_torch.shape) == 3:
            registered = fast_3D_interp_torch(flo_torch, II2, JJ2, KK2)
        else:
            registered = fast_3D_interp_field_torch(flo_torch, II2, JJ2, KK2)
        img = nib.Nifti1Image(registered.numpy(), proxyref.affine)
        nib.save(img, outFile)

    # If both a affmat- and warp-file have been defined, apply the warping accordingly
    else:
        proxyfield = nib.load(warp)
        M = np.load(affmat)

        f2r_field = torch.tensor(np.array(proxyfield.dataobj))

        II, JJ, KK = np.meshgrid(np.arange(proxyref.shape[0]), np.arange(proxyref.shape[1]), np.arange(proxyref.shape[2]), indexing='ij')
        II, JJ, KK = torch.tensor(II, device='cpu'), torch.tensor(JJ, device='cpu'), torch.tensor(KK, device='cpu')
        affine = torch.tensor(np.linalg.inv(proxyfield.affine) @ M @ proxyref.affine, device='cpu')
        II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

        FIELD = fast_3D_interp_field_torch(f2r_field, II2, JJ2, KK2)
        RAS_X = FIELD[:, :, :, 0]
        RAS_Y = FIELD[:, :, :, 1]
        RAS_Z = FIELD[:, :, :, 2]

        affine = torch.tensor(np.linalg.inv(proxyflo.affine), device='cpu')
        II4 = affine[0, 0] * RAS_X + affine[0, 1] * RAS_Y + affine[0, 2] * RAS_Z + affine[0, 3]
        JJ4 = affine[1, 0] * RAS_X + affine[1, 1] * RAS_Y + affine[1, 2] * RAS_Z + affine[1, 3]
        KK4 = affine[2, 0] * RAS_X + affine[2, 1] * RAS_Y + affine[2, 2] * RAS_Z + affine[2, 3]

        flo_torch = torch.tensor(np.array(proxyflo.dataobj))
        if len(flo_torch.shape) == 3:
            registered = fast_3D_interp_torch(flo_torch, II4, JJ4, KK4)
        else:
            registered = fast_3D_interp_field_torch(flo_torch, II4, JJ4, KK4)

        img = nib.Nifti1Image(registered.numpy(), proxyref.affine)
        nib.save(img, outFile)

def register2MNI_fsl(fslDir, inFile, outFile, affmat, warp):
    """ This function registers an image (or time-series of images) to MNI152 T1 2mm. If no affmat is defined, it only warps (i.e. it assumes that the data has been registerd to the structural scan associated with the warp-file already). If no warp is defined either, it only resamples the data to 2mm isotropic if needed (i.e. it assumes that the data has been registered to a MNI152 template). In case only an affmat file is defined, it assumes that the data has to be linearly registered to MNI152 (i.e. the user has a reason not to use non-linear registration on the data).

    Parameters
    ---------------------------------------------------------------------------------
    fslDir:     Full path of the bin-directory of FSL
    inFile:     Full path to the data file (nii.gz) which has to be registerd to MNI152 T1 2mm
    outFile:    Full path of the output file
    affmat:     Full path of the mat file describing the linear registration (if data is still in native space)
    warp:       Full path of the warp file describing the non-linear registration (if data has not been registered to MNI152 space yet)

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    melodic_IC_mm_MNI2mm.nii.gz merged file containing the mixture modeling thresholded Z-statistical maps registered to MNI152 2mm """


    # Import needed modules
    import os
    import subprocess

    # Define the MNI152 T1 2mm template
    fslnobin = fslDir.rsplit('/', 2)[0]
    ref = os.path.join(fslnobin, 'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz')

    # If the no affmat- or warp-file has been specified, assume that the data is already in MNI152 space. In that case only check if resampling to 2mm is needed
    if (len(affmat) == 0) and (len(warp) == 0):
        # Get 3D voxel size
        pixdim1 = float(subprocess.getoutput('%sfslinfo %s | grep pixdim1 | awk \'{print $2}\'' % (fslDir, inFile)))
        pixdim2 = float(subprocess.getoutput('%sfslinfo %s | grep pixdim2 | awk \'{print $2}\'' % (fslDir, inFile)))
        pixdim3 = float(subprocess.getoutput('%sfslinfo %s | grep pixdim3 | awk \'{print $2}\'' % (fslDir, inFile)))

        # If voxel size is not 2mm isotropic, resample the data, otherwise copy the file
        if (pixdim1 != 2) or (pixdim2 != 2) or (pixdim3 != 2):
            os.system(' '.join([os.path.join(fslDir, 'flirt'),
                                ' -ref ' + ref,
                                ' -in ' + inFile,
                                ' -out ' + outFile,
                                ' -applyisoxfm 2 -interp trilinear']))
        else:
            os.system('cp ' + inFile + ' ' + outFile)

    # If only a warp-file has been specified, assume that the data has already been registered to the structural scan. In that case apply the warping without a affmat
    elif (len(affmat) == 0) and (len(warp) != 0):
        # Apply warp
        os.system(' '.join([os.path.join(fslDir, 'applywarp'),
                            '--ref=' + ref,
                            '--in=' + inFile,
                            '--out=' + outFile,
                            '--warp=' + warp,
                            '--interp=trilinear']))

    # If only a affmat-file has been specified perform affine registration to MNI
    elif (len(affmat) != 0) and (len(warp) == 0):
        os.system(' '.join([os.path.join(fslDir, 'flirt'),
                            '-ref ' + ref,
                            '-in ' + inFile,
                            '-out ' + outFile,
                            '-applyxfm -init ' + affmat,
                            '-interp trilinear']))

    # If both a affmat- and warp-file have been defined, apply the warping accordingly
    else:
        os.system(' '.join([os.path.join(fslDir, 'applywarp'),
                            '--ref=' + ref,
                            '--in=' + inFile,
                            '--out=' + outFile,
                            '--warp=' + warp,
                            '--premat=' + affmat,
                            '--interp=trilinear']))

def classification(outDir, maxRPcorr, edgeFract, HFC, csfFract):
    """ This function classifies a set of components into motion and
    non-motion components based on four features;
    maximum RP correlation, high-frequency content, edge-fraction and CSF-fraction

    Parameters
    ---------------------------------------------------------------------------------
    outDir:     Full path of the output directory
    maxRPcorr:  Array of the 'maximum RP correlation' feature scores of the components
    edgeFract:  Array of the 'edge fraction' feature scores of the components
    HFC:        Array of the 'high-frequency content' feature scores of the components
    csfFract:   Array of the 'CSF fraction' feature scores of the components

    Return
    ---------------------------------------------------------------------------------
    motionICs   Array containing the indices of the components identified as motion components

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    classified_motion_ICs.txt   A text file containing the indices of the components identified as motion components """

    # Import required modules
    import numpy as np
    import os

    # Classify the ICs as motion or non-motion

    # Define criteria needed for classification (thresholds and hyperplane-parameters)
    thr_csf = 0.10
    thr_HFC = 0.35
    hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])

    # Project edge & maxRPcorr feature scores to new 1D space
    x = np.array([maxRPcorr, edgeFract])
    proj = hyp[0] + np.dot(x.T, hyp[1:])

    # Classify the ICs
    motionICs = np.squeeze(np.array(np.where((proj > 0) + (csfFract > thr_csf) + (HFC > thr_HFC))))

    # Put the feature scores in a text file
    np.savetxt(os.path.join(outDir, 'feature_scores.txt'),
               np.vstack((maxRPcorr, edgeFract, HFC, csfFract)).T)

    # Put the indices of motion-classified ICs in a text file
    txt = open(os.path.join(outDir, 'classified_motion_ICs.txt'), 'w')
    if motionICs.size > 1:  # and len(motionICs) != 0: if motionICs is not None and
        txt.write(','.join(['{:.0f}'.format(num) for num in (motionICs + 1)]))
    elif motionICs.size == 1:
        txt.write('{:.0f}'.format(motionICs + 1))
    txt.close()

    # Create a summary overview of the classification
    txt = open(os.path.join(outDir, 'classification_overview.txt'), 'w')
    txt.write('\t'.join(['IC',
                         'Motion/noise',
                         'maximum RP correlation',
                         'Edge-fraction',
                         'High-frequency content',
                         'CSF-fraction']))
    txt.write('\n')
    for i in range(0, len(csfFract)):
        if (proj[i] > 0) or (csfFract[i] > thr_csf) or (HFC[i] > thr_HFC):
            classif = "True"
        else:
            classif = "False"
        txt.write('\t'.join(['{:d}'.format(i + 1),
                             classif,
                             '{:.2f}'.format(maxRPcorr[i]),
                             '{:.2f}'.format(edgeFract[i]),
                             '{:.2f}'.format(HFC[i]),
                             '{:.2f}'.format(csfFract[i])]))
        txt.write('\n')
    txt.close()

    return motionICs

def denoising(fslDir, inFile, outDir, melmix, denType, denIdx):
    """ This function classifies the ICs based on the four features;
    maximum RP correlation, high-frequency content, edge-fraction and CSF-fraction

    Parameters
    ---------------------------------------------------------------------------------
    fslDir:     Full path of the bin-directory of FSL
    inFile:     Full path to the data file (nii.gz) which has to be denoised
    outDir:     Full path of the output directory
    melmix:     Full path of the melodic_mix text file
    denType:    Type of requested denoising ('aggr': aggressive, 'nonaggr': non-aggressive, 'both': both aggressive and non-aggressive
    denIdx:     Indices of the components that should be regressed out

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    denoised_func_data_<denType>.nii.gz:        A nii.gz file of the denoised fMRI data"""

    # Import required modules
    import os
    import numpy as np

    # Check if denoising is needed (i.e. are there components classified as motion)
    check = denIdx.size > 0

    if check == 1:
        # Put IC indices into a char array
        if denIdx.size == 1:
            denIdxStrJoin = "%d"%(denIdx + 1)
        else:
            denIdxStr = np.char.mod('%i', (denIdx + 1))
            denIdxStrJoin = ','.join(denIdxStr)

        # Non-aggressive denoising of the data using fsl_regfilt (partial regression), if requested
        if (denType == 'nonaggr') or (denType == 'both'):
            os.system(' '.join([os.path.join(fslDir, 'fsl_regfilt'),
                                '--in=' + inFile,
                                '--design=' + melmix,
                                '--filter="' + denIdxStrJoin + '"',
                                '--out=' + os.path.join(outDir, 'denoised_func_data_nonaggr.nii.gz')]))

        # Aggressive denoising of the data using fsl_regfilt (full regression)
        if (denType == 'aggr') or (denType == 'both'):
            os.system(' '.join([os.path.join(fslDir, 'fsl_regfilt'),
                                '--in=' + inFile,
                                '--design=' + melmix,
                                '--filter="' + denIdxStrJoin + '"',
                                '--out=' + os.path.join(outDir, 'denoised_func_data_aggr.nii.gz'),
                                '-a']))
    else:
        print("  - None of the components were classified as motion, so no denoising is applied (a symbolic link to the input file will be created).")
        if (denType == 'nonaggr') or (denType == 'both'):
            os.symlink(inFile, os.path.join(outDir, 'denoised_func_data_nonaggr.nii.gz'))
        if (denType == 'aggr') or (denType == 'both'):
            os.symlink(inFile, os.path.join(outDir, 'denoised_func_data_aggr.nii.gz'))


# ------------------------- #
# --- PET Preprocessing --- #
# ------------------------- #
def get_fwhm(pet_image):

    try:
        pet_psf = pd.read_csv(join(os.environ['PYTHONPATH'], 'data', 'models', 'pet_scanner_psf.csv'))
        pet_psf = pet_psf.set_index(['Manufacturer', 'Scanner'])

        model = pet_image.entities['Manufacturer']
        machine = pet_image.entities['ManufacturersModelName']
        if model == 'GE':
            if 'Discovery' in machine and 'STE' in machine:
                machine = 'DiscSTE_RP'
            elif 'Discovery' in machine and 'ST' in machine:
                machine = 'DiscST_RP'
            elif 'Discovery' in machine and '600' in machine:
                machine = 'Disc600_RP'
            elif 'Discovery' in machine and '690' in machine:
                machine = 'Disc690_RP'
            elif 'Discovery' in machine and 'RX' in machine:
                machine = 'DiscRX_RP'
            elif 'Discovery' in machine and 'LS' in machine:
                machine = 'DiscLS_RP'
            elif 'Advance' in machine:
                machine = 'Advance_RP'

        if model == 'Philips':
            if 'gemini' in machine.lower() and 'TF' in machine:
                machine = 'GemTF Sharp'
            elif 'vereos' in machine.lower():
                return [4.5, 4.5, 4.5]
            elif 'ingenuity' in machine.lower():
                return [5, 5, 5]
            else:
                return [3, 3, 3]

        if model == 'Siemens':
            if 'biograph' in machine.lower() and 'mCT' in machine:
                machine = 'Biograph mCT'
            elif '1080' in machine:
                machine = 'BioGraph 1080'
            elif '1094' in machine.lower():
                machine = 'BioGraph TruePoint (1093/1094)'
            elif 'truepoint' in machine.lower():
                machine = 'BioGraph TruePoint (1093/1094)'
            elif '962' in machine:
                machine = 'ECAT Exact HR+'


        row = pet_psf.loc[model, machine]
        fwhm = (float(row['XY smoothing']), float(row['XY smoothing']), float(row['Z smoothing']))

    except:
        fwhm = [6, 6, 6]

    return fwhm

def psf_gaussian(vx_size=(1, 1, 1), fwhm=(5, 5, 5), hradius=8):
    '''
    Separable kernels for Gaussian convolution executed on the GPU device
    The output kernels are in this order: z, y, x
    '''

    # if voxel size is given as scalar, interpret it as an isotropic
    # voxel size.
    if isinstance(vx_size, (float, int)):
        vx_size = [vx_size, vx_size, vx_size]

    # the same for the Gaussian kernel
    if isinstance(fwhm, (float, int)):
        fwhm = [fwhm, fwhm, fwhm]

    # avoid zeros in FWHM
    fwhm = [x + 1e-3 * (x <= 0) for x in fwhm]

    xSig = (fwhm[2] / vx_size[2]) / (2 * (2 * np.log(2))**.5)
    ySig = (fwhm[1] / vx_size[1]) / (2 * (2 * np.log(2))**.5)
    zSig = (fwhm[0] / vx_size[0]) / (2 * (2 * np.log(2))**.5)

    # get the separated kernels
    x = np.arange(-hradius, hradius + 1)
    xKrnl = np.exp(-(x**2 / (2 * xSig**2)))
    yKrnl = np.exp(-(x**2 / (2 * ySig**2)))
    zKrnl = np.exp(-(x**2 / (2 * zSig**2)))

    # normalise kernels
    xKrnl /= np.sum(xKrnl)
    yKrnl /= np.sum(yKrnl)
    zKrnl /= np.sum(zKrnl)

    krnl = np.array([zKrnl, yKrnl, xKrnl], dtype=np.float32)

    # for checking if the normalisation worked
    # np.prod( np.sum(krnl,axis=1) )

    # return all kernels together
    return krnl

def conv_separable(vol, knl, output=None):
    """
    Args:
      vol(ndarray): Can be any number of dimensions `ndim`
        (GPU requires `ndim <= 3`).
      knl(ndarray): `ndim` x `width` separable kernel
        (GPU requires `width <= 17`).
      dev_id(int or bool): GPU device ID to try [default: 0].
        Set to `False` to force CPU fallback.
    """
    assert vol.ndim == len(knl)
    assert knl.ndim == 2

    for dim in range(len(knl)):
        h = knl[dim].reshape((1,) * dim + (-1,) + (1,) * (len(knl) - dim - 1))
        vol = ndi.convolve(vol, h, output=output, mode='constant', cval=0.)
    return vol

def pvc_yang(input_im, input_seg, kernel, iter=5):
    '''
    Partial volume correction using iterative Yang method.
    Arguments:
        imgIn: input image which is blurred due to the PSF of the scanner
        krnl: shift invariant kernel of the PSF
        imgSeg: segmentation into regions starting with 0 (e.g., background)
          and then next integer numbers
        itr: number of iteration (default 5)
    '''

    dim = input_im.shape
    m = np.unique(input_seg).astype('int32')
    m_a = np.zeros((len(m), iter), dtype=np.float32)

    for it_jr, jr in enumerate(m):
        m_a[it_jr, 0] = np.mean(input_im[input_seg == jr])

    # init output image
    imgOut = np.copy(input_im)

    # iterative Yang algorithm:
    for i in range(0, iter):
        # piece-wise constant image
        imgPWC = imgOut
        imgPWC[imgPWC < 0] = 0
        for jr in m:
            imgPWC[input_seg == jr] = np.mean(imgPWC[input_seg == jr])

        # blur the piece-wise constant image
        imgSmo = conv_separable(imgPWC, kernel)

        # correction factors
        imgCrr = np.ones(dim, dtype=np.float32)
        imgCrr[imgSmo > 0] = imgPWC[imgSmo > 0] / imgSmo[imgSmo > 0]
        imgOut = input_im * imgCrr
        for it_jr, jr in enumerate(m):
            m_a[it_jr, i] = np.mean(imgOut[input_seg == jr])

    return imgOut, m_a

def group_regions_seg(t1w_seg_array):
    new_t1w_seg_array = np.zeros_like(t1w_seg_array)
    for lab, name in labels.APARC_DICT.items():
        new_t1w_seg_array[t1w_seg_array == lab] = 3

    # Cingulate:
    for lab, name in labels.APARC_DICT.items():
        if 'cingulate' in name:
            new_t1w_seg_array[t1w_seg_array == lab] = 1003

    # Temporal:
    for lab, name in labels.APARC_DICT.items():
        if 'temporal' in name:
            new_t1w_seg_array[t1w_seg_array == lab] = 1007
        elif 'fusiform' in name:
            new_t1w_seg_array[t1w_seg_array == lab] = 1007
        elif 'entorhinal' in name:
            new_t1w_seg_array[t1w_seg_array == lab] = 1007
        elif 'bankssts' in name:
            new_t1w_seg_array[t1w_seg_array == lab] = 1007
        elif 'bankssts' in name:
            new_t1w_seg_array[t1w_seg_array == lab] = 1007

    # Parietal:
    for lab, name in labels.APARC_DICT.items():
        if 'temporal' in name:
            new_t1w_seg_array[t1w_seg_array == lab] = 1009
        elif 'supramarginal' in name:
            new_t1w_seg_array[t1w_seg_array == lab] = 1009
        elif 'postcentral' in name:
            new_t1w_seg_array[t1w_seg_array == lab] = 1009
        elif 'precuneus' in name:
            new_t1w_seg_array[t1w_seg_array == lab] = 1009

    # Subcortical
    new_t1w_seg_array[t1w_seg_array == 10] = 10
    new_t1w_seg_array[t1w_seg_array == 11] = 10
    new_t1w_seg_array[t1w_seg_array == 12] = 10
    new_t1w_seg_array[t1w_seg_array == 13] = 10
    new_t1w_seg_array[t1w_seg_array == 18] = 10
    new_t1w_seg_array[t1w_seg_array == 26] = 10
    new_t1w_seg_array[t1w_seg_array == 28] = 10
    new_t1w_seg_array[t1w_seg_array == 49] = 10
    new_t1w_seg_array[t1w_seg_array == 50] = 10
    new_t1w_seg_array[t1w_seg_array == 51] = 10
    new_t1w_seg_array[t1w_seg_array == 52] = 10
    new_t1w_seg_array[t1w_seg_array == 54] = 10
    new_t1w_seg_array[t1w_seg_array == 58] = 10
    new_t1w_seg_array[t1w_seg_array == 60] = 10

    # Cerebral WM
    new_t1w_seg_array[t1w_seg_array == 2] = 2
    new_t1w_seg_array[t1w_seg_array == 41] = 2

    # Hippocampus
    new_t1w_seg_array[t1w_seg_array == 17] = 17
    new_t1w_seg_array[t1w_seg_array == 53] = 17

    # Brain stem
    new_t1w_seg_array[t1w_seg_array == 16] = 16

    # Cerebellar WM
    new_t1w_seg_array[t1w_seg_array == 7] = 7
    new_t1w_seg_array[t1w_seg_array == 46] = 7

    # Cerebellar GM
    new_t1w_seg_array[t1w_seg_array == 8] = 8
    new_t1w_seg_array[t1w_seg_array == 47] = 8

    # CSF
    new_t1w_seg_array[t1w_seg_array == 24] = 24
    new_t1w_seg_array[t1w_seg_array == 14] = 24
    new_t1w_seg_array[t1w_seg_array == 15] = 24
    new_t1w_seg_array[t1w_seg_array == 4] = 24
    new_t1w_seg_array[t1w_seg_array == 5] = 24
    new_t1w_seg_array[t1w_seg_array == 43] = 24
    new_t1w_seg_array[t1w_seg_array == 44] = 24

    return new_t1w_seg_array
