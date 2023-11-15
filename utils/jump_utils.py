import pdb

import nibabel as nib
import numpy as np

from utils import def_utils



def initialize_graph_linear(pairwise_centroids, affine_filepath, pairwise_timepoints=None, ok_centr=None):
    # https://www.cse.sc.edu/~songwang/CourseProj/proj2004/ross/ross.pdf

    refCent, floCent = pairwise_centroids

    if ok_centr is not None:
        refCent = refCent[:, ok_centr > 0]
        floCent = floCent[:, ok_centr > 0]

    trans_ref = np.mean(refCent, axis=1, keepdims=True)
    trans_flo = np.mean(floCent, axis=1, keepdims=True)

    refCent_tx = refCent - trans_ref
    floCent_tx = floCent - trans_flo

    cov = refCent_tx @ floCent_tx.T
    u, s, vt = np.linalg.svd(cov)
    I = np.eye(3)
    if np.prod(np.diag(s)) < 0:
        I[-1, -1] = -1

    Q = vt.T @ I @ u.T

    # Full transformation

    Tr = np.eye(4)
    Tr[:3, 3] = -trans_ref.squeeze()

    Tf = np.eye(4)
    Tf[:3, 3] = trans_flo.squeeze()

    R = np.eye(4)
    R[:3,:3] = Q

    aff = Tf @ R @ Tr

    np.save(affine_filepath, aff)

    if pairwise_timepoints is not None:
        ref_proxy, flo_proxy = pairwise_timepoints
        data = np.array(flo_proxy.dataobj)
        v2r = np.linalg.inv(aff) @ flo_proxy.affine
        proxy_reg = nib.Nifti1Image(data, v2r)
        proxy_reg = def_utils.vol_resample(ref_proxy, proxy_reg, mode='nearest')
        nib.save(proxy_reg, affine_filepath + '.nii.gz')

def create_template_space(linear_image_list):

    boundaries_min = np.zeros((len(linear_image_list), 3))
    boundaries_max = np.zeros((len(linear_image_list), 3))
    margin_bb = 5
    for it_lil, lil in enumerate(linear_image_list):

        if isinstance(lil, nib.nifti1.Nifti1Image):
            proxy = lil
        else:
            proxy = nib.load(lil)
        mask = np.asarray(proxy.dataobj)
        header = proxy.affine
        idx = np.where(mask > 0)
        vox_min = np.concatenate((np.min(idx, axis=1), [1]), axis=0)
        vox_max = np.concatenate((np.max(idx, axis=1), [1]), axis=0)

        minR, minA, minS = np.inf, np.inf, np.inf
        maxR, maxA, maxS = -np.inf, -np.inf, -np.inf

        for i in [vox_min[0], vox_max[0] + 1]:
            for j in [vox_min[1], vox_max[1] + 1]:
                for k in [vox_min[2], vox_max[2] + 1]:
                    aux = np.dot(header, np.asarray([i, j, k, 1]).T)

                    minR, maxR = min(minR, aux[0]), max(maxR, aux[0])
                    minA, maxA = min(minA, aux[1]), max(maxA, aux[1])
                    minS, maxS = min(minS, aux[2]), max(maxS, aux[2])

        minR -= margin_bb
        minA -= margin_bb
        minS -= margin_bb

        maxR += margin_bb
        maxA += margin_bb
        maxS += margin_bb

        boundaries_min[it_lil] = [minR, minA, minS]
        boundaries_max[it_lil] = [maxR, maxA, maxS]
        # boundaries_min += [[minR, minA, minS]]
        # boundaries_max += [[maxR, maxA, maxS]]

    # Get the corners of cuboid in RAS space
    minR = np.mean(boundaries_min[..., 0])
    minA = np.mean(boundaries_min[..., 1])
    minS = np.mean(boundaries_min[..., 2])
    maxR = np.mean(boundaries_max[..., 0])
    maxA = np.mean(boundaries_max[..., 1])
    maxS = np.mean(boundaries_max[..., 2])

    template_size = np.asarray(
        [int(np.ceil(maxR - minR)) + 1, int(np.ceil(maxA - minA)) + 1, int(np.ceil(maxS - minS)) + 1])

    # Define header and size
    template_vox2ras0 = np.asarray([[1, 0, 0, minR],
                                    [0, 1, 0, minA],
                                    [0, 0, 1, minS],
                                    [0, 0, 0, 1]])


    # VOX Mosaic
    II, JJ, KK = np.meshgrid(np.arange(0, template_size[0]),
                             np.arange(0, template_size[1]),
                             np.arange(0, template_size[2]), indexing='ij')

    RR = II + minR
    AA = JJ + minA
    SS = KK + minS
    rasMosaic = np.concatenate((RR.reshape(-1, 1),
                                AA.reshape(-1, 1),
                                SS.reshape(-1, 1),
                                np.ones((np.prod(template_size), 1))), axis=1).T

    return rasMosaic, template_vox2ras0,  tuple(template_size)

def get_aff(bids_loader, entities):
    aff_entities = { 'scope': 'jump-reg', 'desc': 'aff', 'extension': 'npy'}
    im_entities = {'subject': entities['subject'], 'session': entities['session'], 'suffix': entities['suffix']}

    if 'run' in entities.keys():
        im_entities['run'] = entities['run']

    affine_file = bids_loader.get(**{**aff_entities, **im_entities})
    if len(affine_file) != 1:
        return
    affine_matrix = np.load(affine_file[0])

    if np.linalg.matrix_rank(affine_matrix) != 4:
        return None
    else:
        return affine_matrix