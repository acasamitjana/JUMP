import nibabel as nib
import torch
from tensorflow import keras
import tensorflow as tf

from src.layers import SpatialInterpolation
from utils import synthmorph_utils, fn_utils
from utils.labels import *

def integrate_svf(svf_mri, factor=2, inverse=False, int_end=1):
    svf = np.array(svf_mri.dataobj)
    if inverse: svf = -svf
    if svf.shape[0] == 3:
        svf = np.transpose(svf, axes=(1, 2, 3, 0))

    Saff = svf_mri.affine
    Daff = Saff.copy()
    for c in range(3):
        Daff[:-1, c] = Daff[:-1, c] / factor
    Daff[:-1, -1] = Daff[:-1, -1] - np.matmul(Daff[:-1, :-1], 0.5 * (np.array([factor]*3) - 1))

    integrator = keras.Sequential([synthmorph_utils.VecInt(method='ss', int_steps=7, int_end=int_end)])
    upscaler = keras.Sequential([synthmorph_utils.RescaleTransform(factor)]) if factor != 1 else lambda x: x

    warp_pos_small = integrator(tf.convert_to_tensor(svf[np.newaxis]))
    f2r_field = np.squeeze(upscaler(warp_pos_small))

    flow_mri = nib.Nifti1Image(f2r_field, Daff.astype('float32'))

    return flow_mri

def vol_resample(ref_proxy, flo_proxy, proxysvf=None, proxyflow=None, mode='bilinear', device='cpu', return_np=False):
    interp_func_flow = SpatialInterpolation(padding_mode='zeros', mode='bilinear').to(device)
    if mode == 'distance':
        interp_func = fn_utils.compute_distance_map_nongrid
    else:
        interp_func = SpatialInterpolation(padding_mode='border', mode=mode).to(device)

    ref_v2r = (ref_proxy.affine).astype('float32')
    target_v2r = (flo_proxy.affine).astype('float32')

    image = np.array(flo_proxy.dataobj)
    if len(image.shape) == 3: image = torch.tensor(image[np.newaxis, np.newaxis], device=device).float()
    elif len(image.shape) == 4: image = torch.tensor(np.transpose(image, axes=(3, 0, 1, 2))[np.newaxis], device=device).float()
    else: raise ValueError('Image must be 3-D or 4-D (channels in the last dimensions).')

    ii = np.arange(0, ref_proxy.shape[0], dtype='int32')
    jj = np.arange(0, ref_proxy.shape[1], dtype='int32')
    kk = np.arange(0, ref_proxy.shape[2], dtype='int32')

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

    del ii, jj, kk

    II = torch.tensor(II, device='cpu')
    JJ = torch.tensor(JJ, device='cpu')
    KK = torch.tensor(KK, device='cpu')

    if proxysvf is not None or proxyflow is not None:
        if proxysvf is not None: proxyflow = integrate_svf(proxysvf)

        flow = np.array(proxyflow.dataobj)
        flow_v2r = proxyflow.affine
        flow_v2r = flow_v2r.astype('float32')

        if len(flow.shape) == 4: flow = flow[np.newaxis]
        if flow.shape[-1] == 3: flow = np.transpose(flow, axes=(0, 4, 1, 2, 3))
        flow = torch.tensor(flow)

        affine = torch.tensor(np.linalg.inv(flow_v2r) @ ref_v2r)
        vM_ref_svf_I = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        vM_ref_svf_J = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        vM_ref_svf_K = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

        vM = torch.unsqueeze(torch.stack((vM_ref_svf_I, vM_ref_svf_J, vM_ref_svf_K), dim=0), 0)
        flow_res = interp_func_flow(flow, vM)

        vM_flo_svf_I = vM_ref_svf_I + flow_res[0, 0]
        vM_flo_svf_J = vM_ref_svf_J + flow_res[0, 1]
        vM_flo_svf_K = vM_ref_svf_K + flow_res[0, 2]

        affine = torch.tensor(np.linalg.inv(target_v2r) @ flow_v2r)
        vM_flo_I = affine[0, 0] * vM_flo_svf_I + affine[0, 1] * vM_flo_svf_J + affine[0, 2] * vM_flo_svf_K + affine[0, 3]
        vM_flo_J = affine[1, 0] * vM_flo_svf_I + affine[1, 1] * vM_flo_svf_J + affine[1, 2] * vM_flo_svf_K + affine[1, 3]
        vM_flo_K = affine[2, 0] * vM_flo_svf_I + affine[2, 1] * vM_flo_svf_J + affine[2, 2] * vM_flo_svf_K + affine[2, 3]
        vM_flo = torch.unsqueeze(torch.stack((vM_flo_I, vM_flo_J, vM_flo_K), axis=0), 0)

    else:
        affine = torch.tensor(np.linalg.inv(target_v2r) @ ref_v2r)
        vM_flo_I = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        vM_flo_J = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        vM_flo_K = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]
        vM_flo = torch.unsqueeze(torch.stack((vM_flo_I, vM_flo_J, vM_flo_K), axis=0), 0)


    if mode == 'distance':
        reg_image = interp_func(np.squeeze(image.cpu().detach().numpy()),
                                np.squeeze(vM_flo.cpu().detach().numpy()),
                                labels_lut={k: it_k for it_k, k in enumerate(POST_ARR)})

    else:
        reg_image = interp_func(image, vM_flo)
        reg_image = np.squeeze(reg_image.cpu().detach().numpy())
        if len(reg_image.shape) == 4: reg_image = np.transpose(reg_image, axes=(1, 2, 3, 0))

    del vM_flo

    if return_np:
        return reg_image
    else:
        return nib.Nifti1Image(reg_image, ref_proxy.affine)

def vol_resample_fast(ref_proxy, flo_proxy, proxyflow=None, mode='bilinear', device='cpu', return_np=False):

    ref_v2r = (ref_proxy.affine).astype('float32')
    target_v2r = (flo_proxy.affine).astype('float32')

    ii = np.arange(0, ref_proxy.shape[0], dtype='int32')
    jj = np.arange(0, ref_proxy.shape[1], dtype='int32')
    kk = np.arange(0, ref_proxy.shape[2], dtype='int32')

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

    del ii, jj, kk

    II = torch.tensor(II, device='cpu')
    JJ = torch.tensor(JJ, device='cpu')
    KK = torch.tensor(KK, device='cpu')

    if proxyflow is not None:

        flow_v2r = proxyflow.affine
        flow_v2r = flow_v2r.astype('float32')

        affine = torch.tensor(np.linalg.inv(flow_v2r) @ ref_v2r)
        II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

        flow = np.array(proxyflow.dataobj)
        if flow.shape[-1] == 3: flow = np.transpose(flow, axes=(3, 0, 1, 2))
        flow = torch.tensor(flow)

        FIELD = synthmorph_utils.fast_3D_interp_field_torch(flow, II2, JJ2, KK2)
        II3 = II2 + FIELD[:, :, :, 0]
        JJ3 = JJ2 + FIELD[:, :, :, 1]
        KK3 = KK2 + FIELD[:, :, :, 2]


        affine = torch.tensor(np.linalg.inv(flow_v2r) @ flow_v2r)
        II4 = affine[0, 0] * II3 + affine[0, 1] * JJ3 + affine[0, 2] * KK3 + affine[0, 3]
        JJ4 = affine[1, 0] * II3 + affine[1, 1] * JJ3 + affine[1, 2] * KK3 + affine[1, 3]
        KK4 = affine[2, 0] * II3 + affine[2, 1] * JJ3 + affine[2, 2] * KK3 + affine[2, 3]


    else:
        affine = torch.tensor(np.linalg.inv(target_v2r) @ ref_v2r)
        II4 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ4 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK4 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]


    image = np.array(flo_proxy.dataobj)
    if len(flo_proxy.shape) == 3:
        reg_image = synthmorph_utils.fast_3D_interp_torch(torch.tensor(image), II4, JJ4, KK4, mode)
    else:
        reg_image = synthmorph_utils.fast_3D_interp_field_torch(torch.tensor(image), II4, JJ4, KK4)

    reg_image = reg_image.numpy()

    if return_np:
        return reg_image
    else:
        return nib.Nifti1Image(reg_image, ref_proxy.affine)
