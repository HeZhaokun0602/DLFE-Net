import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import make_grid
import math


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu()
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    tensor = torch.clamp(tensor, 0, 1)
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        #img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB

        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round().astype(np.uint8)

    return img_np


def calculate_metrics(reference, enhanced):
    # 初始化LPIPS模型
    loss_fn = lpips.LPIPS(net='alex').to('cuda')

    # 转换图像格式
    reference_tensor = torch.from_numpy(reference).float().permute(2, 0, 1).unsqueeze(0).to('cuda') / 255.0
    enhanced_tensor = torch.from_numpy(enhanced).float().permute(2, 0, 1).unsqueeze(0).to('cuda') / 255.0

    # 计算PSNR
    psnr_value = psnr(reference, enhanced)

    # 计算SSIM
    ssim_value = ssim(reference, enhanced, multichannel=True)

    # 计算LPIPS
    lpips_value = loss_fn(reference_tensor, enhanced_tensor).item()

    return {
        "psnr": round(psnr_value, 4),
        "ssim": round(ssim_value, 4),
        "lpips": round(lpips_value, 4)
    }