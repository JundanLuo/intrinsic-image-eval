# intrinsic-image-eval
# https://github.com/JundanLuo/intrinsic-image-eval
# Evaluation tools for intrinsic image decomposition.


import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def rgb_to_srgb(rgb, gamma=1.0/2.2):
    return rgb.clip(min=0.0) ** gamma


def srgb_to_rgb(srgb, gamma=1.0/2.2):
    return srgb.clip(min=0.0) ** (1.0 / gamma)


def numpy_to_tensor(img):
    assert isinstance(img, np.ndarray) and img.dtype == np.float32, \
        f"img should be np.ndarray with dtype np.float32"
    if img.ndim == 3:  # HWC
        img = np.transpose(img, (2, 0, 1))  # CHW
    elif img.ndim == 4:  # BHWC
        img = np.transpose(img, (0, 3, 1, 2))  # BCHW
    else:
        assert False
    return torch.from_numpy(img).contiguous().to(torch.float32)


def get_scale_alpha(image: torch.tensor, mask: torch.tensor, src_percentile: float, dst_value: float):
    assert image.ndim == mask.ndim == 3, "Only supports image: [C, H, W]"
    if mask.sum() < 1.0:
        return torch.tensor([1.0]).to(image.device, torch.float32)
    vis = image.detach()
    src_value = vis.mean(dim=0)  # HW
    mask = mask.min(dim=0)[0]  # HW
    src_value = src_value[mask > 0.999].quantile(src_percentile)
    alpha = 1.0 / src_value.clamp(min=1e-5) * dst_value
    return alpha


def plot_images(_images, titles=None, figsize_base=4, columns=3, cmap="gray",
                font_size=10, axis_label_size=8, HW_ratio=1.0, show=True):
    num_images = len(_images)
    rows = (num_images + columns - 1) // columns
    figsize = (figsize_base * columns, int(figsize_base * rows * HW_ratio))
    fig, axs = plt.subplots(rows, columns, figsize=figsize)
    axs = np.array(axs).ravel()  # Make sure axs is always a 1D array
    for i, img in enumerate(_images):
        if torch.is_tensor(img):
            img = img.cpu()
            img = img.numpy().transpose(1, 2, 0)
        if img.dtype == np.float32:
            img = img.clip(min=0.0, max=1.0)
        elif img.dtype == np.uint8:
            img = img.clip(min=0, max=255)
        axs[i].imshow(img, cmap=cmap)
        if titles is not None:
            axs[i].set_title(titles[i], fontsize=font_size)
        axs[i].tick_params(axis='both', which='major', labelsize=axis_label_size)
    for i in range(len(_images), rows * columns):
        axs[i].axis('off')
    plt.tight_layout()
    if show:
        plt.show()
    return plt


def save_image(img: torch.tensor or np.array or list, path: str, **kwargs):
    if torch.is_tensor(img):
        assert img.ndim in [2, 3, 4], f"img should be 2D, 3D or 4D tensor, but got {img.ndim}."
        torchvision.utils.save_image(img, path)
    elif isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        if img.ndim == 3:
            img = img.permute(2, 0, 1)  # HWC -> CHW
        elif img.ndim == 4:
            img = img.permute(0, 3, 1, 2)  # BHWC -> BCHW
        return save_image(img, path, **kwargs)
    elif isinstance(img, list):
        _img = []
        for x in img:
            if x.ndim == 4 and x.shape[0] == 1:
                x = x.squeeze(0)
            assert x.ndim == 3, f"img should be 3D tensor, but got {x.ndim}."
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            _img.append(x)
        torchvision.utils.save_image(_img, path, **kwargs)
    else:
        assert False, f"img should be torch.Tensor or np.ndarray, but got {type(img)}."


def read_image(path: str, type: str, inf_v=None, check_nan=True):
    """ Read image from path """
    MAX_8bit = 255.0
    MAX_16bit = 65535.0
    # Read image
    assert os.path.exists(path), f"image {path} not exists."
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED|cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
    assert img is not None, f"Load image {path} failed."
    if check_nan:
        assert not np.any(np.isnan(img)), f"image {path} contains NaN values."
    # Convert to float32 [0, 1]
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / MAX_16bit
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / MAX_8bit
    elif img.dtype == np.float32:
        pass
    else:
        assert False, f"Not supported dtype {img.dtype} for image {path}."
    if np.any(np.isinf(img)):
        if inf_v is None:
            warnings.warn(f"image {path} contains Inf values.")
        else:
            img[np.isinf(img)] = inf_v  # set inf to a large value
    # Check image shape and convert to RGB
    assert img.ndim < 4, f"Image should be 2D or 3D, but got {img.ndim}."
    if img.ndim == 3:
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        assert img.shape[-1] == 3 or img.shape[-1] == 1, \
            f"image {path} should be gray or RGB, but got shape {img.shape}."
    elif img.ndim == 2:
        img = img[:, :, np.newaxis]  # HW -> HWC
    else:
        assert False, f"image {path} has wrong dimension {img.ndim}."
    # Convert to specified type
    if type == "numpy":
        pass
    elif type == "tensor":
        img = torch.from_numpy(img.copy()).to(torch.float32).permute(2, 0, 1)  # HWC -> CHW
    else:
        raise NotImplementedError(f"Type {type} is not implemented.")
    return img