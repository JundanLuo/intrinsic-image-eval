# intrinsic-image-eval
# https://github.com/JundanLuo/intrinsic-image-eval
# Evaluation tools for intrinsic image decomposition.


import argparse
import os
import re

import torch
from torchvision.transforms import functional as F


from utils.prediction_loader import *
from utils import util, metrics_intrinsic_images

# Invalid ARAP samples excluded in the IntrinsicDiffusion paper's evaluation.
exclude_list = ["amsterdam", "redhead", "butterfly", "office", "caterpillar", "strawberries", "tiger"]


def get_meter(name: str, compute_siMSE=True, compute_siLMSE=True, compute_DSSIM=True):
    return metrics_intrinsic_images. \
        SI_IntrinsicImageMetricsMeter(name, compute_siMSE, compute_siLMSE, compute_DSSIM, "val")


def compute_dense_errors(dataset_name, file_list_path, data_dir, loader: PredictionLoader, out_dir,
                         log_interval=5, log_meter="reflectance"):
    assert dataset_name in ["ARAP"], f"Undefined dataset: {dataset_name}"

    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Create metrics meter
    meters = {
        "reconstr": get_meter("reconstruction", True, False, False),
        "reflectance": get_meter("reflectance", True, True, True),
        "shading": get_meter("shading", True, True, True),
    }

    # Load data list
    with open(file_list_path) as f:
        data_list = f.readlines()
    data_list = [f.strip() for f in data_list]
    data_list.sort()
    print(f"Total {len(data_list)} images in {file_list_path}")

    # Exclude invalid images
    _data_list = data_list.copy()
    data_list = []
    for img_name in _data_list:
        if any([s in img_name for s in exclude_list]):
            print(f"Exclude {img_name}")
        else:
            data_list.append(img_name)
    data_list.sort()
    print(f"Total {len(data_list)} images after excluding invalid images in {file_list_path}")

    # Evaluate
    for i in range(len(data_list)):
        img_name = os.path.basename(data_list[i]).split('.')[0]

        # Load input image
        img_rgb_path = os.path.join(data_dir, "input", f"{img_name}.png")
        gt_r_path = os.path.join(data_dir, "reflectance", f"{img_name}_albedo.png")
        if "light" in gt_r_path:
            gt_r_path = re.sub(r'light\d+_', '', gt_r_path)
        gt_s_path = os.path.join(data_dir, "shading", f"{img_name}_shading.png")
        mask_path = os.path.join(data_dir, "mask", f"{img_name}_alpha.png")
        if "light" in mask_path:
            mask_path = re.sub(r'light\d+_', '', mask_path)
        img_rgb = util.read_image(img_rgb_path, "tensor")
        gt_r = util.read_image(gt_r_path, "tensor")
        gt_s = util.read_image(gt_s_path, "tensor")
        mask = util.read_image(mask_path, "tensor")
        mask = (mask > 0.5).to(torch.float32)
        assert img_rgb.shape == gt_r.shape == gt_s.shape == mask.shape, \
            f"img_rgb.shape: {img_rgb.shape}, gt_r.shape: {gt_r.shape}, gt_s.shape: {gt_s.shape}, " \
            f"mask.shape: {mask.shape}, but must be the same."

        # Load estimated intrinsic images and reconstruct the input
        pred_space = "linear-rgb"
        pred_r, pred_s = loader.get_ARAP_pred_rs(img_name, pred_space)
        assert pred_r.ndim == 3 and pred_s.ndim == 3, \
            f"pred_r.ndim: {pred_r.ndim}, pred_s.ndim: {pred_s.ndim}, but must be [C, H, W]"
        assert pred_r.shape == pred_s.shape, \
            f"pred_r.shape: {pred_r.shape}, pred_s.shape: {pred_s.shape}, but must be the same."
        pred_r = util.numpy_to_tensor(pred_r)  # [C, H, W]
        pred_s = util.numpy_to_tensor(pred_s)  # [C, H, W]
        t_h, t_w = pred_r.shape[1], pred_r.shape[2]
        assert max(t_h, t_w) > 1000, f"Evaluate 1K images, but shape of {img_name} prediction {pred_r.shape} is too small."

        # resize
        img_rgb, gt_r, gt_s, mask = \
            (F.resize(x, size=(t_h, t_w), interpolation=F.InterpolationMode.BILINEAR, antialias=True).unsqueeze(0)
             for x in [img_rgb, gt_r, gt_s, mask])  # [1, C, H, W]
        mask = (mask > 0.99).to(torch.float32)
        pred_r = pred_r.unsqueeze(0)  # [1, C, H, W]
        pred_s = pred_s.unsqueeze(0)  # [1, C, H, W]

        # Compute metrics
        mask_reconstr = mask * (img_rgb.max(dim=1, keepdim=True)[0] < 0.99).to(torch.float32)  # exclude saturated pixels
        mask_gt_r = mask
        mask_gt_s = mask * (gt_s.max(dim=1, keepdim=True)[0] < 0.99).to(torch.float32)
        reconstructed_rgb = (pred_r * pred_s).clamp(min=0)  # [1, C, H, W]
        meters["reconstr"].update(reconstructed_rgb, img_rgb, mask_reconstr)
        meters["reflectance"].update(pred_r, gt_r, mask_gt_r)
        meters["shading"].update(pred_s, gt_s, mask_gt_s)

        # Visualize
        if (log_interval > 0) and (i % log_interval == 0):
            target_list = [img_rgb[0], gt_r[0], gt_s[0]]
            pred_list = [reconstructed_rgb[0], pred_r[0], pred_s[0]]
            mask_list = [mask_reconstr[0], mask_gt_r[0], mask_gt_s[0]]
            vis_list = []
            for g, p, m in zip(target_list, pred_list, mask_list):
                assert g.shape == p.shape == m.shape, \
                    f"g.shape: {g.shape}, p.shape: {p.shape}, m.shape: {m.shape}, but must be the same."
                alpha = metrics_intrinsic_images.scale_matching(p[None], g[None], m[None]).item()
                vis_img = p.mul(alpha).clamp(min=0, max=1)  # [C, H, W]
                vis_list.append(vis_img)
            vis_all = target_list + vis_list + mask_list
            plt = util.plot_images(vis_all,
                                   titles=["Input", "gt R", "gt S",
                                           f"Reconstr pred",
                                           f"Pred reflectance ({pred_space})", f"Pred shading ({pred_space})",
                                           "Mask reconstr", "Mask gt R", "Mask gt S"],
                                   columns=3, figsize_base=6, HW_ratio=(vis_all[0].shape[1] / vis_all[0].shape[2]),
                                   show=False)
            plt.savefig(os.path.join(out_dir, f"{img_name}_rs.jpg"))
            plt.close()
            keys = ["reconstr", "r", "s"]
            for k, img in zip(keys, vis_list):
                util.save_image(img, os.path.join(out_dir, f"{img_name}_{k}.jpg"))

            # Print
            rs = meters[log_meter].get_results()
            print(f"Evaluate {log_meter} on {dataset_name}: {i+1}/{len(data_list)}, "
                  f"si_MSE: {rs.si_MSE: .6f}, "
                  f"si_LMSE: {rs.si_LMSE: .6f}, "
                  f"DSSIM: {rs.DSSIM: .6f}")

    # Result
    print(f"Test data list: {file_list_path}")
    with open(os.path.join(out_dir, f"{dataset_name}_dense_errors.txt"), 'w') as f:
        for k, m in meters.items():
            rs = m.get_results()
            print(f"{k}: \n"
                  f"si_MSE: {rs.si_MSE: .6f}, "
                  f"si_LMSE: {rs.si_LMSE: .6f}, "
                  f"DSSIM: {rs.DSSIM: .6f}\n")
            f.write(f"{k}: \n"
                    f"si_MSE: {rs.si_MSE: .6f}, "
                    f"si_LMSE: {rs.si_LMSE: .6f}, "
                    f"DSSIM: {rs.DSSIM: .6f}\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="ARAP",
        choices=["ARAP"],
        help="Dataset to be evaluated",
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        default="./data/ARAP",
        help="Path to test data",
        type=str,
    )
    parser.add_argument(
        "--outdir",
        default="./out/compute_dense_errors",
        metavar="FILE",
        help="Path to save visualized reconstructed images",
        type=str,
    )
    parser.add_argument(
        "--method",
        default=None,
        required=True,
        metavar="FILE",
        help="Method to be evaluated",
        type=str,
    )
    parser.add_argument(
        "--log_interval",
        default=20,
        metavar="N",
        help="Print log every log_interval images if log_interval > 0",
        type=int,
    )
    parser.add_argument(
        "--log_meter",
        default="reflectance",
        choices=["reconstr", "reflectance", "shading", "shading_hdr"],
        help="Meter to be print during evaluation",
        type=str,
    )
    parser.add_argument(
        "--label",
        default=None,
        metavar="FILE",
        help="Label of the method to be evaluated",
        type=str,
    )

    args = parser.parse_args()
    assert args.dataset in ["ARAP"], f"Undefined dataset: {args.dataset}"
    args.file = {
        "ARAP": "./benchmark/ARAP/ARAP_data_list.txt",
    }[args.dataset]
    print(f"\nTest on the {args.dataset} dataset:\n"
          f"\t test list file path: {args.file}"
          f"\t dataset dir: {args.data_dir}")
    for p in [args.file, args.data_dir]:
        if not os.path.exists(p):
            print(f"Not exists: {p}")
            exit(0)

    loader_dicts = {
        # "PIE-Net_2022": Das_2022_PIE_Net_Loader("./previous_works/Das_2022_PIE-Net"),
        # "OrdinalShading_2023": Careaga_2023_OrdinalShading_Loader("./previous_works/Careaga_2023_OrdinalShading"),
        # "Zhu_2022": Zhu_2022_InverseMonteCarlo_Loader(
        #     "previous_works/Zhu_2022_Learning-based Inverse Rendering of Complex Indoor Scenes with Differentiable Monte Carlo Raytracing"),
        "Luo_2023_CRefNet": Luo_2023_CRefNet_Loader(
            "./previous_works/Luo_2023_CRefNet/final_real"),
        "Luo_2024_IntrinsicDiffusion": Luo_2024_IntrinsicDiffusion_Loader("previous_works/Luo_2024_IntrinsicDiffusion/")
    }
    assert args.method in loader_dicts.keys(), \
        f"Undefined method: {args.method}"

    out_dir = os.path.join(args.outdir, args.method)
    if args.label is not None:
        out_dir = os.path.join(out_dir, args.label)
    print(f"\nEvaluate {args.method} at {loader_dicts[args.method].dir}. Visualize images to {out_dir} ......")
    compute_dense_errors(args.dataset, args.file, args.data_dir, loader_dicts[args.method], out_dir,
                         log_interval=args.log_interval, log_meter=args.log_meter)
