import pickle
import json
import argparse
from collections import namedtuple
from skimage.transform import resize
from skimage import io

from utils.average_meter import AverageMeter
from utils.prediction_loader import *
from benchmark.IIW import metrics_iiw


class WHDRAverageMeter(object):
    Result = namedtuple("Result", ["WHDR", "WHDR_eq", "WHDR_ineq"])

    def __init__(self, name: str):
        self.name = name
        self.whdr_meter = AverageMeter("WHDR")
        self.whdr_eq_meter = AverageMeter("WHDR_eq")
        self.whdr_ineq_meter = AverageMeter("WHDR_ineq")

    def update(self, whdr, whdr_eq, whdr_ineq, count, count_eq, count_ineq):
        self.whdr_meter.update(whdr, count)
        self.whdr_eq_meter.update(whdr_eq, count_eq)
        self.whdr_ineq_meter.update(whdr_ineq, count_ineq)

    def get_results(self):
        return self.Result(WHDR=self.whdr_meter.avg, WHDR_eq=self.whdr_eq_meter.avg, WHDR_ineq=self.whdr_ineq_meter.avg)

    def __str__(self):
        return f"WHDR {self.whdr_meter.avg: .6f}, " \
               f"WHDR_eq {self.whdr_eq_meter.avg: .6f}, " \
               f"WHDR_ineq {self.whdr_ineq_meter.avg: .6f}"


exclude_list = [118507, 118508, 118509, 118510, 118511, 118512, 118513, 118514, 118515, 118516, 118517]


def evaluate_predictions(file_list_path, iiw_dir, eq_delta, color_space, loader: PredictionLoader, evaluate_high_res):
    images_list = pickle.load(open(file_list_path, "rb"))

    whdr_meter = WHDRAverageMeter(f"WHDR({color_space})")

    for j in range(0, 3):
        img_list = images_list[j]
        for i in range(len(img_list)):
            id = str(img_list[i].split('/')[-1][0:-7])
            if evaluate_high_res and int(id) in exclude_list:
                continue
            img_path = os.path.join(iiw_dir, "data", f"{id}.png")
            judgement_path = os.path.join(iiw_dir, "data", f"{id}.json")
            judgements = json.load(open(judgement_path))

            img = np.float32(io.imread(img_path)) / 255.0
            o_h, o_w = img.shape[0], img.shape[1]

            pred_r = loader.get_iiw_pred_r(id, color_space, use_high_res=evaluate_high_res)
            # pred_r = resize(pred_r, (o_h, o_w),
            #                 order=1, preserve_range=True, anti_aliasing=True)
            if evaluate_high_res:
                assert max(pred_r.shape[0], pred_r.shape[1]) > 1000, f"pred_r.shape: {pred_r.shape}, not high resolution"

            (whdr, _), (whdr_eq, valid_eq), (whdr_ineq, valid_ineq) =\
                metrics_iiw.compute_whdr(pred_r, judgements, eq_delta)
            print(f"ID {id}: WHDR {whdr: .6f}, WHDR_eq {whdr_eq: .6f}, WHDR_ineq {whdr_ineq: .6f}")
            whdr_meter.update(whdr,    whdr_eq if valid_eq else 0, whdr_ineq if valid_ineq else 0,
                                   1,       1 if valid_eq else 0,       1 if valid_ineq else 0)
            if i % 100 == 0:
                print(f"Evaluate {j}-{i}: \n"
                      f"\t{whdr_meter.name}: {whdr_meter} \n"
                      )

    print(f"\n{whdr_meter.name}: {whdr_meter}")
    return whdr_meter.get_results()


# python compute_iiw_whdr.py --method CRefNet_2023 --t 0.20
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="./benchmark/IIW/iiw_test_img_batch.p",
        metavar="FILE",
        help="Path to test list file",
        type=str,
    )
    parser.add_argument(
        "--iiwdir",
        default="./data/iiw-dataset",
        metavar="FILE",
        help="Path to test list file",
        type=str,
    )
    parser.add_argument(
        "--method",
        default="CRefNet_2023",
        metavar="FILE",
        help="Method to be evaluated",
        type=str,
    )
    parser.add_argument(
        "--t",
        default=0.20,
        metavar="Number",
        help="Equality threshold",
        type=float,
    )
    parser.add_argument(
        "--outdir",
        default="./out/compute_iiw_whdr",
        metavar="FILE",
        help="Path to save log files",
        type=str,
    )
    parser.add_argument(
        "--use_high_res",
        action="store_true",
        default=False,
        help="Whether or not to evaluate on high resolution images",
    )
    parser.add_argument(
        "--color_space",
        default="linear-rgb",
        choices=["srgb", "linear-rgb"],
        help="Color space to be evaluated",
        type=str,
    )

    args = parser.parse_args()
    print(f"\ntest list file path:{args.file} ")
    print(f"IIW image directory: {args.iiwdir}")
    for p in [args.file, args.iiwdir]:
        if not os.path.exists(p):
            print(f"Not exsists: {p}")
            exit(0)

    loader_dicts = {
        # "Li_2018_CGI_full": Li_2018_CGI_Loader("./previous_works/Li_2018_CGIntrinsics/CGI+IIW+SAW/cgi_iiw"),
        # "Li_2018_CGI_cgi": Li_2018_CGI_Loader("./previous_works/Li_2018_CGIntrinsics/CGI/cgi_iiw"),
        "Luo_2020_NIID-Net": Luo_2020_NIID_Net_Loader("./previous_works/Luo_2020_NIID-Net"),
        # "PIE-Net_2022": Das_2022_PIE_Net_Loader("./previous_works/Das_2022_PIE-Net"),
        # "OrdinalShading_2023": Careaga_2023_OrdinalShading_Loader("./previous_works/Careaga_2023_OrdinalShading"),
        # "Luo_2023_CRefNet-E_2023": Luo_2023_CRefNet_Loader(
        #     "./previous_works/Luo_2023_CRefNet/crefnet-e/"),
        "Luo_2023_CRefNet": Luo_2023_CRefNet_Loader(
            "./previous_works/Luo_2023_CRefNet/final_real/"),
        # "Zhu_2022": Zhu_2022_InverseMonteCarlo_Loader("previous_works/Zhu_2022_Learning-based Inverse Rendering of Complex Indoor Scenes with Differentiable Monte Carlo Raytracing"),
        "Luo_2024_IntrinsicDiffusion": Luo_2024_IntrinsicDiffusion_Loader("previous_works/Luo_2024_IntrinsicDiffusion/")
    }
    assert args.method in loader_dicts.keys(), f"Undefined method: {args.method}"

    print(f"\nEvaluate {args.method} (high_res={args.use_high_res}) with threshold {args.t} in "
          f"{args.color_space} color space.")
    whdr_results = evaluate_predictions(args.file, args.iiwdir, args.t, args.color_space, loader_dicts[args.method],
                                        args.use_high_res)

    out_dir = os.path.join(args.outdir, args.method)
    if args.method == "ours":
        p = loader_dicts[args.method].dir
        p = p.split("results_siggraph/")[-1].replace("/", "_")
        out_dir = os.path.join(out_dir, p)
    out_dir = os.path.join(out_dir, f"t_{args.t}_high_res_{args.use_high_res}_color_{args.color_space}")
    os.makedirs(out_dir, exist_ok=True)
    log_file_path = os.path.join(out_dir, "whdr_log.txt")
    print(f"\nSave log file to: {log_file_path}")
    with open(log_file_path, "w") as f:
        f.write(str(args))
        f.write("\n")
        f.write(f"WHDR({args.color_space}) {whdr_results.WHDR: .3f}\n"
                f"WHDR_eq {whdr_results.WHDR_eq: .3f}\n"
                f"WHDR_ineq {whdr_results.WHDR_ineq: .3f}\n")

