# intrinsic-image-eval
# https://github.com/JundanLuo/intrinsic-image-eval
# Evaluation tools for intrinsic image decomposition.


from abc import ABC, abstractmethod
import os
import h5py
import numpy as np

from utils import util


class PredictionLoader(ABC):
    raw_dir = None
    image_dir = None
    img_postfix = None
    # use_iiw_high_res = False

    @abstractmethod
    def get_iiw_pred_r(self, id, space, use_high_res):
        '''Please implement in subclass'''
        raise NotImplemented

    @abstractmethod
    def get_iiw_pred_s(self, id, space, use_high_res):
        '''Please implement in subclass'''
        raise NotImplemented

    @abstractmethod
    def get_pred_rs_img_path(self, id, use_high_res):
        '''Please implement in subclass'''
        raise NotImplemented

    def get_ARAP_pred_rs(self, filename, space):
        '''Please implement in subclass'''
        raise NotImplemented

    def set_img_dir(self, img_dir, img_postfix):
        assert img_postfix in ["png", "jpg", "jpeg"], f"Unknown image postfix {img_postfix}"
        self.image_dir = img_dir
        self.img_postfix = img_postfix

    # def set_use_iiw_high_res(self, use_high_res):
    #     self.use_iiw_high_res = use_high_res


class Li_2018_CGI_Loader(PredictionLoader):
    DEFAULT_GAMMA = 1.0 / 2.2  # the default gamma value used in Li et al. 2018

    def __init__(self, dir):
        self.dir = dir
        self.raw_dir = os.path.join(self.dir, "release_iiw")
        self.image_dir = os.path.join(self.dir, "release_iiw_images")
        self.img_postfix = "png"

    def get_iiw_pred_r(self, id, space, use_high_res):
        assert not use_high_res, "not implemented"
        assert space in ["srgb", "linear-rgb"]
        pred_path = os.path.join(self.raw_dir, f"{id}.png.h5")
        hdf5_file_read = h5py.File(pred_path, 'r')
        pred_R = hdf5_file_read.get('/prediction/R')
        pred_R = np.array(pred_R).astype(np.float32)
        hdf5_file_read.close()
        if space == "srgb":
            pass
        elif space == "linear-rgb":
            pred_R = util.srgb_to_rgb(pred_R, gamma=self.DEFAULT_GAMMA)
        else:
            raise ValueError(f"Unknown color space {space}")
        return pred_R

    def get_iiw_pred_s(self, id, space, use_high_res):
        assert not use_high_res, "not implemented"
        assert space in ["srgb", "linear-rgb"]
        pred_path = os.path.join(self.raw_dir, f"{id}.png.h5")
        hdf5_file_read = h5py.File(pred_path, 'r')
        pred_S = hdf5_file_read.get('/prediction/S')
        pred_S = np.array(pred_S)[:, :, None].astype(np.float32)  # [H, W, C=1]
        hdf5_file_read.close()
        if space == "srgb":
            pass
        elif space == "linear-rgb":
            pred_S = util.srgb_to_rgb(pred_S, gamma=self.DEFAULT_GAMMA)
        else:
            raise ValueError(f"Unknown color space {space}")
        return pred_S

    def get_pred_rs_img_path(self, id, use_high_res):
        assert not use_high_res, "not implemented"
        r_img_path = os.path.join(self.image_dir, f"{id}-r.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}-s.{self.img_postfix}")
        return r_img_path, s_img_path


class Luo_2020_NIID_Net_Loader(PredictionLoader):
    DEFAULT_GAMMA = 1.0 / 2.2  # the default gamma value used in Luo et al. 2020

    def __init__(self, dir):
        self.dir = dir
        self.raw_dir = os.path.join(self.dir, "IIW_test_low_resolution/raw")
        self.image_dir = os.path.join(self.dir, "IIW_test_low_resolution")
        self.img_postfix = "png"

    def get_iiw_pred_r(self, id, space, use_high_res):
        assert not use_high_res, "not implemented"
        assert space in ["srgb", "linear-rgb"]
        pred_R_path = os.path.join(self.raw_dir, f"{id}_pred_R.npy")
        if not os.path.exists(pred_R_path):
            print(f"Warning: {pred_R_path} not exists.")
            return None
        pred_R = np.load(pred_R_path).astype(np.float32)
        if space == "srgb":
            pred_R = util.rgb_to_srgb(pred_R, gamma=self.DEFAULT_GAMMA)
        elif space == "linear-rgb":
            pass
        else:
            raise ValueError(f"Unknown color space {space}")
        return pred_R

    def get_iiw_pred_s(self, id, space, use_high_res):
        assert not use_high_res, "not implemented"
        assert space in ["srgb", "linear-rgb"]
        pred_S_path = os.path.join(self.raw_dir, f"{id}_pred_S.npy")
        pred_S = np.load(pred_S_path).astype(np.float32)
        if space == "srgb":
            pred_S = util.rgb_to_srgb(pred_S, gamma=self.DEFAULT_GAMMA)
        elif space == "linear-rgb":
            pass
        else:
            raise ValueError(f"Unknown color space {space}")
        return pred_S

    def get_pred_rs_img_path(self, id, use_high_res):
        assert not use_high_res, "not implemented"
        r_img_path = os.path.join(self.image_dir, f"{id}_R.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}_S.{self.img_postfix}")
        return r_img_path, s_img_path


class Luo_2023_CRefNet_Loader(PredictionLoader):
    DEFAULT_GAMMA = 1.0 / 2.2  # the default gamma value used in CRefNet paper

    def __init__(self, dir):
        self.dir = dir
        self.arap_dir = os.path.join(self.dir, "ARAP")
        self.iiw_dir = os.path.join(self.dir, "IIW")
        self.iiw_high_res_dir = os.path.join(self.iiw_dir, "high-resolution")
        self.iiw_low_res_dir = os.path.join(self.iiw_dir, "low-resolution")
        self.image_dir = os.path.join(self.iiw_low_res_dir, "images")
        self.img_postfix = "jpg"
        # self.raw_dir = os.path.join(self.dir, "raw_pred_IIW")

    def get_iiw_pred_r(self, id, space, use_high_res):
        assert space in ["srgb", "linear-rgb"]
        iiw_dir = self.iiw_high_res_dir if use_high_res else self.iiw_low_res_dir
        iiw_dir = os.path.join(iiw_dir, "raw")
        pred_r_path = os.path.join(iiw_dir, f"{id}_r.npy")
        pred_r = np.load(pred_r_path).astype(np.float32)
        if space == "srgb":
            pred_r = util.rgb_to_srgb(pred_r, gamma=self.DEFAULT_GAMMA)
        elif space == "linear-rgb":
            pass
        else:
            raise ValueError(f"Unknown color space {space}")
        return pred_r

    def get_iiw_pred_s(self, id, space, use_high_res):
        assert not use_high_res, "not implemented"
        assert space in ["srgb", "linear-rgb"]
        iiw_dir = self.iiw_high_res_dir if use_high_res else self.iiw_low_res_dir
        iiw_dir = os.path.join(iiw_dir, "raw")
        pred_s_path = os.path.join(iiw_dir, f"{id}_s.npy")
        pred_s = np.load(pred_s_path).astype(np.float32)
        if space == "srgb":
            pred_s = util.rgb_to_srgb(pred_s, gamma=self.DEFAULT_GAMMA)
        elif space == "linear-rgb":
            pass
        else:
            raise ValueError(f"Unknown color space {space}")
        return pred_s

    def get_pred_rs_img_path(self, id, use_high_res):
        assert not use_high_res, "not implemented"
        iiw_dir = self.iiw_high_res_dir if use_high_res else self.iiw_low_res_dir
        r_img_path = os.path.join(self.image_dir, f"{id}_r.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}_s.{self.img_postfix}")
        return r_img_path, s_img_path

    def get_ARAP_pred_rs(self, filename, space):
        assert space in ["linear-rgb"]
        pred_r_path = os.path.join(self.arap_dir, f"{filename}_r.npy")
        pred_s_path = os.path.join(self.arap_dir, f"{filename}_s.npy")
        pred_r = np.load(pred_r_path).astype(np.float32).clip(min=0)
        pred_s = np.load(pred_s_path).astype(np.float32).clip(min=0)
        return pred_r, pred_s


class Wang_2019_Discriminative_Loader(PredictionLoader):
    def __init__(self, dir):
        self.dir = dir
        self.image_dir = os.path.join(self.dir, "test-imgs_ep12_results")
        self.img_postfix = "png"

    def get_iiw_pred_r(self, id, space, use_high_res):
        '''Please implement in subclass'''
        raise NotImplemented

    def get_iiw_pred_s(self, id, space, use_high_res):
        assert False

    def get_pred_rs_img_path(self, id, use_high_res):
        assert not use_high_res, "not implemented"
        r_img_path = os.path.join(self.image_dir, f"{id}_r.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}_sr.{self.img_postfix}")
        return r_img_path, s_img_path


class Bi_2015_L1smoothing_Loader(PredictionLoader):
    def __init__(self, dir):
        self.dir = dir
        self.image_dir = os.path.join(self.dir, "our_result")
        self.img_postfix = "png"

    def get_iiw_pred_r(self, id, space, use_high_res):
        '''Please implement in subclass'''
        raise NotImplemented

    def get_iiw_pred_s(self, id, space, use_high_res):
        assert False

    def get_pred_rs_img_path(self, id, use_high_res):
        assert not use_high_res, "not implemented"
        r_img_path = os.path.join(self.image_dir, f"{id}-r.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}-s.{self.img_postfix}")
        return r_img_path, s_img_path


class Das_2022_PIE_Net_Loader(PredictionLoader):
    DEFAULT_GAMMA = 1.0 / 2.2

    def __init__(self, dir):
        self.dir = dir
        self.iiw_dir = os.path.join(self.dir, "IIW_test_data")
        self.iiw_low_res_dir = os.path.join(self.iiw_dir, "low_resolution")
        self.iiw_high_res_dir = os.path.join(self.iiw_dir, "high_resolution")
        self.raw_dir = self.dir
        self.image_dir = self.iiw_low_res_dir
        self.arap_dir = os.path.join(self.dir, "ARAP")
        self.img_postfix = "png"

    def get_iiw_pred_r(self, id, space, use_high_res):
        assert space in ["srgb", "linear-rgb"]
        iiw_dir = self.iiw_high_res_dir if use_high_res else self.iiw_low_res_dir
        pred_R_path = os.path.join(iiw_dir, f"{id}_pred_alb.npy")
        pred_R = np.load(pred_R_path).astype(np.float32)
        if space == "srgb":
            assert False, "Das et al. do not use gamma correction."
            pred_R = util.rgb_to_srgb(pred_R, gamma=self.DEFAULT_GAMMA)
        elif space == "linear-rgb":
            pass
        else:
            raise ValueError(f"Unknown color space {space}")
        return pred_R

    def get_iiw_pred_s(self, id, space, use_high_res):
        assert not use_high_res, "not implemented"
        assert space in ["srgb", "linear-rgb"]
        pred_S_path = os.path.join(self.iiw_dir, f"{id}_pred_shd.npy")
        pred_S = np.load(pred_S_path).astype(np.float32)[:, :, np.newaxis]
        if space == "srgb":
            assert False, "Das et al. do not use gamma correction."
            pred_S = util.rgb_to_srgb(pred_S, gamma=self.DEFAULT_GAMMA)
        elif space == "linear-rgb":
            pass
        else:
            raise ValueError(f"Unknown color space {space}")
        return pred_S

    def get_pred_rs_img_path(self, id, use_high_res):
        assert not use_high_res, "not implemented"
        r_img_path = os.path.join(self.image_dir, f"{id}_pred_alb.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}_pred_shd.{self.img_postfix}")
        return r_img_path, s_img_path

    def get_ARAP_pred_rs(self, filename, space):
        assert space in ["linear-rgb"]
        pred_r_path = os.path.join(self.arap_dir, f"{filename}_pred_alb.npy")
        pred_s_path = os.path.join(self.arap_dir, f"{filename}_pred_shd.npy")
        pred_r = np.load(pred_r_path).astype(np.float32).clip(min=0)
        pred_s = np.load(pred_s_path).astype(np.float32).clip(min=0)
        pred_s = pred_s[:, :, np.newaxis]
        pred_s = np.repeat(pred_s, 3, axis=2)
        return pred_r, pred_s


class Careaga_2023_OrdinalShading_Loader(PredictionLoader):
    DEFAULT_GAMMA = 1.0 / 2.2

    def __init__(self, dir):
        self.dir = dir
        self.arap_dir = os.path.join(self.dir, "ARAP")
        self.iiw_low_res_dir = os.path.join(self.dir, "IIW_test_low_resolution")
        self.iiw_high_res_dir = os.path.join(self.dir, "IIW/high_resolution")
        # self.raw_dir = self.dir
        # self.image_dir = self.iiw_high_res_dir if self.use_iiw_high_res else self.iiw_low_res_dir
        self.image_dir = self.iiw_low_res_dir
        self.img_postfix = "png"

    def get_iiw_pred_r(self, id, space, use_high_res):
        assert space in ["srgb", "linear-rgb"]
        iiw_dir = self.iiw_high_res_dir if use_high_res else self.iiw_low_res_dir
        pred_R_path = os.path.join(iiw_dir, f"{id}_r.npy")
        pred_R = np.load(pred_R_path).astype(np.float32)
        if space == "srgb":
            pred_R = util.rgb_to_srgb(pred_R, gamma=self.DEFAULT_GAMMA)
        elif space == "linear-rgb":
            pass
        else:
            raise ValueError(f"Unknown color space {space}")
        return pred_R

    def get_iiw_pred_s(self, id, space, use_high_res):
        assert not use_high_res, "not implemented"
        assert space in ["srgb", "linear-rgb"]
        assert False, "not implemented"
        pred_S_path = os.path.join(self.raw_dir, f"{id}_pred_shd.npy")
        pred_S = np.load(pred_S_path).astype(np.float32)[:, :, np.newaxis]
        if space == "srgb":
            pred_S = util.rgb_to_srgb(pred_S, gamma=self.DEFAULT_GAMMA)
        elif space == "linear-rgb":
            pass
        else:
            raise ValueError(f"Unknown color space {space}")
        return pred_S

    def get_pred_rs_img_path(self, id, use_high_res):
        assert not use_high_res, "not implemented"
        r_img_path = os.path.join(self.image_dir, f"{id}_r.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}_s.{self.img_postfix}")
        return r_img_path, s_img_path

    def get_ARAP_pred_rs(self, filename, space):
        assert space in ["linear-rgb"]
        pred_r_path = os.path.join(self.arap_dir, f"{filename}_r.npy")
        pred_s_path = os.path.join(self.arap_dir, f"{filename}_s.npy")
        pred_r = np.load(pred_r_path).astype(np.float32).clip(min=0)
        pred_s = np.load(pred_s_path).astype(np.float32).clip(min=0)
        pred_s = pred_s[:, :, np.newaxis]
        pred_s = np.repeat(pred_s, 3, axis=2)
        return pred_r, pred_s


class Zhu_2022_InverseMonteCarlo_Loader(PredictionLoader):
    DEFAULT_GAMMA = 1.0 / 2.2

    def __init__(self, dir):
        self.dir = dir
        self.arap_dir = os.path.join(self.dir, "ARAP")
        self.iiw_low_res_dir = os.path.join(self.dir, "IIW_test_low_resolution")
        self.iiw_high_res_dir = os.path.join(self.dir, "IIW/high_resolution")
        # self.raw_dir = self.dir
        # self.image_dir = self.iiw_high_res_dir if self.use_iiw_high_res else self.iiw_low_res_dir
        self.img_postfix = "png"

    def get_iiw_pred_r(self, id, space, use_high_res):
        assert space in ["srgb", "linear-rgb"]
        iiw_dir = self.iiw_high_res_dir if use_high_res else self.iiw_low_res_dir
        pred_R_path = os.path.join(iiw_dir, f"{id}_r.npy")
        pred_R = np.load(pred_R_path).astype(np.float32)
        if space == "srgb":
            pred_R = util.rgb_to_srgb(pred_R, gamma=self.DEFAULT_GAMMA)
        elif space == "linear-rgb":
            pass
        else:
            raise ValueError(f"Unknown color space {space}")
        return pred_R

    def get_iiw_pred_s(self, id, space, use_high_res):
        assert False, f"Zhu's work does not predict shading"

    def get_pred_rs_img_path(self, id, use_high_res):
        assert not use_high_res, "not implemented"
        assert False, "not implemented"
        r_img_path = os.path.join(self.image_dir, f"{id}_r.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}_s.{self.img_postfix}")
        return r_img_path, s_img_path

    def get_ARAP_pred_rs(self, filename, space):
        assert space in ["linear-rgb"]
        pred_r_path = os.path.join(self.arap_dir, f"{filename}_r.npy")
        # pred_s_path = os.path.join(self.arap_dir, f"{filename}_s.npy")
        pred_r = np.load(pred_r_path).astype(np.float32).clip(min=0)
        # pred_s = np.load(pred_s_path).astype(np.float32).clip(min=0)
        return pred_r, np.ones_like(pred_r)


class Luo_2024_IntrinsicDiffusion_Loader(PredictionLoader):
    DEFAULT_GAMMA = 1.0 / 2.2

    def __init__(self, dir):
        self.dir = dir
        self.arap_dir = os.path.join(self.dir, "ARAP/evaluate_folder/raw_pred")
        self.iiw_low_res_dir = os.path.join(self.dir, "IIW_test_low_resolution/evaluate_folder")
        self.img_postfix = "png"
        self.image_dir = os.path.join(self.iiw_low_res_dir, "split")

    def get_iiw_pred_r(self, id, space, use_high_res):
        assert not use_high_res, "not implemented"
        assert space in ["srgb", "linear-rgb"]
        raw_pred_dir = os.path.join(self.iiw_low_res_dir, "raw_pred")
        pred_r_path = os.path.join(raw_pred_dir, f"{id}_r.npy")
        pred_r = np.load(pred_r_path).astype(np.float32)
        if space == "srgb":
            pred_r = util.rgb_to_srgb(pred_r, gamma=self.DEFAULT_GAMMA)
        elif space == "linear-rgb":
            pass
        else:
            raise ValueError(f"Unknown color space {space}")
        return pred_r

    def get_iiw_pred_s(self, id, space, use_high_res):
        assert not use_high_res, "not implemented"
        assert space in ["srgb", "linear-rgb"]
        raw_pred_dir = os.path.join(self.iiw_low_res_dir, "raw_pred")
        pred_s_path = os.path.join(raw_pred_dir, f"{id}_s.npy")
        pred_s = np.load(pred_s_path).astype(np.float32)
        if space == "srgb":
            pred_s = util.rgb_to_srgb(pred_s, gamma=self.DEFAULT_GAMMA)
        elif space == "linear-rgb":
            pass
        else:
            raise ValueError(f"Unknown color space {space}")
        return pred_s

    def get_pred_rs_img_path(self, id, use_high_res):
        assert not use_high_res, "not implemented"
        image_dir = self.image_dir
        r_img_path = os.path.join(image_dir, f"{id}_r.{self.img_postfix}")
        s_img_path = os.path.join(image_dir, f"{id}_s.{self.img_postfix}")
        return r_img_path, s_img_path

    def get_ARAP_pred_rs(self, filename, space):
        assert space in ["linear-rgb"]
        pred_r_path = os.path.join(self.arap_dir, f"{filename}_r.npy")
        pred_s_path = os.path.join(self.arap_dir, f"{filename}_s.npy")
        # pred_r = util.read_image(pred_r_path, "numpy")
        # pred_s = util.read_image(pred_s_path, "numpy")
        pred_r = np.load(pred_r_path).astype(np.float32).clip(min=0)
        pred_s = np.load(pred_s_path).astype(np.float32).clip(min=0)
        # pred_s = pred_s[:, :, np.newaxis]
        # pred_s = np.repeat(pred_s, 3, axis=2)
        return pred_r, pred_s


class General_Loader(PredictionLoader):
    def __init__(self, dir, img_postfix="png"):
        assert img_postfix in ["png", "jpeg", "jpg"]
        self.dir = dir
        self.image_dir = self.dir
        self.img_postfix = img_postfix

    def get_iiw_pred_r(self, id, space, use_high_res):
        '''Please implement in subclass'''
        raise NotImplemented

    def get_iiw_pred_s(self, id, space, use_high_res):
        assert False

    def get_pred_rs_img_path(self, id, use_high_res):
        assert not use_high_res, "not implemented"
        r_img_path = os.path.join(self.image_dir, f"{id}-r.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}-s.{self.img_postfix}")
        return r_img_path, s_img_path


class InputLoader(object):
    def __init__(self, dir, img_postfix="png"):
        self.dir = dir
        self.data_dir = os.path.join(self.dir)
        self.img_postfix = img_postfix

    def get_input_img_path(self, id):
        return os.path.join(self.data_dir, f"{id}.{self.img_postfix}")

    def set_img_dir(self, data_dir, img_postfix):
        assert img_postfix in ["png", "jpg", "jpeg"], f"Unknown image postfix {img_postfix}"
        self.data_dir = data_dir
        self.img_postfix = img_postfix
