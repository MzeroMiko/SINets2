# modified from https://github.com/lartpang/PySODMetrics

import os, argparse, time
import cv2
from tqdm import tqdm
from Src.py_sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure


def metric(pred_root, mask_root):
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    _MAE = MAE()

    mask_name_list = sorted(filter(lambda x: x.find('.png')!=-1, os.listdir(mask_root)))
    # for mask_name in tqdm(mask_name_list):
    for mask_name in mask_name_list:
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        _MAE.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = _MAE.get_results()["mae"]

    results = {
        "Smeasure": sm.round(3),
        "wFmeasure": wfm.round(3),
        "MAE": mae.round(3),
        "adpEm": em["adp"].round(3),
        "meanEm": em["curve"].mean().round(3),
        "maxEm": em["curve"].max().round(3),
        "adpFm": fm["adp"].round(3),
        "meanFm": fm["curve"].mean().round(3),
        "maxFm": fm["curve"].max().round(3),
    }
    
    gettime = lambda :time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'[INFO] => [{gettime()}] => [METRIC DONE: {pred_root}]')
    print(results)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_root', type=str, default='./Dataset/TestDataset/COD10K_all_cam/GT')
    parser.add_argument('--pred_root', type=str, default='./Result/TransSINet_COD10K_default/COD10K_all_cam')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = get_args()
    metric(opt.pred_root, opt.mask_root)
    
