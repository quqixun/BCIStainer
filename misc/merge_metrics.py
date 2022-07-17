import os
import numpy as np
import pandas as pd


if __name__ == '__main__':

    input_root = '../outputs'
    output_dir = '../assets'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'metrics.csv')

    metrics_list = []
    for exp in os.listdir(input_root):
        exp_dir = os.path.join(input_root, exp)
        for pred in os.listdir(exp_dir):
            pred_dir = os.path.join(exp_dir, pred)
            data_path = os.path.join(pred_dir, 'metrics.csv')
            if not os.path.isfile(data_path):
                continue

            data = pd.read_csv(data_path)

            psnr_avg = data['psnr'].mean()
            psnr_std = data['psnr'].std()
            ssim_avg = data['ssim'].mean()
            ssim_std = data['ssim'].std()

            metrics_list.append([
                exp, pred, data_path,
                psnr_avg, psnr_std, ssim_avg, ssim_std
            ])

    columns = ['exp', 'pred', 'metrics', 'psnr_avg', 'psnr_std', 'ssim_avg', 'ssim_std']
    metrics_df = pd.DataFrame(data=metrics_list, columns=columns)
    metrics_df.sort_values(by=['psnr_avg'],  ascending=False, inplace=True)
    metrics_df.to_csv(output_path, index=False)
