import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

from catalyst.dl.utils import load_checkpoint
import segmentation_models_pytorch as smp

from models import CustomNet
from utils import predict_batch
from utils.utils import mask2rle, post_process
from utils.config import load_config
from datasets import make_loader
from transforms import get_transforms


def run_cls(config_file_cls):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ------------------------------------------------------------------------------------------------------------
    # 1. classification inference
    # ------------------------------------------------------------------------------------------------------------
    config = load_config(config_file_cls)

    model = CustomNet(config.model.encoder, config.data.num_classes)

    testloader = make_loader(
        data_folder=config.data.test_dir,
        df_path=config.data.sample_submission_path,
        phase='test',
        batch_size=config.test.batch_size,
        num_workers=config.num_workers,
        transforms=get_transforms(config.transforms.test),
        num_classes=config.data.num_classes,
    )

    model.to(config.device)
    model.eval()

    checkpoint = load_checkpoint(f"{config.work_dir}/checkpoints/best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    all_fnames = []
    all_predictions = []
    with torch.no_grad():
        for i, (batch_fnames, batch_images) in enumerate(tqdm(testloader)):
            batch_images = batch_images.to(config.device)
            batch_preds = predict_batch(model, batch_images, tta=config.test.tta, task='cls')

            all_fnames.extend(batch_fnames)
            all_predictions.append(batch_preds)

    all_predictions = np.concatenate(all_predictions)

    np.save('all_preds', all_predictions)
    df = pd.DataFrame(data=all_predictions, index=all_fnames)

    df.to_csv('cls_preds.csv', index=False)
    df.to_csv(f"{config.work_dir}/cls_preds.csv", index=False)


def run_seg(config_file_seg):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ------------------------------------------------------------------------------------------------------------
    # 2. segmentation inference
    # ------------------------------------------------------------------------------------------------------------
    config = load_config(config_file_seg)

    model = getattr(smp, config.model.arch)(
        encoder_name=config.model.encoder,
        encoder_weights=config.model.pretrained,
        classes=config.data.num_classes,
        activation=None,
    )

    if os.path.exists('cls_preds.csv'):
        testloader = make_loader(
            data_folder=config.data.test_dir,
            df_path='cls_preds.csv',
            phase='filtered_test',
            batch_size=config.test.batch_size,
            num_workers=config.num_workers,
            transforms=get_transforms(config.transforms.test)
        )
    else:
        testloader = make_loader(
            data_folder=config.data.test_dir,
            df_path=config.data.sample_submission_path,
            phase='test',
            batch_size=config.test.batch_size,
            num_workers=config.num_workers,
            transforms=get_transforms(config.transforms.test)
        )

    model.to(config.device)
    model.eval()

    checkpoint = load_checkpoint(f"{config.work_dir}/checkpoints/best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    if os.path.exists(config.work_dir + '/threshold_search.json'):
        with open(config.work_dir + '/threshold_search.json') as json_file:
            data = json.load(json_file)
        df = pd.DataFrame(data)
        min_sizes = list(df.T.idxmax().values.astype(int))
        print('load best threshold from validation:', min_sizes)
    else:
        min_sizes = config.test.min_size
        print('load default threshold:', min_sizes)

    predictions = []
    with torch.no_grad():
        for i, (batch_fnames, batch_images) in enumerate(tqdm(testloader)):
            batch_images = batch_images.to(config.device)
            batch_preds = predict_batch(model, batch_images, tta=config.test.tta)

            for fname, preds in zip(batch_fnames, batch_preds):
                if config.data.num_classes == 4:
                    for cls in range(preds.shape[0]):
                        mask = preds[cls, :, :]
                        mask, num = post_process(mask, config.test.best_threshold, min_sizes[cls])
                        rle = mask2rle(mask)
                        name = fname + f"_{cls + 1}"
                        predictions.append([name, rle])
                else:  # == 5
                    for cls in range(1, 5):
                        mask = preds[cls, :, :]
                        mask, num = post_process(mask, config.test.best_threshold, min_sizes[cls])
                        rle = mask2rle(mask)
                        name = fname + f"_{cls}"
                        predictions.append([name, rle])

    # ------------------------------------------------------------------------------------------------------------
    # submission
    # ------------------------------------------------------------------------------------------------------------
    df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
    df.to_csv(config.work_dir + "/submission.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Severstal')
    parser.add_argument('--cls_config', dest='config_file_cls',
                        help='configuration file path',
                        default=None, type=str)
    parser.add_argument('--seg_config', dest='config_file_seg',
                        help='configuration file path',
                        default=None, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config_file_cls != None:
        print('classification inference Severstal Steel Defect Detection.')
        run_cls(args.config_file_cls)
    if args.config_file_seg != None:
        print('segmentation inference Severstal Steel Defect Detection.')
        run_seg(args.config_file_seg)


if __name__ == '__main__':
    main()
