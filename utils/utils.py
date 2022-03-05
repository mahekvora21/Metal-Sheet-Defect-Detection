import json
import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
from catalyst.utils import load_checkpoint
from catalyst.utils import set_global_seed, prepare_cudnn

from models import CustomNet
from .config import load_config


def dict_to_json(dict_obj, file_name):
    with open(file_name, 'w') as fp:
        json.dump(dict_obj, fp)


def seed_all(SEED):
    set_global_seed(SEED)
    prepare_cudnn(deterministic=True)


def prepare_train_directories(config):
    out_dir = config.train.dir
    os.makedirs(os.path.join(out_dir, 'checkpoint'), exist_ok=True)


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4)'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32)  # float32 is V.Imp
    # 4:class 1ï½ž4 (ch:0ï½ž3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return fname, masks


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def load_model(config_path):
    config = load_config(config_path)
    print(config.checkpoint_path)

    if config.model.arch == 'Classification':
        model = CustomNet(config.model.encoder, config.data.num_classes, pretrained=False)
    else:
        # create segmentation model with pre-trained encoder
        model = getattr(smp, config.model.arch)(
            encoder_name=config.model.encoder,
            encoder_weights=None,
            classes=config.data.num_classes,
            activation=None,
        )

    model.to(config.device)
    model.eval()

    checkpoint = load_checkpoint(config.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
