from __future__ import print_function, division

import os
from os.path import join as pj
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import h5py
from tqdm import tqdm

from models import build_image_encoder, preprocess_input


# Funtions
def list_dir_with_full_paths(dir_path):
    dir_abs_path = os.path.abspath(dir_path)
    return sorted([os.path.join(dir_abs_path, file_name) for file_name in os.listdir(dir_abs_path)])


def pad_to_square_image(image):
    width, height = image.size
    size = max(width, height)
    new_image = Image.new('RGB', (size, size), 'black')
    new_image.paste(image, ((size - width) // 2, (size - height) // 2))
    return new_image


def preprocess_caption(caption):
    caption = caption.replace('.', '')
    caption = caption.strip()
    caption = '<s> {} <e>'.format(caption)
    caption = caption.lower()

    return caption


# Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--image-height', type=int, default=224,
                    help='image height')
parser.add_argument('--image-width', type=int, default=224,
                    help='image width')
parser.add_argument('--n-channels', type=int, default=3,
                    help='number of channels in image')
parser.add_argument('--prepared-dataset-dir', type=str,
                    help='path to dir where prepared datasets will be stored')
parser.add_argument('--images-dir', type=str,
                    help='path where images are stored')
parser.add_argument('--captions-path', type=str,
                    help='path where file with captions is stored')
parser.add_argument('--val-ratio', default=0.05, type=float,
                    help='ratio of validation dataset. 0 if not splitting is not needed')

args = parser.parse_args()

if __name__ == '__main__':
    image_encoder = build_image_encoder(args.image_height, args.image_width, args.n_channels)

    print('Loading images ...')
    image_paths = list_dir_with_full_paths(args.images_dir)
    image_names = np.asarray(list(map(os.path.basename, image_paths)), dtype='bytes')

    images = []
    for image_path in tqdm(image_paths):
        image = Image.open(image_path)

        # preprocess
        image = pad_to_square_image(image)
        image = image.resize((args.image_width, args.image_height))
        images.append(np.asarray(image, dtype='uint8'))

    images = np.asarray(images)

    print('Extracting images\' features ...')
    image_features = image_encoder.predict(preprocess_input(np.asarray(images, dtype='float')))

    print('Loading captions ...')
    captions_df = pd.read_table(args.captions_path, names=['image_name', 'caption'])
    captions_df['image_name'] = captions_df['image_name'].apply(lambda x: x[:-2])

    captions = []
    for image_name, g in captions_df.groupby('image_name'):
        image_captions = g['caption'].values
        image_captions = list(map(preprocess_caption, image_captions))
        captions.append(image_captions)

    captions = np.asarray(captions, dtype='bytes')

    print('Splitting data into train/val ...')
    train_indexes, val_indexes = train_test_split(
        list(range(len(images))),
        test_size=args.val_ratio, random_state=0
    )

    print('Save datasets on disk ...')
    for indexes, name in [(train_indexes, 'train'), (val_indexes, 'val')]:
        with h5py.File(pj(args.prepared_dataset_dir, 'flickr8k_{}.h5'.format(name)), 'w') as fout:
            fout['images'] = images[indexes]
            fout['image_features'] = image_features[indexes]
            fout['image_names'] = image_names[indexes]
            fout['captions'] = captions[indexes]
