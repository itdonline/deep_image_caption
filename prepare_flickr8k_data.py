from __future__ import print_function, division

import os
from os.path import join as pj
import argparse
from collections import OrderedDict
import json

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import h5py
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

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


def create_vocabulary(captions):
    words = []
    for caption in captions.flatten():
        words.extend(caption.split())

    unique_words, word_counts = np.unique(sorted(words), return_counts=True)
    vocabulary = OrderedDict((word, i) for i, word in enumerate(unique_words))

    return vocabulary

def get_max_caption_length(captions):
    max_caption_length = 0
    for caption in captions.flatten():
        caption_splitted = caption.split()
        if len(caption_splitted) > max_caption_length:
            max_caption_length = len(caption_splitted)
    return max_caption_length


def label_encode_caption(caption, vocabulary):
    label_encoded_caption = []
    for word in caption.split():
        label_encoded_caption.append(vocabulary[word])

    return label_encoded_caption


def preprocess_captions(captions):
    vocabulary = create_vocabulary(captions)
    max_caption_length = get_max_caption_length(captions)

    preprocessed_captions = []
    for image_captions in captions:
        preprocessed_image_captions = []
        for caption in image_captions:
            caption = label_encode_caption(caption, vocabulary)
            preprocessed_image_captions.append(caption)

        preprocessed_image_captions = pad_sequences(
            preprocessed_image_captions,
            maxlen=max_caption_length, padding='post', value=vocabulary[b'<e>']
        )
        preprocessed_captions.append(preprocessed_image_captions)

    return np.asarray(preprocessed_captions), vocabulary


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
parser.add_argument('--extract-image-features', default=1, type=int,
                    help='if 1 then extract images\' features')

args = parser.parse_args()

if __name__ == '__main__':
    if args.extract_image_features:
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

    if args.extract_image_features:
        print('Extracting images\' features ...')
        image_features = image_encoder.predict(preprocess_input(np.asarray(images, dtype='float')))

    print('Loading captions ...')
    captions_df = pd.read_table(args.captions_path, names=['image_name', 'caption'])
    captions_df['image_name'] = captions_df['image_name'].apply(lambda x: x[:-2])

    # wipe out incorrect names
    captions_df = captions_df[captions_df['image_name'].str.contains('.jpg$')]

    # check order
    assert np.array_equal(np.asarray(captions_df['image_name'].unique(), dtype='bytes'), image_names)

    captions = []
    for image_name, group in captions_df.groupby('image_name'):
        image_captions = group['caption'].values
        image_captions = list(map(preprocess_caption, image_captions))
        captions.append(image_captions)

    captions = np.asarray(captions, dtype='bytes')

    print('Preprocessing captions')
    preprocessed_captions, vocabulary = preprocess_captions(captions)

    print('Splitting data into train/val ...')
    train_indexes, val_indexes = train_test_split(
        list(range(len(images))),
        test_size=args.val_ratio, random_state=0
    )

    print('Saving datasets on disk ...')
    for indexes, name in [(train_indexes, 'train'), (val_indexes, 'val')]:
        with h5py.File(pj(args.prepared_dataset_dir, 'flickr8k_{}.h5'.format(name)), 'w') as fout:
            fout['images'] = images[indexes]

            if args.extract_image_features:
                fout['image_features'] = image_features[indexes]

            fout['image_names'] = image_names[indexes]
            fout['captions'] = captions[indexes]
            fout['preprocessed_captions'] = preprocess_captions[indexes]

    print('Saving vocabulary on disk ...')
    with open(pj(args.prepared_dataset_dir, 'vocabulary'), 'w') as fout:
        json.dump(vocabulary, fout)