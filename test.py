import argparse

# Arguments. Load arguments before all libs because of GPU/CPU suuport
parser = argparse.ArgumentParser()

parser.add_argument('--train-dataset-path', type=str,
                    help='path to train dataset')
parser.add_argument('--val-dataset-path', type=str,
                    help='path to val dataset')
parser.add_argument('--beam-size', type=int, default=1,
                    help='Beam size')
parser.add_argument('--max-caption-length', type=int, default=30,
                    help='Maximal length of caption')
parser.add_argument('--use-gpu', type=int, default=0,
                    help='1 if use GPU, else if zero, than computations will be held on CPU')
parser.add_argument('--model-path', type=str, default='',
                    help='If \'\', than last found in ./experiments dir model weigths will be used')
parser.add_argument('--results-dir', type=str, default='./results',
                    help='Path to dir, where results of testing will be held')
args = parser.parse_args()

import os
if not args.use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # disable GPU

import shutil

from tensorflow.python.client import device_lib

from glob import iglob
from tqdm import tqdm
import nltk

from faker import Faker

# Local imports
from data_manager import DataManager
from models import build_caption_model, build_image_encoder
from caption_generator import CaptionGenerator


if __name__ == '__main__':
    print(device_lib.list_local_devices())

    # load data managers
    dm_train = DataManager(args.train_dataset_path)
    dm_val = DataManager(args.val_dataset_path, caption_length=dm_train.caption_length)

    # set model path
    if args.model_path == '':
        model_paths = list(iglob('./experiments/**/*.hdf5', recursive=True))
        model_paths = sorted(model_paths, key=lambda x: os.path.getctime(x))
        model_path = model_paths[-1]  # getting most fresh weights
    else:
        model_path = args.model_path
    print('Model path: {}'.format(model_path))

    # load image encoder and caption generator
    image_encoder = build_image_encoder(dm_train.image_height, dm_train.image_width, dm_train.n_channels)
    caption_generator = CaptionGenerator(model_path, image_encoder, dm_train.vocabulary, dm_train.caption_length)

    # calculate bleu score
    predicted_captions = []
    true_caption_lists = []

    for image, caption_list in tqdm(dm_val.flow_test(return_encoded_captions=False), total=len(dm_val.images)):
        predicted_caption = caption_generator.generate_captions(image,
            max_caption_length=args.max_caption_length, beam_size=args.beam_size
        )[0]
        predicted_captions.append(predicted_caption)
        
        true_caption_lists.append(caption_list)
        
        bleu_score = nltk.translate.bleu_score.corpus_bleu(true_caption_lists, predicted_captions)
        print('Current bleu score is {:.04}'.format(bleu_score))

    # save test results
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # generating random name for results
    fake = Faker()
    random_name = fake.name()

    result_dir = os.path.join(args.results_dir, '{} [{:.04}]'.format(random_name, bleu_score))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    shutil.copy(model_path, result_dir)
    with open(os.path.join(result_dir, 'bleu_score.txt'), 'w') as fout:
        fout.write(str(bleu_score))

    print('Results are saved in {}'.format(result_dir))




