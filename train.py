from __future__ import print_function, division

from os.path import join as pj
import argparse

from keras.callbacks import ModelCheckpoint, TensorBoard

# Local imports
from experiments import Experiment
from data_manager import DataManager
from models import build_caption_model


# Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--train-dataset-path', type=str,
                    help='path to train dataset')
parser.add_argument('--val-dataset-path', type=str,
                    help='path to val dataset')
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=99999,
                    help='number of training epochs')
parser.add_argument('--embedding-dim', type=int, default=128,
                    help='embedding dimension size')
parser.add_argument('--image-features-dim', type=int, default=-1,
                    help='dimension of image features. If -1 then training is performed on images')

parser.add_argument('--checkpoint-period', type=int, default=1,
                    help='dump-period of checkpoints')

args = parser.parse_args()

if __name__ == '__main__':
    # setup experiment
    experiment = Experiment('./experiments')
    experiment.add_dir('checkpoints')
    experiment.add_dir('tensorboard')

    # setup data managers
    return_image_features = args.image_features_dim != -1
    dm_train = DataManager(args.train_dataset_path,
                           return_image_features=return_image_features)
    dm_val = DataManager(args.val_dataset_path, caption_length=dm_train.caption_length,
                         return_image_features=return_image_features)

    # setup model
    if args.image_features_dim == -1:  # train on images
        caption_model = build_caption_model(args.embedding_dim, dm_train.caption_length, len(dm_train.vocabulary),
                                            image_height=dm_train.image_height, image_width=dm_train.image_width,
                                            n_channels=dm_train.n_channels)
    else:
        caption_model = build_caption_model(args.embedding_dim, dm_train.caption_length, len(dm_train.vocabulary),
                                            image_features_dim=args.image_features_dim)

    print(caption_model.summary())

    # setup callbacks
    callbacks = []

    model_checkpoint_callback = ModelCheckpoint(
        pj(experiment.dirs['checkpoints'], 'checkpoint-{epoch:04d}-loss[{val_loss:.2f}].hdf5'),
        monitor='val_loss',
        save_best_only=True,
        period=args.checkpoint_period,
        verbose=1
    )
    callbacks.append(model_checkpoint_callback)

    tensorboard_callback = TensorBoard(
        log_dir=experiment.dirs['tensorboard'],
        batch_size=1
    )
    callbacks.append(tensorboard_callback)

    # train
    caption_model.fit_generator(
        dm_train.flow(batch_size=args.batch_size, shuffle=True),
        steps_per_epoch=dm_train.n_samples // args.batch_size, epochs=args.epochs,
        validation_data=dm_val.flow(batch_size=args.batch_size, shuffle=False),
        validation_steps=dm_val.n_samples // args.batch_size,
        callbacks=callbacks, verbose=1
    )
