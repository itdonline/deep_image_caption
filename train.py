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
parser.add_argument('--embedding-dim', type=int, default=4096,
                    help='embedding dimension size')

parser.add_argument('--checkpoint-period', type=int, default=1,
                    help='dump-period of checkpoints')

args = parser.parse_args()

if __name__ == '__main__':
    # setup experiment
    experiment = Experiment('./experiments')
    experiment.add_dir('checkpoints')
    experiment.add_dir('tensorboard')

    # setup data managers
    dm_train = DataManager(args.train_dataset_path)
    dm_val = DataManager(args.val_dataset_path, caption_length=dm_train.caption_length)

    # setup model
    caption_model = build_caption_model(args.embedding_dim, dm_train.caption_length, len(dm_train.vocabulary),
                                        dm_train.image_height, dm_train.image_width, dm_train.n_channels)

    # setup callbacks
    callbacks = []

    model_checkpoint_callback = ModelCheckpoint(
        pj(experiment.dirs['checkpoints'], 'checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5'),
        verbose=1,
        period=args.checkpoint_period
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
        steps_per_epoch=dm_train.n_samples // args.batch_size // 10, epochs=args.epochs,
        validation_data=dm_val.flow(batch_size=args.batch_size, shuffle=False),
        validation_steps=dm_val.n_samples // args.batch_size // 10,
        callbacks=callbacks, verbose=1
    )
