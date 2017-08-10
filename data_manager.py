from __future__ import print_function, division


import itertools
import pickle

import numpy as np
from keras.preprocessing.sequence import pad_sequences


class DataManager(object):
    def __init__(self, dataset_path, caption_length=None):
        with open(dataset_path, 'rb') as fin:
            dataset = pickle.load(fin)

        self.images = dataset['images']
        self.image_height, self.image_width, self.n_channels = self.images.shape[1:]

        self.encoded_captions = np.array([x[0] for x in dataset['encoded_captions']])  # TODO TRAIN ON ALL CAPTURES
        self.vocabulary = dataset['vocabulary']
        self.inversed_vocabulary = dict((v, k) for k, v in self.vocabulary.items())

        if caption_length is None:
            self.caption_length = max(map(len, self.encoded_captions))
        else:
            self.caption_length = caption_length

        self.n_samples = sum(map(len, self.encoded_captions))  # useful for determining steps per epoch

    def flow(self, batch_size=32, shuffle=True):
        n_onjects_in_batch = 0
        batch_images, batch_captions, batch_next_words = [], [], []
        while True:
            if shuffle:
                shuffled_indexes = np.random.permutation(np.arange(len(self.images)))
                self.images = self.images[shuffled_indexes]
                self.encoded_captions = self.encoded_captions[shuffled_indexes]

            for image, caption in zip(self.images, self.encoded_captions):
                for i in range(len(caption) - 1):
                    batch_images.append(image)
                    batch_captions.append(caption[:i + 1])

                    next_word = np.zeros(len(self.vocabulary))
                    next_word[caption[i + 1]] = 1
                    batch_next_words.append(next_word)

                    n_onjects_in_batch += 1

                    if n_onjects_in_batch == batch_size:
                        batch = self._prepare_batch(batch_images, batch_captions, batch_next_words)
                        yield batch

                        n_onjects_in_batch = 0
                        batch_images, batch_captions, batch_next_words = [], [], []

    def _prepare_batch(self, images, captions, next_words):
        images = np.asanyarray(images)
        captions = pad_sequences(
            captions,
            maxlen=self.caption_length, padding='post',
            value=self.vocabulary['<p>']
        )
        next_words = np.asarray(next_words)

        return [[images, captions], next_words]

    def _get_max_caption_length(self, captions):
        max_caption_length = 0
        for caption in itertools.chain.from_iterable(captions):
            caption_splitted = caption.split()
            if len(caption_splitted) > max_caption_length:
                max_caption_length = len(caption_splitted)
        return max_caption_length
