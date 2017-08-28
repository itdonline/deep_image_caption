from __future__ import print_function, division


import itertools
import pickle
import random

import numpy as np

from keras.preprocessing.sequence import pad_sequences


class DataManager(object):
    def __init__(self, dataset_path, caption_length=None):
        with open(dataset_path, 'rb') as fin:
            dataset = pickle.load(fin)

        self.images = dataset['images']
        self.image_height, self.image_width, self.n_channels = self.images.shape[1:]

        self.image_features = dataset['image_features']

        self.caption_lists = dataset['captions']
        self.encoded_caption_lists = dataset['encoded_captions']

        self.vocabulary = dataset['vocabulary']
        self.inversed_vocabulary = dict((v, k) for k, v in self.vocabulary.items())

        if caption_length is None:
            self.caption_length = self._get_max_caption_length(self.caption_lists)
        else:
            self.caption_length = caption_length

        self.n_samples = self._count_n_samples(self.caption_lists)  # useful for determining steps per epoch


    def flow(self, batch_size=32, shuffle=True, return_image_features=True, multiple_captions_per_image=True):
        n_onjects_in_batch = 0
        batch_images, batch_encoded_captions, batch_next_words = [], [], []

        images = self.image_features if return_image_features else self.images
        encoded_caption_lists = self.encoded_caption_lists

        while True:
            if shuffle:
                shuffled_indexes = np.random.permutation(np.arange(len(images)))
                images = images[shuffled_indexes]
                encoded_caption_lists = [encoded_caption_lists[i] for i in shuffled_indexes]

            for image, encoded_caption_list in zip(images, encoded_caption_lists):
                # for i in range(len(caption) - 1):
                #     batch_images.append(image)
                #     batch_captions.append(caption[:i + 1])
                
                #     next_word = np.zeros(len(self.vocabulary))
                #     next_word[caption[i + 1]] = 1
                #     batch_next_words.append(next_word)
                batch_images.append(image)

                # choose caption from avaliable captions for current image
                encoded_caption = random.choice(encoded_caption_list) if multiple_captions_per_image else encoded_caption_list[0]
                
                index = np.random.randint(0, len(encoded_caption) - 1)
                batch_encoded_captions.append(encoded_caption[:index + 1])

                # setting target value
                next_word = np.zeros(len(self.vocabulary))
                next_word[encoded_caption[index + 1]] = 1
                batch_next_words.append(next_word)

                n_onjects_in_batch += 1

                if n_onjects_in_batch == batch_size:
                    batch = self._prepare_batch(batch_images, batch_encoded_captions, batch_next_words)
                    yield batch

                    n_onjects_in_batch = 0
                    batch_images, batch_encoded_captions, batch_next_words = [], [], []

    def flow_test(self, return_encoded_captions=False):
        caption_lists = self.encoded_caption_lists if return_encoded_captions else self.caption_lists
        for image, caption_list in zip(self.images, caption_lists):
            yield image, caption_list

    def _prepare_batch(self, images, encoded_captions, next_words):
        images = np.asarray(images)
        encoded_captions = pad_sequences(
            encoded_captions,
            maxlen=self.caption_length, padding='post',
            value=self.vocabulary['<p>']
        )
        next_words = np.asarray(next_words)

        return [[images, encoded_captions], next_words]

    def _get_max_caption_length(self, captions):
        max_caption_length = 0
        for caption in itertools.chain.from_iterable(captions):
            caption_splitted = caption.split()
            if len(caption_splitted) > max_caption_length:
                max_caption_length = len(caption_splitted)
        return max_caption_length

    def _count_n_samples(self, captions):
        return len(list(map(len, itertools.chain.from_iterable(captions))))
