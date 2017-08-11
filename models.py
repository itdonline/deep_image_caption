from keras.applications import ResNet50, VGG16
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, RepeatVector, LSTM, Embedding, TimeDistributed, Merge


def build_image_encoder(image_height, image_width, n_channels):
    base_model = VGG16(weights='imagenet')
    image_encoder = Model(inputs=base_model.input,
                          outputs=base_model.get_layer('fc2').output)

    return image_encoder


def build_caption_model(embedding_dim, caption_length, vocabulary_size, image_features_dim=-1,
                        image_height=224, image_width=224, n_channels=3):
    image_model = Sequential()

    if image_features_dim == -1:  # images' features were not extracted before
        image_encoder = build_image_encoder(image_height, image_width, n_channels)
        image_encoder.trainable = False
        image_model.add(image_encoder)
        image_model.add(Dense(embedding_dim, activation='relu'))
    else:
        image_model.add(Dense(embedding_dim, input_dim=image_features_dim, activation='relu'))

    image_model.add(RepeatVector(caption_length))

    language_model = Sequential()
    language_model.add(Embedding(vocabulary_size, 256, input_length=caption_length))
    language_model.add(LSTM(256,return_sequences=True))
    language_model.add(TimeDistributed(Dense(embedding_dim)))

    caption_model = Sequential()
    caption_model.add(Merge([image_model, language_model], mode='concat'))
    caption_model.add(LSTM(256, return_sequences=False))
    caption_model.add(Dense(vocabulary_size, activation='softmax'))

    caption_model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                          metrics=['accuracy'])

    return caption_model