from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import (Flatten, Dense, RepeatVector, LSTM, GRU,
                          Embedding, TimeDistributed, Merge, Dropout,
                          Input, Masking)
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import concatenate
from keras.regularizers import l2
from keras import optimizers


def build_image_encoder(image_height, image_width, n_channels):
    base_model = applications.ResNet50(weights='imagenet', include_top=False,
                                       input_shape=(image_height, image_width, n_channels))
    base_model_last_layer = base_model.layers[-1]

    image_encoder_output = Flatten()(base_model_last_layer.output)

    image_encoder = Model(inputs=base_model.input,
                          outputs=image_encoder_output)

    return image_encoder


def build_caption_model(embedding_dim, caption_length, vocabulary_size, image_features_dim=-1,
                        image_height=224, image_width=224, n_channels=3, regularizer=1e-8):
    # image model
    if image_features_dim == -1:  # images' features were not extracted before
        image_input = Input(shape=(image_height, image_width, n_channels), name='image')

        image_encoder = build_image_encoder(image_height, image_width, n_channels)
        image_encoder.trainable = False

        image_x = image_encoder(image_input)
        image_x = Dense(embedding_dim, activation='relu')(image_x)
    else:
        image_input = Input(shape=(image_features_dim,), name='image')
        image_x = Dense(embedding_dim, input_dim=image_features_dim, activation='relu')(image_input)
        
    image_x = RepeatVector(caption_length)(image_x)
    image_x = Dropout(0.5)(image_x)

    # text model
    text_input = Input(shape=(caption_length,), name='text')
    text_x = Embedding(vocabulary_size, 512, input_length=caption_length)(text_input)
    text_x = LSTM(
        512,
        dropout=0.2, recurrent_dropout=0.2,
        recurrent_regularizer=l2(regularizer),
        kernel_regularizer=l2(regularizer),
        bias_regularizer=l2(regularizer),
        return_sequences=True
    )(text_x)
    text_x = TimeDistributed(Dense(embedding_dim))(text_x)
    text_x = Dropout(0.5)(text_x)

    # caption model
    caption_x = concatenate([image_x, text_x])
    caption_x = LSTM(
        512,
        dropout=0.2, recurrent_dropout=0.2,
        recurrent_regularizer=l2(regularizer),
        kernel_regularizer=l2(regularizer),
        bias_regularizer=l2(regularizer),
        return_sequences=False
    )(caption_x)

    caption_output = Dense(vocabulary_size, activation='softmax')(caption_x)

    caption_model = Model(inputs=[image_input, text_input],
                          outputs=caption_output)
    caption_model.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.RMSprop(clipvalue=1.0),
                          metrics=['accuracy'])

    return caption_model