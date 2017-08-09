from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import Flatten


def build_image_encoder(image_height, image_width, n_channels):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(image_height, image_width, n_channels))
    flatten_output = Flatten()(base_model.output)
    model = Model(base_model.input, flatten_output)

    return model
