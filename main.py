
from keras.backend.tensorflow_backend import set_session
from Model import Model_Multi, Model_Single
import tensorflow as tf

session_config = tf.ConfigProto()
session_config.gpu_options.visible_device_list = "0"
session_config.gpu_options.allow_growth = True
set_session(tf.Session(config=session_config))

if __name__ == "__main__":

    model_1 = Model_Multi()
    model_1.train_predict()

