
import tensorflow as tf
from matplotlib import pyplot as plt

generator = tf.keras.models.load_model("model.keras")

generator.summary()