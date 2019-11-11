"""
    Input Pipeline
"""

# pylint: disable=E1101

import os
import random
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import cv2

import utils


class ImageGenerator():
  """
      Image generator class
  """

  def __init__(self, batch_size=64, height=28, width=28):
    """
        Initialization
    """
    self.batch_size = batch_size
    self.height = height
    self.width = width
    self.shape = (height, width, 3)
    self.num_classes = 3

  def get_num_classes(self):
    """
        Get number of classes
    """
    return self.num_classes

  def generator(self):
    """
        Generates batches of images for keras fit_generator
    """
    while True:
      # Create tuple of batch
      batch = (
          np.zeros((self.batch_size, self.height, self.width, 3), np.uint8),
          np.zeros((self.batch_size, self.num_classes), np.uint8))
      for i in range(self.batch_size):
        channel = random.randint(0, self.num_classes - 1)
        batch[0][i][:, :, channel] = random.randint(0, 255)
        batch[1][i][channel] = 1
      yield batch

  def generator_tf_data(self):
    """
        Generates single image sample to make it tf.data compatible
    """
    while True:
      # Create tuple of sample
      sample = (np.zeros((self.height, self.width, 3), np.uint8),
                np.zeros((self.num_classes), np.uint8))
      channel = random.randint(0, self.num_classes - 1)
      sample[0][:, :, channel] = random.randint(0, 255)
      sample[1][channel] = 1
      yield sample

  def create_dataset(self):
    """
        Create Dataset from generator using tf.data API
    """
    output_shapes = ((self.height, self.width, 3),
                     (self.num_classes))
    dataset = tf.data.Dataset.from_generator(
        generator=self.generator_tf_data, output_types=(
            tf.uint8, tf.uint8), output_shapes=output_shapes)
    return dataset

  def train_input_function(self):
    """
        Input function for training estimator
    """
    dataset = self.create_dataset()
    dataset = dataset.batch(self.batch_size)
    iterator = dataset.make_one_shot_iterator()
    features_tensors, labels = iterator.get_next()
    features = {'input_1': features_tensors}
    labels = tf.reshape(labels, (self.batch_size, self.num_classes))
    return features, labels

  def eval_input_function(self):
    """
       Input function for evaluate estimator
    """
    gen = self.generator()
    batch = next(gen)
    dataset = tf.data.Dataset.from_tensor_slices(batch)
    dataset = dataset.batch(self.batch_size)
    iterator = dataset.make_one_shot_iterator()
    features_tensors, labels = iterator.get_next()
    features = {'input_1': features_tensors}
    return features, labels

  def generate_and_save(self, data_dir, num_images):
    """
        Generate and saves image in data directory
    """
    data_dir = utils.make_valid_dir(data_dir)
    try:
      os.mkdir(data_dir)
    except FileExistsError:
      pass
    # color_dict = {0:'Red', 1:'Green', 2:'Blue'}
    color_dict = {0: 'Blue', 1: 'Green', 2: 'Red'}  # For cv2
    gen = self.generator()
    for key in color_dict:
      try:
        os.mkdir(data_dir + color_dict[key])
      except FileExistsError:
        pass
    for _ in tqdm(range(num_images // self.batch_size)):
      batch = next(gen)
      for count in range(self.batch_size):
        img = batch[0][count]
        label = color_dict[np.argmax(batch[1][count])]
        cv2.imwrite(
            data_dir +
            label +
            '/' +
            utils.random_string() +
            '.jpg',
            img)


def main():
  """
      Function to generate and save color images and make index.pkl
  """
  gen = ImageGenerator()
  data_dir = '../data/colors/'
  gen.generate_and_save(data_dir=data_dir, num_images=1000)
  utils.make_image_index(data_dir=data_dir)


if __name__ == "__main__":
  main()
