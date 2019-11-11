"""
    TfRecord Utils
"""
# pylint: disable=E0611, R0913

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import load_img

import utils


class TfrecordData():
  """
      Class consists of Functions to convert data to TFRecords
  """

  def __init__(self,
               data_dir,
               num_class,
               img_size,
               batch_size=64,
               num_epochs=1):
    """
        Initialization
    """
    self.data_dir = data_dir
    self.img_size = img_size
    self.num_class = num_class
    self.batch_size = batch_size
    self.num_epochs = num_epochs

  def convert_index_to_data(self, list_of_files):
    """
        Convert paths to data in form (feature paths, labels)
    """
    random.shuffle(list_of_files)
    paths = [self.data_dir + file for file in list_of_files]
    labels = [utils.get_label(file) for file in list_of_files]
    tokens = utils.convert_labels_to_tokens(
        labels)  # Possible to set num_class
    if self.num_class == 2:
      return paths, np.array(tokens)
    y_categorical = tf.keras.utils.to_categorical(tokens, self.num_class)
    return paths, y_categorical

  def convert_batch(self, image_paths, labels, out_path):
    """
        Convert Images at image_paths to tfRecords at output
    """
    writer = tf.python_io.TFRecordWriter(out_path)

    for count, label in enumerate(labels):
      img = np.array(
          (load_img(
              image_paths[count],
              target_size=(
                  self.img_size,
                  self.img_size))))
      g_labels = label.astype(np.uint8)
      example = tf.train.Example(features=tf.train.Features(
          feature={'image': utils.bytes_feature(img.tostring()),
                   'labels': utils.bytes_feature(g_labels.tostring())
                   }))

      writer.write(example.SerializeToString())

    writer.close()

  def convert(self, image_paths, labels, out_dir):
    """
        Convert images to tfRecords and write them to out_dir
    """
    try:
      os.mkdir(out_dir)
    except FileExistsError:
      print('Warning: The output directory already exists')
    out_dir = utils.make_valid_dir(out_dir)
    total_images = len(image_paths)
    index = 0
    while index + self.batch_size <= total_images:
      image_paths_batch = image_paths[index:index + self.batch_size]
      labels_batch = labels[index:index + self.batch_size]
      out_path = out_dir + 'batch-' + \
          str(index / self.batch_size).zfill(12) + '.tfrecords'
      self.convert_batch(image_paths_batch, labels_batch, out_path)
      print('\rSerialized batch: ' + str(index // self.batch_size), end='')
      index += self.batch_size

  def make_tf_records(self, out_dir):
    """
        Make tf records of dataset
    """
    list_of_files = utils.load_indexes(data_dir=self.data_dir)
    paths, labels = self.convert_index_to_data(list_of_files=list_of_files)
    self.convert(image_paths=paths,
                 labels=labels,
                 out_dir=out_dir)

  def parser(self):
    """
        Parser to map dataset in order to decode tfrecord
    """
    def inner_parser(record):
      """
          Decode tfrecord
      """
      featdef = {
          'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
          'labels': tf.FixedLenFeature(shape=[], dtype=tf.string),
      }

      example = tf.parse_single_example(record, featdef)
      image = tf.decode_raw(example['image'], tf.uint8)
      image = tf.reshape(image, (self.img_size, self.img_size, 3))
      label = tf.decode_raw(example['labels'], tf.uint8)
      return image, label
    return inner_parser

  def tfrecord_iterator(self, list_of_data_path):
    """
        Read tf.data using tfrecords
    """
    dataset = tf.data.TFRecordDataset(list_of_data_path)
    dataset = dataset.map(self.parser())
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(self.batch_size)
    dataset = dataset.repeat(self.num_epochs)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    labels = tf.reshape(labels, (self.batch_size, self.num_class))
    features = {'input_1': features}
    return features, labels

  def train_input_function(self):
    """
        Train input function
    """
    def input_fn():
      """
          Input function
      """
      data_dir = utils.make_valid_dir(self.data_dir)
      list_of_data_path = [
          data_dir + tfrecord for tfrecord in os.listdir(data_dir)]
      features, labels = self.tfrecord_iterator(list_of_data_path)
      return features, labels
    return input_fn

  def eval_input_function(self, filename):
    """
        Evaluate the estimator over the given .tfrecord
    """
    def input_fn():
      """
          Input function
      """
      return self.tfrecord_iterator([filename])
    return input_fn


def main():
  """
      Function to create tf records
  """
  data_dir = '../data/colors/'
  tf_record = TfrecordData(data_dir=data_dir,
                           img_size=28,
                           num_class=3)
  tf_record.make_tf_records(out_dir='../data/TFRecords_colors')


if __name__ == "__main__":
  main()
