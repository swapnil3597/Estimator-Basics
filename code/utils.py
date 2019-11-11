"""
    Utilities
"""

import os
import pickle
import random
import string

import tensorflow as tf


def make_valid_dir(directory):
  """
      Make the given path a valid directory path
  """
  if directory[len(directory) - 1] != '/':
    directory += '/'
  return directory


def get_label(path_string):
  """
      Get label from file index
  """
  label = ''
  for char in path_string:
    if char == '/':
      break
    label += char
  return label


def bytes_feature(value):
  """
      Get byte features
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
  """
      Get int64 feature
  """
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def random_string():
  """
      Generate a random string of fixed length
  """
  string_length = random.randint(20, 30)
  letters = string.ascii_lowercase
  return ''.join(random.choice(letters) for i in range(string_length))


def make_image_index(data_dir):
  """
      Function to make 'index.pkl' using data directory
  """
  data_dir = make_valid_dir(data_dir)
  list_of_files = list()
  walker = os.walk(data_dir)
  _, labels, _ = next(walker)
  for label in labels:
    list_of_files += [label + '/' +
                      file for file in os.listdir(data_dir + label)]
  pickle.dump(list_of_files, open(data_dir + 'index.pkl', 'wb'))


def load_indexes(data_dir):
  """
      Load index file in data_dir
  """
  try:
    list_of_files = pickle.load(open(data_dir + 'index.pkl', 'rb'))
  except FileNotFoundError:
    print("'index.pkl' not present, please run 'make_image_index' first")
  return list_of_files


def convert_labels_to_tokens(labels_list):
  """
      Convert labels to tokens
  """
  unique_labels = list(set(labels_list))
  label_dict = dict(zip(unique_labels, list(range(len(unique_labels)))))
  tokens = [label_dict[label] for label in labels_list]
  return tokens
