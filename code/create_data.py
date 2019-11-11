''' Generates dataset and create Tfrecord file'''
import argparse
import utils
from input import ImageGenerator
from tfrecord_utils import TfrecordData


def get_args():
  '''
  Takes arguments on command line
  Returns : Dictionary
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--img-size',
      type=int,
      required=True,
      help='size of image')
  parser.add_argument(
      '--data-dir',
      type=str,
      required=True,
      help='directory path for dataset')
  parser.add_argument(
      '--num-class',
      type=int,
      required=True,
      help='Num of classes')
  parser.add_argument(
      '--num-images',
      type=int,
      default=1000,
      help='Num of Images to generate')
  parser.add_argument(
      '--batch-size',
      type=int,
      default=64,
      help='Number of images in single batch')
  parser.add_argument(
      '--num-epochs',
      type=int,
      default=1,
      help='Num of epochs')
  return parser.parse_args()


def main():
  '''
  main function
  '''
  gen = ImageGenerator()
  args = get_args()
  gen.generate_and_save(data_dir=args.data_dir, num_images=args.num_images)
  utils.make_image_index(data_dir=args.data_dir)
  print("*" * 70)
  print("input file processed.")
  print("*" * 70)
  tf_record = TfrecordData(data_dir=args.data_dir,
                           img_size=args.img_size,
                           num_class=args.num_class,
                           batch_size=args.batch_size,
                           num_epochs=args.num_epochs)
  tf_record.make_tf_records(out_dir='../data/TFRecords_colors')


if __name__ == "__main__":
  main()
