"""
    Run training experiment
"""
# pylint: disable=E1101
import tensorflow as tf

from input import ImageGenerator
from tfrecord_utils import TfrecordData
import models

# tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)


def train_model_keras():
  """
      Train model using keras API
  """
  datagen = ImageGenerator(batch_size=64)
  generator = datagen.generator()
  model = models.build_graph(
      input_shape=datagen.shape,
      output_dim=datagen.num_classes)
  model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(lr=0.01),
                metrics=['accuracy', models.precison, models.recall])
  callbacks = models.get_callbacks()
  model.fit_generator(generator, steps_per_epoch=1000, callbacks=callbacks)
  results = models.validate(model)
  print(results)


def train_and_evaluate_keras_estimator():
  """
      Train model using keras-estimator API
  """
  datagen = ImageGenerator()
  model = models.build_graph(
      input_shape=datagen.shape,
      output_dim=datagen.num_classes)
  model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(lr=0.01),
                metrics=['accuracy', models.precison, models.recall])
  estimator_model = tf.keras.estimator.model_to_estimator(
      keras_model=model, model_dir='../logs_keras_estimator/')
  estimator_model.train(input_fn=datagen.train_input_function, steps=100)
  results = estimator_model.evaluate(input_fn=datagen.eval_input_function)
  print('\nKeras estimator Results:', results)


def train_and_evaluate_custom_estimator():
  """
      Train model using custom estimator API
  """
  datagen = ImageGenerator()
  custom_estimator_model = tf.estimator.Estimator(
      model_fn=models.get_model_function(
          learning_rate=0.01),
      model_dir='../logs_custom_estimator/')
  custom_estimator_model.train(
      input_fn=datagen.train_input_function, steps=1000)
  results = custom_estimator_model.evaluate(
      input_fn=datagen.eval_input_function)
  preds = custom_estimator_model.predict(input_fn=datagen.eval_input_function)
  print('\nCustom estimator Results:', results, preds)


def train_and_evaluate_custom_estimator_tfrecord():
  """
      Train model using custom estimator with tfrecord input pipeline
  """
  data_dir = '../data/TFRecords_colors/'
  tf_record_data = TfrecordData(data_dir=data_dir,
                                num_class=3,
                                img_size=28,
                                num_epochs=100)
  custom_estimator_model = tf.estimator.Estimator(
      model_fn=models.get_model_function(
          learning_rate=0.01),
      model_dir='../logs_custom_estimator_tfrecord/')
  custom_estimator_model.train(
      input_fn=tf_record_data.train_input_function(), steps=200)
  results = custom_estimator_model.evaluate(
      input_fn=tf_record_data.eval_input_function(
          filename=data_dir + 'batch-0000000000.0.tfrecords'))
  print('\nCustom estimator with tfrecord Results:', results)


if __name__ == "__main__":
  print('Select the mode you want to run:\n' +
        '1. Basic Keras API\n' +
        '2. Keras Estimator API\n' +
        '3. Custom Estimator\n' +
        '4. Custom Estimator with tfrecord input')
  MODE = int(input('Enter the mode: '))
  if MODE == 1:
    train_model_keras()
  elif MODE == 2:
    train_and_evaluate_keras_estimator()
  elif MODE == 3:
    train_and_evaluate_custom_estimator()
  elif MODE == 4:
    train_and_evaluate_custom_estimator_tfrecord()
  else:
    print('Please select correct mode')
