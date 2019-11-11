"""
    Model function
"""
# pylint: disable=E1101, E0401, C0103

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import Model
from sklearn.metrics import accuracy_score

from input import ImageGenerator


def precison(y_true, y_pred):
  """
    Custom metric for precision
    TP / (TP+FP)
  """
  threshold = tf.convert_to_tensor(0.3)
  y_p = tf.dtypes.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)
  num_ones_pred = tf.math.count_nonzero(y_p)  # TP + FP
  FP = tf.math.count_nonzero(
      tf.dtypes.cast(
          tf.math.greater(
              y_p - y_true,
              tf.convert_to_tensor(0.0)),
          y_pred.dtype))
  return (num_ones_pred - FP) / num_ones_pred


def recall(y_true, y_pred):
  """
    Custom metric for recall
    TN / (TN+FN)
  """
  threshold = tf.convert_to_tensor(0.3)
  y_p = tf.dtypes.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)
  num_ones_pred = tf.math.count_nonzero(y_p)
  total = tf.dtypes.cast(tf.size(y_p), num_ones_pred.dtype)
  num_zeros_pred = total - num_ones_pred  # TN + FN
  TN = total - tf.math.count_nonzero(y_p + y_true)
  return TN / num_zeros_pred


def f1_score(y_true, y_pred):
  """
    Custom metric for f1-score
    Function definitions for precision and recall needed
  """
  P = precison(y_true, y_pred)
  R = recall(y_true, y_pred)
  return 2 * P * R / (P + R)


def build_graph(input_shape, output_dim):
  """
      Create model graph
  """
  Input = tf.keras.layers.Input(shape=input_shape)
  layer = Conv2D(32, (3, 3), input_shape=input_shape,
                 padding='valid', activation='relu')(Input)
  layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)

  layer = Conv2D(64, (3, 3), activation='relu', padding='valid')(layer)
  layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)

  layer = Flatten()(layer)
  layer = tf.keras.layers.Dense(64, activation="relu")(layer)
  preds = Dense(output_dim, activation='softmax')(layer)

  model = Model(Input, preds)
  return model


def compute_graph(features, output_dim):
  """
    Compute model predictions
  """
  layer = Conv2D(32, (3, 3), input_shape=(None, None, None, 3),
                 padding='valid', activation='relu')(features)
  layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)

  layer = Conv2D(64, (3, 3), activation='relu', padding='valid')(layer)
  layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)

  layer = Flatten()(layer)
  layer = tf.keras.layers.Dense(64, activation="relu")(layer)
  preds = Dense(output_dim, activation='softmax')(layer)

  return preds


def get_model_function(learning_rate=0.01):
  """
      Function which returns custom estimator model function
      Inputs: Hyper-parameters (learning_rate)
      Outputs: custom estimator model function
  """
  def model_function(features, labels, mode):
    """
        Model function for custom estimator
    """
    features['input_1'] = tf.cast(features['input_1'], tf.float32)
    labels = tf.cast(labels, tf.float32)
    preds = compute_graph(
        features['input_1'],
        output_dim=3)
    predictions = {
        'class': tf.argmax(input=preds, axis=1),
        'probabilities': preds
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions)

    # Calculate Loss and Accuracy (for both TRAIN and EVAL modes)
    loss = tf.math.reduce_mean(
        tf.keras.losses.categorical_crossentropy(
            y_true=labels, y_pred=preds))

    accuracy = tf.metrics.accuracy(labels=labels, predictions=preds)
    precision = tf.metrics.precision(labels=labels, predictions=preds)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, train_op=train_op,
          training_hooks=[get_logging_hooks(loss)])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        'accuracy': accuracy,
        'precision': precision}
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)
  return model_function


def get_logging_hooks(loss):
  """
      Get logging hook for training estimator
  """
  logging_hook = tf.train.LoggingTensorHook({'loss': loss},
                                            every_n_iter=10)
  return logging_hook


def get_callbacks():
  """
      Specify callbacks for given model
  """
  callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
      tf.keras.callbacks.TensorBoard(log_dir='../logs')]
  return callbacks


def validate(model):
  """
      Validate the given model over randomely generated samples
      Validation metric = accuracy, precision, recall
  """
  datagen = ImageGenerator(batch_size=64)
  validation_generator = datagen.generator()
  validation_batch = next(validation_generator)
  y_pred = model.predict(validation_batch)
  y_true = validation_batch[1]
  # Compute Accuracy, Precision, Recall
  _precison = precison(y_true, y_pred)
  _recall = recall(y_true, y_pred)
  _accuracy = accuracy_score(y_true, y_pred > 0.5)
  return {
      'accuracy': _accuracy,
      'precison': _precison,
      'recall': _recall
  }
