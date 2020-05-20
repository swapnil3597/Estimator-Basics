

import tensorflow as tf
import numpy as np
import pandas as pd

def get_args():
  """Define the task arguments with the default values.
  Returns:
      experiment parameters
  """

  args_parser = argparse.ArgumentParser()

  args_parser.add_argument(
      '--model-dir',
      help="""
        Model logs and weights directory
      """,
      required=True,
      type=str
  )


  return args_parser.parse_args()

args = get_args()
model_dir = args.model_dir

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)


train_y = train.pop('Species')
test_y = test.pop('Species')

# The label column has now been removed from the features.
train.head()

def input_evaluation_set():
    features = {'SepalLength': np.array([6.4, 5.0]),
                'SepalWidth':  np.array([2.8, 2.3]),
                'PetalLength': np.array([5.6, 3.3]),
                'PetalWidth':  np.array([2.2, 1.0])}
    labels = np.array([2, 1])
    return features, labels


def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)



# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))



# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3,
    model_dir=model_dir)


# Train the Model.
# classifier.train(
#     input_fn=lambda: input_fn(train, train_y, training=True),
#     steps=5000)



# eval_result = classifier.evaluate(
#     input_fn=lambda: input_fn(test, test_y, training=False))


train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train, train_y, training=True),
                                    max_steps=200)

eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(test, test_y, training=False),
                                  steps=5,
                                  start_delay_secs=2,
                                  throttle_secs=5)

tf.estimator.train_and_evaluate(classifier,
                                train_spec,
                                eval_spec)



