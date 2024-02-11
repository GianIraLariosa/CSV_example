import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


pd.options.display.max_rows = 10
# pd.options.display.float_format = "{:.1f}".format

training_df = pd.read_csv('CarPricesPrediction.csv')
# training_df["Mileage"] /= 1000.0
# training_df["Price"] /= 1000.0

training_df['time'] = training_df['Year'] - 2009

# @title Define the functions that build and train a model
def build_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    model.add(tf.keras.layers.Dense(units=1,
                                    input_shape=(1,)))

    # Compile the model topography into code that TensorFlow can efficiently
    # execute. Configure training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, df, feature, label, epochs, batch_size):
    """Train the model by feeding it data."""

    # Feed the model the feature and the label.
    # The model will train for the specified number of epochs.
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=batch_size,
                        epochs=epochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the error for each epoch.
    hist = pd.DataFrame(history.history)

    # To track the progression of training, we're going to take a snapshot
    # of the model's root mean squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


#@title Define the plotting functions
def plot_the_model(trained_weight, trained_bias, feature, label):
  """Plot the trained model against 200 random training examples."""

  # Label the axes.

  plt.xlabel(feature)
  plt.ylabel(label)
  # plt.xlim(0, 15)
  # plt.ylim(0, 35)

  # Create a scatter plot from 200 random points of the dataset.
  random_examples = training_df.sample(n=200)
  price = np.array(random_examples[feature], dtype=float)
  year = np.array(random_examples[label], dtype=int)
  plt.scatter(price, year)

  # Create a red line representing the model. The red line starts
  # at coordinates (x0, y0) and ends at coordinates (x1, y1).
  x0 = 0
  y0 = trained_bias[0]
  x1 = (random_examples[feature].max())
  y1 = (trained_bias + (trained_weight * x1))

  # # Extract scalar value from y1 if it's in the form of an array
  # if isinstance(y1, np.ndarray):
  y1 = y1[0][0]
  #
  # if x1 > random_examples[feature].max():
  #     x1 = random_examples[feature].max()
  #
  # if y1 > random_examples[label].max():
  #     y1 = random_examples[label].max()

  print("x0:", x0)
  print("y0:", y0)
  print("x1:", x1)
  print("y1:", y1)
  plt.plot([x0, x1], [y0, y1], c='r')

  # Render the scatter plot and the red line.
  plt.show()


def plot_the_loss_curve(epochs, rmse):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.97, rmse.max()])
  plt.show()


def predict_house_values(n, feature, label, my_model):
  """Predict house values based on a feature."""

  batch = training_df[feature][989:989 + n]
  predicted_values = my_model.predict_on_batch(x=batch)

  print("feature   label          predicted")
  print("  value   value          value")
  print("          in thousand$   in thousand$")
  print("--------------------------------------")
  for i in range(n):
    print ((training_df[feature][989 + i],
                                   training_df[label][989 + i],
                                   predicted_values[i][0] ))