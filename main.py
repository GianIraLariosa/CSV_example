import pandas as pd
import functions as fn

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Import the dataset.
training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

# Scale the label.
training_df["median_house_value"] /= 1000.0

# Print the first rows of the pandas DataFrame.
print(training_df)

# Get statistics on the dataset.
training_df.describe()



#print("Defined the build_model and train_model functions.")

#print("Defined the plot_the_model and plot_the_loss_curve functions.")

# The following variables are the hyperparameters.
learning_rate = 0.01
epochs = 30
batch_size = 30

# Specify the feature and the label.a
my_feature = "total_rooms"  # the total number of rooms on a specific city block.
my_label="median_house_value" # the median value of a house on a specific city block.
# That is, you're going to create a model that predicts house value based
# solely on total_rooms.

# Discard any pre-existing version of the model.
my_model = None

# Invoke the functions.
my_model = fn.build_model(learning_rate)
weight, bias, epochs, rmse = fn.train_model(my_model, training_df,
                                         my_feature, my_label,
                                         epochs, batch_size)

print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias )

fn.plot_the_model(weight, bias, my_feature, my_label)
fn.plot_the_loss_curve(epochs, rmse)
# print(weight, " ", bias, " ", my_feature, " ", my_label)
# print(epochs, " ", rmse)


fn.predict_house_values(10, my_feature, my_label, my_model)