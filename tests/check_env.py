""" Archie Luxton, aol21, https://github.com/ese-msc-2021/irp-aol21, 2022 """
import warnings
import pandas as pd
import sys
from torch.utils.data import TensorDataset, DataLoader # noqa
import time
import numpy as np
import math

print("   _      _      _    \n"
      " >(.)__ <(.)__ =(.)__ \n"
      "  (___/  (___/  (___/ \n")

print("##################################")
print("#### WELCOME TO DeepUQ (DUQ)! ####")
print("##################################")

print("   _      _      _    \n"
      " >(.)__ <(.)__ =(.)__ \n"
      "  (___/  (___/  (___/ \n")


print("Note: Using a very small network with a very small number of epochs")
print("If dialog stalls, please press return")


print("\n#### IMPORTING MODULES AND PRE-PROCESSING DATA ####\n")


sys.path.append('../duq/')
try:
    from duq import pre, post
    from duq import mc_dropout as MC
except ImportError or ModuleNotFoundError:
    import pre
    import post
    import mc_dropout as MC

# Suppresses sklearn warning about PCA.. Is this ok?
warnings.filterwarnings("ignore", category=UserWarning)

# Set seeds
seed = 1234                  # Assign a value to the seed
# Set the seed for 'random', 'np.random', 'torch.manual_seed' and
# 'torch.cuda.manual_seed_all'
pre.set_seed(seed)

# Which column(s) does the the dependent variable(s) that we're interested
# in sit? list
y_cols = [6]
# Which column(s) is the independent variable(s) (features) in? list
x_cols = [0, 1, 2, 3, 4, 5]
# this is used if we want to sort and split the data by a particular parameter
component = "n/a"

# Import the csv file and save as DataFrame
filepath = '../../irp-aol21/data/all_data.csv'
df_orig = pd.read_csv(filepath, header=None)
df_orig.columns = [
    'nbays_x',
    'nbays_y',
    'nbays_z',
    'bay_width_x',
    'bay_width_y',
    'bay_width_z',
    'modal_freq_1',
    'modal_freq_2',
    'modal_freq_3',
    'modal_freq_4',
    'modal_freq_5',
    'modal_freq_6']

# Calculate the mean and standard deviation of the original dataset
data_mean = df_orig.mean()
data_std = df_orig.std()

# Visualise full dataset in reduced dimensional space
_, components = post.PCA_transformdata(
    df_orig.iloc[:, x_cols], return_components=True)


ood_lims_in = [[0., 3.], [0., 3.], [0., 3.], [0., 5.], [0., 5.], [0., 5.]]
train_lims_all = [[3., 10], [3., 10], [3., 10], [5., 10], [5., 10], [5., 10]]
ood_lims_out = [[10, 25], [10, 25], [10, 30], [10, 12], [10, 12], [10, 12]]

TRAIN, VAL, TEST = pre.split_by_bounds(df=df_orig,
                                       x_cols=x_cols,
                                       y_cols=y_cols,
                                       train_lims_all=train_lims_all,
                                       ood_lims_in=ood_lims_in,
                                       ood_lims_out=ood_lims_out,
                                       data_mean=data_mean,
                                       data_std=data_std,
                                       PCA_components=components,
                                       val_split=0.1,
                                       verbose=True,
                                       plots=False,
                                       figsize=(8, 8))  # ,
# save_image=True,
# savename="../reports/final_report_images/splitting_data/manual_bounds.eps",
# saveformat="eps")

x_train, y_train, train_data, train_indices = TRAIN
x_val, y_val, val_data, val_indices = VAL
x_test, y_test, test_data, test_indices = TEST

y_train_unnorm = pre.unnormalise(
    y_train,
    data_mean[y_cols].values,
    data_std[y_cols].values)
y_val_unnorm = pre.unnormalise(
    y_val,
    data_mean[y_cols].values,
    data_std[y_cols].values)
y_test_unnorm = pre.unnormalise(
    y_test,
    data_mean[y_cols].values,
    data_std[y_cols].values)


# Save the model so we can load it and perform inference later
savename = "../trained_models/for_prediction/MC_Dropout"

parameters = dict(
    # Specific to this method
    drop_prob=0.3,
    num_samples=150,

    # Generic Hyperparameters
    num_epochs=100,
    batch_size=50,              # Batch size for training data
    lr=5e-2,                   # Learning rate
    weight_decay=None,          # Weight decay

    # Model architecture
    input_dim=len(x_cols),      # Number of input neurons
    output_dim=len(y_cols),     # Number of output neurons
    num_units=10,              # Number of neurons per hidden layer
    num_layers=5,

    # Data
    # Which column(s) contain the dependent variable(s) / label(s)
    y_cols=y_cols,
    # Which column(s) contain the independent variable(s) / feature(s)
    x_cols=x_cols,

    # Logigng only
    component=None,            # Which parameter are we sorting by (as an int)?
    # Name of the component we're sorting by (as a string)
    sortby=None,
    model_name="MC_Dropout_Final",    # For logging only
    criterion_name="MSELoss",   # For logging only
    optimiser_name="Adam",      # For logging only
    # How much we're splitting from top and bottom of sorted training set for
    # test set
    cutoff_percentile=None,
    # How much we're splitting from the train set (minus the test set), as a
    # float between 0-1
    val_split=None,
    seed=seed,                   # Random seed used (for logging only)
    ood_lims_in=ood_lims_in,     # Inner OOD limits
    train_lims_all=train_lims_all,  # Training set limits
    ood_lims_out=ood_lims_out    # Outer OOD limits
)

assertstr = "Please ensure that the number of output neurons is correct! "
f"There should be {len(y_cols)}"
assert parameters['output_dim'] == len(y_cols), assertstr
assert parameters['input_dim'] == len(x_cols), assertstr

# Whether or not to log the run to Weights and Biases
# (https://wandb.ai/home). Requires an account
wandb_mode = False

if wandb_mode:
    import wandb
    wandb.login()
    wandb.init(
        config=parameters,
        entity="archieluxton",
        project=parameters['model_name'])


# Instantiate a model class of type MC Dropout
model = MC.MC_Dropout(train_data=train_data,
                      parameters=parameters,
                      val_data=val_data,
                      data_mean=data_mean,
                      data_std=data_std,
                      wandb_mode=wandb_mode)


print("\n\n#### TRAINING MODEL ####\n")
start = time.time()

net, train_loss, val_loss = model.train_model(LLP=False)

end = time.time()
print(f"Time taken: {end-start}s")


# TRAIN THE MODEL
net = model.train_model(LLP=False)

num_samples = 10000
x_value = np.array([5, 5, 5, 10, 10, 10])

s, m, sd = model.make_prediction(x_value=x_value,
                                 model=model,
                                 data_mean=data_mean,
                                 data_std=data_std,
                                 verbose=False,
                                 plots=False,
                                 num_samples=num_samples)


print("\n\n#### TESTING RESULTS ####\n")


assert math.isclose(sd.item(), 0.04184993, abs_tol=0.001), f"Standard deviation is {sd.item()} when it should be approximately 0.038." # noqa
assert math.isclose(m.item(), 0.339122, abs_tol=0.001), f"Standard deviation is {m.item()} when it should be approximately 0.038." # noqa

print("\n#### Testing successful!\n")


print(f"Using X = {x_value}:\n\tModal frequency = {m.item():.3f} Hz, +/- "
      f"{sd.item():.3f} Hz (+/-{100*sd.item()/m.item():.2f}%)")
