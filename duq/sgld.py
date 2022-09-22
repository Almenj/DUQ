""" Archie Luxton, aol21, https://github.com/ese-msc-2021/irp-aol21, 2022 """
import os  # Avoids the OpenMP error in conjunction with the below
import numpy as np
import torch
import torch.nn as nn
import sys
from livelossplot import PlotLosses
import time
from torch.utils.data import TensorDataset, DataLoader # noqa
import warnings
from scipy.special import logsumexp
from scipy.spatial.distance import euclidean
from scipy.stats import norm
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict  # For dynamic layers

# If the DUQ package is installed, import pre from it
# Otherwise, do a relative import (assuming pre is in the same folder as this)
try:
    from duq import pre
except ImportError or ModuleNotFoundError:
    import pre

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device = ", device)
if device.type == "cpu":
    cuda = False
else:
    cuda = True

# Prevent an error about OpenMP popping up when importing torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Suppress the future warning in sns distplot
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress the numpy warnings about indexing arrays
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class SGLD_Model(nn.Module):
    """
    A class to define the network implementing Dropout.
    Inference is performed later by a MC sampling method.
    """

    def __init__(self, input_dim, output_dim, num_units, num_layers):
        super(SGLD_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # n hidden layers
        layers = OrderedDict()

        # Input layer
        layers["0"] = nn.Linear(input_dim, num_units)

        for i in range(1, num_layers - 1):
            layers[str(i)] = nn.Linear(num_units, num_units)

        # Output layer
        layers[str(num_layers - 1)] = nn.Linear(num_units, output_dim)

        self.layers = nn.Sequential(layers)

        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)
        x = self.layers[self.num_layers - 1](x)  # OUTPUT LAYER
        return x


class SGLD():
    """
    A class that contains all the model parameters, hyperparameters and methods
    for MC Dropout.
    """

    def __init__(self, parameters, train_data, data_mean,
                 data_std, val_data=None, wandb_mode=False):
        self.parameters = parameters
        # Specific to this method
        self.burnin_epochs = self.set_defaults('burnin_epochs', 500)
        self.num_networks = self.set_defaults('num_networks', 30)
        self.noise_multiplier = self.set_defaults('noise_multiplier', 1)
        self.anneal_gamma = self.set_defaults('anneal_gamma', None)
        # Generic hyperparameters
        self.num_epochs = self.set_defaults('num_epochs', 1000)
        self.batch_size = self.set_defaults('batch_size', len(train_data))
        self.lr = self.set_defaults('lr', 1e-4)
        self.weight_decay = self.set_defaults(
            'weight_decay', None)     # =None to turn off L2 regularisation
        # Model architecture
        self.input_dim = parameters['input_dim']
        self.output_dim = parameters['output_dim']
        self.num_units = parameters['num_units']
        self.num_layers = self.set_defaults('num_layers', 3)
        a = "num_layers includes input and output layers! "
        f"Please ensure num_layers > 2. Currently {self.num_layers}."
        assert self.num_layers > 2, a
        # Data
        self.y_cols = parameters['y_cols']
        self.x_cols = parameters['x_cols']
        # Logging only
        self.sortby = self.set_defaults('sortby')
        self.component = self.set_defaults('component')
        self.model_name = self.set_defaults('model_name', "SGLD")
        self.criterion_name = self.set_defaults('criterion_name')
        self.optimiser_name = self.set_defaults('optimiser_name')
        self.cutoff_percentile = self.set_defaults('cutoff_percentile')
        self.val_split = self.set_defaults('val_split')
        self.seed = self.set_defaults('seed')

        self.train_data = train_data
        self.val_data = val_data
        self.data_mean = data_mean
        self.data_std = data_std

        self.y_std = self.data_std[self.y_cols].values  # [0]
        self.y_mean = self.data_mean[self.y_cols].values  # [0]
        self.x_mean = self.data_mean[self.x_cols].values
        self.x_std = self.data_std[self.x_cols].values

        # Mainly for re-training purposes
        self.optimiser = None
        self.criterion = None
        self.net = None
        self.logs = None

        self.wandb_mode = wandb_mode
        if self.wandb_mode:
            # Import wandb in a global scope if it exists
            try:
                global wandb
                import wandb
            except ImportError or ModuleNotFoundError:
                print("Weights and Biases API (wandb) isn't installed. Please install it to use. Turning wandb_mode to False.", file=sys.stderr) # noqa
                self.wandb_mode = False

    def set_defaults(self, param_name, default="Not set"):
        """ Set default parameters if they haven't been entered """
        if self.parameters[param_name] is None:
            return default
        else:
            return self.parameters[param_name]

    def add_param_noise(self, net, multiplier=1):
        """
        Cycle through all parameters and add Gaussian noise
        as per the SGLD method.

        Parameters
        ----------
        net : nn.Module
            The network
        multiplier : float, optional
            Scalar to multiply the noise by on each weight. The default is 1.

        Returns
        -------
        None.

        """
        for p in net.parameters():
            if p.requires_grad:
                shape = p.data.shape
                noise = multiplier * norm(0, self.lr).rvs(torch.numel(p.data))
                noise = torch.Tensor(noise).reshape(shape).to(
                    device)  # Reshape the noise and turn into a tensor
                # Add the noise to each parameter
                p.data.add_(noise)

    def train(self, net, optimiser, criterion, data_loader):
        """
        Method for training the neural network.

        Parameters
        ----------
        net : nn.Module
            The network class
        optimiser : torch.optim
            Optimiser
        criterion : torch.nn.modules.loss
            Loss function
        data_loader : torch.DataLoader
            Loader for training data

        Returns
        -------
        torch.Tensor
            Average training loss per datapoint

        """
        net.train()
        train_loss = 0.
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            preds = net(X)  # .detach()
            loss = torch.sqrt(criterion(preds, y))  # RMSE
            loss.backward()
            train_loss += loss * X.size(0)
            optimiser.step()
            self.add_param_noise(net, multiplier=self.noise_multiplier)
        return train_loss / len(data_loader.dataset)

    def validate(self, net, criterion, data_loader):
        """
        Method for calculating the loss on a validation set.

        Parameters
        ----------
        net : nn.Module
            The network class
        criterion : torch.nn.modules.loss
            Loss function
        data_loader : torch.DataLoader
            Loader for training data

        Returns
        -------
        torch.Tensor
            Average validation loss per datapoint

        """
        net.eval()
        validation_loss = 0.
        for X, y in data_loader:
            with torch.no_grad():
                X, y = X.to(device), y.to(device)
                preds = net(X)
                loss = torch.sqrt(criterion(preds, y))
                # multiply by batch size
                validation_loss += loss * X.size(0)
        # Divide to get average loss per datapoint
        return validation_loss / len(data_loader.dataset)

    def evaluate_point(self, net, X):
        """
        Method for performing a single prediction using the NN.

        Parameters
        ----------
        net : nn.Module
            The network class
        X : torch.Tensor
            Datapoint to assess

        Returns
        -------
        torch.Tensor or NumPy array
            Corresponding prediction for the given X point

        """
        net.eval()
        if torch.is_tensor(X):
            tens = True
        else:
            tens = False
            X = torch.Tensor(X)
        with torch.no_grad():
            X = X.to(device)
            preds = net(X).squeeze()

        if tens:
            return preds
        else:
            return preds.numpy()

    def count_parameters(self, net):
        """
        Count the number of trainable parameters in the neural network.

        Parameters
        ----------
        net : nn.Module
            Network to count the parameters of

        Returns
        -------
        float
            Number of trainable parameters in net

        """
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    def PLL(self, pred, true, l_scale=10**(-2)):
        """
        Calculate the Predictive Log-likelihood as per equation 8 (and
        equation 22 in Appendix 4.4) of "Dropout as a Bayesian Approximation:
        Representing Model Uncertainty in Deep Learning" (Y. Gal, 2016,
        University of Cambridge). Predictive log likelihood captures how well
        a model fits the data, with larger values indicating a better model
        fit.
        Not currently being used.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted value of y
        true : torch.Tensor
            True value of y
        l : float
            Length scale of data. Default = 10**(-2) per paper

        Returns
        -------
        PLL: float
            Predictive log-likelihood

        """
        # Number of datapoints
        if pred.ndim > 1:
            N = pred.shape[0]
        else:
            N = 1
        # In paper, (1-p_1) is the probability that a neuron gets dropped
        p_1 = (1 - self.drop_prob)

        # Precision parameter
        tau = (l_scale**2 * p_1) / (2 * N * self.weight_decay)

        # Number of stochastic forward passes performed
        T = self.num_samples

        return logsumexp(-0.5 * tau * euclidean(true, pred)) - \
            np.log(T) - 0.5 * np.log(2 * np.pi) - 0.5 * np.log(tau**(-1))

    def train_model(self, LLP=True, checkpoint=False, checkpoint_path=None):
        """
        Routine to train the model and return the trained network.

        Returns
        -------
        net : nn.Module
            The final trained network
        train_loss : float
            Final training loss after training
        validation_loss : float
            Final validation loss after training

        """
        # Set the random seed
        pre.set_seed(self.seed)

        # Create an empty list to store the saved networks
        self.nets = []

        # NumPy array, stores which epochs should be saved
        self.epoch_saves = np.linspace(
            self.burnin_epochs +
            1,
            self.num_epochs,
            self.num_networks +
            1).astype(int)  # Which epochs to save and sample weights from

        # If we've enabled Weights and Biases, set the config
        if self.wandb_mode:
            config = wandb.config  # noqa

        # Instantiate a new net class
        self.net = SGLD_Model(input_dim=self.input_dim,
                              output_dim=self.output_dim,
                              num_units=self.num_units,
                              num_layers=self.num_layers).to(device)

        if self.weight_decay is None:
            self.optimiser = torch.optim.Adam(
                self.net.parameters(), lr=self.lr)
        else:
            self.optimiser = torch.optim.Adam(
                self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay) # noqa
        self.criterion = nn.MSELoss()

        if self.anneal_gamma is not None:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimiser, gamma=self.anneal_gamma)   # Per the SGLD paper

        assert self.train_data is not None, "You need to enter training data!"
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0)

        if self.val_data is not None:
            val_loader = DataLoader(
                self.val_data,
                batch_size=len(
                    self.val_data),
                shuffle=False,
                num_workers=0)

        # Make sure that we aren't setting the batch size higher than the
        # amount of available data
        assert len(self.train_data) >= self.batch_size, f"Batch size should be no more than {len(self.train_data)}, but it is {self.batch_size}." # noqa

        if self.wandb_mode:
            wandb.watch(self.net)  # noqa
            wandb.watch(self.criterion)

        if LLP:
            liveloss = PlotLosses()
        start = time.time()

        for i in range(self.num_epochs + 1):
            logs = {}
            train_loss = self.train(
                self.net,
                self.optimiser,
                self.criterion,
                train_loader)
            if self.anneal_gamma is not None:
                scheduler.step()  # Change the LR according to the scheduler
            if self.val_data is not None:
                validation_loss = self.validate(
                    self.net, self.criterion, val_loader)
                logs['val_' + 'loss'] = validation_loss.detach().cpu()
            logs['' + 'loss'] = train_loss.detach().cpu()
            if LLP:
                liveloss.update(logs)
            if self.wandb_mode:
                wandb.log(logs)  # noqa
            if i in self.epoch_saves and i > self.burnin_epochs:
                self.nets.append(copy.deepcopy(self.net))
            if LLP and (i % 500 == 0):
                liveloss.draw()
        end = time.time()

        # Save model so we can pick it up later
        if checkpoint:
            assert checkpoint_path is not None, "Please enter a save path for the training checkpoint." # noqa
            torch.save({'model': self.net.state_dict(),
                        'optimiser': self.optimiser.state_dict(),
                        'loss': train_loss,
                        'epoch': i,
                        }, checkpoint_path)

        print(f"Time elapsed: {end-start:.2f}s.")

        print("Number of trainable model parameters: "
              f"{self.count_parameters(self.net)}, number of training "
              f"samples: {len(self.train_data)}")
        print(f"Used batches of {self.batch_size}.\n")

        if self.wandb_mode:
            wandb.unwatch(self.net)  # noqa
            wandb.unwatch(self.criterion)  # noqa

        if self.val_data is not None:
            return self.net, train_loss, validation_loss
        else:
            return self.net, train_loss

    def retrain_model(self, _train_data, _batch_size, _epochs,
                      checkpoint_path=None, save_checkpoint=False):
        """
        Retrain an existing model class using some new data. Works by
        loading up a saved checkpoint. Also gives the option to save a new
        checkpoint so we can train upon this too.

        Parameters
        ----------
        _train_data : torch.TensorDataset
            New training data
        _batch_size : int
            New batch size
        _epochs : int
            How many epochs to train for (in addition to the previously
            executed training)
        save_checkpoint : bool, optional
            Whether or not to save another checkpoint after this
            additional training. The default is False.

        Returns
        -------
        net : nn.Module
            The final trained network
        _train_loss : float
            Final training loss after re-training
        validation_loss : float
            Final validation loss after re-training

        """
        # Set the random seed
        pre.set_seed(self.seed)

        # Create an empty list to store the saved networks
        self.nets = []

        # NumPy array, stores which epochs should be saved
        epoch_saves = np.linspace(self.burnin_epochs + 1, _epochs,self.num_networks + 1).astype(int)  # Which epochs to save and sample weights from

        # If we've enabled Weights and Biases, set the config
        if self.wandb_mode:
            config = wandb.config  # noqa

        if self.anneal_gamma is not None:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimiser, gamma=self.anneal_gamma)   # Per the SGLD paper

        assert _train_data is not None, "You need to enter training data!"
        _train_loader = DataLoader(
            _train_data,
            batch_size=_batch_size,
            shuffle=True,
            num_workers=0)

        if self.val_data is not None:
            val_loader = DataLoader(
                self.val_data,
                batch_size=len(
                    self.val_data),
                shuffle=False,
                num_workers=0)

        # Make sure that we aren't setting the batch size higher than the
        # amount of available data
        assert len(_train_data) >= _batch_size, f"Batch size should be no more than {len(_train_data)}, but it is {_batch_size}." # noqa

        if self.wandb_mode:
            wandb.watch(self.net, self.criterion)  # noqa

        liveloss = PlotLosses()
        start = time.time()

        for i in range(_epochs + 1):
            logs = {}
            _train_loss = self.train(
                self.net,
                self.optimiser,
                self.criterion,
                _train_loader)
            if self.anneal_gamma is not None:
                scheduler.step()  # Change the LR according to the scheduler
            if self.val_data is not None:
                validation_loss = self.validate(
                    self.net, self.criterion, val_loader)
                logs['val_' + 'loss'] = validation_loss.detach().cpu()
            logs['' + 'loss'] = _train_loss.detach().cpu()
            liveloss.update(logs)
            if self.wandb_mode:
                wandb.log(logs)  # noqa
            if i in epoch_saves and i > self.burnin_epochs:
                self.nets.append(copy.deepcopy(self.net))
            if i % 500 == 0:
                liveloss.draw()
        end = time.time()

        if save_checkpoint:
            assert checkpoint_path is not None, "Please enter a save path for the training checkpoint." # noqa
            # Save model so we can pick it up later
            torch.save({'model': self.net.state_dict(),
                        'optimiser': self.optimiser.state_dict(),
                        'loss': _train_loss,
                        'epoch': i,
                        }, checkpoint_path)

        print(f"Time elapsed: {end-start:.2f}s.")

        print("Number of trainable model parameters: "
              f"{self.count_parameters(self.net)}, number of training "
              f"samples: {len(_train_data)}")
        print(f"Used batches of {_batch_size}.\n")

        if self.wandb_mode:
            wandb.unwatch(self.net)  # noqa
            wandb.unwatch(self.criterion)  # noqa

        if self.val_data is not None:
            return self.net, _train_loss, validation_loss
        else:
            return self.net, _train_loss

    def generate_samples(self, X, **kwargs):
        """
          Generate a number of predictions that are sampled to generate
          our uncertainty.

          Parameters
          ----------
          X : torch.Tensor
              Input to generate samples for
          **kwargs
              DESCRIPTION.

          Returns
          -------
          samples : NumPy array
              All individual samples generated by successive stochastic forward
              passes through the network.
              If one output, of shape (#samples, #datapoints)
              If more than one output, of shape (#outputs,#samples,#datapoints)
          means : NumPy array
              Array of the mean prediction of each datapoint, based on the
              average of all forward passes for that point.
              Of shape (#outputs, #datapoints)
          stds : NumPy array
              Array of the standard deviation of each datapoint, based on the
              std of all forward passes for that point.
              Of shape (#outputs, #datapoints)

          """
        samples = []

        for network in self.nets:
            preds = network.cpu().forward(X).data.numpy()
            samples.append(preds)
        samples = np.array(samples)
        samples = pre.unnormalise(samples, self.y_mean, self.y_std)

        # means of shape (num_dependent_vars, num_datapoints)
        means = np.zeros((samples.shape[2], samples.shape[1]))
        # means of shape (num_dependent_vars, num_datapoints)
        stds = np.zeros((samples.shape[2], samples.shape[1]))

        # HOW TO SPEED THIS UP?
        for i in range(samples.shape[2]):   # Looping over each dependent var
            for j in range(samples.shape[1]):  # Looping over each datapoint
                means[i, j] = samples[:, j, i].mean()
                stds[i, j] = samples[:, j, i].std()

        # Finally, swap the last axis of samples to the first; makes it cleaner
        # when referencing later on
        samples = np.moveaxis(samples, 2, 0)
        return samples, means, stds

    def run_sampling(self, X, Y, **kwargs):
        """
        Method to call the sampling method, and also return the associated
        correct value for each prediction.

        Parameters
        ----------
        X : torch.Tensor
            Input X values (features) to be passed through the network
        Y : torch.Tensor
            The true y values (labels) of each datapoint
        **kwargs :
            DESCRIPTION.

        Returns
        -------
        samples : NumPy array
            All individual samples generated by successive stochastic forward
            passes through the network.
            If one output, of shape (#samples, #datapoints)
            If more than one output, of shape (#outputs, #samples, #datapoints)
        means : NumPy array
            Array of the mean prediction of each datapoint, based on the
            average of all forward passes for that point.
            Of shape (#outputs, #datapoints)
        stds : NumPy array
            Array of the standard deviation of each datapoint, based on the
            std of all forward passes for that point.
            Of shape (#outputs, #datapoints)
        Y_np : NumPy array
            The true Y values for all datapoints, converted to a NumPy array.
            Of shape (#datapoints,)

        """

        samples, means, stds = self.generate_samples(X)

        samples = np.squeeze(samples)

        if self.output_dim > 1:
            Y_np = np.swapaxes(
                pre.unnormalise(
                    Y.squeeze().numpy(),
                    self.y_mean,
                    self.y_std),
                1,
                0)
        else:
            Y_np = pre.unnormalise(
                Y.squeeze().numpy(), self.y_mean, self.y_std)
        return samples, means, stds, Y_np

    def make_prediction(self, x_value, verbose=False, plots=False, **kwargs):
        """
        Make a single prediction and return the mean and uncertainty
        Note: num_samples is a kwarg so we can keep calling a generic
        function in duq.pre

        Parameters
        ----------
        x_value : TYPE
            DESCRIPTION.
        model : TYPE
            DESCRIPTION.
        data_mean : TYPE
            DESCRIPTION.
        data_std : TYPE
            DESCRIPTION.
        verbose : TYPE, optional
            DESCRIPTION. The default is False.
        plots : TYPE, optional
            DESCRIPTION. The default is False.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        samples_pred : TYPE
            DESCRIPTION.
        means_pred : TYPE
            DESCRIPTION.
        stds_pred : TYPE
            DESCRIPTION.

        """

        if verbose:
            print(f"Input X: {x_value.numpy()}")
        x_value = torch.Tensor(
            pre.normalise(
                x_value,
                self.x_mean,
                self.x_std)).float()

        # Returns unnormalised data
        samples_pred, means_pred, stds_pred = self.generate_samples(x_value)

        if plots:
            sns.kdeplot(data=samples_pred[0])
            sns.rugplot(data=samples_pred[0])
            plt.xlabel("Frequency (Hz)")
            plt.grid()
            plt.title("Posterior prediction for a single X input\nKDE with rug") # noqa

        if verbose:
            err_percent = 100 * stds_pred.item() / means_pred.item()
            print(f"Using:\n\tX (normalised) = {x_value.numpy()}\n\tNumber of Samples: {len(self.nets)}\n\tMean prediction: {means_pred.item():.3f}\n\tStandard deviation: +/-{stds_pred.item():.3f} sigma ({err_percent:.2f}% of prediction)\n\n") # noqa
        return samples_pred, means_pred, stds_pred
