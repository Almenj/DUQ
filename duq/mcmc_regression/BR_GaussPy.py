""" Archie Luxton, aol21, https://github.com/ese-msc-2021/irp-aol21, 2022 """
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                               # Used for plotting
from scipy.stats import norm                        # Used throughout
# Used in creating dataset and MCMC
import random
import time                                         # Used for timing MCMC
from tqdm import tqdm                               # Progress bar
from statsmodels.graphics.tsaplots import plot_acf  # For autocorrelation
import scipy.stats.kde as kde                       # For the HDI plots
import pickle                                       # For saving/loading obj
import pandas as pd                             # Used only in BivariateGrid


import os                           # Used for animation
import glob                         # Used for animation
import cv2                          # Used for animation
# sns.set_style('white')
sns.set_theme()                           # Return sns to default style
# sns.set_context('talk')                   # Set style to talk
# sns.set_context('paper')
# sns.set_context('notebook')
sns.set_style("ticks")                    # Add ticks
np.random.seed(123)


def load_data(name):
    with open(name, "rb") as fp:
        return pickle.load(fp)


class BR_Gauss:
    """ Bayesian Linear Regression with Gaussians
    Args:
        N: Number of datapoints to generate
    """

    def __init__(self, dataset):
        # If we instantiated the class with a dataset, set the variables
        self.dataset = dataset
        self.X = dataset[0]
        self.Y = dataset[1]

    def numerator(self,
                  params,
                  X,
                  Y,
                  prior_mean=0.,
                  prior_sd=5.):
        """
        Calculate the numerator of Bayes' Theorem (using natural log)
        (i.e. the log likelihood multiplied by the log prior)

        Args:
            params: List of parameters including sd as the final entry (list)
            X: X data (numpy float array)
            Y: Y data (numpy float array)
            prior_mean: Mean value used for ALL the priors (float)
            prior_sd: Standard deviation used for ALL the priors (float)

        Returns:
            numerator: Product of log likelihood and log prior (float)
        """

        # Get our predicted y values based on these guesses
        # Results in vector of length N (number of datapoints)

        # Assuming params are arranged as:
        # [gauss1_mean, gauss1_sd, gauss2_mean, gauss2_sd, .... sd]
        y_hat = np.zeros(len(X))

        # Stack each gaussian on top of each other
        if self.scaling:
            for i in range(0, (len(params) - 1), 3):
                y_hat += params[i + 2] * \
                    np.exp(-((X - params[i])**2 / (2 * params[i + 1])))
        else:
            for i in range(0, (len(params) - 1), 2):
                y_hat += np.exp(-((X - params[i])**2 / (2 * params[i + 1])))

        # Calculate the log likelihoods
        loglikelihoods = np.log(norm(y_hat, params[-1]).pdf(Y))

        # If there are any -inf values in the array, make them 0
        loglikelihoods[loglikelihoods == float('-inf')] = 0

        # Sum up the log likelihoods (because they're logs)
        sum_ll = np.sum(loglikelihoods)

        if sum_ll == float("inf"):
            print("iteration discarded due to -inf log likelihood!")

        # Priors - Sum up all the theta_N, but also the standard deviation
        prior_sum = 0
        for i in range(len(params)):
            prior_sum += np.log(norm(0, 5).pdf(params[i]))

        return sum_ll + prior_sum

    def MCMC_Metropolis_Hastings(self,
                                 num_gaussians=0,
                                 scaling=False,
                                 scale_init=1,
                                 sd_init=0.1,
                                 initial_params=[],
                                 proposal_width=[],
                                 iterations=10_000,
                                 burnin=1_000,
                                 save=False,
                                 save_name="saved_MCMC"):
        """
        Perform a number of iterations of the MCMC (Metropolis Hastings)
        algorithm. Adapted from
        http://gradientdescending.com/metropolis-hastings-algorithm-from-scratch

        Args:
            theta: List of parameters that will form the first guess (list)
            proposal_width: SD of how far we can search in each iter (float)
            initial_params: Array of the initial guesses to start the MCMC
                chain. Should be length of Poly_degree + 2 (to include the
                intersept and the sd term)
            iterations: How many iterations to perform (int)
            burnin: How many iterations to throw away at the start (int)
            X: X data (numpy float array)
            Y: Y data (numpy float array)
            save: Whether to save the results of the chain (bool)
            save_name: Save name of the chain file

        Returns:
            trace: The value of each parameter in each iteration.
            Size of array is [n_iterations,n_parameters] (numpy float array)
        """

        X = self.X
        Y = self.Y
        self.num_gaussians = num_gaussians
        self.scaling = scaling
        if scaling:
            num_parameters = num_gaussians * 3 + 1
        else:
            num_parameters = num_gaussians * 2 + 1
        print("number of parameters = ", num_parameters)

        # If there are no set initial mean values
        # Guess 0 for everything, and 0.25 proposal width
        if not initial_params:
            # Populate all theta parameters as zeros
            # Evenly space centers of gaussians
            initial_params = np.linspace(X.min(), X.max(), num_parameters)
            if scaling:
                # Scaling factor of 1 for all gaussian
                initial_params[2::3] = scale_init
                # Make standard deviation =1 for all gaussians
                initial_params[1::3] = sd_init
            else:
                # Make standard deviation =1 for all gaussians
                initial_params[1::2] = sd_init
            initial_params[-1] = 2   # Change sd (aleatoric noise) to 1
            print("Initial parameters: \n", initial_params)
            # Populate the sd as 2
        else:
            if len(initial_params) != ((num_gaussians * 2) + 1):
                stri = "Please ensure that the initial_params array is of "
                "size Poly_degree + 2 (to include intersept and sd terms)"
                raise ValueError(stri)
        if not proposal_width:
            # Populate the proposal widths as 0.25
            proposal_width = [0.25 for i in range(num_parameters)]

        # Initiliase the chain
        chain = np.empty((iterations + 1, len(initial_params)))
        chain[0, :] = initial_params
        acceptance = np.zeros((iterations, len(initial_params)))

        start = time.time()
        for i in tqdm(range(iterations)):
            # Take a random sample from each of the new probability
            # distributions (for theta0, theta1 and sd)
            # Each distribution is centered at the previous value in the chain
            theta_star = norm(chain[i, :], proposal_width).rvs(
                len(initial_params))

            # Make sure all standard deviations of the gaussians are positive
            if self.scaling:
                theta_star[1::3] = np.sqrt(theta_star[1::3]**2)
            else:
                theta_star[1::2] = np.sqrt(theta_star[1::2]**2)

            # Calculate an acceptance rate based on the posterior for
            # proposed position and previous position
            r = np.exp((self.numerator(theta_star, X, Y) -
                        np.log(norm(chain[i, :],
                                    proposal_width).pdf(theta_star))) -
                       (self.numerator(chain[i, :], X, Y)) -
                       np.log(norm(theta_star,
                                   proposal_width).pdf(chain[i, :])))

            # If our acceptance rate is bigger than the random number,
            # accept the new position and add it to the chain
            for k in range(len(initial_params)):
                # Calculate a random uniform number between 0 and 1
                # to compare against
                rand_num = random.uniform(0, 1)

                if rand_num < r[k]:
                    chain[i + 1, k] = theta_star[k]
                    acceptance[i, k] = 1
                else:
                    chain[i + 1, k] = chain[i, k]

        # Print acceptance rate and time taken
        end = time.time()
        time_taken = end - start
        print("Acceptance rate: ", np.mean(acceptance[burnin:iterations]))
        print(f"Time taken: {time_taken:.2f} seconds")

        self.chain = chain[burnin:iterations]

        if save:
            with open(save_name, "wb") as fp:
                pickle.dump(self.chain, fp)

        return chain

    def save(self, name):
        with open(name, 'wb') as file:
            pickle.dump(self, file)

    def load(name):
        with open(name, 'rb') as file:
            return pickle.load(file)


class BR_Gauss_plots:
    def __init__(self, chain, param_list=[
                 "theta_0", "theta_1", "sd"], **kwargs):
        self.chain = chain

        if 'scaling' in kwargs:
            self.scaling = kwargs.pop('scaling', '')

    def eval_func(self, params, x):
        """
        Evaluate f(x) given the list of gaussian properties.
        Args:
            params: [B_0, B_1... B_n, sd] (numpy array or list of floats)
            x: Point(s) at which to evaluate the function (float or np array)
        """
        y = np.zeros(len(x))

        if self.scaling:
            for i in range(0, (len(params) - 1), 3):
                y += params[i + 2] * \
                    np.exp(-((x - params[i])**2 / (2 * params[i + 1])))
        else:
            for i in range(0, (len(params) - 1), 2):
                y += np.exp(-((x - params[i])**2 / (2 * params[i + 1])))
        return y

    def parameter_labels(self, **kwargs):
        """
        Return a list of the parameter names
        """
        params_label = []
        for i in range(self.chain.shape[1]):
            if i < (self.chain.shape[1] - 1):
                # Only include sign in first term if it's negative
                params_label.append(f"theta_{i}")
            else:
                # Always include the sign after first term
                params_label.append("sd")
        return params_label

    def trace(self, **kwargs):
        """
        Plot the trace and belief plot
        """
        params_label = self.parameter_labels()
        print("Parameters: ", params_label)

        xticks = np.arange(len(self.chain[:, 0]))
        f, axs = plt.subplots(self.chain.shape[1], 3, figsize=(25, 25))

        # Step through the different paramters
        for i in range(self.chain.shape[1]):
            # Histogram and density plot
            if 'true_params' in dir(self):
                hist_title = (params_label[i] + " mean = " +
                              str(np.round(np.mean(self.chain[:, i]), 2)) +
                              " (true = " + str(self.true_params[i]) + ")")
            else:
                hist_title = (params_label[i] + " mean = " +
                              str(np.round(np.mean(self.chain[:, i]), 2)))

            # Belief Plot
            sns.histplot(y=self.chain[:, i],
                         ax=axs[i, 0],
                         kde=True).set(title=hist_title)

            # Trace plot
            axs[i, 1].plot(xticks, self.chain[:, i], linewidth=0.2)
            trace_title = params_label[i] + " Trace"
            axs[i, 1].title.set_text(trace_title)
            y_lims = axs[i, 1].get_ylim()

            # Mean value vs iterations plot
            mean = np.zeros(self.chain.shape[0])
            # Step through iterations
            for j in range(1, self.chain.shape[0]):
                mean[j] = np.mean(self.chain[0:j, i])
            axs[i, 2].plot(xticks, mean, linewidth=1)
            axs[i, 2].set_ylim(y_lims)
            trace_title = params_label[i] + " Chain Mean vs Iteration Count"
            axs[i, 2].title.set_text(trace_title)

        if 'savename' in kwargs:
            savename = kwargs.pop('savename', '')
            plt.savefig(savename, bbox_inches='tight')

        plt.show()

    def autocorr(self, lags=100, **kwargs):
        """
        Plot the autocorrelation
        From: https://towardsdatascience.com/advanced-time-series-analysis-
        in-python-decomposition-autocorrelation-115aa64f475e

        Args:
            lags: How many lags to plot

        Returns:
            None
        """
        if lags > self.chain.shape[0]:
            print("\n\n## AUTOCORR WARNING: NUMBER OF LAGS SHOULD BE EQUAL OR"
                  " MORE THAN THE NUMBER OF ITERATIONS IN CHAIN "
                  "## \n## ABORING AUTOCORR ##\n\n")
            return 0

        params = []
        for i in range(self.chain.shape[1]):
            if i < (self.chain.shape[1] - 1):
                # Only include sign in first term if it's negative
                params.append(f"theta_{i}")
            else:
                # Always include the sign in string conversion after first term
                params.append("sd")

        f, axs = plt.subplots(self.chain.shape[1], figsize=(15, 25))
        for i in range(self.chain.shape[1]):
            plot_title = params[i] + " Autocorrelation"
            plot_acf(self.chain[:, i],
                     lags=lags,
                     fft=True,
                     use_vlines=True,
                     ax=axs[i])
            axs[i].title.set_text(plot_title)
            plt.xlabel("Lag at k")
            plt.ylabel("Correlation coefficient")

        if 'savename' in kwargs:
            savename = kwargs.pop('savename', '')
            plt.savefig(savename, bbox_inches='tight')

        plt.show()

    def plot_gaussians(self,
                       params,
                       x0,
                       x1):

        x = np.linspace(x0, x1, 500)
        # y0 = np.zeros(len(x))

        if self.scaling:
            for i in range(0, (len(params) - 1), 3):
                y = np.zeros(len(x))
                y += params[i + 2] * \
                    np.exp(-((x - params[i])**2 / (2 * params[i + 1])))
                plt.fill_between(x, y, 0, alpha=0.5)
        else:
            for i in range(0, (len(params) - 1), 2):
                y = np.zeros(len(x))
                y += np.exp(-((x - params[i])**2 / (2 * params[i + 1])))
                plt.fill_between(x, y, 0, alpha=0.5)

    def animate_curves(self,
                       videoname,
                       frames,
                       framerate,
                       X,
                       Y,
                       plot_gaussians=True,
                       num_lines=0,
                       line_opacity=0.01,
                       fill_opacity=0.5,
                       linewidth=1,
                       interval=True,
                       uncert_meth="std",
                       uncertainty=1,
                       xmin=-9999,
                       xmax=9999,
                       ymin=-9999,
                       ymax=9999,
                       conf_resolution=100,
                       fill_colour='green',
                       **kwargs):

        savepath = "C:/Users/aolux/Documents/irp-aol21/curve_animations/"

        # Delete all files in folder
        files = glob.glob(savepath + "*.png")
        for f in files:
            os.remove(f)

        # If the folder doesn't exist, make it
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        """
        Produces an animation for the gaussian regression
        """
        num_iters = self.chain.shape[0]
        end_frame_arr = np.linspace(1, num_iters, frames).astype(int)

        for ind, ends in enumerate(end_frame_arr):
            chain_temp = self.chain[:ends]
            if self.scaling:
                num_gaussians = (chain_temp.shape[1] - 1) / 3
            else:
                num_gaussians = (chain_temp.shape[1] - 1) / 2  # noqa
            # If we haven't entered x limits
            if xmin == -9999 and xmax == 9999:
                xmin = np.min(X) * 0.1
                xmax = np.max(X) * 2

            # If we haven't entered y limits
            if xmin == -9999 and xmax == 9999:
                ymin = np.min(Y)
                ymax = np.max(Y)

            # Define an array of mean values of each parameter
            # (theta0, theta1.... thetaN, sd)
            means = [np.mean(chain_temp[:, i])
                     for i in range(chain_temp.shape[1])]

            plt.figure()
            plt.ioff()

            # Plot the constituent gaussians
            if plot_gaussians:
                self.plot_gaussians(means, xmin, xmax)

            # Plot the mean predicted value of y
            xs = np.linspace(xmin, xmax, conf_resolution)
            ys_mean = self.eval_func(means, xs)
            plt.plot(xs,
                     ys_mean,
                     "k-",
                     linewidth=linewidth,
                     label="Mean Predicted Value")

            # Plot the true value
            if 'true_params' in dir(self):
                ys_exact = self.eval_func(self.true_params, xs)
                plt.plot(xs,
                         ys_exact,
                         "r-.",
                         linewidth=linewidth,
                         label="True Exact Value")

            if interval:
                # Determine the posterior distribution at every x point
                # Shape: (n_datapoints, n_MCMC_iterations)
                posterior = self.calc_posterior_prediction_gaussian(
                    chain_temp, xs)

                # Make new arrays for the top and bottom curves
                y_top = np.zeros(len(xs))
                y_bottom = np.zeros(len(xs))

                # Cycle through all x datapoints
                if uncert_meth == "std":
                    for i in range(len(xs)):
                        std_dev = np.std(posterior[i, :])
                        y_top[i] = ys_mean[i] + uncertainty * std_dev
                        y_bottom[i] = ys_mean[i] - uncertainty * std_dev
                    chart_label = f"$\\pm{uncertainty}\\sigma$ Confidence "
                    "Interval"
                elif uncert_meth == "HPD":
                    for i in range(len(xs)):
                        HPD = self.hpd_grid(sample=posterior[i, :],
                                            alpha=(1 - uncertainty),
                                            roundto=2)[0][0]
                        y_top[i] = HPD[0]
                        y_bottom[i] = HPD[1]
                    chart_label = f"${uncertainty*100}%$ Confidence Interval"
                plt.fill_between(xs,
                                 y_top,
                                 y_bottom,
                                 alpha=fill_opacity,
                                 color=fill_colour,
                                 linewidth=0,
                                 label=chart_label)

            # Plot the rest
            plt.plot(X, Y, '.', label="Raw data")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xlim((xmin, xmax))
            plt.ylim((ymin, ymax))
            plt.title(f"Iteration {ends}")
            plt.legend()
            plt.grid()

            prefix = "0" * (6 - len(str(ind)))  # Works up to 1,000,000 frames
            savename = savepath + "curves_animation" + \
                prefix + str(ind) + ".png"
            plt.ioff()  # Avoid displaying figure`
            plt.savefig(savename, bbox_inches='tight')
            # plt.gca()
            plt.close()

        images = sorted(filter(os.path.isfile,
                               glob.glob(savepath + '*.png')))

        moviename = savepath + videoname

        frame = cv2.imread(os.path.join(savepath, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(
            moviename, fps=framerate, fourcc=0, frameSize=(
                width, height))  # fourcc

        for image in images:
            video.write(cv2.imread(os.path.join(savepath, image)))

        cv2.destroyAllWindows()
        video.release()

    def curves(self,
               X,
               Y,
               plot_gaussians=True,
               num_lines=0,
               line_opacity=0.01,
               fill_opacity=0.5,
               linewidth=1,
               interval=True,
               uncert_meth="std",
               uncertainty=1,
               uncert_bounds=3,
               xmin=-9999,
               xmax=9999,
               ymin=-9999,
               ymax=9999,
               conf_resolution=100,
               fill_colour='green',
               **kwargs):
        """
        Plot a range of curves that have been sampled from the posterior
        distributions for each parameter.
        For each curve, a random sample is taken from the sampled posteriors
        for each parameter.

        Args:
            X: Raw X data (i.e. the targets) (numpy float array)
            Y: Raw Y data (i.e. the labels) (numpy float array)
            num_lines: Number of curves to sample from the posteriors.
                Default is 1000 (int)
            alpha: Opacity of each line. Default is 1% (float)
            linewidth: Width of each sampled curve. Default is 1 (float)
            interval: Whether to draw a confidence interval or not (bool)
            conf: How many std devs from the mean to plot the interval (int)
            xmin: Minimum x value to plot the new samples (float)
            xmax: Maximum x value to plot the new samples (float)
            ymin: Minimum y value to plot the new samples (float)
            ymax: Maximum y value to plot the new samples (float)
            conf_resolution: How many x values to use to plot the confidence
                             intervals. (int)
            fill_colour: What colour to fill in the confidence interval
                         (from matplotlib colour pallette)

        Returns:
            None
        """

        if self.scaling:
            num_gaussians = (self.chain.shape[1] - 1) / 3
        else:
            num_gaussians = (self.chain.shape[1] - 1) / 2 # noqa

        # If we haven't entered x limits
        if xmin == -9999 and xmax == 9999:
            xmin = np.min(X) * 0.1
            xmax = np.max(X) * 2

        # If we haven't entered y limits
        if xmin == -9999 and xmax == 9999:
            ymin = np.min(Y)
            ymax = np.max(Y)

        # Define an array of mean values of each parameter
        # (theta0, theta1.... thetaN, sd)
        means = [np.mean(self.chain[:, i]) for i in range(self.chain.shape[1])]

        # Set up a new figure
        figheight = 15
        if 'figheight' in kwargs:
            figheight = kwargs.pop('figheight', '')

        figwidth = 15
        if 'figwidth' in kwargs:
            figwidth = kwargs.pop('figwidth', '')

        fig = plt.figure(figsize=(figheight, figwidth), dpi=400)

        # Plot the constituent gaussians
        if plot_gaussians:
            self.plot_gaussians(means, xmin, xmax)

        # Sample from the distributions to draw a bunch of lines
        xs = np.linspace(xmin, xmax, conf_resolution)

        # Plot the mean predicted value of y
        ys_mean = self.eval_func(means, xs)
        plt.plot(xs,
                 ys_mean,
                 "k-",
                 linewidth=linewidth,
                 label="Mean Surrogate Prediction")

        # Plot the true value
        if 'true_params' in dir(self):
            ys_exact = self.eval_func(self.true_params, xs)
            plt.plot(xs,
                     ys_exact,
                     "r-.",
                     linewidth=linewidth,
                     label="True Exact Value")

        if interval:
            # Determine the posterior distribution at every x point
            # Shape: (n_datapoints, n_MCMC_iterations)
            posterior = self.calc_posterior_prediction(self.chain, xs)

            # Make new arrays for the top and bottom curves
            y_top = np.zeros(len(xs))
            y_bottom = np.zeros(len(xs))

            # Cycle through all x datapoints
            if uncert_meth == "std":
                for i in range(len(xs)):
                    std_dev = np.std(posterior[i, :])
                    y_top[i] = ys_mean[i] + uncertainty * std_dev
                    y_bottom[i] = ys_mean[i] - uncertainty * std_dev
                chart_label = f"$\\pm{uncertainty}\\sigma$ Confidence Interval"
            elif uncert_meth == "HPD":
                for i in range(len(xs)):
                    HPD = self.hpd_grid(sample=posterior[i, :],
                                        alpha=(1 - uncertainty),
                                        roundto=2)[0][0]
                    y_top[i] = HPD[0]
                    y_bottom[i] = HPD[1]
                chart_label = f"${uncertainty*100}%$ Confidence Interval"
            if uncert_bounds == 3 and uncert_meth == "std":
                # 3SD
                plt.fill_between(xs,
                                 y_top + 2 * std_dev,
                                 y_bottom - 2 * std_dev,
                                 alpha=fill_opacity,
                                 linewidth=0,
                                 label="$\\pm 3 \\sigma$ Confidence Interval")
                # 2SD
                plt.fill_between(xs,
                                 y_top + std_dev,
                                 y_bottom - std_dev,
                                 alpha=fill_opacity,
                                 linewidth=0,
                                 label="$\\pm 2 \\sigma$ Confidence Interval")
                # 1SD
            plt.fill_between(xs,
                             y_top,
                             y_bottom,
                             alpha=fill_opacity,
                             linewidth=0,
                             label=chart_label)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((xmin, xmax))
        plt.ylim((ymin, ymax))
        plt.legend()
        plt.grid()

        if 'savename' in kwargs:
            savename = kwargs.pop('savename', '')
            plt.savefig(savename, bbox_inches='tight')

        return fig
        plt.show()

    def running_mean(self, param1, param2, **kwargs):
        """
        Bivariate plot showing the average mean value for two parameters
        throughout the iterations.

        Args:
            param1: Index of the first parameter to plot
                    E.g. 0 = intersept, 1 = x coefficient
                    (int)
            param2: Index of the second parameter to plot
                    E.g. 0 = intersept, 1 = x coefficient
                    (int)
        """
        assert isinstance(param1, int)
        assert isinstance(param2, int)

        plt.figure(figsize=(10, 10))
        N = self.chain.shape[0]

        # Plot the 'walk' of the two parameters
        plt.plot([np.mean(self.chain[:i, param1]) for i in range(1, N)],
                 [np.mean(self.chain[:i, param2]) for i in range(1, N)],
                 '.-', linewidth=0.1, markersize=0.5)

        # Plot the starting point of our params
        plt.plot(self.chain[0, param1],
                 self.chain[0, param2],
                 'og',
                 label="Start")

        # Plot the final value of our params
        plt.plot(np.mean(self.chain[:-1, param1]),
                 np.mean(self.chain[:-1, param2]),
                 'ob',
                 label="End")

        # Plot the true value of the two params
        if 'true_params' in dir(self):
            plt.plot(self.true_params[param1],
                     self.true_params[param2],
                     'or',
                     label="True value")

        param_names = self.parameter_labels()
        plt.xlabel("$\theta_0$")
        plt.ylabel("$\theta_1$")
        plt.title(f"Mean Value of {param_names[param1]} and "
                  f"{param_names[param2]} Throughout MCMC Iterations")
        plt.legend()
        plt.grid()

        if 'savename' in kwargs:
            savename = kwargs.pop('savename', '')
            plt.savefig(savename, bbox_inches='tight')

        plt.show()

    # Return a single posterior prediction
    def calc_posterior_prediction(self, chain, x):
        """
        Add the slope and intersept parameters together to get the
        full posterior chain.

        Args:
            chain: The samples of theta_0, theta_1 and sd from MCMC (np array)
            x: The target value we want to evaluate/predict at (float)

        Returns:
            The posterior w.r.t. iteration number (numpy float array)
        """

        # Repeat this step in case the curves haven't been plotted
        n_iterations = chain.shape[0]
        n_parameters = chain.shape[1]

        # Firstly, if x is a single number
        if isinstance(x, int) or isinstance(x, float):
            n_datapoints = 1
            posterior = np.zeros(n_iterations)
            posterior[:] = chain[:, 0]
            if self.scaling:
                for j in range(0, (n_parameters - 1), 3):
                    posterior[:] += chain[:, j + 2] * \
                        np.exp(-((x - chain[:, j])**2 / (2 * chain[:, j + 1])))
            else:
                for j in range(0, (n_parameters - 1), 2):
                    posterior[:] += np.exp(-((x - chain[:, j])
                                           ** 2 / (2 * chain[:, j + 1])))

        # Otherwise, if x is an array
        else:
            n_datapoints = len(x)
            posterior = np.zeros((n_datapoints, n_iterations))
            for i in range(n_datapoints):
                posterior[i, :] = chain[:, 0]
                if self.scaling:
                    for j in range(0, (n_parameters - 1), 3):
                        posterior[i, :] += chain[:, j + 2] * \
                            np.exp(-((x[i] - chain[:, j]) **
                                   2 / (2 * chain[:, j + 1])))
                else:
                    for j in range(0, (n_parameters - 1), 2):
                        posterior[i, :] += np.exp(-((x[i] - chain[:, j])**2 /
                                                    (2 * chain[:, j + 1])))

        return posterior

    def bivariate_grid(self, **kwargs):
        """
        Produce a pairwise relationship plot using seaborn PairGrid
        https://seaborn.pydata.org/tutorial/distributions.html
        """
        # Populate a list of the column (parameter) names
        column_names = self.parameter_labels()
        df = pd.DataFrame(self.chain, columns=column_names)
        g = sns.PairGrid(df)
        g.map_upper(sns.histplot)
        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.histplot, kde=True)

        if 'savename' in kwargs:
            savename = kwargs.pop('savename', '')
            plt.savefig(savename, bbox_inches='tight')

    def hpd_grid(self, sample, alpha, roundto=2):
        """Calculate highest posterior density (HPD) of array for given alpha.
        The HPD is the minimum width Bayesian credible interval (BCI).
        The function works for multimodal distributions, returning more
        than one mode
        Parameters
        From: https://github.com/PacktPublishing/Bayesian-Analysis-
        with-Python/blob/master/Chapter%201/hpd%20(1).py
        ----------

        sample : Numpy array or python list
            An array containing MCMC samples
        alpha : float
            Desired probability of type I error (defaults to 0.05)
        roundto: integer
            Number of digits after the decimal point for the results
        Returns
        ----------
        hpd: array with the lower

        """
        sample = np.asarray(sample)
        sample = sample[~np.isnan(sample)]
        # get upper and lower bounds
        lower = np.min(sample)
        upper = np.max(sample)
        density = kde.gaussian_kde(sample)
        x = np.linspace(lower, upper, 2000)
        y = density.evaluate(x)
        # y = density.evaluate(x, lower, upper) waitting for PR to be accepted
        xy_zipped = zip(x, y / np.sum(y))
        xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
        xy_cum_sum = 0
        hdv = []
        for val in xy:
            xy_cum_sum += val[1]
            hdv.append(val[0])
            if xy_cum_sum >= (1 - alpha):
                break # noqa
        hdv.sort()
        diff = (upper - lower) / 20  # differences of 5%
        hpd = []
        hpd.append(round(min(hdv), roundto))
        for i in range(1, len(hdv)):
            if hdv[i] - hdv[i - 1] >= diff:
                hpd.append(round(hdv[i - 1], roundto))
                hpd.append(round(hdv[i], roundto))
        hpd.append(round(max(hdv), roundto))
        ite = iter(hpd)
        hpd = list(zip(ite, ite))
        modes = []
        for value in hpd:
            x_hpd = x[(x > value[0]) & (x < value[1])]
            y_hpd = y[(x > value[0]) & (x < value[1])]
            modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
        return hpd, x, y, modes


if __name__ == "__main__":

    #######################################
    # INSTANTIATING AND SOLVING THE MODEL #
    #######################################

    # If importing from a csv dataset
    dataset = np.genfromtxt(
        "C:/Users/aolux/Documents/irp-aol21/damped_wave.csv",
        delimiter=',')
    X = dataset[:, 0]
    Y = dataset[:, 1]
    dataset = [X, Y]
    BRG = BR_Gauss(dataset)  # Where dataset = [X, Y]
    scaling = True
    chain = BRG.MCMC_Metropolis_Hastings(
        num_gaussians=5,
        scaling=scaling,
        iterations=500,
        burnin=0)
    plt.figure(figsize=(3, 3), dpi=80)
    plt.plot(X, Y, 'ob', label="Raw (Measured) Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()

    # hi!

    ########################
    # PLOTTING THE RESULTS #
    ########################

    # Instantiate a BLR_plots object based on the results of the MCMC
    # Note: scaling is a kwarg
    Plots = BR_Gauss_plots(BRG.chain, scaling=scaling)

    # Set the true parameters of the plots object (note: optional)
    # Plots.true_params = PR.true_params
    # Plots.true_params=[0.5, -0.7, 0.6, -0.1, 1]

    # Plot the belief and trace plots
    Plots.trace(gaussian=True, scaling=scaling)

    # Plot the autocorrelation plots
    # Plots.autocorr(lags=50)

    # Plot the X,Y data as well as various curves sampled from the posterior
    # Plots.curves(X=PR.dataset[0],
    #             Y=PR.dataset[1],
    #             num_lines=0,
    #             interval=True,
    #             xmin=-10,
    #             xmax=10,
    #             uncert_meth="std",
    #             uncertainty=1,
    #             ymin=-15,
    #             ymax=15)

    Plots.curves(
        X,
        Y,
        ymin=-1,
        ymax=2.5,
        xmin=-1,
        xmax=3,
        figwidth=4,
        interval=True,
        figheight=4,
        uncertainty=1)
    Plots.curves(
        X,
        Y,
        ymin=-1,
        ymax=2.5,
        xmin=-1,
        xmax=3,
        figwidth=4,
        interval=True,
        figheight=4,
        uncertainty=2)
    Plots.curves(
        X,
        Y,
        ymin=-1,
        ymax=2.5,
        xmin=-1,
        xmax=3,
        figwidth=4,
        interval=True,
        figheight=4,
        uncertainty=6)

    Plots.animate_curves(
        videoname="mymovie.avi",
        frames=50,
        framerate=20,
        X=X,
        Y=Y,
        ymin=-1,
        ymax=2.5,
        xmin=-1,
        xmax=3,
        figwidth=4,
        interval=False,
        figheight=4,
        uncertainty=6)
