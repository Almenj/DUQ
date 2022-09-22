""" Archie Luxton, aol21, https://github.com/ese-msc-2021/irp-aol21, 2022 """
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import random
import time
from tqdm import tqdm                               # Progress bar
from statsmodels.graphics.tsaplots import plot_acf  # For autocorrelation
import scipy.stats.kde as kde                       # For the HDI plots
import pickle                                       # For saving/loading obj
sns.set_style('white')
sns.set_context('talk')
np.random.seed(123)


def load_data(name):
    with open(name, "rb") as fp:
        return pickle.load(fp)


class BLR:
    """ Bayesian Linear Regression
    Args:
        N: Number of datapoints to generate
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.X = dataset[0]
        self.Y = dataset[1]

    def create_dataset(self, N=50,
                       true_theta0=2,
                       true_theta1=4,
                       true_sd=2):
        """
        Create a new dataset based on
        f(x) = true_theta0 + true_theta1*x + N(0, true_sd)

        Args:
            N: Number of datapoints to generate (int)
            true_theta0: intersept of true data (float)
            true_theta1: slope of true data (float)
            true_sd: standard deviation of the error (centered at 0) (float)

        Returns:
            X: All x_i values up to i=N (numpy float array)
            Y: All y_i values up to i=N (numpy float array)
        """
        # Sample N number of x values that have a mean of 10 ..
        # and a std dev of sqrt(5)
        X = norm(10, np.sqrt(5)).rvs(N)

        # Sample N number of y values based on slope and intersept ..
        # plus a noise value
        Y = true_theta0 + true_theta1*X + norm(0, true_sd).rvs(N)

        # Create true parameter value array (used later for plotting)
        true_params = [true_theta0, true_theta1, true_sd]
        self.true_params = true_params

        dataset = np.array([X, Y])

        return dataset

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
        # Extract our different parameters
        theta0 = params[0]   # Intersept
        theta1 = params[1]   # Slope
        sd = params[2]   # Standard deviation

        # Get our predicted y values based on these guesses
        # Results in vector of length N (number of datapoints)
        y_hat = theta0 + theta1*X

        # Calculate the log likelihoods
        loglikelihoods = np.log(norm(y_hat, sd).pdf(Y))

        # If there are any -inf values in the array, make them 0
        loglikelihoods[loglikelihoods == float('-inf')] = 0

        # Sum up the log likelihoods (because they're logs)
        sum_ll = np.sum(loglikelihoods)

        # Priors
        theta0_prior = np.log(norm(prior_mean, prior_sd).pdf(theta0))
        theta1_prior = np.log(norm(prior_mean, prior_sd).pdf(theta1))
        sd_prior = np.log(norm(prior_mean, prior_sd).pdf(sd))

        numerator = sum_ll + theta0_prior + theta1_prior + sd_prior
        return numerator

    def MCMC_Metropolis_Hastings(self,
                                 theta=[1, 2, 1],
                                 proposal_width=[0.5, 0.1, 0.25],
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
            iterations: How many iterations to perform (int)
            burnin: How many iterations to throw away at the start (int)
            X: X data (numpy float array)
            Y: Y data (numpy float array)

        Returns:
            trace: The value of each parameter in each iteration.
            Size of array is [n_iterations,n_parameters] (numpy float array)
        """

        X = self.X
        Y = self.Y

        # theta = initial mean values for theta0, theta1 and sd
        # Initiliase the chain
        chain = np.empty((iterations+1, len(theta)))
        chain[0, :] = theta
        acceptance = np.zeros((iterations, len(theta)))

        start = time.time()
        for i in tqdm(range(iterations)):
            # Take a random sample from each of the new probability
            # distributions (for theta0, theta1 and sd)
            # Each distribution is centered at the previous value in the chain
            theta_star = norm(chain[i, :], proposal_width).rvs(3)

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
            for k in range(len(theta)):
                # Calculate a random uniform number between 0 and 1
                # to compare against
                rand_num = random.uniform(0, 1)

                if rand_num < r[k]:
                    chain[i+1, k] = theta_star[k]
                    acceptance[i, k] = 1
                else:
                    chain[i+1, k] = chain[i, k]

        # Print acceptance rate and time taken
        end = time.time()
        time_taken = end-start
        print("Acceptance rate: ", np.mean(acceptance[burnin:iterations]))
        print(f"Time taken: {time_taken:.2f} seconds")
        self.chain = chain[burnin:iterations]

        if save:
            with open(save_name, "wb") as fp:
                pickle.dump(self.chain, fp)

        return chain

    def MCMC_Metropolis(self,
                        theta=[1, 2, 1],
                        proposal_width=[0.5, 0.1, 0.25],
                        iterations=10_000,
                        burnin=1_000,
                        save=False,
                        save_name="saved_MCMC"):
        X = self.X
        Y = self.Y

        # Initiliase the chain
        chain = np.empty((iterations+1, len(theta)))
        chain[0, :] = theta
        acceptance = np.zeros(iterations)

        start = time.time()
        for i in tqdm(range(iterations)):
            # Take a random sample from each of the new probability
            # distributions (for theta0, theta1 and sd)
            # Each distribution is centered at the previous value in the chain
            theta_star = norm(chain[i, :], proposal_width).rvs(3)

            # Calculate an acceptance rate based on the posterior for
            # proposed position and previous position
            r = np.exp(self.numerator(theta_star, X, Y) -
                       self.numerator(chain[i, :], X, Y))

            # Calculate a random uniform number between 0 and 1 to
            # compare against
            rand_num = random.uniform(0, 1)

            # If our acceptance rate is bigger than the random number,
            # accept the new position and add it to the chain
            if rand_num < r:
                chain[i+1, :] = theta_star
                acceptance[i] = 1

            # If the acceptance rate is smaller than the random number,
            # keep the old value in the chain
            else:
                chain[i+1, :] = chain[i, :]

        # Print acceptance rate and time taken
        end = time.time()
        time_taken = end-start
        print("Acceptance rate: ", np.mean(acceptance[burnin:iterations]))
        print(f"Time taken: {time_taken:.2f} seconds")
        self.chain = chain[burnin:iterations]

        if save:
            with open(save_name, "wb") as fp:
                pickle.dump(self.chain, fp)

        return self.chain

    def save(self, name):
        with open(name, 'wb') as file:
            pickle.dump(self, file)

    def load(name):
        with open(name, 'rb') as file:
            return pickle.load(file)


class BLR_plots:
    def __init__(self, chain, param_list=["theta_0", "theta_1", "sd"]):
        self.chain = chain
        self.theta_0 = chain[:, 0]
        self.theta_1 = chain[:, 1]
        self.sd = chain[:, -1]
        self.params = ["theta_0", "theta_1", "sd"]
        self.true_params = [-999, -999, -999]

    def trace(self):
        """
        Plot the trace and belief plot
        """

        xticks = np.arange(len(self.chain[:, 0]))
        f, axs = plt.subplots(self.chain.shape[1], 2, figsize=(15, 15))

        for i in range(self.chain.shape[1]):
            # Histogram and density plot
            if self.true_params != [-999, -999, -999]:
                hist_title = (self.params[i] +
                              " mean = " +
                              str(np.round(np.mean(self.chain[:, i]), 2)) +
                              " (true = " +
                              str(self.true_params[i]) + ")")
            else:
                hist_title = (self.params[i] +
                              " mean = " +
                              str(np.round(np.mean(self.chain[:, i]), 2)))

            sns.histplot(self.chain[:, i],
                         ax=axs[i, 0],
                         kde=True).set(title=hist_title)

            # Trace plot
            axs[i, 1].plot(xticks, self.chain[:, i], linewidth=0.2)
            trace_title = self.params[i] + " Trace"
            axs[i, 1].title.set_text(trace_title)
        plt.show()

    def autocorr(self, lags=100):
        """
        Plot the autocorrelation
        From: https://towardsdatascience.com/advanced-time-series-analysis-
        in-python-decomposition-autocorrelation-115aa64f475e

        Args:
            lags: How many lags to plot

        Returns:
            None
        """
        f, axs = plt.subplots(self.chain.shape[1], figsize=(15, 15))
        for i in range(self.chain.shape[1]):
            plot_title = self.params[i] + " Autocorrelation"
            plot_acf(self.chain[:, i],
                     lags=lags,
                     fft=True,
                     use_vlines=True,
                     ax=axs[i])
            axs[i].title.set_text(plot_title)
            plt.xlabel("Lag at k")
            plt.ylabel("Correlation coefficient")
        plt.show()

    # Sample some parameters from the distributions to draw some lines
    def curves(self,
               X,
               Y,
               num_lines=1000,
               line_opacity=0.01,
               fill_opacity=0.5,
               linewidth=1,
               mode=1,
               interval=True,
               alpha=0.95,
               xmin=-9999,
               xmax=9999,
               fill_colour='green'):
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
            mode: 1 = sample from new normal distributions with a mean and std
                  2 = sample directly from the chain
            interval: Whether to draw a confidence interval or not (bool)
            conf: How many std devs from the mean to plot the interval (int)
            xmin: Minimum x value to plot the new samples (float)
            xmax: Maximum x value to plot the new samples (float)

        Returns:
            None
        """
        # If we haven't entered x limits
        if xmin == -9999 and xmax == 9999:
            xmin = np.min(X)*0.1
            xmax = np.max(X)*2

        # Set up a new figure
        plt.figure(figsize=(10, 10))

        theta0_mean, theta0_sd = norm.fit(self.chain[:, 0])
        theta1_mean, theta1_sd = norm.fit(self.chain[:, 1])

        # Distributions:
        theta0_dist = norm(theta0_mean, theta0_sd)
        theta1_dist = norm(theta1_mean, theta1_sd)

        # Sample from the distributions to draw a bunch of lines
        xs = np.linspace(xmin, xmax)
        if mode == 1:  # Sample from some new normal distributions
            for i in range(num_lines):
                ys = theta0_dist.rvs() + theta1_dist.rvs()*xs
                plt.plot(xs, ys, 'k-', linewidth=linewidth, alpha=line_opacity)
        elif mode == 2:  # Sample from the chain itself
            for i in range(num_lines):
                ys = random.choice(self.chain[:, 0]) + \
                    random.choice(self.chain[:, 1])*xs
                plt.plot(xs, ys, 'k-', linewidth=linewidth, alpha=line_opacity)

        if interval:
            y_top = np.empty(len(xs))
            y_bottom = np.empty(len(xs))
            for i in range(len(xs)):
                y_bottom[i] = norm.interval(alpha=alpha,
                                            loc=theta0_mean +
                                            theta1_mean*xs[i])[0]
                y_top[i] = norm.interval(alpha=alpha,
                                         loc=theta0_mean +
                                         theta1_mean*xs[i])[1]
            plt.fill_between(xs,
                             y_top,
                             y_bottom,
                             alpha=fill_opacity,
                             color=fill_colour,
                             linewidth=0,
                             label=f"{alpha*100}% Confidence Interval")

        # Plot a line at the mean values of the distributions
        y_mean = np.mean(self.chain[:, 0]) + np.mean(self.chain[:, 1])*xs
        plt.plot(xs, y_mean, 'k-', linewidth=linewidth, label="Mean Value")

        plt.plot(X, Y, '.', label="Raw data")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show()

    def bivariate(self):
        """
        Bivariate plot showing the average mean value for theta0 and theta1
        throughout the iterations.

        """
        plt.figure(figsize=(10, 10))
        N = self.chain.shape[0]
        plt.plot([np.mean(self.chain[:i, 0]) for i in range(N)],
                 [np.mean(self.chain[:i, 1]) for i in range(N)],
                 '.-', linewidth=0.1, markersize=0.5)  # Plot all points
        plt.plot(self.chain[0, 0],
                 self.chain[0, 1],
                 'og',
                 label="Start")  # Plot the starting point
        plt.plot(np.mean(self.chain[:-1, 0]),
                 np.mean(self.chain[:-1, 1]),
                 'ob',
                 label="End")  # Plot the final value
        plt.plot(self.true_params[0],
                 self.true_params[1],
                 'or',
                 label="True value")
        plt.xlabel("$\theta_0$")
        plt.ylabel("$\theta_1$")
        plt.title("Mean Value of $\theta_0$ and "
                  "$\theta_1$ Throughout MCMC Iterations")
        plt.legend()
        plt.grid()
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
        # dist_theta_0 = norm(np.mean(chain[:,0]), np.std(chain[:,0]))
        # dist_theta_1 = norm(np.mean(chain[:,1]), np.std(chain[:,1]))

        return chain[:, 0] + chain[:, 1]*x

    # Plot a posterior prediction
    def posterior_prediction(self,
                             x,
                             bin_width=1,
                             alpha=0.05):
        """
        Plot a prediction from the posterior.
        Calculates the posterior using calc_posterior_prediction,
        then compares this with the true mean (if we have it) and draws a
        Highest Density Interval of interval (1-alpha) based on the function
        hpd_grid.

        Args:
            x: The target we want to make a prediction for (float)
            bin_width: Affects the line density plot resolution (float)
            true_val: The true label value for our target (float)
            alpha: Desired probability of type I error (defaults to 0.05)

        Returns:
            None
        """

        # TO DO: Allow the user to enter a function?
        # Crunch some numbers
        true_val = self.true_params[0] + self.true_params[1]*x
        posterior = self.calc_posterior_prediction(self.chain, x)
        mean = np.mean(posterior)
        sd = np.std(posterior)
        xmin = mean-6*sd
        xmax = mean+6*sd

        # Plot
        sns.displot(x=posterior,
                    kind="kde",
                    bw_adjust=bin_width,
                    height=10,
                    aspect=1).set(title=f"Posterior Prediction for"
                                  f"$f(x={x}) = {mean:.2f}$")
        plt.xlim(xmin, xmax)  # 6 sigma either side of the mean
        # Extract the max value from the plot
        density_vals = [h.get_height()
                        for h in sns.distplot(posterior, bins=200).patches]
        plt.vlines(mean,
                   ymin=0,
                   ymax=np.max(density_vals),
                   colors='red',
                   linestyles='--',
                   label="Predicted Value")

        # Plot the true value if we've entered it
        if true_val != -999:
            plt.vlines(true_val,
                       ymin=0,
                       ymax=np.max(density_vals),
                       colors='yellow',
                       linestyles='-',
                       label="True Value")

            plt.title(f"Posterior Prediction for f(x={x}) = "
                      f"{mean:.2f}\nTrue Value = ${true_val:.2f}$")

        # Plot the HDI (Highest Density Interval)
        HDI_ylocation = (np.max(density_vals)/20)
        HDI = self.hpd_grid(sample=posterior, alpha=alpha, roundto=2)[0][0]

        # HDI = norm.interval(alpha=1-alpha, loc=posterior)[0]  # Doesn't work
        # HDI = [np.min(HDI), np.max(HDI)]
        HDI_width = HDI[1] - HDI[0]
        plt.hlines(HDI_ylocation,
                   xmin=HDI[0],
                   xmax=HDI[1],
                   colors='black',
                   linestyles='-',
                   linewidth=5,
                   label=f"{(1-alpha)*100}% HDI")
        plt.text(HDI[0]-HDI_width/5,
                 HDI_ylocation+(HDI_ylocation/8),
                 f"{HDI[0]}")
        plt.text(HDI[1],
                 HDI_ylocation+(HDI_ylocation/8),
                 f"{HDI[1]}")

        plt.legend()
        plt.show()

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
        xy_zipped = zip(x, y/np.sum(y))
        xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
        xy_cum_sum = 0
        hdv = []
        for val in xy:
            xy_cum_sum += val[1]
            hdv.append(val[0])
            if xy_cum_sum >= (1-alpha):
                break # noqa

        hdv.sort()
        diff = (upper-lower)/20  # differences of 5%
        hpd = []
        hpd.append(round(min(hdv), roundto))
        for i in range(1, len(hdv)):
            if hdv[i]-hdv[i-1] >= diff:
                hpd.append(round(hdv[i-1], roundto))
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

    # Load a LR object from file
    # Linear_Regression = BLR.load_data("5000")

    # Create a new BLR object with a dataset
    LR = BLR(BLR.create_dataset(BLR))

    # Sample the posterior to end up with a chain of all parameters
    chain = LR.MCMC_Metropolis_Hastings(iterations=1000, burnin=0)

    # Save the LR object to file
    # LR.save("LR_50000")

    # Load the LR object from file
    # LR = BLR.load("LR_Jupyter_20000")

    ########################
    # PLOTTING THE RESULTS #
    ########################

    # Instantiate a BLR_plots object based on the results of the MCMC
    Plots = BLR_plots(LR.chain)

    # Set the true parameters of the plots object (note: optional)
    Plots.true_params = LR.true_params

    # Plot the belief and trace plots
    Plots.trace()

    # Plot the autocorrelation plots
    Plots.autocorr(lags=1)

    # Plot the X,Y data as well as various curves sampled from the posterior
    Plots.curves(X=LR.dataset[0], Y=LR.dataset[1], num_lines=1000)

    # Plot the values of theta_0 and theta_1 as the iterations progress
    Plots.bivariate()

    # Make a prediction from the posterior distribution
    x = 20
    Plots.posterior_prediction(x=x, alpha=0.05)
