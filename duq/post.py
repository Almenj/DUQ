""" Archie Luxton, aol21, https://github.com/ese-msc-2021/irp-aol21, 2022 """
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import copy
from matplotlib import animation
from sklearn.decomposition import PCA
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d # noqa

# If the DUQ package is installed, import pre from it
# Otherwise, do a relative import (assuming pre is in the same folder as this)
try:
    from duq import pre
except ImportError or ModuleNotFoundError:
    import pre


def calculate_credibility(samples, interval, method):
    """
    Calculate credibility intervals.
    Args:
        Samples:
            Raw samples from the neural network output.
            Shape: (num_samples_drawn_from_NN, num_test_points, 1) (np array)
        Interval:
            E.g. if using HDI: 0.95 (for 95% Highest Density Interval)
            E.g. if using SD: 2 (for +/- 2 standard deviations)
        Method: Either:
            "HDI" - For Highest Density Interval
            "SD" - For Standard Deviation

    Returns:
        Lower, Upper:
            Lower and upper prediction ranges. Includes the
    """
    num_datapoints = samples.shape[1]
    means = np.zeros(num_datapoints)
    if method == "HDI":
        CI = np.zeros((num_datapoints, 2))
        for i in range(num_datapoints):
            means[i] = samples[:, i].mean()
            CI[i, :] = az.hdi(samples[:, i], hdi_prob=interval)
            lower = CI[:, 0]  # Lower confidence interval
            upper = CI[:, 1]  # Upper confidence interval

    elif method == "SD":
        stds = np.zeros(num_datapoints)
        for i in range(num_datapoints):
            means[i] = samples[:, i].mean()
            stds[i] = np.std(samples[:, i])
        # Lower confidence interval (when using SD)
        lower = means - interval * stds
        # Upper confidence interval (when using SD)
        upper = means + interval * stds

    return upper, lower, means

# Plotting error bar plot


def stats(datatype, upper, lower, means, stds, true, N, method="SD",
          uncert=1, sort=False, summary=True, plots=False, **kwargs):
    if sort:
        inds = np.argsort(true, axis=-1)
        true = true[inds]
        upper = upper[inds]
        lower = lower[inds]
        means = means[inds]

    if 'xs' in kwargs:
        xs = kwargs.pop('xs', '')
    else:
        xs = np.arange(0, len(true[:N]))

    # PRODUCE ERROR BAR PLOTS
    if plots:
        # Offset [upper, lower] from the mean value
        asymmetric_error = [(upper[:N] - means[:N]), (means[:N] - lower[:N])]

        plt.figure(figsize=(15, 9))
        plt.errorbar(xs, means[:N], yerr=asymmetric_error, fmt='o')
        plt.plot(xs, true[:N], 'xr', label="True data")
        if method == "HDI":
            plt.title(f"A selection of the first {N} {datatype} points. "
                      f"Using {uncert*100:.0f}% HDI")
        elif method == "SD":
            plt.title(
                f"A selection of the first {N} {datatype} points. "
                f"Using {uncert} sigma")
        plt.ylabel("Concrete Compressive Strength (MPa)")
        if sort:
            plt.xlabel(
                "Data Point Number (Sorted by magnitude of training data)")
        else:
            plt.xlabel("Data Point Number")
        plt.legend()
        plt.grid()

    # Printing some basic stats
    num_below = np.count_nonzero(true < lower)
    num_above = np.count_nonzero(true > upper)
    if summary:
        print(datatype)
        print("----------")
        if method == "HDI":
            print("Total number of data points out of range of prediction "
                  f"({uncert*100:.0f}% HDI): {num_below + num_above} "
                  f"({100*(num_below + num_above)/len(true):.2f}%)")
        elif method == "SD":
            print(
                "\nTotal number of data points out of range of prediction "
                f"({uncert} sigma): {num_below + num_above} "
                f"({100*(num_below + num_above)/len(true):.2f}%)")

        # RMS error in predictions
        RMSe = np.sqrt(np.mean((means - true)**2))
        print("\nAverage RMS error between mean prediction and "
              f"true value: {RMSe:.2f}")

    # How bad are the errors? Out Of Range (OOR)
    OOR_inds_U, OOR_inds_L = (true > upper), (true < lower)

    OOR_means_U, OOR_means_L = means[OOR_inds_U], means[OOR_inds_L]

    OOR_stds_U, OOR_stds_L = stds[OOR_inds_U], stds[OOR_inds_L]

    OOR_true_U, OOR_true_L = true[OOR_inds_U], true[OOR_inds_L]

    OOR_U = np.mean(OOR_true_U - upper[OOR_inds_U])
    OOR_L = np.mean(OOR_true_L - lower[OOR_inds_L])

    OOR_U_percent = 100 * \
        np.mean((OOR_true_U - (OOR_means_U + uncert * OOR_stds_U)) /
                (OOR_means_U + uncert * OOR_stds_U))
    OOR_L_percent = 100 * \
        np.mean((OOR_true_L - (OOR_means_L - uncert * OOR_stds_L)) /
                (OOR_means_L - uncert * OOR_stds_L))

    if summary:
        print("\nOf the true values outside the predicted confidence "
              "intervals:")
        print("\tAverage distance beyond upper region of certainty = "
              f"{OOR_U:.2f} ({OOR_U_percent:.1f}% above the upper "
              "confidence region)")
        print("\tAverage distance below lower region of certainty = "
              f"{OOR_L:.2f} ({OOR_L_percent:.1f}% below the lower "
              "confidence region)")

    # Just plot the predictions that are WRONG (Out Of Range)
    if plots:
        OOR_inds = OOR_inds_U + OOR_inds_L
        OOR_true = true[OOR_inds]
        OOR_means = means[OOR_inds]
        OOR_xs = np.arange(0, len(OOR_means))
        plt.figure(figsize=(15, 9))
        plt.errorbar(
            OOR_xs,
            OOR_means,
            yerr=[
                upper[OOR_inds] -
                OOR_means,
                OOR_means -
                lower[OOR_inds]],
            fmt='o')
        plt.plot(OOR_xs, OOR_true, 'xr', label="True data")
        if method == "SD":
            plt.title(f"{len(OOR_xs)} {datatype} data points were "
                      f"incorrectly predicted. Using {uncert} sigma")
        if method == "HDI":
            plt.title(f"{len(OOR_xs)} {datatype} data points were "
                      f"incorrectly predicted ({uncert*100:.0f}% HDI)")
        plt.ylabel("Concrete Compressive Strength (MPa)")
        if sort:
            plt.xlabel(
                "Data Point Number (Sorted by magnitude of training data)")
        else:
            plt.xlabel("Data Point Number")
        plt.legend()
        plt.grid()

    if summary:
        # Average prediction
        print(
            f"\nAverage prediction: {np.mean(means):.2f}, Average true value:"
            f" {np.mean(true):.2f}")

        # Average estimated error in predictions - HDI
        upper_err = np.mean(upper - means)
        upper_err_percent = 100 * np.mean((upper - means) / means)
        lower_err = np.mean(means - lower)
        lower_err_percent = 100 * np.mean((means - lower) / means)
        if method == "HDI":
            print("\nAverage uncertainty in predictions "
                  f"({uncert*100:.0f}% HDI): +{upper_err:.2f} "
                  f"({upper_err_percent:.1f}%), {lower_err:.2f} "
                  f"({lower_err_percent:.1f}%)")
        elif method == "SD":
            print(f"\nAverage uncertainty in predictions (+/-{uncert} SD): "
                  f"+{upper_err:.2f} ({upper_err_percent:.1f}%), "
                  f"{lower_err:.2f} ({lower_err_percent:.1f}%)")
        print("_______________________________________")


def sort_data(data, sortby=0, inds=[None]):
    data = np.array(data)
    if None in inds:
        inds = np.argsort(data[sortby], axis=-1)
    for i in range(data.shape[0]):
        if len(data[i].shape) > 1:
            # Captures the multi-dimensionality of the samples arrays
            data[i] = data[i][:, inds]
        else:
            data[i] = data[i][inds]       # Everything else
    return data


def count_wrong_preds(samples, true, interval, method, verbose=False):
    """
    Taking into account the upper and low confidence intervals, how many
    predictions did we get wrong? I.e. how many true values are out of the
    confidence region?

    Parameters
    ----------
    samples : array_like
        Array of shape (#samples, #datapoints) containing samples for a dataset
    true : array_like
        Array of shape (#datapoints,) containing the true y values for dataset
    interval : float
        If method = "SD", this is number of standard deviations from mean to
        calc
        If method = "HDI", this is the interval to take (e.g. 0.95 = 95% HDI)
    method : str
        Either "SD" or "HDI"

    Returns
    -------
    pred_low : int
        How many predictions were too low (i.e. true value falls above the
        CI)
    pred_high : int
        How many predictions were too high (i.e. true value falls below the
        CI)

    """
    upper, lower, means = calculate_credibility(samples, interval, method)
    pred_low = (upper < true).sum()
    pred_high = (lower > true).sum()
    if verbose:
        print(f"Low predictions: \t{pred_low}\nHigh predictions: "
              f"\t{pred_high}\n----------------------------\nTotal out "
              f"of range: \t{pred_low+pred_high}")
    return pred_low, pred_high

# Plot clusters of points instead of the bars


def result_plots(all_samples,
                 component_name=None,
                 output_num=None,
                 labels=None,
                 true_inds=None,
                 interval=2,
                 method="SD",
                 all_true=None,
                 sort=False,
                 sortby=0,
                 bar_method="bars",
                 **kwargs):
    """
    Produce error plots for the testing samples, but instead of plotting
    error bars, plot the raw predicted ys
    Args:
        all_samples:
            Samples are the individual predictions 'y' for each X that
            form the prediction distributions
            List containing each dataset
            (e.g. [samples_test, samples_train, samples_val])
            Each element in the list is either a NumPy array of shape
            (#samples, #datapoints) if only 1 output, or
            (#outputs, #samples, #datapoints) if more than 1 output
        component_name:
            Name of the component of interest.
            This is the component that the original DataFrame was sorted by,
            which influences how the test/train/val data is split. Optional.
        labels:
            Names of each of the datasets, e.g. ["Test", "Train", "Val"].
            Optional.
        interval:
            How wide the uncertainty regions are. Depends on 'method' input.
            E.g. if using HDI: 0.95 (for 95% Highest Density Interval)
            E.g. if using SD: 2 (for +/- 2 standard deviations)
        method:
            Either:
            "HDI" - For Highest Density Interval
            "SD" - For Standard Deviation
        all_true:
            The true values of 'y' for each X
            List containing each dataset (e.g.
            [samples_test, samples_train, samples_val])
            Each element in the list is either a NumPy array of shape
            (#datapoints) if only 1 output, or
            (#outputs, #datapoints) if more than 1 output
        sort:
            If True, the function sorts the X values in each dataset by the
            magnitude of their corresponding 'y' value. Individually sorts the
            different datasets (i.e. it doesn't combine them)
            If False, the function plots each datapoint in the order they're
            presented.
        sortby:
            Which output# to sort by. Only applicable when plotting multiple
            outputs (i.e. more than one dependent variable). This ensures
            that the correct y value is plotted for each X, and each X can
            be compared against the different outputs. If int type, the
            corresponding output will be prioritised and plotted first e.g.
            if sortby=1, the second output data will be plotted first.
            If sortby="all", then all the data will be sorted and plotted.
            IMPORTANT: If 'all' is used, any particular datapoint number
            (on the x axis) corresponds to a different datapoint number in
            each output set. They cannot be compared.
        bar_method:
            How to display the confidence intervals.
                "bars": Solid bars without caps, cleaner look
                "streaks": Plot every sample of every test point, more
                interesting but perhaps less clear
    **kwargs:
        true_inds:
            If the user wants to plot the individual datapoints in the order in
            which they appear in the original dataframe (before sorting)

        title: Title of the plot
        ylabel: y axis label
        markersize:
        linewidth:
        output_labels:
        figsize:
        ylim:
        xlim:
    """

    # Check if we've passed more than one output's worth of data
    ndim = np.ndim(all_samples[0])
    num_datasets = len(all_samples)  # E.g. "Test", "Train", "Val" would = 3

    assert ndim == 2 or ndim == 3, "Please check the dimension of 'all_samples' input. Should either be 2 or 3." # noqa
    if ndim == 2:  # One dependent variable / output
        # Count every datapoint across all datasets
        N_total = np.sum([all_samples[i].shape[1]
                         for i in range(num_datasets)])
    elif ndim == 3:  # More than one dependent variable / output
        # Count every datapoints across all datasets and
        N_total = np.sum([all_samples[i][0].shape[1]
                         for i in range(num_datasets)])
        # Number of dependent variables we're plotting
        num_outputs = all_samples[0].shape[0]

    # If we've included true values or indices, make sure the dimensions match
    # with the samples ot made into a function call because everything is a
    # slightly different shape
    # all_ samples: either (#datasets, #outputs, #samples, #datapoints) or
    # (#datasets, #samples, #datapoints)
    # all_true: either (#datasets, #outputs, #datapoints) or
    # (#datasets, #datapoints)
    # true_inds: (#datasets, #datapoints)
    if all_true is not None:
        if ndim == 3:
            for i in range(num_datasets):
                for j in range(num_outputs):
                    assert all_samples[i][j].shape[1] == len(
                        all_true[i][j]), "Shape of all_true and samples does not match. Please check." # noqa
        elif ndim == 2:
            for i in range(num_datasets):
                assert all_samples[i].shape[1] == len(
                    all_true[i]), "Shape of all_true and samples does not match. Please check." # noqa
    if true_inds is not None:
        if ndim == 3:
            for i in range(num_datasets):
                for j in range(num_outputs):
                    assert all_samples[i][j].shape[1] == len(
                        true_inds[i]), "Shape of true_inds and samples does not match. Please check." # noqa
        elif ndim == 2:
            for i in range(num_datasets):
                assert all_samples[i].shape[1] == len(
                    true_inds[i]), "Shape of true_inds and samples does not match. Please check." # noqa

    if labels is None:
        # Fill in the labels with blanks
        labels = [f"Dataset {i}" for i in range(num_datasets)]

    # Make a new x value array to cover the whole span of the plot.
    XS = np.arange(0, N_total)

    if true_inds is not None and sort:
        print("Please note: the 'sort' argument overrules the true_inds input."
              " Not using true_inds")
    elif true_inds is not None and not sort:
        XS = np.concatenate(true_inds)

    # KWARG/ARG ASSIGNMENT
    if 'markersize' in kwargs:
        markersize = kwargs.pop('markersize', '')
    else:
        markersize = 1

    if 'linewidth' in kwargs:
        linewidth = kwargs.pop('linewidth', '')
    else:
        linewidth = 1

    if 'output_labels' in kwargs:
        output_labels = kwargs.pop('output_labels', '')
        if ndim == 3:
            assert len(
                output_labels) == num_outputs, f"Not enough labels given to the outputs. {len(output_labels)} given but should be either 0 or {num_outputs}." # noqa
    else:
        output_labels = [f"Output #{i}" for i in range(num_datasets)]

    if 'figsize' in kwargs:
        figsize = kwargs.pop('figsize', '')
    else:
        figsize = (20, 8)

    if 'title' in kwargs:
        plt_title = kwargs.pop('title', '')
    else:
        plt_title = "Mean estimates and uncertainty for all data"

    ax = plt.figure(figsize=figsize, dpi=350)  # noqa

    if ndim == 2:
        num_samples = all_samples[0].shape[0]
    elif ndim == 3:
        num_samples = all_samples[0].shape[1]

    # Counter target: how many output variables to go through
    if ndim == 3:                 # If we've passed in multiple sets of samples for different dependent variables  # noqa
        if output_num is None:      # We want to plot more than one
            counter_target = num_outputs  # Number of dependent variables
        elif output_num is not None:    # We've chosen to only plot one
            counter_target = 1        # Only do one pass
    else:  # If we're only looking at a single prediction
        counter_target = 1          # Only do one pass

    # Sorting indices, so we can apply the same sorting to different 'outputs'
    sort_inds = [[None]] * num_datasets

    # Swap the dependent variable dimensions around to make the one of
    # interest in the 0th position
    if isinstance(sortby, int) and sortby != 0 and ndim == 3:
        for i in range(len(all_samples)):
            all_samples[i][sortby], all_samples[i][0] = all_samples[i][0], all_samples[i][sortby].copy()  # noqa
            if all_true is not None:
                all_true[i][sortby], all_true[i][0] = all_true[i][0], all_true[i][sortby].copy()  # noqa
            output_labels[sortby], output_labels[0] = output_labels[0], output_labels[sortby] # noqa

    counter = 0
    while counter < counter_target:

        # Extract the current data for the output number of choice
        if ndim == 2:
            # Otherwise we end up editing all_samples too.. This took far too
            # long to spot
            samples = copy.deepcopy(all_samples)
            true = all_true
        # Reassemble samples to only include the current output number
        elif ndim == 3:
            if output_num is None:
                samples = [all_samples[i][counter].copy()
                           for i in range(num_datasets)]
                if all_true is not None:
                    true = [all_true[i][counter].copy()
                            for i in range(num_datasets)]
                else:
                    true = None
            elif output_num is not None:
                output_labels[output_num], output_labels[0] = output_labels[0], output_labels[output_num] # noqa
                samples = [all_samples[i][output_num].copy()
                           for i in range(num_datasets)]
                if all_true is not None:
                    true = [all_true[i][output_num].copy()
                            for i in range(num_datasets)]
                else:
                    true = None
        # Running total of which x position we're currently at
        N_running = 0

        # Loop through the datasets (e.g. test, train, val)
        # This loop shouldn't care whether we're looking at ndim==2 or nim==3
        for i, dataset in enumerate(labels):
            assert samples[i].shape[0] != samples[i].shape[1], "Please ensure number of samples != number of datapoints. Causes a weird bug when converting to np array in the sort_data function. To be fixed later." # noqa
            # How many datapoints are in the current dataset
            N_curr = samples[i].shape[1]

            # Get the upper and lower confidence band values and the mean
            # values
            upper, lower, means = calculate_credibility(
                samples[i], interval, method)

            # Construct array of x values for this dataset only, based on how
            # many datapoints we've processed
            xs = XS[N_running:N_running + N_curr]

            # Sort the data if we've chosen to
            if sort:
                # If we have entered the true y values, sort by them
                if true is not None:
                    if sortby != "all":
                        if None in sort_inds[i]:
                            sort_inds[i] = np.argsort(true[i], axis=-1)
                        true[i], samples[i], upper, lower, means = sort_data(
                            [true[i], samples[i], upper, lower, means], inds=sort_inds[i]) # noqa
                    elif sortby == "all":
                        true[i], samples[i], upper, lower, means = sort_data(
                            [true[i], samples[i], upper, lower, means], sortby=0) # noqa
                # If we have not entered the true y values
                # sort by the mean predictions instead
                elif true is None:
                    if sortby != "all":
                        if None in sort_inds[i]:
                            sort_inds[i] = np.argsort(means, axis=-1)
                        means, samples[i][:], upper, lower = sort_data(
                            [means, samples[i][:], upper, lower], inds=sort_inds[i]) # noqa
                    elif sortby == "all":
                        means, samples[i][:], upper, lower = sort_data(
                            [means, samples[i][:], upper, lower], sortby=0)
                elif not sort and true_inds is not None:
                    # Otherwise, if we've chosen not to sort and given the true
                    # indices, take a slice of those as the current x values
                    xs = true_inds[i]

            # Construct the array that contains the error bar info
            # Offset [upper, lower] from the mean value
            asymmetric_error = [(upper - means), (means - lower)]

            # Possibly nicer looking, but takes a while to process
            if bar_method == "streaks":
                plt.errorbar(
                    xs,
                    means,
                    yerr=asymmetric_error,
                    fmt='ok',
                    capsize=2,
                    elinewidth=0.1,
                    markersize=0,
                    alpha=0.8,
                    label=labels[i])  # markeredgewidth=10)
                [plt.plot(xs, samples[i][j, :], 'ob', alpha=0.1, markersize=markersize * 1.5) for j in range(num_samples)]  # Plot a scatter of the other points # noqa

            # Probably the cleaner option
            elif bar_method == "bars":
                labelstr = f"{labels[i]} Data ({samples[i].shape[1]} points)"
                if ndim == 3 and output_labels is not None:
                    labelstr = f"{output_labels[counter]}: {labels[i]} "
                    f"Data ({samples[i].shape[1]} points)"
                plt.errorbar(xs,
                             means,
                             yerr=asymmetric_error,
                             fmt='o',
                             markersize=markersize,
                             linewidth=linewidth,
                             markeredgecolor='blue',
                             markerfacecolor='blue',
                             label=labelstr)

            # Add to the counter to keep track of where we are on the x axis
            N_running = N_running + N_curr

        # Plot the 'true' values
        if ndim == 3 and output_labels is not None:
            true_labelstr = f"{output_labels[counter]}: True data"
        else:
            true_labelstr = "True data"
        if true is not None:
            plt.plot(
                XS,
                np.concatenate(true),
                'ok',
                label=true_labelstr,
                markersize=markersize)   # Plot the true data if we've passed
        counter += 1

    # Set the ylabel if we've inputted it
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs.pop('ylabel', ''))
    else:
        plt.ylabel("y Value")

    # Set the xlim and ylim if we've inputted them
    if 'ylim' in kwargs:
        plt.ylim(kwargs.pop('ylim', ''))
    if 'xlim' in kwargs:
        plt.xlim(kwargs.pop('ylim', ''))

    # Set the x axis label
    if true_inds is not None:
        if component_name is not None:
            plt.xlabel(
                f"Data point number, ordered by magnitude of: "
                f"{component_name}")
        else:
            plt.xlabel("Data point number")
    else:
        if sort:
            if true is not None:
                plt.xlabel(
                    "Data Point Number (Sorted by magnitude of true "
                    "value of y)")
            elif true is None:
                plt.xlabel(
                    "Data Point Number (Sorted by magnitude of mean "
                    "predicted y)")
        else:
            plt.xlabel("Data Point Number")

    if method == "HDI":
        plt.title(f"{plt_title}. Using {interval*100:.0f}% HDI")
    elif method == "SD":
        plt.title(f"{plt_title}. Using +/-{interval}$\\sigma$")

    curr_fig = plt.gcf()
    plt.legend()
    plt.grid()

    # Swap the dependent variable dimensions back around
    # A bit of a nasty fix but will do for now
    if isinstance(sortby, int) and sortby != 0 and ndim == 3:
        for i in range(len(all_samples)):
            all_samples[i][0], all_samples[i][sortby] = all_samples[i][sortby], all_samples[i][0].copy() # noqa
            if all_true is not None:
                all_true[i][0], all_true[i][sortby] = all_true[i][sortby], all_true[i][0].copy() # noqa
            output_labels[0], output_labels[sortby] = output_labels[sortby], output_labels[0] # noqa

    return curr_fig


def histogram_stats(samples,
                    true,
                    labels,
                    method="SD",
                    interval=1,
                    stats=True,
                    plot_1_xlim=None,
                    plot_1_ylim=None,
                    plot_2_xlim=None,
                    plot_2_ylim=None,
                    plot_3_xlim=None,
                    plot_3_ylim=None,
                    plot_4_xlim=None,
                    plot_4_ylim=None,
                    plot_5_xlim=None,
                    plot_5_ylim=None,
                    **kwargs):

    if 'figsize' in kwargs:
        figsize = kwargs.pop('figsize', '')
    else:
        figsize = (15, 10)

    if 'dp' in kwargs:
        dp = kwargs.pop('dp', '')
        decimal_place = True # noqa
    else:
        dp = 1
        decimal_place = False # noqa

    fig, axs = plt.subplots(
        2, 3, constrained_layout=True, figsize=figsize, dpi=500)

    # Create a new dict to store all the mu and sigma data
    # params = {'mu': {None}, 'sigma':{None}}
    # datasets = {'Test':params.copy(), 'Train':params.copy(), 'OoD':params.copy()} # noqa
    # stats = {'1': datasets.copy(), '2': datasets.copy(), '3': datasets.copy(), '4': datasets.copy()} # noqa
    if stats:
        stats = {'1': {'Validation': {'mu': {}, 'sigma': {}},
                       'Train': {'mu': {}, 'sigma': {}},
                       'Test (OoD)': {'mu': {}, 'sigma': {}}},
                 '2': {'Validation': {'mu': {}, 'sigma': {}},
                       'Train': {'mu': {}, 'sigma': {}},
                       'Test (OoD)': {'mu': {}, 'sigma': {}}},
                 '3': {'Validation': {'mu': {}, 'sigma': {}},
                       'Train': {'mu': {}, 'sigma': {}},
                       'Test (OoD)': {'mu': {}, 'sigma': {}}},
                 '4': {'Validation': {'mu': {}, 'sigma': {}},
                       'Train': {'mu': {}, 'sigma': {}},
                       'Test (OoD)': {'mu': {}, 'sigma': {}}}}

    # Samples drawn relative to the true values
    # Assesses the performance of the network and UQ method
    for i, s in enumerate(samples):
        samples_error = np.abs([s.squeeze()[j, :] - true[i]
                               for j in range(s.shape[0])])
        s_flattened = np.array(samples_error).flatten()
        s_m = np.mean(s_flattened)
        s_sd = np.std(s_flattened)
        if stats:
            stats['1'][labels[i]]['mu'], stats['1'][labels[i]]['sigma'] = s_m, s_sd # noqa
        label = f"{labels[i]} ($\\mu={s_m:{dp}f}$, $\\sigma={s_sd:{dp}f}$)"
        sns.distplot(s_flattened, label=label, ax=axs[0, 0], **kwargs)
    axs[0, 0].legend()
    if plot_1_ylim is not None:
        axs[0, 0].set_ylim((plot_1_ylim))
    if plot_1_xlim is not None:
        axs[0, 0].set_xlim((plot_1_xlim))
    axs[0, 0].set_title("1): Abs Error Between each Sample and True\n"
                        f"{samples[0].shape[0]} Samples per Datapoint")
    axs[0, 0].set_xlabel("Abs Error in Individual Sample")

    # Mean Predicted values realtive to true values
    # Assesses the performance of the network
    for i, s in enumerate(samples):
        _, _, means = calculate_credibility(s, 1, "SD")
        errs = np.abs((means - true[i]))
        errs_m = np.mean(errs)
        errs_sd = np.std(errs)
        if stats:
            stats['2'][labels[i]]['mu'], stats['2'][labels[i]]['sigma'] = errs_m, errs_sd # noqa
        label = f"{labels[i]} ($\\mu={errs_m:{dp}f}$, $\\sigma={errs_sd:{dp}f}$)" # noqa
        sns.distplot(errs, label=label, ax=axs[0, 1], **kwargs)
    axs[0, 1].legend()
    if plot_2_ylim is not None:
        axs[0, 1].set_ylim((plot_2_ylim))
    if plot_2_xlim is not None:
        axs[0, 1].set_xlim((plot_2_xlim))
    axs[0, 1].set_title("2): Abs Err Between Mean Prediction and True")
    axs[0, 1].set_xlabel("Error")

    # Uncertainty region widths (from top to bottom)
    # Assesses the performance of the UQ method
    for i, s in enumerate(samples):
        upper, lower, _ = calculate_credibility(
            s, interval=interval, method=method)
        uncert = np.abs(upper - lower)
        uncert_m = np.mean(uncert)
        uncert_sd = np.std(uncert)
        if stats:
            stats['3'][labels[i]]['mu'], stats['3'][labels[i]]['sigma'] = uncert_m, uncert_sd # noqa
        label = f"{labels[i]} ($\\mu={uncert_m:{dp}f}$, $\\sigma={uncert_sd:{dp}f}$)" # noqa
        sns.distplot(uncert, label=label, ax=axs[0, 2], **kwargs)
    axs[0, 2].legend()
    if plot_3_ylim is not None:
        axs[0, 2].set_ylim((plot_3_ylim))
    if plot_3_xlim is not None:
        axs[0, 2].set_xlim((plot_3_xlim))
    if method == "SD":
        axs[0, 2].set_title(
            f"3): Distribution of Abs Uncertainty Widths\n(Using $\\pm {interval} \\sigma$ Confidence Interval)") # noqa
    elif method == "HDI":
        axs[0, 2].set_title(
            f"3): Distribution of Abs Uncertainty Widths\n(Using {interval*100:.1f}% HDI Confidence Interval)") # noqa
    axs[0, 2].set_xlabel("Uncertainty Region Total Width")

    # Mean predicted values
    # Assesses the quality of the data splitting
    for i, s in enumerate(samples):
        _, _, means = calculate_credibility(s, 1, "SD")
        means_m = np.mean(means)
        means_sd = np.std(means)
        if stats:
            stats['4'][labels[i]]['mu'], stats['4'][labels[i]]['sigma'] = means_m, means_sd # noqa
        label = f"{labels[i]} ($\\mu={means_m:{dp}f}$, $\\sigma={means_sd:{dp}f}$)" # noqa
        sns.distplot(means, label=label, ax=axs[1, 0], **kwargs)
    if plot_4_ylim is not None:
        axs[1, 0].set_ylim((plot_4_ylim))
    if plot_4_xlim is not None:
        axs[1, 0].set_xlim((plot_4_xlim))
    axs[1, 0].legend()
    axs[1, 0].set_title("4): Distribution of Mean Predicted Values")
    axs[1, 0].set_xlabel("Mean Prediction")

    # Distance of true value outside of credible interval
    # Assess the calibration of the uncertainty
    for i, s in enumerate(samples):
        upper, lower, _ = calculate_credibility(
            s, interval=interval, method=method)
        pred_low_ind = (upper < true[i])
        # pred_low_num = pred_low_ind.sum()
        pred_low_amnt = upper[pred_low_ind] - \
            true[i][pred_low_ind]  # Purposefully negative

        pred_high_ind = (lower > true[i])
        # pred_high_num = pred_high_ind.sum()
        pred_high_amnt = lower[pred_high_ind] - true[i][pred_high_ind]

        # wrong_preds_ind = pred_low_ind + pred_high_ind
        wrong_preds_amnt = np.concatenate((pred_low_amnt, pred_high_amnt))
        wrong_preds_m = np.mean(wrong_preds_amnt)
        wrong_preds_sd = np.std(wrong_preds_amnt)

        label = f"{labels[i]} ($\\mu={wrong_preds_m:{dp}f}$, "
        f"$\\sigma={wrong_preds_sd:{dp}f}$)"
        sns.distplot(wrong_preds_amnt, label=label, ax=axs[1, 1], **kwargs)
    axs[1, 1].legend()
    if plot_5_ylim is not None:
        axs[1, 1].set_ylim((plot_5_ylim))
    if plot_5_xlim is not None:
        axs[1, 1].set_xlim((plot_5_xlim))
    axs[1, 1].set_title("5): Dist Between True Value and the CI\n"
                        "If Prediction is Untrustworthy")
    axs[1, 1].set_xlabel("Distance")

    # Grey out the final one
    axs[1, 2].set_facecolor('lightgrey')

    curr_fig = plt.gcf()
    if stats:
        return curr_fig, stats
    else:
        return curr_fig


def produce_random_samples(num_samples):
    num_bays_low = 5
    num_bays_high = 10

    bay_width_low = 5
    bay_width_high = 10

    num_bay_dims = 3

    # Produce a random int between our specified ranges
    def ri():
        return np.random.randint(
            num_bays_low, num_bays_high, size=num_bay_dims)   # Ints for #bays

    # Produce a random float between our specified ranges
    def rf():
        return np.random.uniform(bay_width_low, bay_width_high, [
                                 num_bay_dims])  # floats for bay width

    X = np.array([])
    for i in range(num_samples):
        temp = np.hstack((ri(), rf()))
        X = np.concatenate((X, temp))

    X = np.reshape(X, (num_samples, num_bay_dims * 2))
    return X


def generate_3d_samples(model, params, **kwargs):

    params_flat = [element for sublist in params for element in sublist]
    assert len(
        params_flat) == 12, f"Please ensure the format of 'params' is correct. Total length should be 12, but it's {len(params_flat)}" # noqa

    # Can get away with this syntax for None only
    num_x = num_y = num_z = dim_x = dim_y = dim_z = None
    all_var = [num_x, num_y, num_z, dim_x, dim_y, dim_z]
    # Will save the indices of the parameters that are constant
    consts = [0, 0, 0]
    # Will save the indices of the parameters that will have a range spanned
    ranges = [0, 0, 0]
    consts_ind, rang_ind = 0, 0
    for i in range(len(all_var)):
        if len(params[i]) == 1:
            all_var[i] = params[i][0]
            consts[consts_ind] = i
            consts_ind += 1
        else:
            all_var[i] = np.linspace(params[i][0], params[i][1], params[i][2])
            ranges[rang_ind] = i
            rang_ind += 1

    len1 = len(all_var[ranges[0]])
    len2 = len(all_var[ranges[1]])
    len3 = len(all_var[ranges[2]])

    N = len1 * len2 * len3

    xxx = np.zeros((len1, len2, len3))
    yyy = np.zeros((len1, len2, len3))
    zzz = np.zeros((len1, len2, len3))

    means_pred = np.zeros((len1, len2, len3))
    stds_pred = np.zeros((len1, len2, len3))
    percent_uncert = np.zeros((len1, len2, len3))

    all_labels = ["num_x", "num_y", "num_z", "dim_x", "dim_y", "dim_z"]
    labels = [None] * 3
    for i in range(3):
        labels[i] = all_labels[ranges[i]]
    print(f"Variable parameter / axis labels = {labels}")

    print(f"Cycling through {len1}*{len2}*{len3} = {N} options..")

    for i, x in enumerate(all_var[ranges[0]]):
        for j, y in enumerate(all_var[ranges[1]]):
            for k, z in enumerate(all_var[ranges[2]]):
                xxx[i, j, k] = x
                yyy[i, j, k] = y
                zzz[i, j, k] = z
                X = [None] * 6
                X[ranges[0]] = x
                X[ranges[1]] = y
                X[ranges[2]] = z
                X[consts[0]] = all_var[consts[0]]
                X[consts[1]] = all_var[consts[1]]
                X[consts[2]] = all_var[consts[2]]
                X_norm = torch.Tensor(pre.normalise(X, model.x_mean, model.x_std)) # noqa
                _, means_pred[i, j, k], stds_pred[i, j, k] = model.generate_samples(X_norm, **kwargs) # noqa
                percent_uncert[i, j, k] = 100 * (stds_pred[i, j, k] / means_pred[i, j, k]) # noqa

    return means_pred, stds_pred, percent_uncert, xxx, yyy, zzz, labels, ranges, consts # noqa


def generate_3d_plot(means_pred,
                     stds_pred,
                     percent_uncert,
                     xxx,
                     yyy,
                     zzz,
                     labels,
                     ranges,
                     consts,
                     scale=False,
                     legend_num=5,
                     legend_HDI=0.95,
                     mode="absolute",
                     produce_animation=False,
                     limit=None,
                     animation_name="./animation.mp4",
                     all_domain=[
                         [
                             3, 10], [
                             3, 10], [
                             3, 10], [
                             5, 10], [
                             5, 10], [
                             5, 10]],
                     figsize=(15, 15),
                     scalingpower=None,
                     scalingscalar=None,
                     dpi=200):
    """
    Generate a 3D scatter plot of the uncertainty analysis performed by the
    function generate_3d_samples.

    Parameters
    ----------
    means_pred : array_like
        Mean prediction values of every point in the sample space.
    stds_pred : array_like
        Standard deviation of predictions for every point in the sample space.
    percent_uncert : array_like
        Percentage uncertainty (std/mean) for every point in the sample space.
    xxx : NumPy array
        Meshgrid used for plotting in 3D.
        Of shape (num_steps, num_steps, num_steps). See generate_3d_samples.
    yyy : TYPE
        Meshgrid used for plotting in 3D.
        Of shape (num_steps, num_steps, num_steps). See generate_3d_samples.
    zzz : TYPE
        Meshgrid used for plotting in 3D.
        Of shape (num_steps, num_steps, num_steps). See generate_3d_samples.
    labels : list of str
        The axis labels, format: [x, y, z]. E.g. ["num_x", "num_y", "num_z"]
    ranges : list of int
        A list detailing which parameters are set as ranges.
        1 = num_x, 2 = num_y, 3 = num_z, 4 = dim_x, ....
    consts : list of int
        A list detailing which parameters are set as constants.
        1 = num_x, 2 = num_y, 3 = num_z, 4 = dim_x, ....
    scale : bool
        If True, scale every axis to match the relative dimension of the
        x axis. Loses a bit of resolution as everything gets scaled down,
        but maybe still helpful for nice visualisation.
    legend_num : int, optional
        How many items to include in the legend which links the size of
        markers to the uncertainty. The default is 5.
    legend_HDI : float, optional
        Float between (0,1). Controls the upper and lower extent of the
        legend entries for the uncertainty using Highest Density Interval
        (HDI). The default is 0.95.
    mode : str, optional
        If "absolute", scale the markers according to the stds_pred.
        If "relative", scale markers according to percent_uncert.
        The default is "absolute".
    produce_animation : bool, optional
        Whether to produce an animation that rotates the plot automatically.
        It's quite nice but slow. The default is False.
    limit : float, optional
        The markers can get quite big. If 'limit' is specified, limit
        the scale of all markers to this number. A sensible(ish) number is
        about 4000. The default is None.
    animation_name : str, optional
        Where to save the animation if it's been produced.
        The default is "./animation.mp4".
    domain : list, optional
        The extents of the training data for dimx, dimy and dimz.
        Format is [[minx,maxx],[miny,maxy],[minz,maxz]]
        Default is [[5,10],[5,10],[5,10]]
    figsize : TYPE, optional
        DESCRIPTION. The default is (15,15).
    scalingpower : TYPE, optional
        DESCRIPTION. The default is None.
    scalingscalar : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection='3d')

    if scale:
        def scale_func():
            return np.dot(Axes3D.get_proj(ax), scale)
        # Take this as the basis, scale other axes to match this
        x_scale = 1
        y_scale = yyy.max() / xxx.max()
        z_scale = zzz.max() / xxx.max()
        scale = np.diag([x_scale, y_scale, z_scale, 1.0])
        scale = scale * (1.0 / scale.max())
        scale[3, 3] = 1.0
        ax.get_proj = scale_func

    N = len(xxx.ravel())

    # Functions to scale the size of the markers according to the SD
    if mode == "absolute":
        if scalingpower is None:
            scalingpower = 3
        if scalingscalar is None:
            scalingscalar = 10

        def scaling_func(input_var):
            """ Converts an input uncertainty to a marker size """
            return (input_var * scalingscalar)**scalingpower

        # Plot
        scatter = ax.scatter(xxx, yyy, zzz,
                             c=['r'] * N,
                             marker='o',
                             s=scaling_func(stds_pred))

    elif mode == "relative":
        if scalingpower is None:
            scalingpower = 2
        if scalingscalar is None:
            scalingscalar = 0.5
        percent_uncert = np.abs(percent_uncert)

        def scaling_func(input_var):
            """ Converts an input uncertainty to a marker size """
            return scalingscalar * (input_var**scalingpower)**(1 / 2)

        # Limit the size of the markers
        uncerts = scaling_func(percent_uncert)
        if limit is None:
            limit = 4000
        uncerts[uncerts > limit] = limit

        # Plot
        scatter = ax.scatter(xxx, yyy, zzz, # noqa
                             c=['r'] * N,
                             marker='o',
                             s=uncerts) # noqa

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    domain = [[None, None]] * 3
    for i in range(3):
        domain[i] = all_domain[consts[i]]

    square1 = [[(domain[0][0],
                 domain[1][0],
                 domain[2][0]),
                (domain[0][0],
                 domain[1][1],
                 domain[2][0]),
                (domain[0][1],
                 domain[1][1],
                 domain[2][0]),
                (domain[0][1],
                 domain[1][0],
                 domain[2][0])]]
    square2 = [[(domain[0][0],
                 domain[1][0],
                 domain[2][0]),
                (domain[0][1],
                 domain[1][0],
                 domain[2][0]),
                (domain[0][1],
                 domain[1][0],
                 domain[2][1]),
                (domain[0][0],
                 domain[1][0],
                 domain[2][1])]]
    square3 = [[(domain[0][1],
                 domain[1][0],
                 domain[2][0]),
                (domain[0][1],
                 domain[1][1],
                 domain[2][0]),
                (domain[0][1],
                 domain[1][1],
                 domain[2][1]),
                (domain[0][1],
                 domain[1][0],
                 domain[2][1])]]
    square4 = [[(domain[0][1],
                 domain[1][0],
                 domain[2][1]),
                (domain[0][1],
                 domain[1][1],
                 domain[2][1]),
                (domain[0][0],
                 domain[1][1],
                 domain[2][1]),
                (domain[0][0],
                 domain[1][0],
                 domain[2][1])]]
    square5 = [[(domain[0][0],
                 domain[1][0],
                 domain[2][0]),
                (domain[0][0],
                 domain[1][1],
                 domain[2][0]),
                (domain[0][0],
                 domain[1][1],
                 domain[2][1]),
                (domain[0][0],
                 domain[1][0],
                 domain[2][1])]]
    square6 = [[(domain[0][1],
                 domain[1][1],
                 domain[2][1]),
                (domain[0][0],
                 domain[1][1],
                 domain[2][1]),
                (domain[0][0],
                 domain[1][1],
                 domain[2][0]),
                (domain[0][1],
                 domain[1][1],
                 domain[2][0])]]
    all_squares = [square1, square2, square3, square4, square5, square6]

    for square in all_squares:
        ax.add_collection3d(
            Poly3DCollection(
                square,
                alpha=0.5,
                linewidths=1,
                edgecolors='gray'))

    # Create legend
    if mode == "absolute":
        stds_ravel = stds_pred.ravel()
        CI = az.hdi(stds_ravel, legend_HDI)
        lower, upper = CI[0], CI[1]
        arr = np.linspace(lower, upper, legend_num)
        arr = np.around(arr / (0.005), decimals=0) * \
            (0.005)  # Round to nearest 0.005
        for area in arr:
            plt.scatter(
                [],
                [],
                c='r',
                alpha=0.9,
                s=scaling_func(area),
                label=f"{area:.3f}")
        plt.legend(
            scatterpoints=None,
            frameon=True,
            labelspacing=1,
            title="Uncertainty ($\\sigma$)",
            loc="upper left")

    elif mode == "relative":
        percent_uncert_ravel = percent_uncert.ravel()
        CI = az.hdi(percent_uncert_ravel, legend_HDI)
        lower, upper = CI[0], CI[1]
        arr = np.linspace(lower, upper, legend_num)
        arr = np.around(arr / 5, decimals=0) * 5  # Round to nearest 5%
        for area in arr:
            plt.scatter(
                [],
                [],
                c='r',
                alpha=0.9,
                s=scaling_func(area),
                label=f"{int(area)}%")
        plt.legend(
            scatterpoints=None,
            frameon=True,
            labelspacing=1,
            title="Uncertainty (%)",
            loc="upper left")

    plt.show()

    if produce_animation:
        def animate(frame):
            ax.view_init(20 + 8 * np.sin(frame / 10), frame)
            plt.pause(.001)
            return fig

        anim = animation.FuncAnimation(fig, animate, frames=360, interval=100)

        f = f"./{animation_name}"
        writervideo = animation.FFMpegWriter(fps=30)
        anim.save(f, writer=writervideo)


def PCA_transformdata(x_data, dimensions=3, components=None,
                      return_components=False):
    """
    Perform Prinicipal Component Analysis to reduce the dimension of input
    data X
    Parameters
    ----------
    x_data : torch.Tensor
        Data to reduce
    dimensions : int, default = 3
        Dimensions to reduce to
    verbose : bool
        If True, print the PCA components
    components : array_like
        If
    XXXXX

    Returns
    -------
    x_PC_df : DataFrame
        Transformed data, with columns 'PC1', 'PC2', 'PC3',.. etc

    """
    if torch.is_tensor(x_data):
        x_df = pd.DataFrame(x_data.numpy())
    elif isinstance(x_data, pd.DataFrame):
        x_df = x_data.values
    else:
        x_df = x_data # noqa

    if components is None:
        pca = PCA(n_components=dimensions)  # Create a new PCA object
        # Do the computation to obtain PCA components
        pca.fit(x_data)

    else:
        pca = components

    # Perform the transformation to reduced dimensional space
    x_PC = pca.transform(x_data)

    x_PC_df = pd.DataFrame(
        data=x_PC, columns=[
            f"PC{i+1}" for i in range(dimensions)])

    if return_components:
        return x_PC_df, pca
    else:
        return x_PC_df


def PCA_plot(all_data,
             all_stds=None,
             output_num=0,
             legend_num=5,
             legend_HDI=0.95,
             labels=None,
             colours=['r', 'g', 'b'],
             scalingpower=1,
             scalingfactor=1,
             figsize=(15, 15),
             produce_animation=False,
             animation_name="./animation.mp4",
             save_image=False,
             savename=None,
             saveformat='png',
             legend=True,
             dpi=200,
             **kwargs):
    """
    Plot the reduced dimension dataset in a 3D scatter graph.
    Colour the points according to their dataset, size the points according to
    their corresponding uncertainty prediction.

    Parameters
    ----------
    all_data : list of DataFrames
        All the data to be plotted. List of length equal to the number of
        datasets (e.g. 3 for Train, Test and Val). Each item in the list is
        a DataFrame returned from the function PCA_transformdata.
    all_stds : list of NumPy array
        All the standard deviations for the data being plotted.
        Of shape (1, #datapoints)
    output_num : int
        Which output to plot, e.g. 0=freq1, 1=freq2,...
    legend_num : int, optional
        How many items to include in the legend which links the size of
        markers to the uncertainty. The default is 5.
    legend_HDI : float, optional
        Float between (0,1). Controls the upper and lower extent of the
        legend entries for the uncertainty using Highest Density Interval
        (HDI). The default is 0.95.
    labels : list of str, optional
        Labels for the datasets, e.g. ["Train", "Test", "Val"].
        The default is None.
    colours : list of str, optional
        Colours to plot each dataset. The default is ['r', 'g', 'b'].
    scalingpower : float, optional
        What power to raise the uncertainty by to transform it into a
        marker size (area). The default is 3.
    scalingfactor : float, optional
        What factor to multiply the uncertainty by (before raising to the power
        above) to transform uncertainty to marker size (area).
        The default is 50.
    figsize : tuple, optional
        Tuple of fig size. The default is (15,15).
    produce_animation : bool, optional
        Whether to produce an animation that rotates the plot automatically.
        It's quite nice but slow. The default is False.
    animation_name : str, optional
        Where to save the animation if it's been produced.
        The default is "./animation.mp4".

    Returns
    -------
    None


    """
    if all_stds is not None:
        if isinstance(all_stds, list):
            if all_stds[0].shape[0] >= 1:
                assert output_num < all_stds[0].shape[
                    0], f"Number of outputs inferred from dimension of stds: {all_stds[0].shape[0]}.\nPlease ensure output_num is equal to or less than {all_stds[0].shape[0]-1}." # noqa
        else:
            if all_stds.shape[0] > 1:
                assert output_num < all_stds[0].shape[
                    0], f"Number of outputs inferred from dimension of stds: {all_stds[0].shape[0]}.\nPlease ensure output_num is equal to or less than {all_stds.shape[0]-1}." # noqa

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection='3d')

    # Functions to scale the size of the markers according to the SD
    def scaling_func(data):
        return (data * scalingfactor)**scalingpower

    # First plot creates the scatter object
    def first_plot(data, stds, output_num, label, colour):
        scatter = ax.scatter(data.iloc[:, 0],
                             data.iloc[:, 1],
                             data.iloc[:, 2],
                             c=[colour] * len(data),
                             marker='o',
                             s=scaling_func(stds[output_num]), **kwargs)
        return scatter

    # Subsequent plots plot on the existing object
    def subsequent_plots(data, stds, output_num, label, colour):
        ax.scatter(data.iloc[:, 0],
                   data.iloc[:, 1],
                   data.iloc[:, 2],
                   c=[colour] * len(data),
                   marker='o',
                   s=scaling_func(stds[output_num]), **kwargs)

    stds_ravel = None

    # If we've passed more than one output
    if isinstance(all_data, list):
        num_datasets = len(all_data)
        if labels is None:
            labels = [f"Dataset {i+1}" for i in range(num_datasets)]
        for i in range(num_datasets):
            data = copy.deepcopy(all_data[i])
            if all_stds is None:
                stds = np.full(data.shape[0], 1)
                stds_ravel = np.array(stds).ravel()
            elif all_stds is not None:
                stds = copy.deepcopy(all_stds[i])
                # Keep a running total of all the stds so the legend is correct
                if stds_ravel is None:
                    stds_ravel = all_stds[i].ravel()
                else:
                    stds_ravel = np.concatenate(
                        (stds_ravel, all_stds[i].ravel()))
            if i == 0:
                scatter = first_plot(
                    data, stds, output_num, labels[i], colours[i])
                plt.scatter(
                    [],
                    [],
                    c=colours[i],
                    alpha=1,
                    s=200,
                    label=f"{labels[i]}",
                    **kwargs)
            else:
                subsequent_plots(data, stds, output_num, labels[i], colours[i])
                plt.scatter(
                    [],
                    [],
                    c=colours[i],
                    alpha=1,
                    s=200,
                    label=f"{labels[i]}",
                    **kwargs)

    # If we've passed one input
    elif isinstance(all_data, pd.DataFrame):
        if labels is None:
            labels = "Dataset"
        if all_stds is None:
            all_stds = np.full(all_data.shape[0], 1)
        scatter = first_plot( # noqa
            all_data,
            all_stds,
            output_num,
            labels,
            colours[0])
        plt.scatter(
            [],
            [],
            c=colours[0],
            alpha=1,
            s=200,
            label=f"{labels[0]}",
            **kwargs)

    if legend:
        # Produce a legend with a cross section of sizes from the scatter
        CI = az.hdi(stds_ravel, legend_HDI)
        lower, upper = CI[0], CI[1]
        arr = np.linspace(lower, upper, legend_num)
        arr = np.around(arr / (0.005), decimals=0) * \
            (0.005)  # Round to nearest 0.005
        for area in arr:
            plt.scatter([], [], c='r', alpha=0.9, s=scaling_func(
                area), label=f"$\\sigma$ = {area:.3f}", **kwargs)
        # , title="Uncertainty ($\sigma$)")
        plt.legend(scatterpoints=None, frameon=True, labelspacing=1)

    ax.set_xlabel("Principal Component #1")
    ax.set_ylabel("Principal Component #2")
    ax.set_zlabel("Principal Component #3")

    if produce_animation:
        # Quite slow, but makes a nice animation
        def animate(frame):
            ax.view_init(20 + 8 * np.sin(frame / 10), frame / 2)
            plt.pause(.001)
            return fig

        anim = animation.FuncAnimation(fig, animate, frames=720, interval=20)

        f = r"./animation.mp4"
        writervideo = animation.FFMpegWriter(fps=60)
        anim.save(f, writer=writervideo)

    if save_image:
        plt.savefig(savename, format=saveformat)

    plt.show()

    return ax
