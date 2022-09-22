""" Archie Luxton, aol21, https://github.com/ese-msc-2021/irp-aol21, 2022 """


from scipy.spatial.distance import cdist, euclidean
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader  # noqa
import random

try:
    from duq import post
except ImportError or ModuleNotFoundError:
    import post

"""
A module to primarily import tabular data and process it ready for NN analysis.
"""


def normalise_lims(x, x_mean, x_std):
    """
    Normalises lists of the shape:
    [[A., B], [A., B], [A., B], [A., B], [A., B], [A., B]]
    Where A is a lower bound and B is an upper bound of one of the input
    parameters, i.e.:
    [x1, x2, x3, x4, x5, x6]

    """
    one = normalise(np.array(x)[:, 0], x_mean, x_std).to_numpy()
    two = normalise(np.array(x)[:, 1], x_mean, x_std).to_numpy()
    return np.stack((one, two), axis=-1)


def normalisation_checks(x, mean, std):
    """
    Just run assertion tests on the normalisation.
    The normalisation function is quite easily broken due to different
    data structures doing vectorisation differently, plus wanting to allow
    data input as array_like, single values or a mixture of the two.

    Parameters
    ----------
    x : scalar, array_like, DataFrame, TorchTensor
        Data to be normalised
    mean : scalar, array_like, DataFrame, TorchTensor
        Mean of the data (pre-normalisation)
    std : scalar, array_like, DataFrame, TorchTensor
        Standard deviation of the data (pre-normalisation)

    Returns
    -------
    None.

    """
    # If a std is zero, then we get NaN after normalisation. Check for this.
    if isinstance(std, np.floating) or isinstance(
            std, int) or isinstance(std, float):
        assert std != 0, "Cannot normalise data with 0 standard deviation. Please check data. Std deviation: {data_std}" # noqa
    else:
        assert (std == 0).any(
        ) == False, "Cannot normalise data with 0 standard deviation. Please check data. Std deviation: {data_std}" # noqa

    if isinstance(mean, np.floating) or isinstance(
            mean, int) or isinstance(mean, float):
        # First check: If the mean is just a single number, check that the std
        # is too
        assert (
            isinstance(
                std,
                np.floating) or isinstance(
                std,
                int) or isinstance(
                std,
                float))
    else:
        # Second check: Check generally that the dimensions of mean and std
        # match.
        assert len(mean) == len(
            std), f"Please check the dimensions of mean (len {len(mean)}) and std (len {len(std)})" # noqa

        # Third check: If x is 2D, check that the number of columns is equal to
        # the size of mean (and therefore std)
        if np.ndim(x) == 2:
            assert (
                x.shape[1] == len(mean)) or (
                len(mean) == 1), f"Dimensions of mean and std should either be (1,) or {x.shape}. Instead len(mean) = {len(mean)}, len(std)={len(std)}." # noqa

        # Fourth check: If x is 1D, check the length (still number of columns)
        # is equal to the length of mean (and therefore std)
        elif np.ndim(x) == 1:
            assert (
                len(x) == len(mean)) or (
                len(mean) == 1), f"Dimensions of mean and std should either be (1,) or {x.shape}. Instead len(mean) = {len(mean)}, len(std)={len(std)}." # noqa


def normalise(x, mean, std):
    """
    Noramlise (standardise) some data x based on a mean and standard deviation

    Parameters
    ----------
    x : scalar, array_like, DataFrame, TorchTensor
        Data to be normalised
    mean : scalar, array_like, DataFrame, TorchTensor
        Mean of the data (pre-normalisation)
    std : scalar, array_like, DataFrame, TorchTensor
        Standard deviation of the data (pre-normalisation)

    Returns
    -------
    normalised_x
        Normalised version of the input struct
    """

    normalisation_checks(x, mean, std)

    return (x - mean) / std


def unnormalise(x, mean, std):
    """
    Unnoramlise (standardise) some data x based on a mean and standard
    deviation

    Parameters
    ----------
    x : scalar, array_like, DataFrame, TorchTensor
        Data to be unnormalised
    mean : scalar, array_like, DataFrame, TorchTensor
        Mean of the data (pre-normalisation)
    std : scalar, array_like, DataFrame, TorchTensor
        Standard deviation of the data (pre-normalisation)

    Returns
    -------
    unnormalised_x
        Unnormalised version of the input struct
    """

    normalisation_checks(x, mean, std)

    return (x * std) + mean


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any
    randomness from cuda kernels
    From Imperial College Machine Learning module teaching resources

    Parameters
    ----------
    seed : int
        Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def geometric_median(X, eps=1e-5):
    """
    Find the geometric median of a multi-dimensional dataset.
    From user 'orlp': https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points # noqa
    18th May 2015, 13:56

    Implements the L1-median median as described in "The multivatiate L1-median and associated data depth" (Y. Vardi et al, 1999) # noqa
    Available at: https://www.pnas.org/doi/pdf/10.1073/pnas.97.4.1423

    Parameters
    ----------
    X : scalar, array_like, DataFrame, TorchTensor
        Data to analyse
    eps : float
        Epsilon, convergence criteria

    Returns
    -------
    y1 : NumPy array
        Geometric median of datastet
    """

    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def euclid_dist(df, origin, x_cols):
    """
    A function to find the Euclidean (L2) distance of every point in a
    dataframe from a specified origin.

    Parameters
    ----------
        df : Pandas DataFrame
            DF containing the data. Each column is either an independent or
            dependent variable, and
            each row is a datapoint.
        origin : NumPy array or list
            A 1D array of shape (len(x_cols),) containing the median value of
            every independent variable in the df
        x_cols : list
            A list containing the column indexes of every independent variable

    Returns
    -------
        euclid_dist: NumPy array
            Array containing the L2 distances of each datapoint from the
            geometric median. Of shape (len(df),)
    """
    num_parameters = len(x_cols)  # How many independent variables do we have # noqa
    num_datapoints = len(df)      # Number of data points
    # Create a new 1D array to store each datapoint's distance from median
    euclid_dist = np.zeros(num_datapoints)

    for datapoint in range(0, num_datapoints):
        for param in x_cols:

            # Go through each datapoint and sum the distance of each
            # independent variable from the origin
            euclid_dist[datapoint] += (df[df.columns[param]]
                                       [datapoint] - origin[param])**2

        euclid_dist[datapoint] = np.sqrt(euclid_dist[datapoint])

    return euclid_dist


def process_df_euclid(df, x_cols, data_mean=None,
                      data_std=None, plot=False, median=None):
    """
    Process a dataframe for analysis. This adds a column for the Euclidean
    (L2) distance of each datapoint from the geometric mean (of independent
    variables), and also normalises the dataset according to each parameter's
    mean and standard deviation.

    Parameters
    ----------
        df : Pandas DataFrame
            DF containing the data. Each column is either an independent or
            dependent variable, and
            each row is a datapoint.
        x_cols : list
            A list containing the column indexes of every independent variable
        plot : bool
            Whether or not to plot a histogram showing the distribution of
            Euclidean distances of each datapoint from the L1-median of the
            independent variables (pre-normalisation).

    Returns
    -------
        df : Pandas DataFrame
            A normalised Pandas DataFrame with the L2 distance column added
        data_mean : Pandas Series
            Contains the mean of each column (parameter) in df
        data_std : Pandas Series
            Contains the standard deviation of each column (parameter) in df
    """

    if median is None:
        # Calculate the geometric median and the euclidean distances between
        # every datapoint and the median
        median = geometric_median(df.to_numpy()[:, x_cols])

    euclid_dists = euclid_dist(df, median, x_cols)

    # Visualise the distribution of distances
    if plot:
        sns.distplot(euclid_dists)
        plt.title(
            "Distribution of Euclidean distances\nbetween the L1-median of "
            "independent variables $X_n$.\nPre-normalisation")
        plt.xlabel("Euclidean Distance")

    # Append the distances to the dataframe
    df["L2 Dist of X from Geometric Median"] = euclid_dists.tolist()

    # Normalise the dataframe
    if data_mean is None:
        data_mean = df.mean()
    if data_std is None:
        data_std = df.std()

    df = normalise(df, data_mean, data_std)  # Normalise data

    return df, data_mean, data_std


def extract_ood(df, train_lims_all, test_type="all",
                ood_lims_out=None, ood_lims_in=None, verbose=False):
    """
    From a giant dataset, extract an OOD set and a training set.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing all the data
    train_lims_all : array_like
        Of shape (#x_parameters, 2), contains the lower and upper bounds
        for every x parameter
    test_type : str, optional
        "all": Assumes every datapoint that's not in the training set is
        in the test set.
        "specified": Only put data in test set if it fulfils the requirements
        set out in ood_lims_upper and ood_lims_lower. The default is "all".
    ood_lims_upper : array_like, optional
        Of shape (#x_parameters, 2), contains the lower and upper bounds
        for the inner test set, e.g. the domain within the training set.
        The default is None.
    ood_lims_lower : array_like, optional
        Of shape (#x_parameters, 2), contains the lower and upper bounds
        for the outer test set, e.g. the domain outside (larger than) the
        training set. The default is None.
    verbose : bool, optional
        If True, print the results of splitting. The default is False.

    Returns
    -------
    df_train : Pandas DataFrame
        DataFrame containing the training data.
    df_test: Pandas DataFrame
        DataFrame containing the test (OOD) data.

    """
    # CREATE THE TRAIN DF
    train_inds = np.full(len(df), True)
    for i in range(len(train_lims_all)):
        up = (df[df.columns[i]] >= train_lims_all[i][0])
        low = (df[df.columns[i]] <= train_lims_all[i][1])
        res = up * low
        train_inds = train_inds * res

    df_train = df[train_inds]

    if test_type == "all":
        df_test = df[~train_inds]

    elif test_type == "specified":
        assert ood_lims_in is not None, "Please specify lower test domain"
        assert ood_lims_out is not None, "Please specify upper test domain"

        # Check which datapoints are in the inside OOD
        upper_in = np.full(len(df), True)
        lower_in = np.full(len(df), False)
        for i in range(len(ood_lims_in)):
            # Find rows in current column that are above the lower limit
            # ALL have to be above lower constraint to be in inside OOD
            upper_in = upper_in & (df[df.columns[i]] > ood_lims_in[i][0])

            # Find rows in current column that are below the upper limit
            # ANY have to be below upper contraint to be in the inside OOD
            lower_in = lower_in | (df[df.columns[i]] < ood_lims_in[i][1])

        test_inside_inds = upper_in & lower_in

        # Check which datapoints are in the outside OOD
        upper_out = np.full(len(df), False)
        lower_out = np.full(len(df), True)
        for i in range(len(ood_lims_out)):
            # Find rows in current column that are above the lower limit
            # ANY have to be above lower constraint to be in outside OOD
            upper_out = upper_out | (df[df.columns[i]] > ood_lims_out[i][0])

            # Find rows in current column that are below the upper limit
            # ALL have to be below upper contraint to be in the outside OOD
            lower_out = lower_out & (df[df.columns[i]] < ood_lims_out[i][1])

        test_outside_inds = upper_out & lower_out

        test_inds = test_outside_inds | test_inside_inds

        df_test = df[test_inds]

        used_inds = test_inds | train_inds
        df_unused = df[~used_inds]

    if verbose:
        if test_type == "specified":
            print(f"OOD lower:\t{test_inside_inds.sum()}")
            print(f"OOD upper:\t{test_outside_inds.sum()}")

    if test_type == "all":
        return df_train, df_test
    else:
        return df_train, df_test, df_unused


def dataset_from_df(df, x_cols, y_cols):
    """
    Helper function that takes a DataFrame containing all data,
    and outputs the corresponding dfs for x and y only, x and y values as
    torch.tensors, the corresponding indices leftover from splitting the data,
    and a TensorDataset ready for training.

    Parameters
    ----------
    df : DataFrame
        df containing ALL data for a specific dataset, i.e. all the ys and xs
    x_cols : array_like
        Which columns of the df relate to the independent variables (features)
    y_cols : array_like
        Which columns of the df relate to the dependent variable(s) (labels)

    Returns
    -------
    x_df : DataFrame
        Dataframe containing all X values only
    y_df : DataFrame
        DataFrame containing all Y values only
    x : torch.tensor
        Tensor containing all X values
    y : torch.tensor
        Tensor containing all Y values
    indices : array_like
        The preserved indices of the df
    dataset : TensorDataset
        Containing the x and y values.

    """
    x_df = df.iloc[:, x_cols]
    y_df = df.iloc[:, y_cols]
    x = torch.tensor(x_df.values).float()
    y = torch.tensor(y_df.values).float()
    indices = df.index.values
    dataset = TensorDataset(x, y)
    return x_df, y_df, x, y, indices, dataset


def split_by_PCA_mean(df, x_cols,
                      y_cols,
                      dist,
                      data_mean,
                      data_std,
                      PCA_components,
                      val_split,
                      seed=123,
                      verbose=False,
                      plots=False,
                      **kwargs): # noqa
    """
    Split a dataset according the distance from the geometric mean in
    reduced dimension sapce. Anything within the
    specified dist from mean goes to the training set, anything outside
    goes to the test set. Then, a portion of the training set gets randomly
    shuffled to make the validation set.

    FINISH [XXX]
    """
    # Normalise dataframe with all data
    df_ = normalise(df, data_mean, data_std)

    # Add columns for the PCA (calculated based on the full unnormalised
    # dataset)
    all_PCA_norm = post.PCA_transformdata(
        df_.iloc[:, x_cols], components=PCA_components)
    df_pca = pd.concat([df_, all_PCA_norm], axis=1)

    # Calculate the geometric median
    median = geometric_median(df_pca)

    # Add L2 dist from geometric mean to the dataframe
    euclid_dists = euclid_dist(
        df_pca, median, [
            len(data_mean), len(data_mean) + 1, len(data_mean) + 2])
    df_pca["L2 Dist of X from Geometric Median"] = euclid_dists.tolist()

    # Separate out the ones within our distance
    df_train = df_pca[df_pca[df_pca.columns[-1]] <= dist]
    df_test = df_pca[df_pca[df_pca.columns[-1]] > dist]

    # Reset index in training and test sets. Otherwise breaks later on
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Set up the test dataset etc
    x_test_df, y_test_df, x_test, y_test, test_indices, test_data = dataset_from_df(df_test, x_cols, y_cols) # noqa

    # Split training from validation data
    TRAIN, VAL = split_data(df_train, y_cols=y_cols, x_cols=x_cols,
                            val_split=val_split, seed=seed, test=False)
    x_train, y_train, train_data, train_indices = TRAIN
    x_val, y_val, val_data, val_indices = VAL

    if verbose:
        print(f"len(x_train): {len(x_train)}")
        print(f"len(x_test): {len(x_test)}")
        print(f"len(x_val): {len(x_val)}")
        print(f"Total: {len(x_train) + len(x_test) + len(x_val)} of {len(df)}")

    if plots:
        pca_train = df_train[df_train.columns[len(data_mean):]]
        pca_test = df_test[df_test.columns[len(data_mean):]]
        pca_val = df_train.iloc[val_indices,
                                len(data_mean):].reset_index(drop=True)

        ax = post.PCA_plot([pca_train, pca_test, pca_val],
                           labels=["Train", "Test", "Val"],
                           figsize=(8, 8),
                           legend_num=0,
                           **kwargs)

        ax.scatter(median[len(data_mean)],
                   median[len(data_mean) + 1],
                   median[len(data_mean) + 2],
                   c='k',
                   s=10)

    return [x_train, y_train, train_data, train_indices], [x_val, y_val, val_data, val_indices], [x_test, y_test, test_data, test_indices] # noqa


def split_by_bounds(df,
                    x_cols,
                    y_cols,
                    train_lims_all,
                    data_mean,
                    data_std,
                    ood_lims_in=None,
                    ood_lims_out=None,
                    PCA_components=None,
                    val_split=0.1,
                    seed=123,
                    verbose=False,
                    plots=False,
                    **kwargs):
    """
    Split a dataset according to some fixed bounds. Anything within the
    specified domain goes to the training set, anything outside goes to the
    test set. Then, a portion of the training set gets randomly shuffled
    to make the validation set.

    FINISH [XXX]
    """
    # Set out the training boundary

    df_ = normalise(df, data_mean, data_std)

    # Normalise the training domain
    train_lims_all_norm = normalise_lims(
        train_lims_all, data_mean[x_cols], data_std[x_cols])
    if ood_lims_in is not None and ood_lims_out is not None:
        test_type = "specified"
        ood_lims_in_norm = normalise_lims(
            ood_lims_in, data_mean[x_cols], data_std[x_cols])
        ood_lims_out_norm = normalise_lims(
            ood_lims_out, data_mean[x_cols], data_std[x_cols])
        df_train, df_test, df_unused = extract_ood(df_, # noqa
                                                   train_lims_all_norm,
                                                   test_type=test_type,
                                                   ood_lims_out=ood_lims_out_norm, # noqa
                                                   ood_lims_in=ood_lims_in_norm, # noqa
                                                   verbose=verbose)
        df_unused = df_unused.reset_index(drop=True)
        x_unused_df, y_unused_df, x_unused, y_unused, unused_indices, unused_data = dataset_from_df(df_unused, x_cols, y_cols) # noqa
    else:
        test_type = "all"
        df_train, df_test = extract_ood(
            df_, train_lims_all_norm, test_type=test_type, verbose=verbose)

    # Reset index in training and test sets. Otherwise breaks later on
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Set up the test dataset etc
    x_test_df, y_test_df, x_test, y_test, test_indices, test_data = dataset_from_df(df_test, x_cols, y_cols) # noqa

    # Split training from validation data
    TRAIN, VAL = split_data(df_train, y_cols=y_cols, x_cols=x_cols,
                            val_split=val_split, seed=seed, test=False)
    x_train, y_train, train_data, train_indices = TRAIN
    x_val, y_val, val_data, val_indices = VAL

    if verbose:
        if ood_lims_in is not None and ood_lims_out is not None:
            print(f"Train:\t\t{len(x_train)}")
            print(f"Test:\t\t{len(x_test)}")
            print(f"Val:\t\t{len(x_val)}")
            print(f"Unused:\t\t{len(x_unused)}")
            print(f"Total:\t\t"
                  f"{len(x_train) + len(x_test) + len(x_val) + len(x_unused)}"
                  f" (of {len(df_)})")
        else:
            print(f"Train:\t\t{len(x_train)}")
            print(f"Test:\t\t{len(x_test)}")
            print(f"Val:\t\t{len(x_val)}")
            print(f"Total:\t\t"
                  f"{len(x_train) + len(x_test) + len(x_val)} (of {len(df)})")

    if plots:
        if PCA_components is not None:
            # Get PCA of NORMALISED data, but according to UNNORMALISED PCA
            # components
            x_train_PCA = post.PCA_transformdata(
                x_train, components=PCA_components)
            x_test_PCA = post.PCA_transformdata(
                x_test, components=PCA_components)
            x_val_PCA = post.PCA_transformdata(
                x_val, components=PCA_components)
            if ood_lims_out is not None and ood_lims_in is not None and x_unused.shape[0] != 0: # noqa
                x_unused_PCA = post.PCA_transformdata(
                    x_unused, components=PCA_components)
                post.PCA_plot([x_train_PCA, x_test_PCA, x_val_PCA, x_unused_PCA], # noqa
                              labels=["Train", "Test", "Val", "Unused"],
                              colours=['r', 'g', 'b', 'grey'],
                              legend_num=0,
                              **kwargs)
            else:
                post.PCA_plot([x_train_PCA, x_test_PCA, x_val_PCA],
                              labels=["Train", "Test", "Val"],
                              legend_num=0,
                              **kwargs)

    return [x_train, y_train, train_data, train_indices], [x_val, y_val, val_data, val_indices], [x_test, y_test, test_data, test_indices] # noqa


def split_data(df,
               x_cols,
               y_cols,
               component=None,
               cutoff_percentile=(0.005, 0.05),
               val_split=0.2,
               seed=1234,
               return_df=False,
               verbose=False,
               test=True):
    """
    Splits a dataframe into train, test (i.e. OoD) and validate (i.e. within
    the domain of train data).
    Parameters
    ----------
        df : Pandas Dataframe
            Contains all the raw data (pandas DataFrame)
        component : int
            Which column to sort the data by in the DataFrame
        x_cols : list
            Which columns in the df contain the independent variable(s)
        y_cols : list
            Which columns in the df contain the dependent variable(s)
        cutoff_percentile : tuple, default: (0,005,0.05)
            How many rows from the top and bottom of the df to cut off
            (when sorted by 'component') to make the ood set.
            Defulat: (0.005, 0.05), i.e. 0.5% from bottom and 5% from top
        val_split : float, default: 0.2 (i.e. 20%)
            How much of the dataframe to split off for the validation set.
        seed : int
            Random seed to use for val_split
        return_df  : bool
            If True, return a Pandas DataFrame for each dataset only
        verbose : bool
            If True, print the progress and summary of data pre-processing
        test : bool
            If True, produce a Train, Test and Validation sets
            If False, only produce Train and Validation sets

     Returns
     -------
        [x_train, y_train, train_data, train_indices] : list
            A list of parameters relating to the train dataset.
            x_train : Torch Tensor
                All independent variables X in the training dataset. Of shape (num_datapoints, num_independent_vars) # noqa
            y_train : Torch Tensor
                All dependent variables y in the training dataset. Of shape (num_datapoints, 1) # noqa
            train_data : Torch TensorDataset
                Contains the x_train and y_train Tensors.
            train_indices : NumPy array
                A 1D array of shape (num_datapoints,) that contains the original indices of the now-sorted DataFrame # noqa
        [x_val, y_val, val_data, val_indices] : list
            A list of parameters relating to the validation dataset.
            x_val : Torch Tensor
                All independent variables X in the validation dataset. Of shape (num_datapoints, num_independent_vars) # noqa
            y_val : Torch Tensor
                All dependent variables y in the validation dataset. Of shape (num_datapoints, 1) # noqa
            val_data : Torch TensorDataset
                Contains the x_val and y_val Tensors.
            val_indices : NumPy array
                A 1D array of shape (num_datapoints,) that contains the original indices of the now-sorted DataFrame # noqa
        [x_test, y_test, test_data, test_indices] : list
            A list of parameters relating to the test dataset.
            x_test : Torch Tensor
                All independent variables X in the test dataset. Of shape (num_datapoints, num_independent_vars) # noqa
            y_test : Torch Tensor
                All dependent variables y in the test dataset. Of shape (num_datapoints, 1) # noqa
            test_data : Torch TensorDataset
                Torch TensorDataset containing the x_test and y_test Tensors.
            test_indices : NumPy array
                A 1D array of shape (num_datapoints,) that contains the original indices of the now-sorted DataFrame # noqa
    """

    df_len = len(df)

    # Find out which component we've chosen to sort by
    if component is not None:
        component_name = list(df.columns.values)[component]
        if verbose:
            print("Ordering by: ", component_name)

    # Print out which parameter(s) we wish to have as our dependent variables
    if verbose:
        print("y variable(s): ")
        if isinstance(y_cols, list):
            for value in y_cols:
                print(f"\t {list(df.columns.values)[value]}")
        else:
            print(f"\t{list(df.columns.values)[y_cols]}")

    # Sort dataframe and then cut off the top and bottom values dependent on
    # the percentile chosen
    if test:
        df = df.sort_values(df.columns[component])
    df = df.reset_index(drop=True)
    if test:
        upper = df[df[df.columns[component]] <
                   df[df.columns[component]].quantile(cutoff_percentile[0])]
        lower = df[df[df.columns[component]] >
                   df[df.columns[component]].quantile(1 - cutoff_percentile[1])] # noqa

        # Extract the indices of the top and bottom slices
        upper_indices = df[df.columns[component]] < df[df.columns[component]].quantile(cutoff_percentile[0]) # noqa
        upper_indices = upper_indices[upper_indices == True].index.values # noqa

        lower_indices = df[df.columns[component]] > df[df.columns[component]].quantile(1 - cutoff_percentile[1]) # noqa
        lower_indices = lower_indices[lower_indices == True].index.values # noqa

        if verbose:
            print("Cutting ", len(upper), " values off the top and ", len(lower), " off the bottom of the ordered dataset to create the OoD set.") # noqa

        # Combine upper and lower dfs to test set
        test_data = pd.concat([upper, lower])

        # Delete the test set rows from the main df
        df = df.drop(df[df[df.columns[component]] < df[df.columns[component]].quantile(cutoff_percentile[0])].index) # noqa
        df = df.drop(df[df[df.columns[component]] > df[df.columns[component]].quantile(1 - cutoff_percentile[1])].index) # noqa

    # Split the remaining data to training and validation
    train_data, val_data = train_test_split(
        # Note: this preserves the indices
        df, test_size=val_split, random_state=seed)

    train = dataset_from_df(train_data, x_cols, y_cols)

    # Validation set
    val = dataset_from_df(val_data, x_cols, y_cols)

    if test:
        # Test set
        test = dataset_from_df(test_data, x_cols, y_cols)

    # Print some final stats
    if verbose:
        if test:
            print(f"Split data with {len(train[2])} ({100*len(train[2])/df_len:.2f}%) training points, " # noqa
                  f"{len(val[2])} ({100*len(val[2])/df_len:.2f}%) validation points and "  # noqa
                  f"{len(test[2])} ({100*len(test[2])/df_len:.2f}%) test (OoD) points.") # noqa
        else:
            print(f"Split data with {len(train[2])} ({100*len(train[2])/df_len:.2f}%) training points and" # noqa
                  f"{len(val[2])} ({100*len(val[2])/df_len:.2f}%) validation points") # noqa

    if return_df:
        if test:
            return [train[0], train[1]], [val[0], val[1]], [test[0], test[1]]
        else:
            return [train[0], train[1]], [val[0], val[1]]
    else:
        if test:
            return [train[2], train[3], train[5], train[4]], [
                val[2], val[3], val[5], val[4]], [test[2], test[3], test[5], test[4]] # noqa
        else:
            return [train[2], train[3], train[5], train[4]], [
                val[2], val[3], val[5], val[4]]
