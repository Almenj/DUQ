# Welcome to duq (DeepUQ)!

DeepUQ (`duq`) is a Python package that can construct, train and perform inference on Bayesian Neural Network-based surrogate models. 

It implements Monte Carlo (MC) Dropout (Gal and Ghahramani, 2016) and Stochastic Gradient Langevin Dynamics (SGLD) (Welling and Teh, 2011) to stochastically train feed-forward networks, and then via sampling determine an approximation to the true predictive posterior distribution. The mean of this distribution is the most likely prediction, and the confidence interval (CI) can be returned either using standard deviatio or Highest Density Interval.

# How to get started
If you have Anaconda installed, getting started should be easy. 

## Creating a Conda Environment
1) Clone this repo
2) Navigate where irp-aol21 has been cloned in a terminal
3) Run `conda env create -f environment.yml` to create a new `duq_env` environment from the `environment.yml` file
4) Run `conda activate duq_env` to activate the new environment 
5) To check everything is working, navigate to `irp-aol21/tests` in a terminal window and run `python check_env.py`. The test will prompt the user when it completes successfully.

*If your computer has security certificate limitations:*: 
- The following command should be run in Anaconda prompt before creating an environment: `conda config --set ssl_verify no`
- After this, run `conda env create -f environment_force.yml` to run a modified version of the installation.
- PLEASE CONSULT WITH YOUR IT SPECIALIST BEFORE MAKING ANY CHANGES TO SECURITY POLICY

## Using `pip` and `setup.py`
You can also install it as a package using `pip` so that `duq` can be called from any Python environment, however this is not recommended. To do so:
1) Clone this repo
2) Navigate where irp-aol21 has been cloned in a terminal
3) Run `pip install -e ./` to globally install `duq`. Note that you may need to run `pip install -e ./ --user` to elevate privileges.

## Manually Installing Dependencies
Many of the functions can be run in this package without installing dependencies, or you may already have many of the required packages. You can choose to manually install any dependency you wish to use.

# How to use Grashopper to Perform Inference (using `flask` RESTful API)
1) Navigate to `irp-aol21/duq` in a terminal window. This should contain `app.py`
2) Run `flask run`
3) Open up `DUQ_Test.GH`
4) Test the interface; moving sliders should adjust the geometry and the colouring in Rhino. See the individual panels in Grasshopper for details of the prediction, plus a reference value. 

# How to Perform Inference Using Other Methods
Please see Section 2.2.7 in `irp-aol21/reports/aol21-final-report.pdf` for further details on how to carry out inference.