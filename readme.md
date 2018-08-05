# Bayesian Models of the Area Under Roc Curve

## About

This repository contains Python code for my undergraduate thesis
_Bayesian Models of the Area Under Roc Curve_. It contains two directories:

* `py2`: Contains python2 scripts for Bayesian Analysis
* `ipynb`: Contains IPython notebooks with diagrams for experiment results.

## Introduction

Given the classification results on a test set, one can easily calculate the AUC by definition, that is to add up successive areas of trapezoids under the ROC curve. However, this one-dimensional AUC value does not tell the full story, since it does not contain any information about the sample variance or confidence intervals. An AUC value of 0.8 got from 100 test cases and one of 0.8 from 10000 test cases show no difference from the result, although the later one is intuitively more reliable and has lower degree of uncertainty than the previous one.

An alternative way is to calculate the AUC from a Bayesian view. A Bayesian approach computes the posterior probability distribution directly from a test set without dividing it, and therefore sidesteps the necessity to select parameters for division. It is highly favored in some performance evaluation problems; in particular, C Goutte proposed a probabilistic interpretation of precision, recall and F-score in 2005, and Zhang constructed a multi- class bayesian model for F1 score in 2015.

To determine to what degree an AUC measure is reliable, we develop a Bayesian probabilistic model to estimate the value of it, from which the variance and confidence interval can be easily obtained. We then describe a natural extension of our model for multiple classification problems. We argue that our multi-class model outperforms the frequentist method, especially when some classes have little data.

## Dependency

This repository is a python2 project. To run the project on your machine, a [conda virtual environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)
is recommended. The fastest way to obtain conda is to [install Miniconda](https://conda.io/docs/user-guide/install/index.html), a mini version of Anaconda.
After conda is successfully installed, run the following command to create a virtual environment called "redfish".
```commandline
conda create -n bayesian python=2.7
source activate bayesian
```
You can then install the dependencies inside the condo virtual environment via pip.
```commandline
pip install -r requirements.txt
```

The dependencies include:

* Pandas (Python Data Analysis Library)
* [Scikit-learn](http://scikit-learn.org/stable/)
* [PyMC3](https://github.com/pymc-devs/pymc3)
* Matplotlib
* Seaborn

## Model

See model.pdf if you are interested in the model.
