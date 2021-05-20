# LARNN: Linear Antisymmetric RNN

This pip package provides an implementation of the Linear Antisymmetric RNN (LARNN) cell from [Moe, Remonato, Grøtli and Gravdahl, 2020](http://proceedings.mlr.press/v120/moe20a.html). It also shows how to use these network for unevenly sampled data and flexible online predictions, as described in Moe and Sterud, 2021 (coming in Proceedings of Machine Learning Research vol 144:1–11, 2021). 

To install the `larnn` packge with pip, run 

`git clone https://github.com/SINTEF/Linear-Antisymmetric-RNN` 

`pip install larnn` 


in the terminal.

Importing `larnn` to your python project, gives access the LinearAntisymmetricCell class, as well as larnn.utils, which contain helper functions for shaping data to feed to a RNN with LinearAntisymmetricCells.

Usage examples are given in jupyter notebooks in the `examples` folder. `ex_1_introduction.ipynb` shows how to construct and train a simple LARNN on a one-dimensional example. `ex_2_var_step_size.ipynb` shows how to construct, train and make predictions with a LARNN when the dataset is unevenly sampled. The packages needed for running the provided examples are listed in `requirements.txt`.