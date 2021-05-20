# LARNN: Linear Antisymmetric RNN

This pip package provides an implementation of the Linear Antisymmetric RNN (LARNN) cell from [Moe, Remonato, Grøtli and Gravdahl, 2020](http://proceedings.mlr.press/v120/moe20a.html). 

To install, run 

`git clone url` 
`pip install larnn` 


Importing `larnn` to your python project, gives access the LinearAntisymmetricCell class, as well as larnn.utils, which contain helper functions for shaping data to feed to a RNN with LinearAntisymmetricCells.

Usage examples are given in jupyter notebooks in the `examples` folder. `ex_1_introduction.ipynb` shows how to construct and train a simple LARNN on a one-dimensional example. `ex_2_var_step_size.ipynb` shows how to construct, train and make predictions with a LARNN when the dataset is unevenly sampled.