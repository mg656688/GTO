# GTO_Algorithm

> main.py is the main class that has the main loop and by using the GTO.py "optimizer" and other files, it will work on 36 optimization function listed below:

`` 'Sphere', 'Rastrigin', 'Ackley', 'Griewank', 'Schwefel P2.22',
                       'Rosenbrock', 'Sehwwefel P2.21', 'Quartic', 'Schwefel P1.2', 'Penalized 1',
                       'Penalized 2', 'Schwefel P2.26', 'Step', 'Kowalik', 'Shekel Foxholes',
                       'Goldstein-Price', 'Shekel 5', 'Branin', 'Hartmann 3', 'Shekel 7',
                       'Shekel 10', 'Six-Hump Camel-Back', 'Hartmann 6', 'Zakharov', 'Sum Squares',
                       'Alpine', 'Michalewicz', 'Exponential', 'Schaffer', 'Bent Cigar',
                       'Bohachevsky 1', 'Elliptic', 'Drop Wave', 'Cosine Mixture', 'Ellipsoidal',
                       'Levy and Montalvo 1' ``
> GTO.py is the optimizer 

> dimension.py and bound_X.py provide the boundaries and the dimensions for each optimization function 


> ideal_F provides the ideal fitness for each optimization function 

> benchmark.py provides the 36 optimization functions' formula


# The output will be 2 CSV files :
  - table.CSV shows 6 values for each optimization 
    - ['average fitness', 'std fitness', 'worst fitness', 'best fitness', 'ideal fitness', 'time taken']
  - loss_curves.CSV shows the loss curve values through 500 iteration for each optimization function 

**update 

> We used the Gorilla Troop Optimizer (GTO) as an optimizer for training a simple NN with (NSL-KDD) Network intrusion dataset, in other words we used it find the best weights to achieve a higher accuracy and return the best fitness function value which is the loss function in our case.

[Source Code](https://github.com/ZongSingHuang/Artificial-Gorilla-Troops-Optimizer)

[Artificial gorilla troops optimizer paper](https://onlinelibrary.wiley.com/doi/full/10.1002/int.22535)
