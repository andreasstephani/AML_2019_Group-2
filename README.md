# AML_2019 Coursework, Part 1, Group 2

## Experiments with gradient descent

### Introduction

Gradient descent is an optimization technique which can be used to minimise a function. This is a very important and useful tool in machine learning. Machine Learning models have a loss function, which is a way to determine how well the model has performed given the different values of each of its parameters. Therefore, gradient descent can be used to find the parameter values that minimise the loss function of the model (i.e. find the parameters that give the lowest loss). The procedure involves taking steps from a starting point on the loss function by evaluating the gradient at each step and adjusting the parameters, until the minimum point is reached.

### The Six-Hump Camel Function

In this coursework, it will be desmonstrated how gradient descent can be used to minimise the Six-Hump Camel function which is illustrated in the figure below. The function has two global minimum points at (0.0898,-0.7126) and (-0.0898,0.7126).

![Six-Hump Camel Function](https://user-images.githubusercontent.com/51288218/61081430-1f71bd80-a41f-11e9-883a-a4b582f3c638.PNG)

### Plain Vanilla Gradient Descent

Firstly, plain vanilla gradient descent will be used to minimise the Six-Hump Camel function (i.e. find the global minimum point).  

The graph below shows the loss path until convergence and the loss function using an initial point of (1,1) and a step-size, eta = 0.001.

![Plain Vanilla](https://user-images.githubusercontent.com/51288218/61085184-8eebab00-a427-11e9-8472-f4d60c61d388.PNG)

The plain vanilla algorithm converges at the global minimum in 1839 steps. However, depending on the starting point it can get stuck into a saddle point.(see notebook)

Experiments with different step-sizes(eta) are carried out in order to investigate the behaviour of the plain vanilla algorithm and the number of steps needed for convergence.

![Step-sizes plain vanilla](https://user-images.githubusercontent.com/51288218/61087427-86966e80-a42d-11e9-8c36-337d9737994e.png)

The figures show that as the value of eta increases the steps needed for convergence decrease. However, if eta is high it might cause the algorithm not to converge and this is the case with **eta** = 0.2 (i.e. red points on right graph).
Therefore, plain vanilla diverges if the stepsize is too big and can be slow if it is too small.

### Two variants of Plain Vanilla Gradient Descent




