# AML_2019 Coursework, Part 1, Group 2

## Experiments with gradient descent

### Why is gradient descent important in machine learning?

Gradient descent is an optimization technique which can be used to minimise a function. This is a very important and useful tool in machine learning. Machine Learning models have a loss function, which is a way to determine how well the model has performed given the different values of each of its parameters. Therefore, gradient descent can be used to find the parameter values that minimise the cost function of the model (i.e. find the parameters that give the lowest loss). An example can be a linear regression model where the parameters are the beta coefficients and the cost function is the mean squared error (MSE). Thus, gradient descent could be used to find the beta coefficients that minimise the MSE of the linear regression model.


![Six-Hump Camel Function](https://user-images.githubusercontent.com/51288218/61081430-1f71bd80-a41f-11e9-883a-a4b582f3c638.PNG)

### How does Plain Vanilla Gradient Descent work?

Plain vanilla gradient descent can be used to minimise a function (i.e. find the minimum point). The procedure of doing this starts by choosing a starting point that lies within the domain of the chosen function. The very first step is usually a downhill movement from the starting point towards the direction specified by the gradient at that point. Then, the gradient at the new point is recalculated, and another step is taken in the direction that it specifies. For every step, the gradient of the loss function is calculated and the parameters are adjusted in the opposite direction. This process is repeated until the global minimum of the function or a point where no more downhill movement can be made (i.e. a local minimum) is reached.
