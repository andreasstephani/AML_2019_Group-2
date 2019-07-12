# AML_2019 Coursework, Part 1, Group 2

## Experiments with gradient descent

### Introduction - The Six-Hump Camel Function

In this coursework, it will be demonstrated how gradient descent can be used to find the global minimum of the Six-Hump Camel function which is illustrated in the figure below. The function has two global minimum points at (0.0898,-0.7126) and (-0.0898,0.7126).

![Six-Hump Camel Function](https://user-images.githubusercontent.com/51288218/61081430-1f71bd80-a41f-11e9-883a-a4b582f3c638.PNG)

### Plain Vanilla Gradient Descent

Firstly, plain vanilla gradient descent will be used to minimise the Six-Hump Camel function (i.e. find the global minimum point).  

The figures below show the loss path until convergence and the loss function for two different starting points. 

![pv](https://user-images.githubusercontent.com/51288218/61093431-691fcf80-a442-11e9-8f67-e5d6fdd710a0.png)

Starting Point|eta|Steps|Minimum found
---|---|---|---|
(1,1)|0.001|1839|(-0.0898 , 0.7126) 
(2,1)|0.001|1562|(1.6071 , 0.5687)

From the figures and table above it can be concluded that plain vanilla can find the global minimum but it can also get stuck at a saddle point depending on the starting point. 

Experiments with different step-sizes(eta) are carried out in order to investigate the behaviour of the plain vanilla algorithm and the number of steps needed for convergence.

![Step-sizes plain vanilla](https://user-images.githubusercontent.com/51288218/61087427-86966e80-a42d-11e9-8c36-337d9737994e.png)

eta|Steps|Minimum found
---|---|---|
0.001|1839|(-0.0898 , 0.7126)
0.01|181|(-0.0898 , 0.7126)
0.1|34|(-0.0898 , 0.7126)
0.2|Did not converge| 

The figures show that as the value of eta increases the steps needed for convergence decrease. However, if eta is high it might cause the algorithm not to converge and this is the case with **eta** = 0.2 (i.e. red points on right graph).
Therefore, plain vanilla diverges if the step-size is too big and can be slow if it is too small.

### Two variants of Gradient Descent

Gradient descent has the limitation of long running time because it uses the whole set of data to determine the next step. Therefore, stochastic gradient descent (SGD) is used which uses a subset of the data to find the next step. As a result, SGD sometimes cannot find the global minimum but it can get a very close approximation. 

For this project, experiments with Nesterov's Accelarated Gradient(NAG) and ADAM are performed.

#### Nesterov's Accelarated Gradient NAG

![NAG 1](https://user-images.githubusercontent.com/51288218/61094317-9a9a9a00-a446-11e9-8497-366b1cd7d4f5.png)

Starting Point|eta|Steps|Minimum found
---|---|---|---|
(1,1)|0.001|371|(-0.0833 , 0.6614) 
(2,1)|0.001|399|(-0.0833 , 0.6614)

NAG does not find the global minimum of the function but it finds a very close approximation and as it can be seen by the figures above it goes over the saddle point that plain vanilla was stuck.

![NAG STEPSIZE](https://user-images.githubusercontent.com/51288218/61094278-67580b00-a446-11e9-9241-d5621cc591bd.png)

eta|Steps|Minimum found
---|---|---|
0.001|371|(-0.0833 , 0.6614)
0.01|167|(-0.0833 , 0.6614)
0.05|50|(0.0833 , -0.6614)
0.08|27| (0.0833 , -0.6614)

It can be observed that as the value of eta increases the steps needed for convergence decrease and NAG still converges to a very close approximation of the global minimum. However, it can be noticed that there are some fluctuations on the right figure and the curves are not as smooth as the plain vanilla. Also, it is worth noting that 0.05 and 0.08 give the same approximation of the global minimum but with a different sign. The Six-Hump camel function has two global minimums with opposite signs which implies that as eta increases NAG can move and find an approximation of the second global minimum.

#### ADAM

![ADAM](https://user-images.githubusercontent.com/51288218/61094775-7dff6180-a448-11e9-8f93-978faed663b4.png)

Starting Point|eta|Steps|Minimum found
---|---|---|---|
(1,1)|0.001|1734|(-0.0898 , 0.7126 
(2,1)|0.001|1371|(1.6071 , 0.5687)

ADAM finds the global minimum of the function but also gets stuck at a saddle point like plain vanilla.

![ADAM 5](https://user-images.githubusercontent.com/51288218/61095102-ffa3bf00-a449-11e9-8ad4-2a17e65ed2f7.png)

eta|Steps|Minimum found
---|---|---|
0.001|1734|(-0.0898 , 0.7126)
0.01|539|(-0.0898 , 0.7126)
0.1|488|(-0.0898 , 0.7126)
0.2|487| (-0.0898 , 0.7126)

As the value of eta increases the steps needed for convergence decrease. However, it can be observed that the number of steps do not decrease as much as for plain vanilla and NAG. Also there are some fluctuations and the curves are not as smooth as the plain vanilla. Compared with the other two flavours, ADAM does not have the same decrease on its number of steps as eta increases (i.e. takes longer for 'large' eta compared to NAG and plain vanilla) but it finds the global minimum of the function.

###Conclusion



