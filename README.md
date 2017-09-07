# Gradient Descent Method Implementation

I havent done any Feature Engineering per se. All the results are according to the original Feature Matrix.

The result obtained through the L2 closed form solution gave a much better 'w' vector than the gradient descent method.
The error was around 5.25 whereas the gradient descent error was around 6.45.

I tried various values for lambda including 10, 1, 0.1, 0.01, 0.001 and 0.0001. I found that 0.0001 was the best fit in
all the cases.

I took the 3 different values of p in (1,2] as 1.25, 1.5 and 1.75. The learning rate came to be the same in all the cases.
The lambda value also came out to be the same.

The code takes little over an hour to run. 