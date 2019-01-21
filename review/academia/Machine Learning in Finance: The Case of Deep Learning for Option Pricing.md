# Machine Learning in Finance: The Case of Deep Learning for Option Pricing

> Journal of Investment Management 2017
[Paper Link](https://srdas.github.io/Papers/BlackScholesNN.pdf)
[Python code](https://srdas.github.io/DLBook/DeepLearningWithPython.html#option-pricing)


## Idea
1. Without having knowledge on option pricing, we can still price the option.
2. We can add more parameters later on.


## Network/Model
* Input: 6 parameters
* Hidden: 4 hidden layers of 100 neurons each
* Output: one single exponential number (The final output layer comprises a single output neuron which we set to be the standard exponential function exp(·) because we need the output of the neural net to be non-negative with certainty, as option prices cannot take negative values.


## Other Training setting
* Activation function: LeakyReLU, ELU, ReLU, ELU (The neurons at each layer are chosen based on different “activation” functions)
* Dropout rate: 25%
* Loss function: MSE
* batch size: 64
* epochs: 10


## Data ** Simulated Data **
In order to create data for the assessment of how a deep neural net would learn this equation, we simulated a range of call option prices using a range of parameters shown
* training + validation

## Result
* In-sample: RMSE is 0.0112, ±1% of the strike. The average percentage pricing error (error divided by option price) is 0.0420, i.e., 4%.
* Out-of-sample: RMSE is also 0.0112, ±1% of the strike. The average percentage pricing error (error divided by option price) is 0.0421, i.e., 4%

## Strengthness
* Have python code.


## Weakness
* not actual data
