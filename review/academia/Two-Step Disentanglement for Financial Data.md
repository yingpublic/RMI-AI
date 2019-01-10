# Two-Step Disentanglement for Financial Data
> 2017 Sep, Facebook AI Research, [Paper Link](https://arxiv.org/pdf/1709.00199.pdf)

## What's the disentanglement?
> Try to seperate two different types of factors: 
> 1. specified factors: factors which are relevant for the classification task/label
> 2. unspecified factors: factors that are not correlated

## Idea
* Use adversarial training 
* classifier S: trained to predict the specified factors. The activations of S are then used to capture the specified
component of each sample.
* network Z: trained to recover the complimentary component
* A first loss on Z ensures that the output of both networks together (S and Z) is able to reconstruct the original sample
* A second loss, which is based on a third, adversarial, network, ensures that Z does not encode the specified factors
* The algorithm makes very weak assumptions about the distribution of the specified and unspecified factors.

## Network/Model
> * Encoding part: chose S and Z to be vectors of real numbers (rather than a one-hot vector)
> 1. deterministic encoder: X -> to specified components S = EncS(X)
> 2. deterministic encoder: X -> to unspecified components Z = EncS(X)

> > * Q: How to train S encoder EncS?
> > * A:  Fig. 1(a). Input: X, encodes it to a vector S, and then runs the S classifier on S to obtain the labels Y

> 

## Data
> * simulated CAPM data
> * daily returns of stocks


## Strength


## Weakness and my two cents
