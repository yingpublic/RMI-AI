# Deeply Learning Derivatives
> 2018 Arxiv


## Example
Call option on a basket of stocks

## Strengthness
* Speed: compute valuation 1M times faster than MC
* Combine MC & DNN: using small numbers of Monte Carlo paths in the training set is very effective and the the neural network learns to average out the random error component of the Monte Carlo model found in the training set.
* 寫的條理分明（未細看）


## Future Work (eight points in the paper)
1. Applying transfer learning to accelerate the training of related derivative products
2. Using DNNs to approximate both valuation model and model calibration steps independently as a pipeline
3. Providing risk sensitivities through back propagation of neural networks as an alternative or complement to algorithmic differentiation
