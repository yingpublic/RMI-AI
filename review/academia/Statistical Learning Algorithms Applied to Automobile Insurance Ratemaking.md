# Statistical Learning Algorithms Applied to Automobile Insurance Ratemaking
> 作者有Bengio!!!
> [paper](https://www.iro.umontreal.ca/~vincentp/Publications/itii.pdf)

## Data
* North American automobile insurer

## Strengthness
1. Compare several different categories of model including Ensemble: 
> Linear Regressions, GLMs, Decision Trees, Neural Networks and Support Vector Machines
2. Customized risk tolerance to which group of ppl they want to take
3. Considering actuarial fairness

4. Prove Var-Bias Delimma btw models:
> At this point it is interesting to consider that choosing among models may be guided by two different objectives, which sometimes yield to different answers: an operational objective (which model will yield to the best decisions/predictions on new data), or a ”modeling” objective (which model better describes the true underlying nature of the data). We will show an example in which the two approaches yield to different statistical tests and the operational approach yields to more conser- vative decisions (chooses simpler models). Another example of the difference between the two approaches is illustrated by the case of ridge regression: there is a regularized (biased) regression that brings better out-of-sample expected predictions than the maximum likeli- hood (unbiased) estimator.
