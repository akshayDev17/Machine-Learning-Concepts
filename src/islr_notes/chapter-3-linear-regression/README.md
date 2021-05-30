# Table of Contents

1. [Introduction](#introduction)
2. [Errors](#lr-error)
3. [Optimal Linear Regression](#optimal-lr)
4. [Inaccuracies in Sample mean and True Mean](#sample-mean-vs-true-mean)
5. [Standard Error of Sample mean](#standard-error-mu)
6. [Standard errors for the beta-parameters](#standard-error-beta)
   1. [Confidence Intervals](#ci)
7. [Residuals - SSR](#ssr)
8. [Sum of squared totals - SST](#sst)
9. [Relation between Residuals, SSE and SST](#sst,sse,ssr)
10. [Null Hypothesis](#nh)
11. [t-distribution](#t-distribution)
    1. [Introduction](#t-intro)
    2. [t-statistic of the sample-data of a random variable](#t-statistic)
    3. [p-value test for simple linear regression slope parameter](#p-value)
    4. [Obtaining T-distribution from the T-statistic](#distribution-from-statistic)
    5. [What's a t-test?](#t-test)
12. [Root Mean Square Error - Bessel's correction](#bessel)
13. R-squared statistic
    1. [Introduction](#r2-intro)
    2. [Adjusted R-squared](#adjusted-r2)
    3. [Better metrics than R-squared](#better-than-r2)
14. 





# Introduction<a name="introduction"></a>

- prediction function f-hat of X is of the form beta-0 + beta-1 times X. this yields y-hat-sub-i, the predicted value of the dependent variable.
- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7B%5Ctextbf%7B%20True%20function%3A%20%7D%7D%20y_i%20%3D%20%5Cbeta_0%20&plus;%20%5Cbeta_1.x_i%20&plus;%20%5Cin_i%20%5Cnewline%20%5Ctextrm%7B%5Ctextbf%7BPredictive%20function%7D%3A%20%7D%5Chat%7By%7D_i%20%3D%20%5Chat%7B%5Cbeta_0%7D%20&plus;%20%5Chat%7B%5Cbeta_1%7Dx_i)
- this is known as simple linear regression, wherein only 1 feature exists, when multiple features exist, then the linear regression is called as multiple linear regression.



# Errors<a name="lr-error"></a>

- difference between the actual and predicted value of the predictand , for a sample. ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?e_i%20%3D%20y_i%20-%20%5Chat%7By_i%7D)
- sum of squared errors is adding up the squares of these sample-errors  ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7BSSE%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20e_i%5E2%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cleft%28y_i%20-%20%5Chat%7By_i%7D%20%5Cright%20%29%5E2)

# Optimal Linear Regression<a name="optimal-lr"></a>

- the problem becomes to find the optimal values of beta-0 and beta-1 such that the SSE is minimized.
- this can be achieved by replacing ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7By_i%7D%20%3D%20%5Chat%7B%5Cbeta_0%7D%20&plus;%20%5Chat%7B%5Cbeta_1%7Dx_i) in the expression for SSE and then differentiating the resulting expression with respect to beta-hat-0 and beta-hat-1.
- <img src="sse-derivation.png" />
- 



# Inaccuracies in Sample mean and True Mean<a name="sample-mean-vs-true-mean"></a>

- Since we never can have all possible samples, the dataset mean, also known as the sample mean(mu-cap, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cmu%7D) ), **which is the mean of all true predictand values for the dataset**, will never be equal to the true mean(mu, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cmu))
  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cmu%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum%5Climits_%7Bi%3D1%7D%5En%20y_i%20%2C%20%5Cmu%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum%5Climits_%7Bi%3D1%7D%5EN%20y_i)
  - But still, to yield a good model, its expected that an average of mu-cap( ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cmu%7D)) across all possible samples equal mu( ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cmu)).
  - On the basis of one particular set of observations **mu-cap( ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cmu%7D))** might overestimate mu( ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cmu)), and on the basis of another set of observations,  **mu-cap( ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cmu%7D))** might underestimate mu( ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cmu)). 
  - But if we could average a huge number of estimates of mu-cap obtained from a huge number of sets of observations, then this average would exactly equal mu. 
- if we estimate the beta-parameters, on the basis of a particular data set, then our estimates won’t be exactly equal to the true value of beta-parameters, i.e. their values in the true function. 
  - But if we could average the estimates obtained over a huge number of data sets, then the average of these estimates would be spot on!



# Standard Error of Sample mean<a name="standard-error-mu"></a>

- How accurate is the sample mean  **mu-cap( ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cmu%7D))**  as an estimate of mu( ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cmu))? we answer this question by computing the standard error of  **mu-cap( ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cmu%7D))** , written as SE( **mu-cap( ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cmu%7D))** ). 
- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Chat%7B%5Cmu%7D%20%26%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum%5Climits_%7Bi%3D1%7D%5En%20y_i%20%5C%5C%20SE%28%5Chat%7B%5Cmu%7D%29%20%26%3D%20var%5Cleft%28%5Cfrac%7B1%7D%7Bn%7D%5Csum%5Climits_%7Bi%3D1%7D%5En%20y_i%20%5Cright%29%20%3D%20%5Cfrac%7B1%7D%7Bn%5E2%7Dvar%5Cleft%28%5Csum%5Climits_%7Bi%3D1%7D%5En%20y_i%20%5Cright%29%20%5C%5C%20%26%3D%20%5Cfrac%7B1%7D%7Bn%5E2%7D%20%5Cleft%5B%5Csum%5Climits_%7Bi%3D1%7D%5En%20var%28y_i%29%20&plus;%202.%5Csum%5Climits_%7Bi%3D1%7D%5En%5Csum%5Climits_%7Bj%3Di&plus;1%7D%5En%20cov%28y_i%2C%20y_j%29%5Cright%5D%20%5C%5C%20%26%5Ctextrm%7Bassuming%20all%20%7D%20y_i%20%5Ctextrm%7Bare%20independent%2C%20which%20they%20practically%20are%7D%2C%20cov%28y_i%2C%20y_j%29%20%3D%200%20%5C%5C%20%26%3D%20%5Cfrac%7B1%7D%7Bn%5E2%7D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%5Csigma%5E2%20%5C%5C%20%5Cmathbf%7BSE%28%5Chat%7B%5Cmu%7D%29%7D%20%26%3D%20%5Cmathbf%7B%5Cfrac%7B%5Csigma%5E2%7D%7Bn%7D%7D%20%5Cend%7Balign*%7D)
- sigma is the standard deviation of each of the realizations y-sub-i of Y.
- its assumed that all y-sub-i's are independent and identically distributed, here y-1 just means the first sample's Y value, it doesn't mean a particular value, as the dataset could be shuffled row-wise, hence y-1 can have any of the sample values. similar is the argument for all other y-sub-i's.
- this deviation shrinks with n, the more observations we have, the smaller the standard error of  **mu-cap( ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cmu%7D))** .



# Standard errors for the beta-parameters<a name="standard-error-beta"></a>

- since y-sub-i = beta-0 + beta-1 times x-sub-i + epsilon-sub-i, where beta-0 and beta-1 are assumed to be constants.
  - additionally, its also assumed that X-sub-i are also all constants, because in reality, for the given ML problem, those are constant, and aren't randomly decided.
  - what is randomly occurring is the epsilon-sub-i , i.e. the irreducible error term.
  - moreover, Y-sub-i is also unknown for the test-samples, and also its relation with beta-0 + beta-1 times X-sub-i is unknown, hence its safe to assume that since epsilon-sub-i is random, Y-sub-i is also random.







![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta%7D_1%20%3D%20%5Cfrac%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%28x_i-%5Cbar%7Bx%7D%29%28y_i-%5Cbar%7By%7D%29%7D%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cleft%28%20x_i%20-%20%5Cbar%7Bx%7D%20%5Cright%20%29%5E2%7D%20%3D%20%5Cfrac%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%28x_i-%5Cbar%7Bx%7D%29.y_i%20-%20%5Csum%5Climits_%7Bi%3D1%7D%5En%28x_i-%5Cbar%7Bx%7D%29.%5Cbar%7By%7D%7D%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cleft%28%20x_i%20-%20%5Cbar%7Bx%7D%20%5Cright%20%29%5E2%7D%20%5Cnewline%20%5CRightarrow%20%5Cfrac%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%28x_i-%5Cbar%7Bx%7D%29.y_i%20-%20%5Cbar%7By%7D%5Csum%5Climits_%7Bi%3D1%7D%5En%28x_i-%5Cbar%7Bx%7D%29%7D%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cleft%28%20x_i%20-%20%5Cbar%7Bx%7D%20%5Cright%20%29%5E2%7D%20%3D%20%5Cfrac%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%28x_i-%5Cbar%7Bx%7D%29.y_i%20-%20%5Cbar%7By%7D.%28n%5Cbar%7Bx%7D%20-%20n%5Cbar%7Bx%7D%29%7D%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cleft%28%20x_i%20-%20%5Cbar%7Bx%7D%20%5Cright%20%29%5E2%7D%20%5Cnewline%20%5Cmathbf%7B%5Chat%7B%5Cbeta%7D_1%20%3D%20%5Cfrac%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%28x_i-%5Cbar%7Bx%7D%29.y_i%7D%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cleft%28%20x_i%20-%20%5Cbar%7Bx%7D%20%5Cright%20%29%5E2%7D%7D)

![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Chat%7B%5Cbeta%7D_1%20%26%3D%20%5Cfrac%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%28x_i-%5Cbar%7Bx%7D%29.y_i%7D%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cleft%28%20x_i%20-%20%5Cbar%7Bx%7D%20%5Cright%20%29%5E2%7D%20%5CRightarrow%20%5Ctextrm%7Bvar.%7D%28%5Chat%7B%5Cbeta%7D_1%29%20%3D%20%5Ctextrm%7Bvar.%7D%5Cleft%28%20%5Cfrac%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%28x_i-%5Cbar%7Bx%7D%29.y_i%7D%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cleft%28%20x_i%20-%20%5Cbar%7Bx%7D%20%5Cright%20%29%5E2%7D%20%5Cright%20%29%20%5C%5C%20%5Ctextrm%7Bvar.%7D%28%5Chat%7B%5Cbeta%7D_1%29%20%26%3D%20%5Ctextrm%7Bvar.%7D%5Cleft%28%20%5Cfrac%7B1%7D%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cleft%28%20x_i%20-%20%5Cbar%7Bx%7D%20%5Cright%20%29%5E2%7D.%5Csum%5Climits_%7Bi%3D1%7D%5En%20%28x_i-%5Cbar%7Bx%7D%29.y_i%20%5Cright%20%29%20%3D%20%5Cleft%28%20%5Cfrac%7B1%7D%7B%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cleft%28%20x_i%20-%20%5Cbar%7Bx%7D%20%5Cright%20%29%5E2%7D%20%5Cright%20%29%5E2%20.%5Ctextrm%7Bvar.%7D%5Cleft%28%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%28x_i-%5Cbar%7Bx%7D%29.y_i%20%5Cright%20%29%20%5Cend%7Balign*%7D)

* since all X-sub-i's are constant, their mean, X-bar, will also be constant, hence the sum of squares of the difference between X-sub-i and X-bar over all i's, from 1 to n, is also constant, and hence can be pulled out from the expression of the variance.
* <img src="variance-beta_hat-1-p1.png" />
  <img src="variance-beta_hat-1-p2.png" />
  * under the assumption that any 2 error terms are uncorrelated, i.e. all error have a pairwise 0 correlation.
* <img src="variance-beta0-derivation-1.png" /> 

<img src="beta-0-p3.png" />

- Notice that the standard error for the slope parameter, i.e. beta-1-hat, is smaller when the x-sub-i are more spread out; intuitively we have more leverage to estimate a slope when this is the case.



## Confidence Intervals<a name="ci"></a>

- A 95 % confidence interval is defined as a range of values such that with 95 % interval probability, the range will contain the true unknown value of the parameter.
- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?C.I.%20%3D%20%5Cleft%5B%20%5Chat%7B%5Cbeta_1%7D%20-%202.SE%28%5Chat%7B%5Cbeta_1%7D%29%2C%20%5Chat%7B%5Cbeta_1%7D%20&plus;%202.SE%28%5Chat%7B%5Cbeta_1%7D%29%20%5Cright%20%5D)
- 





# Residuals<a name="ssr"></a>

- the part of the variance of predictand values that is left unexplained by the regression model. ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7BSSR%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cleft%28%20%5Chat%7By_i%7D%20-%20%5Cbar%7By%7D%20%5Cright%20%29%5E2)



# Sum of squared totals<a name="sst"></a>

- the entire variance of the predictands. ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7BSST%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cleft%28%20y_i%20-%20%5Cbar%7By%7D%20%5Cright%20%29%5E2)



# Relation between Residuals, SSE and SST<a name="sst,sse,ssr"></a>

- SST = SSE + SSR.
- <img src="sst-sse-ssr.png" />







# Null Hypothesis<a name="nh"></a>

-  The most common hypothesis test involves testing the null hypothesis of  *H0 : There is no relationship between X and Y* .
- This is compared to the alternative hypothesis alternative hypothesis *Ha : There is some relationship between X and Y* .
- basically, H0 means ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbeta_1%20%3D%200) , and Ha means ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbeta_1%20%5Cne%200)
- to prove the alternative hypothesis, β1 should be sufficiently far from zero that we can be confident that β1 is non-zero. 
  - If ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?S.E.%5Cleft%28%20%5Chat%7B%5Cbeta_1%7D%20%5Cright%20%29) is small, then even relatively small values of ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D) may provide strong evidence that ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbeta_1%20%5Cne%200) .
  - if ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?S.E.%5Cleft%28%20%5Chat%7B%5Cbeta_1%7D%20%5Cright%20%29) is large, then ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D) must be large in absolute value in order for us to reject the null hypothesis
- hence, a t-statistic is computed to determine to the null hypothesis.
  ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bt-statistic%20%3D%20%7D%20%5Cfrac%7B%5Chat%7B%5Cbeta_1%7D%20-%200%7D%7B%5Ctextrm%7BS.E.%7D%5Cleft%28%5Chat%7B%5Cbeta_1%7D%20%5Cright%20%29%7D)
  - the larger the t-statistic, the farther is 0 from the ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D) value. the smaller the probability value of the estimate of ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D) to take up significant values, and thus the more likely it is of Y being linearly dependent on X.
  - the smaller the t-statistic, the higher chance of ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D) to be 0, and thus the larger chance of Y being not linearly dependent on X. 
  - we say estimate, because ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D) is estimated only for the current sample, but a different sample, i.e. a different dataset is bound to give a different ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D).
- The probability of observing any number equal to |t| or larger in absolute value, assuming β1= 0 is called **p-value**.
- 



# t-distribution<a name="t-distribution"></a>



## Introduction<a name="t-intro"></a>



- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20f%28t%29%20%3D%20c%5Cleft%281%20&plus;%20%5Cfrac%7Bt%5E2%7D%7B%5Cnu%7D%20%5Cright%20%29%5E%7B-%5Cleft%28%5Cfrac%7B1&plus;%5Cnu%7D%7B2%7D%20%5Cright%20%29%7D)
  - t is the variable value, and ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cnu) is the degrees of freedom.
  - c is the constant used so as to normalize this function, i.e. area under the curve of this function = 1.
- this exhibits fatter tails
  - Our use of an estimate for the  variance instead of an known/assumed value interjects another source of  variation into the T-statistic that a Z-statistic doesn't have. 
  - Hence  the values of the T-statistic will vary more wildly and hence it's  distribution has fatter tails which represent that more spread out  variation.



## t-statistic of the sample-data of a random variable<a name="t-statistic"></a> 

- Assume that we know that X is a random variable drawn from a normal distribution , such that we only know the mean value of that normal distribution.
- we only have n samples drawn from this distribution, and we wish to know an *approximated normal distribution*.
- calculate the sample mean and the sample variance.
  ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Cbar%7Bx%7D%20%26%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum%5Climits_%7Bi%3D1%7D%5En%20x_i%20%5C%5C%20var.%28x%29%20%26%3D%20%5Cfrac%7B1%7D%7Bn-1%7D%5Csum%5Climits_%7Bi%3D1%7D%5En%20%28x_i%20-%20%5Cbar%7Bx%7D%29%5E2%20%5C%5C%20%5Cend%7Balign*%7D)
  - Check [Bessel's correction](#bessel) for **why the denominator has n-1 instead of n**.
- the following random variable t, also called the t-statistic of the random variable X, samples a t-distribution with n-1 total degrees of freedom.
  ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20t%20%26%3D%20%5Cfrac%7B%5Cbar%7Bx%7D%20-%20%5Cmu%7D%7BVar.%28x%29/%5Csqrt%7Bn%7D%7D%20%5C%5C%20%5Cend%7Balign*%7D)
  - here, the t-distribution is ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B200%7D%20f%28t%29%20%3D%20c%5Cleft%281&plus;%5Cfrac%7Bt%5E2%7D%7Bn-1%7D%5Cright%29%5E%7B%20-%5Cleft%28%20%5Cfrac%7Bn%7D%7B2%7D%20%5Cright%20%29%7D)
  - [prove that this r.v. samples the above t-distribution function](https://www.freecodecamp.org/news/the-t-distribution-a-key-statistical-concept-discovered-by-a-beer-brewery-dbfdc693184/).



- 



## p-value test for simple linear regression slope parameter<a name="p-value"></a>

- for ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D) , to prove the fact that Y is independent of X, i.e. to prove the null hypothesis as being true, 
- Typical p-value cutoff for rejecting the null hypothesis is 0.005.



## Obtaining T-distribution from the T-statistic<a name="distribution-from-statistic"></a>

- A single t-test produces a single t-value. 
- Now, imagine the following  process. 
  - First, let’s assume that the null hypothesis is true for the population. 
  - Now, suppose we repeat our study many times by drawing many random  samples of the same size from this population. 
  - Next, we perform [t-tests](#t-test) on all of the samples and plot the distribution of the t-values. 
  - This  distribution is known as a sampling distribution, which is a type of  probability distribution.



## What's a t-test?<a name="t-test"></a>

- checks if for 2 sample-sets of same size drawn from a population have means that are reliably different from each other.
- its the ratio of variance between the groups to the variance within the groups.
- hence, the t-statistic for ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D) is actually this, where the numerator assumes 2 distributions, the one that is available, that is used to estimate ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D) and the other distribution has assumed that ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D) = 0.
  - the S.E. is obtained from the sample considered, i.e. the first distribution in the numerator.
- 







# Root Mean Square Error - Bessel's correction<a name="bessel"></a>

- ![equation](https://latex.codecogs.com/gif.latex?RSE%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7D%7Bn-2%7DRSS%7D%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7D%7Bn-2%7D%5Csum%5Climits_%7Bi%3D1%7D%5En%20%28y_i%20-%20%5Chat%7By_i%7D%29%5E2%7D)
- The RSE is an estimate of the standard deviation of ![equation](https://latex.codecogs.com/gif.latex?%5Cin) , the irreducible error. 
- instead of ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?s_n) we use ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?s_%7Bn-1%7D) as the unbiased estimator of the standard deviation of the population, where both terms are calculated from the given sample, and  ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?s_n) is the sample variance.
- [reference - wikipedia](https://en.wikipedia.org/wiki/Bessel%27s_correction)
  - this will be the proof for when degrees of freedom = n-1, i.e. only when the random variable exists, and nothing *depends* on it.
- <img src="bessel-correction.png" />
- hence, for a random variable having a **normal distribution**, such that a sample-space of data is chosen from the population space
  - has the sample variance as a biased estimator of the true variance.
  - to get the unbiased estimation of the true variance, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?S_%7Bn-1%7D) is to be calculated, rather than ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?S_%7Bn%7D).
- the variance will almost always be smaller when calculated using the sum of squared distances to the sample mean, compared to using the sum of  squared distances to the population mean. 
- **The one exception** to this is  when the **sample mean** happens to be **equal to** the **population mean**, in  which case **the variance is also equal**.
- RSE value by itself depends upon the value of the dependent variable, hence its **better to use it as** a **percentage of mean value of** the **dependent variable**.





# ![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B300%7D%20%5Cmathbf%7BR%5E2%7D) statistic



## Introduction<a name="r2-intro"></a>

- A value between 0 and 1, this **metric of accuracy** is independent of the scale of the dependent variable.
- ![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20R%5E2%20%3D%20%5Cfrac%7B%5Ctextrm%7Bvariance%20explained%20by%20the%20model%7D%7D%7B%5Ctextrm%7Btotal%20variance%20of%20data%7D%7D%20%3D%20%5Cfrac%7BTSS%20-%20RSS%7D%7BTSS%7D%20%3D%201%20-%20%5Cfrac%7BRSS%7D%7BTSS%7D)
  - TSS measures the total variance in the response Y , and can be squares thought of as the amount of variability inherent in the response before the regression is performed. 
  - In contrast, RSS measures the amount of variability that is left unexplained after performing the regression. 
  - Hence, **TSS - RSS** measures the  amount of variability in the response that is explained (or removed) by performing the regression.
-  A number near 0 indicates the linear model is wrong, or the inherent, i.e. irreducible error, ![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csigma%5E2%28%5Cin%29)  is high, or both. 
- **Except for linear regression**, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20R%5E2) is a terrible metric to measure the accuracy of a regression analysis.
  - The basic problem with  ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20R%5E2)  is that it depends on too many things (**even when adjusted in multiple  regression**), but most especially on the variance of the independent  variables and the variance of the residuals.  
  - Normally it tells us *nothing* about *linearity* or *strength of relationship* or even *goodness of fit* for comparing a sequence of models. 
- when the independent variables are set to standard values, essentially controlling for the effect of their variance, ![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%201-R%5E2) is really a proxy for the variance of the residuals(residual error), suitably standardized.



## Adjusted ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B200%7D%20R%5E2)<a name="adjusted-r2"></a>

- The only cases where there's much of a difference between ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20R%5E2)  and the adjusted-![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20R%5E2)  are when you are using *loads* of parameters compared to the data (p is comparable to n).  
- A dataset of 1,000 data points, sliced up into 18 small datasets, the slicing added only 18 parameters; the  adjustments to ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20R%5E2) wouldn't even affect the second decimal place, except possibly in the  end segments where there were only a few dozen data points: and it would *lower* them.





## Better metrics than ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B200%7D%20R%5E2)<a name="better-than-r2"></a>

- For **model selection** you can look to <font color="red">AIC and BIC = ?</font>.
- For expressing the adequacy of a model, look at the variance of the residuals. 