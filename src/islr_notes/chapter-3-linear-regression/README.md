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
14. [Multiple Linear Regression](#mlr)
    1. [F-statistic](#f)
       1. [Degrees of Freedom in Statistics](#dof)





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
  
- ### Proof<a name="t-and-normal-distributions-proof"></a>

  - since our dataset is subject to change, so is our variance of the dataset.
  - we **assume** that we have the **prior knowledge** of the **expected value of** the dependent variable, **Y**.
  - hence, for a given value of variance(a.k.a. the variance of the irreducible error), Y is normally distributed, thus we have a conditional distribution, where ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?P%28Y%7C%5Csigma%5E2%29) is a normal distribution. 
  - the variance parameter, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Csigma%5E2),  is assumed to be distributed according to the **probability density function** of the **Gamma random variable ** 
  - Gamma Function : ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5CGamma%28z%29%20%3D%20%28z-1%29%20%5Ctimes%20%5CGamma%28z-1%29)
    - the gamma function can be thought of as a generalized factorial function, where ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bf%7D%28z%29%20%3D%20%28z-1%29%21%20%3D%20%28z-1%29%20%5Ctimes%20%5Ctextrm%7Bf%7D%28z-1%29)
    - the other representation of this gamma function is : ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5CGamma%28n%2C%20h%29%20%3D%20%5Cint%5Climits_%7Bh%7D%5E%7B%5Cinfty%7Dx%5E%7Bn-1%7De%5E%7B-x%7Ddx), where n and h are hyperparameters that influence the distribution of this function.
    - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Ctextrm%7Bln%7D%28n%21%29%20%26%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20ln%28i%29%20%5Capprox%20%5Cint%5Climits_%7B1%7D%5En%20ln%28x%29dx%20%3D%20xln%28x%29-x%7C%5En_1%5C%5C%20%26%3D%20nln%28n%29%20-%20n%20-%20%280%20-%201%29%20%3D%20nln%28n%29%20-%20n%20&plus;%201%20%5Capprox%20nln%28n%29%20-%20n%20%5C%5C%20n%21%20%26%3D%20%5Cint%20%5Climits_%7B0%7D%5E%7B%5Cinfty%7D%20e%5E%7B-x%7Dx%5En%20dx%20%5C%2C%5C%2C%5C%2C%20%5Ccdots%20%5C%2C%5C%2C%5C%2C%20%5Ctextrm%7Bintegral%20definition%20of%20factorial%7D%5C%5C%20%5Cfrac%7Bd%5Cleft%28%20ln%5Cleft%28n%21%20%5Cright%20%29%20%5Cright%20%29%7D%7Bdx%7D%20%26%3D%20%5Cfrac%7Bd%5Cleft%28%20%5Cint%20%5Climits_%7B0%7D%5E%7B%5Cinfty%7D%20ln%5Cleft%28e%5E%7B-x%7Dx%5En%20%5Cright%20%29dx%20%5Cright%20%29%7D%7Bdx%7D%20%3D%20%5Cfrac%7Bd%20%5Cleft%28%20ln%28e%5E%7B-x%7Dx%5En%29%5Cright%20%29%7D%7Bdx%7D%20%5C%5C%20%26%3D%20%5Cfrac%7Bd%5Cleft%28%20nlnx%20-%20x%20%5Cright%29%7D%7Bdx%7D%20%3D%20%5Cfrac%7Bn%7D%7Bx%7D%20-%201%20%5C%5C%20%5Cmathbf%7B%5Cfrac%7Bd%5Cleft%28%20ln%28e%5E%7B-x%7Dx%5En%29%20%5Cright%20%29%7D%7Bdx%7D%7D%20%26%3D%20%5Cmathbf%7B%5Cfrac%7Bn%7D%7Bx%7D%20-%201%7D%20%5CRightarrow%20ln%5Cleft%28e%5E%7B-x%7Dx%5En%20%5Cright%29%20%3D%20nln%5Cleft%28x%20%5Cright%29%20-%20x%20%5C%5C%20%5Cend%7Balign*%7D)
    - the `n ln(x) - x` is significant only in the regions where x and n are comparable, or else the function just evaluates to n.
      ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%26%5Ctextrm%7Bthis%20expression%20evaluates%20to%20n%2C%20except%20when%20x%20is%20comparable%20to%20n%7D%20%5C%5C%20%5Ctextrm%7B%20using%20%7D%20x%20%26%3D%20n%20&plus;%20%5Cin%20%5C%2C%5C%2C%5C%2C%20%2C%20%5C%2C%5C%2C%20%7C%5Cin%7C%20%3C%3C%3C%20n%5C%5C%20ln%5Cleft%28e%5E%7B-x%7Dn%5Ex%20%5Cright%20%29%20%26%3D%20nlnx%20-%20x%20%3D%20nln%5Cleft%28n%20&plus;%5Cin%20%5Cright%20%29%20-n-%5Cin%20%5C%5C%20nln%5Cleft%28n&plus;%5Cin%20%5Cright%20%29%20%26%3D%20nln%5Cleft%28n%5Cleft%281&plus;%5Cfrac%7B%5Cin%7D%7Bn%7D%20%5Cright%20%29%20%5Cright%20%29%20%3D%20nln%28n%29%20&plus;%20nln%5Cleft%28%201&plus;%5Cfrac%7B%5Cin%7D%7Bn%7D%5Cright%20%29%20%5C%5C%20ln%281&plus;x%29%20%26%3D%20x%20-%20%5Cfrac%7Bx%5E2%7D%7B2%7D%20&plus;%20%5Cfrac%7Bx%5E3%7D%7B3%7D%20%5Ccdots%5Ccdots%20%7Cx%7C%20%3C%3C%201%20%5C%5C%20nln%5Cleft%28n&plus;%5Cin%20%5Cright%20%29%20%26%3D%20nln%28n%29%20&plus;%20nln%5Cleft%28%201&plus;%5Cfrac%7B%5Cin%7D%7Bn%7D%5Cright%20%29%20%3D%20nln%28n%29%20&plus;%20n%5Cleft%5B%20%5Cfrac%7B%5Cin%7D%7Bn%7D%20-%20%5Cfrac%7B%5Cleft%28%20%5Cfrac%7B%5Cin%7D%7Bn%7D%20%5Cright%29%5E2%7D%7B2%7D%20&plus;%20%5Cfrac%7B%5Cleft%28%20%5Cfrac%7B%5Cin%7D%7Bn%7D%20%5Cright%29%5E3%7D%7B3%7D%20%5Ccdots%5Ccdots%20%5Cright%20%5D%20%5Capprox%20nln%28n%29%20&plus;%20%5Cin%20-%20%5Cfrac%7B%5Cin%5E2%7D%7B2n%7D%20%5C%5C%20ln%5Cleft%28e%5E%7B-x%7Dn%5Ex%20%5Cright%20%29%20%26%3D%20nln%28n%29%20&plus;%20%5Cin%20-%20%5Cfrac%7B%5Cin%5E2%7D%7B2n%7D%20-n-%5Cin%20%3D%20ln%5Cleft%28n%5En.e%5E%7B-n%7D%20%5Cright%20%29%20%5CRightarrow%20ln%5Cleft%28e%5E%7B-x%7Dn%5Ex%20%5Cright%20%29%20%3D%20ln%5Cleft%28%20n%5En.e%5E%7B-n%7D.e%5E%7B-%5Cfrac%7B%5Cin%5E2%7D%7B2n%7D%7D%20%5Cright%20%29%20%5C%5C%20%5Cend%7Balign*%7D))
    - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20e%5E%7B-x%7Dx%5En%20%26%3D%20n%5En.e%5E%7B-n%7D.e%5E%7B-%5Cfrac%7B%5Cin%5E2%7D%7B2n%7D%7D%20%5CRightarrow%20n%21%20%3D%20%5Cint%5Climits_%7Bi%3D0%7D%5E%7B%5Cinfty%7D%20e%5E%7B-x%7Dx%5En%20dx%20%5C%2C%5C%2C%5C%2C%28x%20%3D%20n&plus;%5Cin%20%5CRightarrow%20dx%20%3D%20d%5Cin%20%29%5C%5C%20%26%5CRightarrow%20%5Cint%5Climits_%7B0%7D%5E%7B%5Cinfty%7D%20n%5En.e%5E%7B-n%7D.e%5E%7B-%5Cfrac%7B%5Cin%5E2%7D%7B2n%7D%7D%20d%5Cin%20%3D%20n%5En.e%5E%7B-n%7D%20%5Cint%5Climits_%7B%5Cmathbf%7B-n%7D%7D%5E%7B%5Cinfty%7D%20e%5E%7B-%5Cfrac%7B%5Cin%5E2%7D%7B2n%7D%7D%20d%5Cin%20%5C%5C%20%7B%5Ccolor%7BRed%7D%20%5Cint%5Climits_%7B%5Cmathbf%7B-n%7D%7D%5E%7B%5Cinfty%7D%20e%5E%7B-%5Cfrac%7B%5Cin%5E2%7D%7B2n%7D%7D%20d%5Cin%7D%20%26%5Capprox%20%7B%5Ccolor%7BRed%7D%20%5Cint%5Climits_%7B-%5Cinfty%7D%5E%7B%5Cinfty%7D%20e%5E%7B-%5Cfrac%7B%5Cin%5E2%7D%7B2n%7D%7D%20d%5Cin%20%5C%2C%5C%2C%5C%2C%5Ccdots%20n%20%5Crightarrow%20%5Cinfty%20%7D%20%3D%20%5Csqrt%7B2%5Cpi%20n%7D%5C%5C%20%5Ctherefore%20%5C%2C%5C%2C%20%2C%20%5C%2C%5C%2C%20n%21%20%26%3D%20n%5En.e%5E%7B-n%7D.%5Csqrt%7B2%5Cpi%20n%7D%20%5CRightarrow%20%5Cmathbf%7B%5CGamma%28n&plus;1%29%7D%20%3D%20%5Cmathbf%7Bn%5En.e%5E%7B-n%7D.%5Csqrt%7B2%5Cpi%20n%7D%7D%20%5C%5C%20%5Cend%7Balign*%7D)
    - hence, we can see that gamma function by itself is deterministic, hence no point in assuming that gamma function is itself a p.d.f.
    - [Source - Stirling's approximation](https://mathworld.wolfram.com/StirlingsApproximation.html) 
    - the following gives the p.d.f. of a gamma distributed random variable:
      - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20p.d.f.%28z%3B%20n%29%20%26%3D%20z%5E%7Bn-1%7D%20n%5En%20%5Cfrac%7B%5Ctextrm%7Bexp%7D%5Cleft%28-zn%20%5Cright%20%29%7D%7B%5CGamma%28n%29%7D%20%5C%5C%20%5Cend%7Balign*%7D)
      - we **assume** that ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Csigma%5E2) is **gamma-distributed random variable**, with **n = n/2** , ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20p.d.f.%28z%3B%20n%2C%201%29%20%26%3D%20%5Cfrac%7B%28n/2%29%5E%7B%28n/2%29%7D%20%7D%7B%5CGamma%28n/2%29%7D%20z%5E%7Bn/2-1%7D%20%5Ctextrm%7Bexp%7D%5Cleft%28%20-zn/2%20%5Cright%20%29%20%3D%20c.z%5E%7Bn/2-1%7D%20%5Ctextrm%7Bexp%7D%5Cleft%28%20-zn/2%20%5Cright%20%29%5C%5C%20%5Cend%7Balign*%7D)
  - 
  - [refer this for proof](https://www.statlect.com/probability-distributions/student-t-distribution) <font color="red">Sunday left!!!</font>

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
- the variance will almost always be smaller when calculated using the sum of squared distances to the sample mean, compared to using the sum of  squared distances to the population mean.  [check here as to why exactly n-2 appears in the denominator of RSE](#f)
- <font color="red">Write proof for RSE in simple linear regression as well.</font>
- **The one exception** to this is  when the **sample mean** happens to be **equal to** the **population mean**, in  which case **the variance is also equal**.
- RSE value by itself depends upon the value of the dependent variable, hence its **better to use it as** a **percentage of mean value of** the **dependent variable**.
- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20RSE%20%3D%20%5Csqrt%7B%5Cfrac%7BRSS%7D%7Bn-p-1%7D%7D%20%5C%5C%20%5Cend%7Balign*%7D)
  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20E%5Cleft%5B%20RSS%20%5Cright%20%5D%20%3D%20%28n-p-1%29%5Csigma%5E2%20%5CRightarrow%20%5Csigma%5E2%20%3D%20%5Csqrt%7B%5Cfrac%7BE%20%5Cleft%5BRSS%20%5Cright%20%5D%7D%7Bn-p-1%7D%7D%20%5C%5C%20%5Cend%7Balign*%7D)
  - hence, for p=1(simple linear regression), ![equation](https://latex.codecogs.com/gif.latex?RSE%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7D%7Bn-2%7DRSS%7D%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7D%7Bn-2%7D%5Csum%5Climits_%7Bi%3D1%7D%5En%20%28y_i%20-%20%5Chat%7By_i%7D%29%5E2%7D)





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

- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?R%5E2) is the square of the correlation between the response(Y) and the fitted linear model(![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7BY%7D)), i.e. ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?Corr%28Y%2C%20%5Chat%7BY%7D%29%5E2).
- It turns out that ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?R%5E2) will always increase when more variables are added to the model, even if those variables are only weakly associated with the  response.
  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20R%5E2%20%3D%201%20-%20%5Cfrac%7BRSS%7D%7BTSS%7D%20%26%3D%20%5Cfrac%7BTSS%20-%20RSS%7D%7BTSS%7D%20%5C%5C%20E%5Cleft%5BTSS%20-%20RSS%20%5Cright%20%5D%20%26%3D%20%28p&plus;1%29%5Csigma%5E2%20%5C%5C%20E%5Cleft%5BTSS%20%5Cright%20%5D%20%26%3D%20%28n%29%5Csigma%5E2%20%5C%5C%20%5Cend%7Balign*%7D)
  - hence ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?R%5E2) is directly dependent on p, as p increases, so does the ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?R%5E2).



## Adjusted ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B200%7D%20R%5E2)<a name="adjusted-r2"></a>

- The only cases where there's much of a difference between ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20R%5E2)  and the adjusted-![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20R%5E2)  are when you are using *loads* of parameters compared to the data (p is comparable to n).  
- A dataset of 1,000 data points, sliced up into 18 small datasets, the slicing added only 18 parameters; the  adjustments to ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20R%5E2) wouldn't even affect the second decimal place, except possibly in the  end segments where there were only a few dozen data points: and it would *lower* them.





## Better metrics than ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B200%7D%20R%5E2)<a name="better-than-r2"></a>

- For **model selection** you can look to <font color="red">AIC and BIC = ?</font>.
- For expressing the adequacy of a model, look at the variance of the residuals. 





# Multiple Linear Regression<a name="mlr"></a>



## F-statistic<a name="f"></a>

- while t-statistic is used to  tell you if a *single* variable is statistically significant, this test is used to test if a *group* of variables are jointly significant.

- hence, all predictor variables need to be checked, which is why f-statistic is preferred in multiple linear regression.

- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7BNull%20hypothesis%20%3A%7D%20H_0%20%3A%20%5Cbeta_i%20%3D%200%20%5Cforall%20i%20%5Cin%20%5C%7B1%2C%202%2C%20%5Ccdots%20p%5C%7D%20%5Cnewline%20%5Ctextrm%7BAlternative%20hypothesis%20%3A%7D%20H_%7B%5Calpha%7D%20%3A%20%5Cexists%20i%2C%20i%20%5Cin%20%5C%7B1%2C%202%2C%20%5Ccdots%20p%5C%7D%5C%2C%2C%20%5C%2C%5C%2C%20%5Cbeta_i%20%5Cne%200)

- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?F-statistic%20%3D%20%5Cfrac%7B%28TSS-RSS%29/p%7D%7BRSS/%28n-p-1%29%7D%20%3D%20%5Cfrac%7B%28n-p-1%29%5Cleft%28%20%5Csum%20%5Climits_%7Bi%3D1%7D%5En%20%28y_i%20-%20%5Cbar%7By%7D%29%5E2%20-%20%28y_i%20-%20%5Chat%7By_i%7D%29%5E2%20%5Cright%20%29%7D%7Bp%5Cleft%28%20%5Csum%20%5Climits_%7Bi%3D1%7D%5En%20%28y_i%20-%20%5Chat%7By_i%7D%29%5E2%20%5Cright%20%29%7D)

- it can be easily proved, with the assumption that the linear model(linear regression) is used, that ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?E%5Cleft%5B%5Cfrac%7BRSS%7D%7Bn-p-1%7D%20%5Cright%20%5D%20%3D%20%5Csigma%5E2)
  - we had earlier seen this for [simple linear regression](#bessel), where p=1

  - ### Degrees of Freedom in Statistics<a name="dof"></a>

    - degrees of freedom is the effective number of parameters of the model required to **even estimate the regression**, i.e. the slope(a vector of dimensions p) and the coefficient.

    - [source for intuitive explanation of degrees of freedom](https://www.youtube.com/watch?v=4otEcA3gjLk&list=WL&index=37&t=236s)
      
      - this video explains that y=mx+c needs exactly 2 points to define a line, x being the single-feature independent variable and y being the dependent variable.
      - this causes the model to yield ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20R%5E2) =1 , hence the relation between y and x cannot be assessed, and only after the addition of the 3rd point is when the model gains some *freedom* to assess the strength of the relation between y and x 
      - **Note** : we are trying to **draw the best-fit line**, i.e. minimum sum of least squares.
      - anything less than this imparts freedom for the regression line itself, i.e. for n = 1 point, infinite number of lines can pass through this point, hence infinite values of m and c, hence **no best-fit line possible**.
      - The problem is that when we have more parameters than observations(n > p), there is a risk of overfitting the training dataset
      
    - we assume the feature vector X to be ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?X_%7Bn%20%5Ctimes%20p%7D) thus the weight vector becomes ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BB%7D_%7Bp%20%5Ctimes%201%7D) , and thus we differentiate p times, w.r.t each basis(or component/dimension of the weight vector) , thus obtain p constraints .

    - as [machinelearningmastery](https://machinelearningmastery.com/degrees-of-freedom-in-machine-learning/) says, : "*degrees of freedom = number of independent values – number of statistics*" , here the **number of independent values is n**(since n dependent variables are to be predicted) and **p number of statistics**  , i.e. number of constraints employed.
      - for instance, in [simple linear regression](#optimal-lr), we get the 2 listed constraints , hence the remaining degrees of freedom becomes *n-2* , which is also seen in its corresponding expression of [RMSE](#bessel).
      - **Note: ** in simple linear regression, it looks as if X is ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?X_%7Bn%20%5Ctimes%201%7D) but its actually ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?X_%7Bn%20%5Ctimes%202%7D) since X actually is appended with a column vector of the same dimensionality will all elements as 1, i.e. ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?X_%7Bn%20%5Ctimes%202%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20X_%7Bn%20%5Ctimes%201%7D%20%26%201_%7Bn%20%5Ctimes%201%7D%20%5Cend%7Bbmatrix%7D) 
      
    - The mathematical definition of d.o.f. is ![img](https://miro.medium.com/max/204/1*ZS5LFdgnR8fTnjYe7HUcNg.png) , 
      - for instance, for the claim that *the predicted values of each sample is the **sample mean*** , with the mean being , ![img](https://miro.medium.com/max/458/1*YmTBgecEu8u1Nm_ofzLyzg.png), the d.o.f. is ![img](https://miro.medium.com/max/582/1*caUcAmt2tCSOtlGP1qX9Rw.png) 
      - thus, the randomly samples vector ![equation](https://latex.codecogs.com/gif.latex?Y%20-%20%5Cbar%7BY%7D) has the total number of degrees as n-1, since after randomly sampling the value of ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbar%7By%7D) , it  becomes fixed, hence only the variables ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?y_1) to ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?y_%7Bn-1%7D) are free to move, since the ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?y_n) variable is linearly dependent on these n-1 values and the fixed-mean-value.
      
    - coming back to our definition of multiple linear regression, the final estimation mapping is ![img](https://miro.medium.com/max/471/1*x3TetJ_fDMbhaGd5onE-aA.png) 
      - here, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7BY%7D) is the predicted values for all samples, and H is called the hat-matrix(since it produces the predicted values, i.e. *hatting the Y*)
      - this H is ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?H_%7Bn%20%5Ctimes%20n%7D) , and for instance, n = 2, then ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20H%20%26%3D%20%5Cbegin%7Bbmatrix%7D%20a_1%20%26%20a_2%20%5C%5C%20a_3%20%26%20a_4%20%5Cend%7Bbmatrix%7D%2C%20Y%20%3D%20%5Cbegin%7Bbmatrix%7D%20y_1%20%5C%5C%20y_2%20%5Cend%7Bbmatrix%7D%20%5C%5C%20HY%20%26%3D%20%5Cbegin%7Bbmatrix%7D%20a_1y_1%20&plus;%20a_2%20&plus;%20y_2%20%5C%5C%20a_3y_1%20&plus;%20a_4y_2%20%5Cend%7Bbmatrix%7D%20%5CRightarrow%20%5Cfrac%7B%5Cpartial%20HY%7D%7B%5Cpartial%20Y%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cpartial%20a_1y_1%20&plus;%20a_2y_2%7D%7B%5Cpartial%20y_1%7D%20%5C%5C%20%5Cfrac%7B%5Cpartial%20a_3y_1%20&plus;%20a_4y_2%7D%7B%5Cpartial%20y_2%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20a_1%20%5C%5C%20a_4%20%5Cend%7Bbmatrix%7D%20%5C%5C%20%5Ctextrm%7Bdiv.%28H%29%7D%20%26%3D%20%5Cnabla_Y.H%20%3D%20%5Cbegin%7Bbmatrix%7D%20a_1%20%5C%5C%20a_4%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cmathbf%7Btrace%28H%29%7D%20%5C%5C%20%5Cend%7Balign*%7D)
      - this hat-matrix is
        - symmetric in nature, i.e. ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?H%20%3D%20H%5ET)
        - idempotent in nature, i.e. ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?H%5Ea%20%3D%20%5Cunderbrace%7BH%5Ccdot%20H%20%5Ccdot%20H%20%5Ccdots%7D_%7B%5Ctextrm%7B%20%27a%27%20times%20product%7D%7D%20%3D%20H%2C%20a%20%5Cin%20Z%5E%7B&plus;%7D)
        - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7BTr.%7D%5Cleft%28A_%7Bn%20%5Ctimes%20n%7D%5Ccdot%20B_%7Bn%20%5Ctimes%20n%7D%20%5Cright%29%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Csum%5Climits_%7Bj%3D1%7D%5En%20a_%7Bij%7Db_%7Bji%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Csum%5Climits_%7Bj%3D1%7D%5En%20b_%7Bji%7Da_%7Bij%7D%20%5Cnewline%20%5Ctextrm%7Bflip%20summations%2C%20i.e.%20evaluate%20i%20first%2C%20then%20j%7D%20%5Cnewline%20%5Csum%5Climits_%7Bj%3D1%7D%5En%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20b_%7Bji%7Da_%7Bij%7D%20%3D%20BA%20%5Cnewline%20Tr.%28ABC%29%20%3D%20Tr.%28A%28BC%29%29%20%3D%20Tr.%28BCA%29%20%5Cnewline%20Tr.%28%28AB%29C%29%20%3D%20Tr.%28C%28AB%29%29%20%3D%20Tr.%28CAB%29)
        - this states that for multiplication of **square matrices**, the trace of the product is **invariant in cyclic permutations**.
        - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?H%20%3D%20X%5Cleft%28X%5ETX%20%5Cright%20%29%5E%7B-1%7DX%5ET%20%5CRightarrow%20Tr.%28H%29%20%3D%20Tr.%28X%5Cleft%28X%5ETX%20%5Cright%20%29%5E%7B-1%7DX%5ET%29%20%3D%20Tr.%28X%5ETX%5Cleft%28X%5ETX%20%5Cright%20%29%5E%7B-1%7D%29%20%5Cnewline%20X%5ETX%20%3D%20%28X%5ETX%29_%7Bp%20%5Ctimes%20p%7D%20%5CRightarrow%20Tr.%28H%29%20%3D%20Tr.%28I_%7Bp%20%5Ctimes%20p%7D%29%20%3D%20p)
  
- [mathematical explanation of degrees of freedom and divergence function](https://towardsdatascience.com/the-official-definition-of-degrees-of-freedom-in-regression-faa04fd3c610)
  
- we have to use the [definition](https://math.stackexchange.com/questions/626732/linear-regression-degrees-of-freedom-of-sst-ssr-and-rss) of degrees of freedom
  
- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?SST%20%3D%20SSE%20&plus;%20SSR%20%5CRightarrow%20d.f.%28SST%29%20%3D%20d.f.%28SSE%29%20&plus;%20d.f.%28SSR%29%20%5CRightarrow%20d.f.%28SSE%29%20%3D%20d.f.%28SST%29%20-%20d.f.%28SSE%29%20%5Cnewline%20%5Cbegin%7Balign*%7D%20d.f.%28SST%29%20%3D%20d.f.%28y%20-%20%5Cbar%7By%7D%29%20%26%3D%20d.f.%5Cleft%28%5Cbegin%7Bbmatrix%7D%20y_1%20-%20%5Cfrac%7By_1%20&plus;%20y_2%20&plus;%20%5Ccdots%20y_n%7D%7Bn%7D%20%5C%5C%20y_2%20-%20%5Cfrac%7By_1%20&plus;%20y_2%20&plus;%20%5Ccdots%20y_n%7D%7Bn%7D%20%5C%5C%20%5Cvdots%20%5C%5C%20y_n%20-%20%5Cfrac%7By_1%20&plus;%20y_2%20&plus;%20%5Ccdots%20y_n%7D%7Bn%7D%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cfrac%7B%5Cpartial%20%5Cleft%28y_i%20-%20%5Cfrac%7B%5Csum%5Climits_%7Bj%3D1%7D%5En%20y_j%7D%7Bn%7D%20%5Cright%20%29%7D%7B%5Cpartial%20y_i%7D%20%5C%5C%20%26%3D%20n%20-%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cfrac%7B%5Cpartial%20%5Cleft%28%5Csum%5Climits_%7Bj%3D1%7D%5En%20y_j%20%5Cright%29%7D%7B%5Cpartial%20y_i%7D%20%3D%20n-1%20%5C%5C%20d.f.%28SSR%29%20%3D%20d.f.%28%5Chat%7By%7D%20-%20%5Cbar%7By%7D%29%20%26%3D%20d.f.%28X.%5Chat%7B%5Cbeta%7D%20-%20%5Cbar%7By%7D%29%20%3D%20d.f.%28X%5Chat%7B%5Cbeta%7D%29%20-%20d.f.%28%5Cbar%7By%7D%29%20%5C%5C%20d.f.%28X%20%5Ccdot%20%5Chat%7B%5Cbeta%7D%29%20-%201%20%3D%20d.f.%28X%5Ccdot%20%28X%5ETX%29%5E%7B-1%7DX%5ET%29%20-%201%20%26%3D%20d.f.%28X%5ETX%20%28X%5ETX%29%5E%7B-1%7D%29%20-%201%20%3D%20d.f.%28I_%7Bp&plus;1%20%5Ctimes%20p&plus;1%7D%20-%201%29%20%3D%20p&plus;1-1%20%5C%5C%20%5Ctherefore%20%5C%2C%2C%5C%2C%20d.f.%28SSE%29%20%3D%20n-1%20-%20p%20%26%3D%20n-p-1%20%5C%5C%20%5Cend%7Balign*%7D)
  
- here the dimensionality is p+1 , since we have included an extra value in ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta%7D) (for bias, i.e. ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_0%7D)) , and thus an extra column of all 1's in X, thus making X = n x p+1.
  
- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20Y%20%26%3D%20X%5Ccdot%20%5Cbeta%20&plus;%20%5Cin%20%5C%5C%20%5Chat%7BY%7D%20%3D%20HY%20%3D%20X%28X%5ETX%29%5E%7B-1%7DX%5ET%5Ccdot%20Y%20%26%3D%20X%28X%5ETX%29%5E%7B-1%7DX%5ET%20%5Ccdot%20%5Cleft%28X%5Ccdot%20%5Cbeta%20&plus;%20%5Cin%20%5Cright%20%29%20%3D%20X%5Cbeta%20&plus;%20H%20%5Cin%20%5C%5C%20Y%20-%20%5Chat%7BY%7D%20%3D%20%5Cin%20-%20H%20%5Cin%20%26%3D%20%5C%2C%20%5Cleft%28I%20-%20H%20%5Cright%20%29%5Cin%5C%5C%20%5Cleft%28Y%20-%20%5Chat%7BY%7D%20%5Cright%20%29%5ET%20%3D%20%5Cleft%28%20%5Cleft%28I%20-%20H%20%5Cright%20%29%20%5Cin%20%5Cright%20%29%5ET%20%26%3D%20%5Cin%5ET%20%28I-H%29%5ET%20%3D%20%5Cin%5ET%28I%20-%20H%29%20%5C%5C%20SSE%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%28y_i%20-%20%5Chat%7By_i%7D%29%5E2%20%3D%20%28Y%20-%20%5Chat%7BY%7D%29%5ET%28Y%20-%20%5Chat%7BY%7D%29%20%26%3D%20%5Cin%5ET%28I%20-%20H%29%20%5Ccdot%20%5Cleft%28I%20-%20H%20%5Cright%20%29%5Cin%20%3D%20%5C%2C%5C%2C%20%5Cin%5ET%20%28I-H%29%20%5Cin%20%5C%5C%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cin_i%20%5Csum%5Climits_%7Bj%3D1%7D%5En%20h%27_%7Bij%7D%20%5Cin_%7Bj%7D%20%26%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Csum%5Climits_%7Bj%3D1%7D%5En%20h%27_%7Bij%7D%20%5Cin_i%20%5Cin_%7Bj%7D%20%5C%5C%20%5Cend%7Balign*%7D)
    ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20SSE%20%3D%20%5Cin%5ET%20%5Cin%20-%20%5Cin%5ET%20H%20%5Cin%20%5C%5C%20E%5Cleft%5BSSE%20%5Cright%20%5D%20%3D%20E%5Cleft%5B%20%5Cin%5ET%20%5Cin%20-%20%5Cin%5ET%20H%20%5Cin%20%5Cright%20%5D%20%3D%20E%5B%5Cin%5ET%20%5Cin%5D%20-%20E%5B%5Cin%5ET%20H%20%5Cin%5D%20%5C%5C%20E%5B%5Cin%5ET%20%5Cin%5D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20E%5B%5Cin_i%20%5E%202%5D%20%3D%20n%20%5Csigma%5E2%20%5C%5C%20%5Cin%5ET%20H%20%5Cin%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Cin_%7Bi%7D%20%5Cleft%28%5Csum%5Climits_%7Bj%3D1%7D%5En%20h_%7Bij%7D%20%5Cin_%7Bj%7D%20%5Cright%29%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Csum%5Climits_%7Bj%3D1%7D%5En%20h_%7Bij%7D%20%5Cin_%7Bi%7D%20%5Cin_%7Bj%7D%20%5C%5C%20E%5B%5Cin%5ET%20H%20%5Cin%5D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Csum%5Climits_%7Bj%3D1%7D%5En%20h_%7Bij%7D%20E%5Cleft%5B%20%5Cin_i%20%5Cin_%7Bj%7D%20%5Cright%20%5D%20%5C%5C%20%7B%5Ccolor%7Bred%7D%20%5Ctextrm%7B%20remember%20that%20h%27%20is%20made%20up%20of%20X%2C%20and%20is%20square%2C%20idempotent%20and%20symmetric%20in%20nature%7D%7D%20%5C%5C%20E%5B%5Cin_i%20%5Cin_j%5D%20%3D%200%20%5Ctextrm%7B%20for%20%7D%20i%20%5Cne%20j%20%2C%20%3D%20%5Csigma%5E2%20%5Ctextrm%7B%20for%20%7D%20i%20%3D%20j%20%5C%5C%20E%5B%5Cin%5ET%20H%20%5Cin%5D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20h_%7Bii%7D%20%5Csigma%5E2%20%3D%20%5Csigma%5E2%5Csum%5Climits_%7Bi%3D1%7D%5En%20h_%7Bii%7D%20%3D%20%5Csigma%5E2Tr.%28H%29%20%3D%20%5Csigma%5E2.%28p&plus;1%29%20%5C%5C%20E%5BSSE%5D%20%3D%20n%5Csigma%5E2%20-%20%5Csigma%5E2.%28p&plus;1%29%20%3D%20%5Csigma%5E2.%28n-p-1%29%20%5C%5C%20%5Csigma%5E2%20%3D%20%5Cfrac%7BE%5BSSE%5D%7D%7Bn-p-1%7D%20%5C%5C%20%5Cend%7Balign*%7D)
  
- along with using linear regression model, when assuming the null-hypothesis, it can also be proved that ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?E%5Cleft%5B%20%5Cfrac%7BSST%20-%20SSE%7D%7Bp%7D%20%5Cright%20%5D%20%3D%20%5Csigma%5E2)

  - <img src="expected_tss-rss_1.png" />
  - <img src="expected_tss-rss_2.png" />
  - <img src="expected_tss-rss_3.png" />

- 

- [F-statistic vs adjusted R2](http://www.cmaxxsports.com/ec228/Adjusted%20R%5E2%20and%20F%20Statistics%20v1.2.pdf) <font color="red">remaining!!!</font>



## Other Assessments of the model<a name="other-assess"></a>



- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Ctextrm%7BThe%20coefficient%20estimates%20%7D%20%5Chat%7B%5Cbeta%7D_0%2C%20%5Chat%7B%5Cbeta%7D_1%2C%20.%20.%20.%20%2C%20%5Chat%7B%5Cbeta%7D_p%20%5Ctextrm%7B%20are%20estimates%20for%20%7D%20%5Cbeta_0%2C%20%5Cbeta_1%2C%20.%20.%20.%20%2C%20%5Cbeta_p.%20%5Cend%7Balign*%7D)
  - We can compute a confidence interval in order to determine how close ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%7BY%7D) will be to f(X), i.e. the true function.
- on factoring in the irreducible error of the true function, prediction intervals are used instead of confidence intervals.
  - 
- 