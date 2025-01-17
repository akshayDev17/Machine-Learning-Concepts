[Reference link](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/)



# Table of Contents

1. [Challenges faced with imbalanced classification](#challenges)
2. [Confusion Matrix](#confusion-matrix)
3. [Imbalance Detection](#imbalance-detection)
   1. [Gini Coefficient](#gini-coefficient)
4. [Classifier Evaluation Metrics](#taxonomy)
   1. [Threshold Metrics](#tm)
   2. [Ranking metrics](#rm)
   3. [Probabilistic metrics](#pm)
5. [Summary](#summary)


# Challenges faced with imbalanced classifications<a name="challenges"></a>

*  imbalanced classification problems typically rate classification errors with the minority class as more important than those with the majority class. 
* As such performance metrics may be needed that focus on the minority class, which is made challenging because it is the minority class where we lack observations required to train an effective model.

# Confusion Matrix<a name="confusion-matrix"></a>
1. <img src="type1_vs_type2.png" width=400/>
2. Basically, FP = Type1, FN = Type2.

# Imbalanced-Classification Detection<a name="imbalance-detection"></a>
- need to have statistical metrics that tell us whether a particular dataset has imbalance or not.
- for instance, take 2 cases with 3 classes:
   1. case-1: 0.4, 0.4,0.2 is the class proportions
   2. case-1: 0.7, 0.25,0.05 is the class proportions
   3. which case is imbalanced?

## Gini Coefficient<a name="gini-coefficient"></a>
- uses class proportions(proportions of samples in this class): $p_i$
- works well with atomic classes(no subgroup within a class), and **non-multi-label classes**, i.e. a sample cannot have multiple classes.
- Algo
   - sort all $p_i's$
   - calculate cumulative $p_i$ for every i: $C_i$
   - $G.C. = \dfrac{\sum\limits_{i=1}^N(2i-N-1)(C_i - C_{i-1})}{N}$, N: total number of classes.
   - this has a value within 0 and 1.
   - values **towards 0: relatively balanced, towards 1: relatively imbalanced**.
   - for the (0.4, 0.4, 0.2) case, $C_i = (0.2, 0.6, 1), G.C. = \frac{1}{3}\left(0.2(-2)+0.4(0)+0.4(2)\right) = 0.4/3 = 0.1333$, hence this is a relatively balanced dataset.
   - for the (0.7, 0.25, 0.05) case, $C_i = (0.05, 0.3, 1), G.C. = \frac{1}{3}\left(0.05(-1)+0.25(1)+0.7(3)\right) = 2.3/3 = 0.7667$, hence this is a relatively imbalanced dataset.

# Classifier Evaluation Metrics<a name="taxonomy"></a>

1. Threshold Metrics
2. Ranking Metrics
3. Probability Metrics.



*Several machine learning researchers have identified three families of evaluation metrics used in the context of classification. These are the threshold metrics (e.g., accuracy and F-measure), the ranking methods and metrics (e.g., receiver operating characteristics (ROC) analysis and AUC), and the probabilistic metrics (e.g., root-mean-squared error).* - Page 189, [Imbalanced Learning: Foundations, Algorithms, and Applications](https://amzn.to/32K9K6d), 2013.



## Threshold Metrics for imbalanced classification<a name="tm"></a>

* Metrics based on a threshold and a qualitative understanding of error

1. **Accuracy** = Correct Predictions / Total Predictions

2. **Error** = Incorrect Predictions / Total Predictions

3. classification accuracy is almost universally ![equation](https://latex.codecogs.com/gif.latex?%7B%5Ccolor%7BRed%7D%20%5Ctextrm%7Binappropriate%7D%7D) for imbalanced classification

   1. reason is, a high accuracy (or low error) is achievable by a no skill model that only predicts the majority class.
   2. [Failure of Classification Accuracy for Imbalanced Class Distributions](https://machinelearningmastery.com/failure-of-accuracy-for-imbalanced-class-distributions/)

4. majority class is typically referred to as the negative outcome (e.g. such as “*no change*” or “*negative test result*“), and the minority class is typically referred to as the positive outcome (e.g. “*change*” or “*positive test result*“).

5. confusion matrix for a binary (two-class) classification problem

6. **Sensitivity** = TruePositive / (TruePositive + FalseNegative)
   Specificity is the complement to sensitivity, or the true negative rate, and summarises how well the negative class was predicted.

7. **Specificity** = TrueNegative / (FalsePositive + TrueNegative)

8. For imbalanced classification, the sensitivity might be more interesting than the specificity(given the definition of the positive class).

   1. Sensitivity and Specificity can be combined into a single score that balances both concerns, i.e. their geometric mean, ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7BG-mean%7D%20%3D%20%5Csqrt%7B%5Ctextrm%7Bsensitivity%7D%5Ctimes%5Ctextrm%7Bspecificity%7D%7D)

9. The Fbeta-measure measure is an abstraction of the F-measure where the balance of precision and recall in the calculation of the harmonic mean is controlled by a coefficient called *beta*.

   - **Fbeta-Measure** = ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%281&plus;%5Cbeta%5E%7B%5Ctextrm%7B2%7D%7D%29%5Ctimes%5Ctextrm%7Bprecision%7D%5Ctimes%20%5Ctextrm%7Brecall%7D%7D%7B%5Cbeta%5E%7B%5Ctextrm%7B2%7D%7D%5Ctimes%20%5Ctextrm%7Bprecision%7D%20&plus;%20%5Ctextrm%7Brecall%7D%7D)

   - **for multi-class classification**
     - ***Precision*** is calculated as the sum of true positives across all classes divided by the sum of true positives and false positives across all classes. \
         **How precise were you in your minority class prediction.**
         - majority class is the negative class, but there are two positive minority classes: class 1 and class 2, for instance, 1:1:100 minority to majority class ratio, $ = \dfrac{TP_c}{TP_c+FP_c}$
         - **weighted is shit in imbalanced tasks**, it uses proportion ratio of that class as weights. **better with balanced datasets**.
         - **macro-precision** = **arithmetic mean** of precision per class.

     - ***Recall*** is calculated as sum of true positives across all classes divided by the sum of true positives and false negatives across all classes.\
         **out of actual minority class samples, how much were you able to correctly classify as that minority class.**
         - $ = \dfrac{TP_c}{TP_c+FN_c}$

10. Other threshold metrics like *Kappa, Macro-Average Accuracy, Mean-Class-Weighted Accuracy, Optimized Precision, Adjusted Geometric Mean, Balanced Accuracy* also exist, but the ones covered before are the most popular.

11. ![equation](https://latex.codecogs.com/gif.latex?%7B%5Ccolor%7BRed%7D%20%5Ctextrm%7BLimitation%3A%7D%7D) assumes distribution of classes, especially the minority class(es) should be the same in train and test sets

    1. it may be so that the distribution might be entirely reversed, or else the proportions may be almost comparable, making it a *balanced* classification problem



## Ranking metrics<a name="rm"></a>

1. classifier predicts a score or a probability of class membership
2. most commonly used ranking metric is the ROC Curve
   1. TPR vs. FPR graph
3. precision-recall curve could also be used
   1. *Precision-recall curves (PR curves) are recommended for highly skewed domains where ROC curves may provide an excessively optimistic view of the performance.* -  [A Survey of Predictive Modelling under Imbalanced Distributions](https://arxiv.org/abs/1505.01658), 2015.
   2. **focuses on** the performance of the classifier on the **minority class**.
   3. similar to the AUC-ROC curve, predictions are made at different threshold values of minority class probability, thus giving different values of precision and recall, which are then plotted to obtain this curve 
   4. <img src="images/pr-auc.png"/>
   5. A no-skill classifier will be a horizontal line on the plot with a precision that is proportional to the number of positive examples in the dataset(*constant precision*).
      1. For a **balanced** dataset this will be **0.5**( TP = FP, cannot distinguish between whether a sample is correctly classified or wrongly into the minority class)  
   6. A perfect classifier is represented by a point in the top right.
   7. **helpful** diagnostic tool for evaluating a **single classifier** but ![equation](https://latex.codecogs.com/gif.latex?%7B%5Ccolor%7BRed%7D%20%5Ctextrm%7Bchallenging%7D%7D) for ![equation](https://latex.codecogs.com/gif.latex?%7B%5Ccolor%7BRed%7D%20%5Ctextrm%7Bcomparing%7D%7D) classifiers.



## Probabilistic Metrics<a name="pm"></a>

* measuring the deviation from the true probability
* assessment of the reliability of the classifiers not only measuring when they fail but whether they have selected the wrong class with a high or low probability
* classifiers such as MLE trained using a probabilistic framework, probabilities are already calibrated, for instance logistic regression classifiers
* Many nonlinear classifiers are not trained under a probabilistic framework and therefore require their probabilities to be calibrated against a dataset prior to being evaluated via a probabilistic metric. 
  * Examples might include support vector machines and k-nearest neighbors.
* log-loss = ![equation](https://latex.codecogs.com/gif.latex?-%28%5Ctextrm%7Bylog%28%7D%5Chat%7B%5Ctextrm%7By%7D%7D%29%5C%2C&plus;%5C%2C%281-%5Ctextrm%7By%29log%281-%7D%5Chat%7B%5Ctextrm%7By%7D%7D%29%29)
  * for multiple minority classes, ![equation](https://latex.codecogs.com/gif.latex?%5Csum%5Climits_%7B%5Ctextrm%7Bc%7D%5C%2C%20%5Cepsilon%5C%2C%20%5Ctextrm%7Bminority-class-set%7D%7D%20-%5Ctextrm%7By%7D_%7B%5Ctextrm%7Bc%7D%7D%5Ctextrm%7Blog%28%7D%5Chat%7B%5Ctextrm%7By%7D_%7B%20%5Ctextrm%7Bc%7D%7D%7D%29)
* 



### Brier Score<a name="bs"></a>

1. focused more on the positive class
2. more preferable than log loss, which is focused on the entire probability distribution.
3. mean squared error between the expected probabilities for the positive class and the predicted probabilities, ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Ctextrm%7B1%7D%7D%7B%5Ctextrm%7BN%7D%7D%5Csum%5Climits_%7B%5Ctextrm%7Bi%7D%3D1%7D%5E%7B%5Ctextrm%7BN%7D%7D%28%5Chat%7B%5Ctextrm%7By%7D%7D_%7B%20%5Ctextrm%7Bi%7D%7D-%5Ctextrm%7By%7D_%7B%20%5Ctextrm%7Bi%7D%7D%29%5E%7B%5Ctextrm%7B2%7D%7D)
4. a perfect classifier, **brier score = 0.0**
5. differences in Brier score for different classifiers can be very small
   1. hence scaling required
   2. the brier score of a no-skill classifier can be taken as a reference, to then calculate the **Brier Skill Score(BSS)** ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7BBSS%20%3D%201%20-%20%7D%5Cfrac%7B%5Ctextrm%7BBrier%20Score%7D%7D%7B%5Ctextrm%7BBrier%20Reference%7D%7D)
   3. hence **perfect classifier, BSS = 1**




# Summary<a name="summary"></a>
<img src="images/taxonomy.png"/>



![equation](https://latex.codecogs.com/gif.latex?%7B%5Ccolor%7BRed%7D%20%5Ctextrm%7BNote%3A%20F-1/F-0.5/F-2%20all%20point%20to%20F-%7D%5Cbeta%5C%2C%20%5Ctextrm%7Bwhere%20%7D%5Cbeta%5C%2C%20%3D%5C%2C%5Ctextrm%7B1%2C%200.5%2C%202%7D%7D)

* models that predict probabilities, but require crisp class labels also included.

# Interesting Scenarios

## Undersampled model test precision/recall
1. Given undersampling performed on a dataset and a model trained on that, how would precision and recall be on a test set?
2. ***Points to note:***
   1. Test set won't undergo any specific sampling, hence will be imbalanced.
   2. a model trained on undersampled/oversampled dataset will assume that samples belonging to both data occur equally frequently, hence would have the tendency of predicting more than required samples as the positive(minority) class.
3. hence a lot of false positives will be observed.
4. This means **test precision < Train precision**.
5. For test recall, it could remain relatively unchanged, or even improve slightly, since neither of TP nor FN is affected due to this special training scenario.
   1. recall could improve because the model could capture more true positives, as part of an increased volume of positive label prediction.


## When does FP matter more than FN?
- precision becomes more relevant than recall
- an example: **Email Spam classification**, since a non-spam email (which could be important) shouldn't be forcefully classified as a spam.
   - missing work-related emails or urgent messages due to them being in the spam folder.
   - poor user experience, user has to always check the spam folder to see whether an important mail has ended up there.
   - A spam email misclassified as not-spam might still annoy the user but can often be ignored or manually deleted without major harm.

## When does FN matter more than FP?
- recall is more relevant than precision

