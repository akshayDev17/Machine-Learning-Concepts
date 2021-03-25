# Lime-Model-Interpretation

1. [paper](#paper)
2. [install lime](#install_lime)
   1. [model agnostic approach](#maa)
3. [alternatives to lime](#alternatives)
4. [Dataset description](#dd)
5. [Problem statement](#ps)





## Proposal Paper<a name="paper"></a>

[Why Should I Trust You?](https://arxiv.org/pdf/1602.04938.pdf)



## Install Lime<a nam="install_lime"></a>

```conda install -c conda-forge lime```

[Documentation](https://lime-ml.readthedocs.io/en/latest/)

* LIME stands for Local Interpretable Model-agnostic Explanations.
* local means that explanations can be provided for single samples, but not for all samples at a time.
  * this is evidenced by the `data_row – 1d numpy array, corresponding to a row`  argument in the `LimeTabularExplainer.explain_instance` method, [here in the docs](https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_tabular.LimeTabularExplainer.explain_instance).
* Interpretable indicates human interpretability.



### model-agnostic approach<a name="maa"></a>

* When studying a machine learning problem, the underlying  structure in the data may or may not be described by one type of model  or the other.
* The model-agnostic approach consists in using machine learning models to study the underlying structure without assuming that  it can be accurately described by the model because of its nature. 
  * basically the entire interrelation between features that affect the dependent variable(s) cannot be explained by the model since the model is assumed to be inaccurate and inadequate under this approach.
* This  avoids introducing a potential bias in the interpretation. 
* A model-agnostic approach pretty much requires that several different techniques are used for the same task, the task being to explore the relationship between all  the features amongst themselves, and with the dependent variable(s).



## Alternatives to LIME<a name="alternatives"></a>

1. PDP - partial dependent plots
2. SHAP - SHapley Additive exPlanations
3. CAM - Class Activation Mapping.



[Reference Book for Interpretable Machine learning](http://ganj-ie.iust.ac.ir:8081/images/6/69/Interpretable-machine-learning.pdf)



## Dataset description<a name="dd"></a>

* bank-details of customers.
* the name is "Churn_Modelling.csv"
  * churning in this context means the rate at which clients using this bank will exit from doing so, which is obviously bad since banking is all about interest and commission earning.



Following are the columns.

1. RowNumber - sample number.
2. CustomerId - self-explanatory
3. Surname - self-explanatory
4. CreditScore - whether the client is worthy of being lent to.
5. Geography - which country the client lives in.
6. Gender - self-explanatory.
7. Age - self-explanatory(may be linked to credit score.)
8. Tenure - the time from which they have been banking with this bank.
9. Balance - self-explanatory.
10. NumOfProducts - number of products the bank offered, that a client is **subscribed** to.
11. HasCrCard - whether the client has a credit card or not.
12. IsActiveMember - actively uses this account.
13. EstimatedSalary - self-explanatory.
14. Exited - whether this client has closed this account with the bank(1) or not(0).



## Problem Statement<a name="ps"></a>

Predict, based on the details of a client, whether the client will **churn out** or not.



## Interpretation<a name="LIME-interpretation"></a>

* as observed in the code cell :

  ```python
  exp = interpretor.explain_instance(
      data_row=x_test.iloc[4],
      predict_fn=clf.predict_proba
  )
  ```

  * prediction probabilities are printed due to the classifier predicting this test sample, but the reasoning(as seen in the following markdown cell) is provided by the LIME framework.



## Workings of LIME<a name="lime_works"></a>

* fit a local, simple and interpretable model around the instance to be explained.
  * this can be done, since we have already supplied the `Explainer` the information about the aim of the classification model
    * `training_data`, `feature_names` gives the names of the independent features, `mode = classification`  tells the `explainer` that the task to be performed.
  * 
* 





