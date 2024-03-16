# Introduction
1. Bootstrapped Aggregation of several decision trees.(`n_decision_trees`, i.e. `n_estimators`: tunable hyperparameter.)
2. **Row Sampling**: Dataset with D samples and E estimators, each estimator gets a certain number of $n_e$ samples.
    1. Samples could be provided in a *with replacement*(different trees could be trained on the same sample) or *without replacement*(ensures no duplicacy) strategy.
3. **Feature Sampling**: rather than division of rows, divide on features, and this could also happen in a *with replacement* or *without replacement* strategy.
4. This creation of subsets using *replacement* strategy: **Bootstrapping**.

# Improvements on Classic Decision Tree
1. Classic DT: High variance low bias
2. this model: comprised of many DTs having low bias.
    1. noisy data points get distributed amongst each tree(rather than all going to the same tree in the case of a classic DT) of the forest.
    2. hence less noise -> less variance.
    3. high variance can be detected by comparing train/test accuracies/mses/rmses/maes etc.
        1. if one is significantly different than the other, overfitting i.e. rote learning has taken place, which is a sign of high variance.

# Other advantages
1. *random* column sampling at split time
    1. not present in BaggingRegressor: have predetermined max_features, and once chosen, won't be replaced for an underlying estimator. The different estimators making up a bagging regressor may have distinct features.
    2. on the other hand, splitting is done on randomly selected `max_features` features at each node of splitting.