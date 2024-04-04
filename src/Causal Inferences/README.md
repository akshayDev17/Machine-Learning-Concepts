# Table of Contents
1. [RCT](#rct)
    1. [Challenges](#rct-challenges)
        1. [Confounders](#confounders)
        2. [Selection Bias](#selection-bias)
        3. [Counterfactuals](#counterfactuals)
    2. [Assumptions](#assumptions)
        1. [Causal Markov Condition](#causal_markov_condition)
        2. [Stable Unit Treatment Value Assumption](#sutva)
        3. [Ignorability](#ignorability)
    3. [Measuring Treatment Effects](#mte)
2. Uplift Modelling
    1. [Introduction](#uplift_modelling)
    2. [Meta-Learning Techniques](#meta_learners)
    3. [Causal Decision Trees](#cdt)
3. Propensity Score
    1. [Introduction](#propensity_score)
    2. [Reason for Propensity](#reason_for_propensity)
4. Direct Uplift Estimation Techniques
    1. [Introduction](#duet)
    2. [Uplift Classification](#uc)
    3. [Estimating ITE vs Uplift Classification](#et_vs_uc)
    4. [Class Transformation Approach](#cta)
5. [Causal inference is same as Uplift modelling?](#ci_vs_um)
6. Impact of Covariate shift on Uplift modelling
    1. [Introduction](#covariate-shift-impact)
7. [Watchlist](#watchlist)
8. [Papers](#papers)
9. [Practicals](#practicals)
10. [Pending Resources](#pending_resources)

# Randomized Control Tests<a name="rct"></a>
- Problem statement: send targetted ads/emails to people inorder to increase purchases.
- 2 groups:
    - Treatment group: receives emails.
    - Control group: doesn't receive any emails.
- **Randomized** because selected participants are randomly assigned a group.
    - hence, **probability** of being **assigned to** a **treatment/control group** is the same, i.e. **0.5**.
- Places where these cannot be run
    - Setting up the RCT is almost impossible: **billboard ads**.
    - Experiment takes too long.
- Resolution:
    - use past data.
    - <font color="red">Question: How to use past data if the RCT is for an entirely original/new thing?</font>
        - for instance, assume that data for food-delivery ad-recommendations exists.
        - the same customer basis cannot be used for a diamond-ad-recommendation because most of the samples would have a different label value.

## Challenges<a name="rct-challenges"></a>

## Confounders<a name="confounders"></a>
1. Features that influence:
    1. efficacy of treatment.
    2. availability of treatment(whether the features impact a person's ability to avail the treatment).

## Selection Bias<a name="selection-bias"></a>
1. Sample selected doesn't represent population.
2. for instance, sample = aged 20-30, population: avg age 45.

## Counterfactuals<a name="counterfactuals"></a>
1. What would've happened if a person who was given the treatment not be given the treatment instead? Would they've naturally cured?
2. Similarly, would a person who wasn't given the treatment be given it and be cured? or be uncured?

## Assumptions<a name="assumptions"></a>

### Causal Markov Condition<a name="causal_markov_condition"></a>
- <font color="red" size=4>What's this?</font>
- Causal graphs
    - <img src="causal_graph.png" width=400 />
    - <img src="causal_graph_simplified.png" width=400 />

### Stable Unit Treatment Value Assumption<a name="sutvs"></a>
- Assume a communicable disease.
- what if the person receiving treatment is in contact with a person not receiving it? Can they not cure in this case, even on getting the treatment?
- In the context of ad-recommendation, can the person being recommended the ad *pursuade* the one not being recommended for a purchase?
    - Likewise, can the person being recommended the ad *be dissuaded* by the one not being recommended from making a purchase?
- <font color="red" size=4>Study more on this !!</font>

### Ignorability<a name="ignorability"></a>
- no additional confounders exist that could've affected the treatment/outcome.

## Measuring Treatment Effects<a name="mte"></a>
- Treatment Effects can stem from dependency on a lot of confounders.
- in addition to this, counterfactuals estimation will also be required.
- Once both exists, **Average Treatment Effect(ATE)** is the *arithmetic mean* of *Individual Treatment Effect*.
    - which is the difference in treatment outcome and control outcome.
    - $ATE = \mathcal{E}\left[Y_i(1)\right] - \mathcal{E}\left[Y_i(0)\right]$
    - $Y_i(1)$ means for people who received treatment, what was the outcome(1: cured, 0: not-cured).
- **CATE(Conditional ATE)**: based on condition(s) imposed on confounder(s), whats ATE?
    - <img src="cate.png" width=400 />

# Uplift Modelling<a name="uplift_modelling"></a>
- in the targetted-ads task, the people that make a purchase on receiving an email/ad are called **persuadables**.
- $Y$ = 0 or 1(made a purchase), $W$ = 0 or 1(was sent an ad/email), $X$: feature vector definining a person.
- ITE = $P(Y_i=1|W_i=1) - P(Y_i=1|W_i=0)$
    - a person having $W_i = 0$ can't have another entry in the $W_i=1$ since either an ad/email was recommended/sent or not.
    - hence for people with $W_i=0$ the first term needs to be estimated, and for people with $W_i=1$ the second one.
    - also notice why $Y_i=1$ in both terms. That's because we only look at persuadables/people who do make a purchase.
        - not interested in those who don't/didn't.
    
## Meta-Learning Techniques<a name="meta_learners"></a>
- ML to detect hidden patterns in people-data.
- reiterating from above section, **2 models are required**: if an ad is sent, chances of purchase , if an ad is not sent, chances of purchase.
    - first model takes people with $W_i=1$, the other with $W_i=0$.
    - both will output probabilities since log-loss will be used.
- Since the logloss probabilites may not be the actual probabilities, calibration on them is performed.
    - <font color="red" size=3>What's Calibration?</font> [Codemporium's Video](https://www.youtube.com/watch?v=5zbV24vyO44&t=2s)
- It could happen that model-1 outputs very high and very low probabilities(magnitude-wise), and model-2 outputs medium-low and medium-high ones.
    - the AUC-ROC curve can justify highly accuracy for both.
    - this would mean higher ITEs than reality.
    - the reverse could be true, meaning lower/negative ITEs than reality.
    - Hence the need to combine all samples into a single model.
    - ***Class Transformation Approach*** is hence used.
- See these noteboooks to know about the 3 main kinds of uplift learners: S, T, X.
- a meta-learner is a combination of such learner(s).
- **THE MAIN TASK** of **META LEARNER** is to **ESTIMATE INDIVIDUAL TREATMENT EFFECT**, not correctly predict potential outcome as much.
- Stacking, blending, or weighted averaging of individual treatment effect models.

## Causal Decision Trees<a name="cdt"></a>
- [Codemporium YT video](https://www.youtube.com/watch?v=IEj8uzIG7C8)

# Propensity Score<a name="propensity_score"></a>

## Reason<a name="reason_for_propensity"></a>
- availability of treatment(in observation data) depends on confounders(X), and is never truly random($P(W)=0.5$)
- hence a distribution $P(W|X)$ exists.
- to *rectify* this *bias*, propensity score is introduced.
- $e(W_i) = P(W_i=1|X_i)$, i.e. probability of getting the treatment(or being recommended an ad) w.r.t. its confounders.
- the expression for ATE = E\[ITE\] = $E[P(Y_i=1|W_i=1)] - E[P(Y_i=1|W_i=0)]$.
- on using propensity, $ATE = E\left[\frac{W_iY_i}{e(W_i)}\right] - E\left[\frac{(1-W_i)Y_i}{(1-e(W_i))}\right]$
    - uncured samples, i.e. $Y_i=0$ won't contribute to any terms.
    - untreated but cured won't contribute to the first term. Hence, **only treated units** would contribute to **first term**.
    - hence $ATE = E\left[\frac{Y_i}{e(X_i)}\right] - E\left[\frac{Y_i}{(1-e(X_i))}\right] = E\left[\frac{Y_i}{P(W_i=1|X_i)}\right] - E\left[\frac{Y_i}{(P(W_i=0|X_i))}\right] = \frac{1}{n}\sum\limits_{i, t_i=1} \frac{y_i}{p(t_i=1|x_i)} - \frac{1}{n}\sum\limits_{i, t_i=0} \frac{y_i}{p(t_i=0|x_i)}$, n: total units or samples, not necessary that all samples contribute in a non-zero way, but doesn't mean they won't be taken while calculating average.
    - $\frac{1}{n}\sum\limits_{i, t_i=1} \frac{y_i}{p(t_i=1|x_i)} = \frac{1}{n_1}\sum\limits_{i, t_i=1} \dfrac{p(t_i=1).p(y_i)}{p(t_i=1|x_i)} = $
        - now this expression adheres to the confounder-led bias in treatment, 
    - remember that for any random variable $x \rightarrow p(x), E[f(x)] = \int xf(x)dx$
    - for the new and old forms of expected values, the probability distribution of X is different.
        - in the first expression, its not conditional.
        - for the $2^{nd}$ expression, its actually $x|W_i=1$, and $x|W_i=0$
    - $E\left[\frac{Y_i.P(W_i=1)}{P(W_i=1|X_i)}\right] = \int \frac{Y_i.P(W_i=1)}{P(W_i=1|X_i)} P(X|W_i=1) dX$
        - using Bayes Theorem: $P(A|B).P(B)=P(B|A).P(A) \Rightarrow P(A)=\frac{P(A|B).P(B)}{P(B|A)}, B=(W_i=1), A=X$
        - $\int Y_i P(X) dX = E[Y_i]$

# Direct Uplift Estimation Techniques<a name="duet"></a>
## Uplift Classification<a name="uc"></a>
1. Correctly predict potential outcome(probability).
2. This probability is of an individual responding positively to a treatment.
3. Trained directly to classify individuals as "responders" or "non-responders" based on their expected treatment response.

## Estimating ITE vs Uplift Classification<a name="et_vs_uc"></a>
1. the following is a snippet from a chat with GPT3.5
2. if theres a positive and a negative response, why/when there wouldn't be the strength of that response? for example, in the promotional offers(treatment) example, if promotional offers did result in a purchase(positive response), then shouldn't we be rather looking as individual treatment effect estimation? in other words, look at the amount of purchase made on being notified of a promotional offer(treated)?
3. ChatGPT
    You raise an important point. In some scenarios, particularly those involving promotional offers or marketing campaigns, the focus may indeed be on estimating the individual treatment effect or the strength of the response to the treatment, rather than just predicting whether a response occurs or not. Let's delve deeper into this distinction:

    1. Binary Response vs. Continuous Response:
    In many uplift modeling applications, such as direct marketing campaigns, the response variable may indeed be binary, representing whether an individual responded positively (e.g., made a purchase) or not to the treatment (e.g., receiving a promotional offer).
    However, in other cases, particularly when the outcome of interest is quantitative or continuous (e.g., amount spent on purchases), the focus shifts to estimating the strength or magnitude of the response to the treatment.
    2. Individual Treatment Effect Estimation:
    When the outcome of interest is continuous or quantitative, estimating the individual treatment effect becomes crucial. This involves quantifying the difference in outcomes between treated and untreated individuals, such as the difference in purchase amounts between those who received a promotional offer and those who did not.
    Techniques such as S-learner, T-learner, or X-learner are commonly used for individual treatment effect estimation in these scenarios. These methods provide insights into how much each individual's outcome changes as a result of receiving the treatment.
    3. Optimizing Treatment Allocation:
    Estimating individual treatment effects allows for more informed decision-making in resource allocation and treatment targeting. By understanding the magnitude of the response to the treatment for each individual, organizations can optimize their strategies to target individuals who are likely to generate the highest return on investment (e.g., highest purchase amounts in response to promotional offers).

In summary, while binary response prediction models (such as classifiers) focus on predicting whether a response occurs or not, individual treatment effect estimation techniques provide a deeper understanding of the strength and magnitude of the response to the treatment, particularly in scenarios where the outcome of interest is continuous or quantitative, such as purchase amounts or revenue generated. This information enables organizations to make more targeted and effective decisions in treatment allocation and intervention strategies.

## Class Transformation Approach<a name="cta"></a>
- Define a new target variable $Z_i$ by combining $W_i, Y_i$
    - $Z_i = W_iY_i + (1-W_i)(1-Y_i)$
    - $\therefore \,, Z_i = \begin{cases} 1 & \textrm{both are 0 or 1} \\ 0 & \textrm{either is 0, but the other is 1} \end{cases}$
    - $Z_i$ is basically XNOR.
    - its 1 if either \<ad was recommended and purchase was made\> OR \<ad wasn't recommended and purchase wasn't made\> (With a small nudge, purchase is made. If the nudge is absent, so is the purchase.)
    - its 0 if either \<ad was recommended and purchase wasn't made\> OR \<ad wasn't recommended and purchase was made\> (Either the ad wasn't good enough, or wasn't necessary.)
- $ITE = 2.P(Z_i=1)-1$. Derivation:
    - **Assumption: Assigning to treatment/control group is random**. Hence $P(W_i=0)=P(W_i=1)=0.5$ (mutually exclusive events, both equal, both add up to 1).
    - Also, $P(Z_i=1) = P(Y_i=1.W_i=1) + P(Y_i=0.W_i=0)$
    - ITE = $P(Y_i=1|W_i=1) - P(Y_i=1|W_i=0)$, using $P(A|B)=\frac{P(A.B)}{P(B)}$
    - ITE = $\frac{P(Y_i=1.W_i=1)}{P(W_i=1)} - \frac{P(Y_i=1.W_i=0)}{P(W_i=0)} \Rightarrow 2\left[P(Y_i=1.W_i=1) - P(Y_i=1.W_i=0) \right]$
    - Now, $P(Y_i=1.W_i=1) + P(Y_i=1.W_i=0) + P(Y_i=0.W_i=1) + P(Y_i=0.W_i=0) = 1 \Rightarrow P(Y_i=1.W_i=0) = 1-\left(P(Y_i=1.W_i=1) + P(Y_i=0.W_i=0) + P(Y_i=0.W_i=1) \right)$
    - ITE = $2\left[P(Y_i=1.W_i=1) - 1 + P(Y_i=1.W_i=1) + P(Y_i=0.W_i=0) + P(Y_i=0.W_i=1) \right] = 2\left[P(Y_i=1.W_i=1) + P(Y_i=0.W_i=0) + P(Y_i=1.W_i=1) + P(Y_i=0.W_i=1) - 1 \right] = 2\left[P(Z_i=1) + P(W_i=1) -1 \right]$
        - either people in the treatment group will be cured($P(Y_i=1)$) or not cured($P(Y_i=0)$).
    - ITE = $2\left[P(Z_i=1) + 0.5 -1 \right] = 2\left[P(Z_i=1) -0.5 \right] = 2P(Z_i=1) -1$

# Causal inference is same as Uplift modelling?<a name="ci_vs_um"></a>
- Causal inference focuses on understanding the *causal relationship between variables*, particularly in observational studies where experiments are not feasible or ethical. 
    - It aims to determine the effect of a treatment or intervention on an outcome by accounting for potential confounding variables.
- Uplift modeling, on the other hand, is a technique used in marketing and personalized interventions to *predict* the *incremental impact of a treatment* or action *on an individual*'s behavior. 
    - It aims to identify the subset of individuals who will respond positively to a treatment, such as a marketing campaign, and those who will not, enabling targeted interventions.

While both causal inference and uplift modeling deal with understanding the effects of interventions, they approach the problem from different perspectives and have distinct methodologies and objectives.

# Impact of Covariate/Dataset shift<a name="covariate-shift-impact"></a>
1. Suppose your original training set had 3600 samples, s.t. age was a feature ranging from 20-30.
2. You test the corresponding uplift classifier out on a group of 100 samples whose age ranges in 35-50.
    1. Firstly, should you even test on such a dataset, that has a large covariate shift(on age) from your original training set?
    2. The answer is no, its not right. following are the reasons when the model considered is a decision tree classifier([`causalml.inference.tree.UpliftTreeClassifier`](https://github.com/uber/causalml/blob/v0.15.0/causalml/inference/tree/uplift.pyx#L333))
    3. **Tree Structure Validity**
        1. Since age ranges in 20-30 in training data, the age space will be partitioned into rectangular regions whose limits are in this range.
        2. The tree may make splits at irrelevant age boundaries or miss important boundaries in the new range, when tested on the new sample-set having the 35-50 age range.
    4. **Missed Interactions**
        1. The trees may have missed learning important interactions between age and other features for the 35-50 range during training.
        2. This can limit the model's representational capacity in the new feature space.

# Watchlist<a name="watchlist"></a>
- [Using LLMs for causal inference](https://www.youtube.com/watch?v=1D5P7RVhHFU)

# Papers<a name="papers"></a>
- [Robust Causal Inference for Incremental Return on Ad Spend with Randomized Paired Geo Experiments](https://research.google/pubs/robust-causal-inference-for-incremental-return-on-ad-spend-with-randomized-paired-geo-experiments/)


# Practicals<a name="practicals"></a>
- [Water Quality Causal Inference by LiNGAM](https://www.kaggle.com/code/sasakitetsuya/water-quality-causal-inference-by-lingam)



# <font color="red">Resources Pending</font><a name="pending_resources"></a>
- https://blog.ml.cmu.edu/2020/08/31/7-causality/
- [Brad Neal - Causal Inference](https://www.youtube.com/@BradyNealCausalInference)
- [Causal inference as a blind spot of data scientists](https://dzidas.com/ml/2023/10/15/blind-spot-ds/)
- [Strong ignorability: confusion on the relationship between outcomes and treatment](https://stats.stackexchange.com/questions/474616/strong-ignorability-confusion-on-the-relationship-between-outcomes-and-treatmen)
- 

# Appendix

## Notebook: `hypothesisTesting.ipynb`
- intro to chi-squared distribution, f-statistic, f-distribution
- what is hypothesis testing w.r.t. null hypothesis in linear regression.

## Notebook: `Feature_Selection_using_FilterSelect_in_causalml.ipynb`
- 3 feature selection methods used by [`causalml`](https://causalml.readthedocs.io/en/latest/examples/feature_selection.html)
    - using F-statistic
    - using likelihood ratio
    - using KL Divergence.
- talks about problems with all the 3 approaches as well.