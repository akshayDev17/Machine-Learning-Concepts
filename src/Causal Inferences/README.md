# Randomized Control Tests
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

## Challenges

### Confounders
1. Features that influence:
    1. efficacy of treatment.
    2. availability of treatment(whether the features impact a person's ability to avail the treatment).

## Selection Bias
1. Sample selected doesn't represent population.
2. for instance, sample = aged 20-30, population: avg age 45.

### Counterfactuals.
1. What would've happened if a person who was given the treatment not be given the treatment instead? Would they've naturally cured?
2. Similarly, would a person who wasn't given the treatment be given it and be cured? or be uncured?

## Assumptions

### Causal Markov Condition
- <font color="red" size=4>What's this?</font>
- Causal graphs
    - <img src="causal_graph.png" width=400 />
    - <img src="causal_graph_simplified.png" width=400 />

### Stable Unit Treatment Value Assumption
- Assume a communicable disease.
- what if the person receiving treatment is in contact with a person not receiving it? Can they not cure in this case, even on getting the treatment?
- In the context of ad-recommendation, can the person being recommended the ad *pursuade* the one not being recommended for a purchase?
    - Likewise, can the person being recommended the ad *be dissuaded* by the one not being recommended from making a purchase?
- <font color="red" size=4>Study more on this !!</font>

### Ignorability
- no additional confounders exist that could've affected the treatment/outcome.

## Measuring Treatment Effects
- Treatment Effects can stem from dependency on a lot of confounders.
- in addition to this, counterfactuals estimation will also be required.
- Once both exists, **Average Treatment Effect(ATE)** is the *arithmetic mean* of *Individual Treatment Effect*.
    - which is the difference in treatment outcome and control outcome.
- **CATE(Conditional ATE)**: based on condition(s) imposed on confounder(s), whats ATE?
    - <img src="cate.png" width=400 />

# Uplift Modelling
- in the targetted-ads task, the people that make a purchase on receiving an email/ad are called **persuadables**.
- $Y$ = 0 or 1(made a purchase), $W$ = 0 or 1(was sent an ad/email), $X$: feature vector definining a person.
- ITE = $P(Y_i=1|W_i=1) - P(Y_i=1|W_i=0)$
    - a person having $W_i = 0$ can't have another entry in the $W_i=1$ since either an ad/email was recommended/sent or not.
    - hence for people with $W_i=0$ the first term needs to be estimated, and for people with $W_i=1$ the second one.
    

## Meta-Learning Techniques
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

### Class Transformation Approach
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

## Causal Decision Trees
- [Codemporium YT video](https://www.youtube.com/watch?v=IEj8uzIG7C8)

# Watchlist
- [Stanford Lectures on ATE](https://www.youtube.com/watch?v=ZA8iOjUR8aY&list=PLxq_lXOUlvQAoWZEqhRqHNezS30lI49G-&index=5)

## Direct Uplift Estimation Techniques

# <font color="red">Resources Pending</font>
- https://blog.ml.cmu.edu/2020/08/31/7-causality/
- [Brad Neal - Causal Inference](https://www.youtube.com/@BradyNealCausalInference)
- [Strong ignorability: confusion on the relationship between outcomes and treatment](https://stats.stackexchange.com/questions/474616/strong-ignorability-confusion-on-the-relationship-between-outcomes-and-treatmen)
- 
