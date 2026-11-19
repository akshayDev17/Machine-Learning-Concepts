# Ensembles without trees: what the literature shows

**Scope.** This note collects published results on ensembles in which **no member is a decision tree**: bagging,
boosting, stacking, voting and plain averaging where every member is a neural network, a linear model, a kernel
machine (SVM), a k-nearest-neighbour (k-NN) classifier, a spline or another smoother. Tree ensembles appear only as
a baseline that a paper compared against. The search was done on 2026-11-19. Every claim below comes from a paper,
abstract or publisher page that was opened during the search. Where only a secondary description was available,
the note says so. It assumes you have read [`../README.md`](../README.md), especially sections 3 (bias and
variance) and 13.

## Table of contents

1. [Key terms](#key-terms)
2. [Foundations: why combining any learner can help](#foundations)
3. [Averaging and bagging neural networks](#nn-averaging)
4. [Boosting learners that are not trees](#boosting-non-tree)
5. [k-NN, SVMs and other stable or strong learners](#stable-strong)
6. [Linear-model ensembles](#linear-ensembles)
7. [Stacking members that are not trees](#stacking)
8. [Extrapolation and distribution shift](#extrapolation)
9. [How this relates to the folder's own experiments](#folder-experiments)
10. [Open questions and gaps](#gaps)
11. [Bibliography](#bibliography)

<a id="key-terms"></a>
## 1. Key terms

- **Member** (or *base learner*): one model inside the ensemble.
- **Stable / unstable learner**: a learner is *unstable* if a small change in the training data can change the fitted
  model a lot ([Breiman, 1996b](#ref-breiman-1996b)). Bagging helps unstable learners only.
- **Diversity**: members make *different* errors. Identical members give nothing to average.
- **Ambiguity**: how much the members disagree with their own average, measured on inputs
  ([Krogh & Vedelsby, 1995](#ref-krogh-1995)).
- **Weak / strong learner**: a weak learner is only slightly better than guessing. A strong one is accurate on its
  own (an SVM, a well-trained network).
- **Dataset shift / out-of-distribution (OOD)**: test inputs differ from the training inputs; extrapolation is one
  kind. **Calibration**: a model that says "90 %" is right 90 % of the time.
- **Smoother / hat matrix**: a fit whose predictions are a fixed linear combination of the training targets,
  $\hat y = H y$. Linear regression, ridge, smoothing splines and kernel smoothers all work this way.

<a id="foundations"></a>
## 2. Foundations: why combining any learner can help

- **The first ensemble theory was written for neural networks, not trees.**
  - [Hansen & Salamon (1990)](#ref-hansen-1990) showed that "ensembles of similar networks" reduce the error left after
    cross-validated tuning.
  - [Opitz & Maclin (1999)](#ref-opitz-1999) summarise the result as follows. If each member errs on an example less
    than 50 % of the time *and* the members err independently, the vote's expected error on that example goes to
    zero as members are added. They add that "such assumptions rarely hold in practice".
  - [Dietterich (2000)](#ref-dietterich-2000) states the condition as "accurate and diverse", citing Hansen & Salamon.
- **Averaging can never be worse than the members' average error, and the gain equals the disagreement.**
  - The ambiguity decomposition of [Krogh & Vedelsby (1995)](#ref-krogh-1995), for squared error, is:
    ensemble error = (weighted) average member error − (weighted) average ambiguity.
  - So the ensemble error is always at most the average member error. It drops further only if the members
    disagree. "We want the networks to disagree!"
  - Caveat from the same paper: if the members are "strongly biased" they agree everywhere, the ambiguity is small,
    and the ensemble is no better than a typical member.
  - [Perrone & Cooper (1993)](#ref-perrone-1993) (abstract only) build a combined regression estimator that is
    "as good or better in the MSE sense than any estimator in the population". They also note that it "utilizes
    local minima": networks stuck in different local minima become useful members.
- **Three reasons any learner can gain** ([Dietterich, 2000](#ref-dietterich-2000)):
  - *statistical*: little data, many hypotheses fit equally well, so averaging lowers the risk of picking a bad one;
  - *computational*: gradient descent in a network can stop in a local optimum, so several starts approximate the
    target better;
  - *representational*: a weighted sum of hypotheses can represent functions that no single member can.
- **Bagging needs unstable members.**
  - [Breiman (1996a)](#ref-breiman-1996a): "The vital element is the instability of the prediction method."
  - He lists neural nets, trees and subset selection in linear regression as unstable, and k-NN methods as stable,
    citing his instability paper ([Breiman, 1996b](#ref-breiman-1996b)).
  - Bagging "can push a good but unstable procedure a significant step towards optimality". On the other hand, "it
    can slightly degrade the performance of stable procedures".
  - [Dietterich (2000)](#ref-dietterich-2000) gives the same split: "Linear regression, nearest neighbor, and linear
    threshold algorithms are generally very stable."
- **The classic head-to-head comparison: networks vs trees** ([Opitz & Maclin, 1999](#ref-opitz-1999); 23 data sets).
  - Bagging networks "is almost always more accurate than a single classifier".
  - Boosting can be *less* accurate than a single classifier, "especially when using neural networks".
  - With 5–30 % added noise, bagging's gain over a single network grows while AdaBoost's shrinks or reverses.
  - A "simple ensemble" (same data, only different random starting weights) was "surprisingly effective, often
    producing results as good as Bagging".
  - For networks, most of the gain had arrived by ten to fifteen members.
  - Example rows from their Table 2 (test error %, single net → simple ensemble → bagging): credit-g 27.9 → 24.7 →
    24.2; diabetes 23.9 → 23.0 → 22.8.
- **"Many could be better than all".** [Zhou et al. (2002)](#ref-zhou-2002) show that a well-chosen subset of trained
  networks can beat averaging all of them. Their GASEN picks members with a genetic algorithm and gave "far smaller
  sizes but stronger generalization" than bagging or boosting.
- **Survey of diversity methods.** [Brown et al. (2005)](#ref-brown-2005) sort the ways of creating diversity into
  *implicit* and *explicit* methods, along three dimensions.

<a id="nn-averaging"></a>
## 3. Averaging and bagging neural networks

- **Deep ensembles: average a few independently trained networks** ([Lakshminarayanan et al., 2017](#ref-lakshminarayanan-2017)).
  - Recipe: M networks (default M = 5), each with its own random initialisation, all trained on the *full* data.
    Each net is trained with a proper scoring rule, a loss that rewards honest probabilities (for example
    log-likelihood).
  - ImageNet (Inception network): top-1 error 22.166 % with M = 1, 19.104 % with M = 5.
  - Toy regression $y = x^3 + \epsilon$ (20 points): ensembling "improves performance, especially as we move farther
    from the observed training data".
  - **Bagging hurt**: "We observed that bagging deteriorated performance in our experiments". A bootstrap sample
    holds only about 63 % unique points.
- **Random starts beat bootstrap samples for deep networks** ([Lee et al., 2015](#ref-lee-2015)).
  - Ensemble accuracy, random initialisation vs bagging: CIFAR-10 80.72 vs 78.40; CIFAR-100 66.51 vs 62.11; ImageNet
    (NiN) 64.08 vs 55.02. The single model scored 77.06, 60.19 and 58.90.
  - The loss of unique data is "the primary negative effect of bagging". Training on 63 % unique data without
    duplicates gave a similar drop.
  - For "convex or shallow models" the trade is different: random starts give too little diversity, so resampling
    is still needed.
- **Fixing the 63 % does not rescue the bootstrap** ([Nixon et al., 2020](#ref-nixon-2020)). Even when each bootstrap
  sample is topped up to cover more unique points, "bootstrap ensembles of deep neural networks yield no benefit
  over simpler baselines" (Wide ResNet, CIFAR-10).
- **Why random starts are enough** ([Fort et al., 2019](#ref-fort-2019)).
  - Networks from different random starts land in "entirely different modes" (separate low-loss regions, giving
    different functions).
  - Points along one training run stay "within a single mode predictions-wise".
- **Cheaper ensembles from one training run.**
  - Snapshot ensembles ([Huang et al., 2017](#ref-huang-2017)) save several minima of one run (cyclic learning rate):
    3.4 % / 17.4 % error on CIFAR-10 / CIFAR-100 "at no additional training cost". Fast geometric ensembling
    ([Garipov et al., 2018](#ref-garipov-2018)) samples low-loss curves between minima and beats snapshots.
  - BatchEnsemble ([Wen et al., 2020](#ref-wen-2020)) shares one weight matrix plus a rank-one factor per member:
    "3X" faster at test time and "3X" less memory at size 4.
  - Caveat: within-run members are less diverse than separately started ones ([Fort et al., 2019](#ref-fort-2019)).
- **Training the members to disagree.** Negative correlation learning ([Liu & Yao, 1999](#ref-liu-1999)) trains all
  networks together, with a penalty in each net's loss that pushes its errors to be negatively correlated with the
  others'.
- **Ensembles under dataset shift** ([Ovadia et al., 2019](#ref-ovadia-2019); image, text and click data).
  - "Deep ensembles seem to perform the best across most metrics and be more robust to dataset shift". M = 5
    "may be sufficient".
  - All methods lose accuracy *and* calibration as the shift grows.
  - Doubling network width did not improve calibration, "suggesting that the good calibration performance of
    ensembles is not due simply to higher capacity".
- **Ensemble or one bigger network?**
  - [Kondratyuk et al. (2020)](#ref-kondratyuk-2020): eight Wide ResNet 16-2 nets reach the 95 % CIFAR-10 accuracy of
    one Wide ResNet 16-8 with 80M vs 150M FLOPs. Two EfficientNet-B3s beat one B4 with fewer FLOPs. The benefit
    grows with model size: ensembles of EfficientNet-B0 "do not come close" to B1's accuracy per FLOP.
  - [Wang et al. (2022)](#ref-wang-2022): committees of off-the-shelf pretrained models "can match or exceed the
    accuracy of state-of-the-art models while being drastically more efficient". For example, an EfficientNet
    cascade reached B7's accuracy 5.4x faster.
  - [Abe et al. (2022)](#ref-abe-2022): the OOD gains of deep ensembles "can be replicated by a (larger) single
    model". Ensemble diversity "does not meaningfully contribute" to OOD uncertainty quality.
- **Survey:** [Ganaie et al. (2022)](#ref-ganaie-2022) review deep ensembles by family.

<a id="boosting-non-tree"></a>
## 4. Boosting learners that are not trees

- **AdaBoost with neural networks works, but how it is fed data matters** ([Schwenk & Bengio, 2000](#ref-schwenk-2000)).
  - About 1.4 % error on online handwritten digits; 1.5 % on UCI letters and 8.1 % on UCI satellite, "significantly
    better than boosted decision trees".
  - Random resampling "is not the main explanation" of boosting's gains. Weighting the cost function works too. This
    differs from bagging, where resampling is essential.
- **Boosting networks is fragile on noisy data** ([Opitz & Maclin, 1999](#ref-opitz-1999); section 2).
- **Gradient boosting with shallow networks needs a "corrective step"** (GrowNet, [Badirli et al., 2020](#ref-badirli-2020)).
  - Members are MLPs with two hidden layers of 16 units; 30 members were enough.
  - After each new member is added, all members are re-trained jointly. Without that step, AUC on Higgs-1M fell
    from 0.8401 to 0.8093.
  - With it, GrowNet beat XGBoost on Higgs (AUC 0.8510 vs 0.8304) and on two learning-to-rank sets. Plain greedy
    boosting of networks was clearly worse than boosting plus joint refitting.
- **Boosting as a way to grow one network.**
  - BoostResNet ([Huang et al., 2018](#ref-huang-2018)) trains a ResNet block by block as boosting "weak module
    classifiers". The training error decays exponentially with depth if each module beats a weak baseline.
  - AdaNet ([Cortes et al., 2017](#ref-cortes-2017)) learns both structure and weights. GrowNet's authors note that
    although its "learning process is boosting-style, the final model is a single NN"
    ([Badirli et al., 2020](#ref-badirli-2020)).
- **Boosting strong learners: weaken them first.**
  - [Li et al. (2008)](#ref-li-2008) report that [Wickramaratna et al. (2001)](#ref-wickramaratna-2001) "show that
    AdaBoost with strong component classifiers is not viable". That paper's own text was not accessible.
  - Li et al.'s AdaBoostSVM starts with RBF-SVMs of large kernel width σ (deliberately weak) and shrinks σ round by
    round. It "can perform as well as SVM", is better on imbalanced data, and beat AdaBoost with trees or networks
    on their benchmarks.
- **L2-boosting of smoothers: the statisticians' view** ([Bühlmann & Yu, 2003](#ref-buhlmann-yu-2003); [Bühlmann & Hothorn, 2007](#ref-buhlmann-hothorn-2007)).
  - L2-boosting means repeatedly fitting the current residuals with squared loss.
  - For a linear smoother with hat matrix $H$, $m$ rounds give the hat matrix $B_m = I - (I - H)^m$. If $H$ can learn
    at all, $B_m \to I$: the fit eventually interpolates the data, so early stopping is required.
  - The number of rounds $m$ acts as the "smoothing or regularization parameter". Variance grows "very slowly" with
    $m$.
  - Boosting a cubic smoothing spline with fixed degrees of freedom (df, the spline's flexibility), stopped at the
    right time, reaches the optimal (minimax) convergence rate and "even adapts to higher-order, unknown
    smoothness".
  - Toy example close to this folder's data: $Y = 0.8X + \sin(6X) + \varepsilon$, $n = 100$. Boosting a df = 4
    spline and tuning one spline's df give "about the same minimum mean squared error", but boosting "overfits much
    more slowly".
  - Design advice, the "low-variance principle": pick a deliberately weak base learner (for example df = 4) and a
    small step size ($\nu = 0.1$).
  - Software: the R package `mboost` boosts componentwise linear models, splines and trees under many losses
    ([Hothorn et al., 2010](#ref-hothorn-2010)).

<a id="stable-strong"></a>
## 5. k-NN, SVMs and other stable or strong learners

- **Bagging 1-nearest-neighbour changes nothing.**
  - [Breiman (1996a)](#ref-breiman-1996a), Table 5, six data sets: the bagged and single error rates are *identical*
    (for example waveform 26.1 vs 26.1, glass 21.6 vs 21.6).
  - Reason: a test case's prediction changes only if its nearest training point is missing from at least half of the
    bootstrap samples. Each point appears in a sample with probability 0.632, so this almost never happens.
- **Smaller resamples do help nearest neighbours** ([Hall & Samworth, 2005](#ref-hall-samworth-2005)).
  - Bagging improves a nearest-neighbour classifier asymptotically only if each resample holds less than 69 % of n
    (with replacement) or 50 % (without).
  - At larger fractions there is "no asymptotic difference". With fractions shrinking to 0 (but resample sizes
    growing), the bagged classifier converges to the optimal Bayes rule.
- **Diversity for k-NN comes from features, not rows.**
  - Random subspace method: each member sees a random subset of the features. It was introduced for trees by
    [Ho (1998)](#ref-ho-1998).
  - [Bay (1998)](#ref-bay-1998) applied it to nearest neighbours (MFS, Multiple Feature Subsets). Bay notes that
    bagging and error-correcting output codes help trees, rule learners and networks but not nearest neighbours.
  - On 25 UCI data sets MFS significantly beat 1-NN, k-NN and nearest neighbours with feature selection, and it
    reduced both bias and variance (from the abstract).
- **SVM ensembles: bag a low-bias SVM** ([Valentini & Dietterich, 2004](#ref-valentini-2004)).
  - Training sets of 100 points. Bagged SVMs differed significantly (0.1 level) from a single SVM in 13 of 20 cases,
    generally for the better. "Lobag" (bagging the lowest-bias SVM) did so in 17 of 20 and beat plain bagging in 9.
  - Even *linear* SVMs gained from bagging on these small samples: Waveform error 0.0939 → 0.0726; Spam
    0.1760 → 0.1352.
- **Open question raised in 2000.** [Dietterich (2000)](#ref-dietterich-2000) asked "whether local algorithms (such
  as radial basis functions and nearest neighbor methods) can be profitably combined via AdaBoost". This search
  found no systematic answer (see [gaps](#gaps)).

<a id="linear-ensembles"></a>
## 6. Linear-model ensembles

- **Boosting a linear least-squares learner ends at ordinary least squares.**
  - [Bühlmann & Hothorn (2007)](#ref-buhlmann-hothorn-2007): L2-boosting with componentwise linear least squares
    "converges to a least squares solution", unique if the design has full rank.
  - Stopped early, it selects variables and gives "shrunken versions of a least squares estimate".
- **Boosting with tiny steps traces (almost) the Lasso path.**
  - Forward stagewise linear regression is L2-boosting with an infinitesimal step. Its solutions are approximately
    the set of Lasso solutions ([Efron et al., 2004](#ref-efron-2004); [Bühlmann & Hothorn, 2007](#ref-buhlmann-hothorn-2007)).
    The two coincide exactly under a "positive cone condition".
  - [Hastie et al. (2007)](#ref-hastie-2007) make this precise: stagewise solves a *monotone* Lasso. The Lasso makes
    the best progress per unit of L1 norm, stagewise per unit of L1 arc length travelled.
  - [Rosset et al. (2004)](#ref-rosset-2004) extend the idea to classification losses. Boosting "approximately (and
    in some cases exactly)" minimises its loss under an L1 constraint, and in the separable case it heads towards
    an L1-margin-maximising separator. This parallels the SVM.
  - Takeaway: the early-stopping time of linear boosting plays the role of the Lasso penalty.
- **Bagging linear models: useless when they are stable, useful when they are not.**
  - [Breiman (1996a)](#ref-breiman-1996a): "linear regression using all variables is a fairly stable procedure". In
    his subset-selection simulation, bagging the near-full models gave *larger* error than not bagging. Bagging won
    only when few variables were kept, the unstable regime.
  - [Skurichina & Duin (2002)](#ref-skurichina-2002), linear discriminants: bagging and the random subspace method
    help at "critical" training sizes, where the sample is barely large enough for the number of features. Boosting
    helps at large training sizes.
  - [Valentini & Dietterich (2004)](#ref-valentini-2004): bagged linear SVMs beat single ones with 100 training
    points (section 5).
- **Why averaging leaves "linear" estimators alone.** [Friedman & Hall (2007)](#ref-friedman-hall-2007) model bagging
  as averaging out random "bumps" in the loss. In their tree simulations it "sharply reduces the nonlinear component
  of variance" and does not change the squared bias. Reading this onto our case (our inference, not their
  statement): a least-squares fit has almost no nonlinear component, so there is little to cancel.

<a id="stacking"></a>
## 7. Stacking members that are not trees

- **Stacking was introduced with nearest-neighbour-style members** ([Wolpert, 1992](#ref-wolpert-1992)).
  - Stacking means training a second-level model on the members' cross-validated predictions.
  - NETtalk (text to phonemes): three metric-based "HERBIE" generalizers, each looking at one letter slot, scored
    23 %, 69 % and 25 %. A HERBIE stacked on top of them reached 88 %.
  - Choosing the best member by cross-validation would have stopped at 69 %.
- **Stacking linear regressions** ([Breiman, 1996c](#ref-breiman-1996c)).
  - Combine members by least squares on cross-validated predictions, with non-negative weights.
  - "Stacking never does worse than selecting the single best predictor."
  - "The biggest gains came when dissimilar sets of predictors were stacked", meaning subset regressions together
    with ridge regressions. Stacking ridge regressions alone gave significant gains only in one correlation
    setting.
  - Breiman also reports "substantial reductions in error" from stacking linear regressions with k-NN predictors.

<a id="extrapolation"></a>
## 8. Extrapolation and distribution shift

- **ReLU networks extrapolate linearly.**
  - [Xu et al. (2021)](#ref-xu-2021), Theorem 1: along any direction from the origin, a two-layer ReLU MLP's
    prediction "quickly converges to a linear function" (proved for very wide networks, the "NTK regime"; confirmed
    in experiments across depths and widths).
  - So ReLU MLPs "do not extrapolate most nonlinear functions" (quadratic, cosine, square root). They do extrapolate
    *linear* targets, provided the training inputs cover all directions (Theorem 2).
  - [Ziyin et al. (2020)](#ref-ziyin-2020) prove the same for ReLU networks of any depth, and that tanh networks
    level off to a constant. Neither "can extrapolate a periodic function". They propose the activation
    $x + \sin^2 x$ as a fix.
  - [Hein et al. (2019)](#ref-hein-2019) explain the classification side: ReLU classifiers are piecewise linear, so
    far from the data they give "almost always high confidence predictions".
- **Ensembles do not automatically become uncertain far away.**
  - [de Mathelin et al. (2023)](#ref-demathelin-2023): deep ensembles "often return overconfident estimates outside
    the training domain". Their fix pushes members towards large weights so that they disagree there.
  - [Jain et al. (2019)](#ref-jain-2019) also push members to disagree on all plausible future inputs. This improved
    OOD performance on 38 protein–DNA regression sets, 9 UCI sets and an image set.
  - [Ovadia et al. (2019)](#ref-ovadia-2019): ensembles were best under shift, yet every method degraded.
  - [Abe et al. (2022)](#ref-abe-2022): an ensemble's OOD performance "is strongly determined by" its in-distribution
    performance.

<a id="folder-experiments"></a>
## 9. How this relates to the folder's own experiments

The experiments are in [`extrapolation_lab.ipynb`](extrapolation_lab.ipynb) and README §13
(1-D, $y = 2x + 3\sin x$, train on $[0, 10]$, test on $[10, 15]$).

- **"Extrapolation is a property of the members, not the ensemble." Confirms and explains.**
  - Members with a known shape past the data pass it to their sum or average. ReLU members continue straight lines,
    tanh members go flat ([Xu et al., 2021](#ref-xu-2021); [Ziyin et al., 2020](#ref-ziyin-2020)).
  - For linear smoothers this is exact: boosting gives another smoother, $B_m = I - (I-H)^m$
    ([Bühlmann & Hothorn, 2007](#ref-buhlmann-hothorn-2007)).
  - No paper found states the general claim for arbitrary members; it follows from the sum/average.
- **Bagged small ReLU nets extrapolate reliably but are nearly straight lines inside. Partly explained.**
  - The linear continuation is what Theorem 1 of [Xu et al. (2021)](#ref-xu-2021) predicts. It was close to the
    truth only because the true trend, $2x$, is linear.
  - Each bootstrap member sees about 63 % unique points, which weakens deep-network members
    ([Lee et al., 2015](#ref-lee-2015); [Lakshminarayanan et al., 2017](#ref-lakshminarayanan-2017);
    [Nixon et al., 2020](#ref-nixon-2020)). Whether this caused the straightness here was not tested.
  - The literature's default for networks is random starts on the full data, not bootstrap samples.
  - Caveat: five seeds agreeing (spread 0.5) shows *reproducibility*, not correctness. Ensembles can agree far from
    the data and still be wrong ([de Mathelin et al., 2023](#ref-demathelin-2023)).
- **Boosted ReLU nets fit the wave but extrapolate unreliably. Consistent; the literature adds two fixes.**
  - Boosting networks is fragile on noisy data ([Opitz & Maclin, 1999](#ref-opitz-1999)), and boosting strong
    members is "not viable" without weakening them ([Li et al., 2008](#ref-li-2008)).
  - Fix 1: a deliberately weak member and a small step ([Bühlmann & Hothorn, 2007](#ref-buhlmann-hothorn-2007)).
  - Fix 2: a joint "corrective" refit of all members ([Badirli et al., 2020](#ref-badirli-2020)).
  - **Contradicts "networks are rarely boosted" if read strictly.** Boosted networks have beaten boosted trees on
    some benchmarks ([Schwenk & Bengio, 2000](#ref-schwenk-2000); [Badirli et al., 2020](#ref-badirli-2020)).
- **tanh nets level off. Confirmed by theorem.** Tanh networks converge to a constant far from the data
  ([Ziyin et al., 2020](#ref-ziyin-2020)). The same paper proves that neither ReLU nor tanh nets continue a wave.
- **Boosted k-NN chases noise; bagged k-NN changes little. Bagging part confirmed; boosting part mostly untested in
  the literature.**
  - Bagged 1-NN equals 1-NN ([Breiman, 1996a](#ref-breiman-1996a)). Bagging helps only with resamples below about
    69 % of n ([Hall & Samworth, 2005](#ref-hall-samworth-2005)). Both results are for 1-NN; the folder used k = 10.
  - Diversity for k-NN is injected through feature subsets instead ([Bay, 1998](#ref-bay-1998)). With one input
    feature, as here, that route does not exist.
  - Boosting k-NN was an open question in [Dietterich (2000)](#ref-dietterich-2000).
- **Bagging linear regression gains nothing. Confirmed, with a limit on how far it generalises.**
  - Confirmed for stable linear fits ([Breiman, 1996a](#ref-breiman-1996a);
    [Dietterich, 2000](#ref-dietterich-2000)). 300 rows and one feature is the stable regime.
  - It does not hold for small samples relative to the number of features. There bagging linear classifiers helps
    ([Skurichina & Duin, 2002](#ref-skurichina-2002); [Valentini & Dietterich, 2004](#ref-valentini-2004)).
  - The README's "bagging stable models gains nothing" should say *stable at this sample size*.
- **`gblinear` = LinearRegression/Ridge/Lasso; boosting OLS returns OLS. Confirmed.** L2-boosting with linear least
  squares converges to the least-squares fit ([Bühlmann & Hothorn, 2007](#ref-buhlmann-hothorn-2007)).
- **Boosting ridge undoes its penalty. Explained by the hat-matrix formula (our derivation).**
  - Ridge's hat matrix has eigenvalues strictly between 0 and 1 on the span of the features. So $(I-H)^m \to$ the
    projection off that span, and $B_m \to$ the OLS hat matrix.
  - Bühlmann & Yu's "iteration m as regularization parameter" ([Bühlmann & Yu, 2003](#ref-buhlmann-yu-2003)) means
    the penalty is replaced by the stopping time. Stopping early would have kept some shrinkage.
- **Boosting lasso keeps its penalty. Not found in the literature.** Lasso is not a linear smoother, so the formula
  above does not apply. The stagewise–Lasso results ([Efron et al., 2004](#ref-efron-2004);
  [Hastie et al., 2007](#ref-hastie-2007)) concern boosting *plain least squares* with small steps, which is a
  different setup.
- **README §13.2 "an ensemble is not automatically better". Confirmed.**
  - The ambiguity decomposition guarantees averaged error ≤ average member error (6.84 < 16.89), not ≤ best member
    ([Krogh & Vedelsby, 1995](#ref-krogh-1995)).
  - Selecting members can beat using all of them ([Zhou et al., 2002](#ref-zhou-2002)).
  - Majority voting needs members that are better than chance and diverse ([Opitz & Maclin, 1999](#ref-opitz-1999)).
  - Stacking near-identical ridge fits gains little ([Breiman, 1996c](#ref-breiman-1996c)).
- **Splines and SVR fit well inside and badly outside. Consistent, not directly studied.** The spline-boosting theory
  ([Bühlmann & Yu, 2003](#ref-buhlmann-yu-2003)) is about accuracy *inside* the data range. No result on boosted
  splines past the edge was found.

<a id="gaps"></a>
## 10. Open questions and gaps

- **Extrapolation of boosted or bagged regressors is barely studied.** Work on ensembles under shift is mostly about
  *classification uncertainty* on images and text ([Ovadia et al., 2019](#ref-ovadia-2019)), not about the value
  predicted past the training range.
- **Seed-to-seed spread of boosted networks past the edge** (the folder's 9.5) has no published counterpart that
  this search found.
- **Boosting a Lasso learner**, as opposed to stagewise boosting of least squares, was not found.
- **Boosting local learners** (k-NN, RBF kernels) is still mostly an open question
  ([Dietterich, 2000](#ref-dietterich-2000)). Only weakened SVMs have a clear positive result
  ([Li et al., 2008](#ref-li-2008)).
- **Member disagreement as an uncertainty signal past the edge.** Deep ensembles widen their uncertainty away from the
  data in a toy cubic ([Lakshminarayanan et al., 2017](#ref-lakshminarayanan-2017)), but can also stay overconfident
  ([de Mathelin et al., 2023](#ref-demathelin-2023)). A natural next experiment for the lab is to plot the bagged
  members' spread over $[10, 15]$.
- **One bigger network vs an ensemble for small tabular regression.** The evidence is for large image models
  ([Kondratyuk et al., 2020](#ref-kondratyuk-2020); [Abe et al., 2022](#ref-abe-2022)).
- **Not verified:** [Perrone & Cooper (1993)](#ref-perrone-1993) (abstract only) and
  [Wickramaratna et al. (2001)](#ref-wickramaratna-2001) (known only through [Li et al., 2008](#ref-li-2008)). The
  experiments in [Liu & Yao (1999)](#ref-liu-1999) were not seen either (portal summary only).

<a id="bibliography"></a>
## 11. Bibliography

- <a id="ref-abe-2022"></a>Abe, T., Buchanan, E. K., Pleiss, G., Zemel, R., & Cunningham, J. P. (2022). Deep ensembles work, but are they necessary? *NeurIPS 2022*. [arXiv:2202.06985](https://arxiv.org/abs/2202.06985)
- <a id="ref-badirli-2020"></a>Badirli, S., Liu, X., Xing, Z., Bhowmik, A., Doan, K., & Keerthi, S. S. (2020). Gradient boosting neural networks: GrowNet. *arXiv preprint*. [arXiv:2002.07971](https://arxiv.org/abs/2002.07971)
- <a id="ref-bay-1998"></a>Bay, S. D. (1998). Combining nearest neighbor classifiers through multiple feature subsets. *Proc. 15th ICML*, 37–45. [ML Anthology](https://mlanthology.org/icml/1998/bay1998icml-combining/)
- <a id="ref-breiman-1996a"></a>Breiman, L. (1996a). Bagging predictors. *Machine Learning*, 24(2), 123–140. [doi:10.1007/BF00058655](https://doi.org/10.1007/BF00058655) (read via the [1994 technical report](https://www.stat.berkeley.edu/~breiman/bagging.pdf))
- <a id="ref-breiman-1996b"></a>Breiman, L. (1996b). Heuristics of instability and stabilization in model selection. *Annals of Statistics*, 24(6), 2350–2383. [doi:10.1214/aos/1032181158](https://doi.org/10.1214/aos/1032181158)
- <a id="ref-breiman-1996c"></a>Breiman, L. (1996c). Stacked regressions. *Machine Learning*, 24(1), 49–64. [doi:10.1007/BF00117832](https://doi.org/10.1007/BF00117832) ([PDF](https://stat.berkeley.edu/sites/default/files/tech-reports/367.pdf))
- <a id="ref-brown-2005"></a>Brown, G., Wyatt, J., Harris, R., & Yao, X. (2005). Diversity creation methods: a survey and categorisation. *Information Fusion*, 6(1), 5–20. [doi:10.1016/j.inffus.2004.04.004](https://doi.org/10.1016/j.inffus.2004.04.004)
- <a id="ref-buhlmann-hothorn-2007"></a>Bühlmann, P., & Hothorn, T. (2007). Boosting algorithms: regularization, prediction and model fitting. *Statistical Science*, 22(4), 477–505. [doi:10.1214/07-STS242](https://doi.org/10.1214/07-STS242) ([arXiv:0804.2752](https://arxiv.org/abs/0804.2752))
- <a id="ref-buhlmann-yu-2003"></a>Bühlmann, P., & Yu, B. (2003). Boosting with the L2 loss: regression and classification. *Journal of the American Statistical Association*, 98(462), 324–339. [doi:10.1198/016214503000125](https://doi.org/10.1198/016214503000125)
- <a id="ref-cortes-2017"></a>Cortes, C., Gonzalvo, X., Kuznetsov, V., Mohri, M., & Yang, S. (2017). AdaNet: adaptive structural learning of artificial neural networks. *Proc. 34th ICML*, PMLR 70, 874–883. [PMLR](https://proceedings.mlr.press/v70/cortes17a.html)
- <a id="ref-demathelin-2023"></a>de Mathelin, A., Deheeger, F., Mougeot, M., & Vayatis, N. (2023). Deep anti-regularized ensembles provide reliable out-of-distribution uncertainty quantification. *arXiv preprint*. [arXiv:2304.04042](https://arxiv.org/abs/2304.04042)
- <a id="ref-dietterich-2000"></a>Dietterich, T. G. (2000). Ensemble methods in machine learning. *Multiple Classifier Systems (MCS 2000)*, LNCS, 1–15. [doi:10.1007/3-540-45014-9_1](https://doi.org/10.1007/3-540-45014-9_1) ([PDF](https://web.engr.oregonstate.edu/~tgd/publications/mcs-ensembles.pdf))
- <a id="ref-efron-2004"></a>Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression. *Annals of Statistics*, 32(2), 407–451. [arXiv:math/0406456](https://arxiv.org/abs/math/0406456)
- <a id="ref-fort-2019"></a>Fort, S., Hu, H., & Lakshminarayanan, B. (2019). Deep ensembles: a loss landscape perspective. *arXiv preprint*. [arXiv:1912.02757](https://arxiv.org/abs/1912.02757)
- <a id="ref-friedman-hall-2007"></a>Friedman, J. H., & Hall, P. (2007). On bagging and nonlinear estimation. *Journal of Statistical Planning and Inference*, 137(3), 669–683. [doi:10.1016/j.jspi.2006.06.002](https://doi.org/10.1016/j.jspi.2006.06.002)
- <a id="ref-ganaie-2022"></a>Ganaie, M. A., Hu, M., Malik, A. K., Tanveer, M., & Suganthan, P. N. (2022). Ensemble deep learning: a review. *Engineering Applications of Artificial Intelligence*. [arXiv:2104.02395](https://arxiv.org/abs/2104.02395)
- <a id="ref-garipov-2018"></a>Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D., & Wilson, A. G. (2018). Loss surfaces, mode connectivity, and fast ensembling of DNNs. *NeurIPS 2018*. [arXiv:1802.10026](https://arxiv.org/abs/1802.10026)
- <a id="ref-hall-samworth-2005"></a>Hall, P., & Samworth, R. J. (2005). Properties of bagged nearest neighbour classifiers. *Journal of the Royal Statistical Society: Series B*, 67(3), 363–379. [doi:10.1111/j.1467-9868.2005.00506.x](https://doi.org/10.1111/j.1467-9868.2005.00506.x)
- <a id="ref-hansen-1990"></a>Hansen, L. K., & Salamon, P. (1990). Neural network ensembles. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 12(10), 993–1001. [doi:10.1109/34.58871](https://doi.org/10.1109/34.58871)
- <a id="ref-hastie-2007"></a>Hastie, T., Taylor, J., Tibshirani, R., & Walther, G. (2007). Forward stagewise regression and the monotone lasso. *Electronic Journal of Statistics*, 1, 1–29. [doi:10.1214/07-EJS004](https://doi.org/10.1214/07-EJS004)
- <a id="ref-hein-2019"></a>Hein, M., Andriushchenko, M., & Bitterwolf, J. (2019). Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem. *CVPR 2019*. [arXiv:1812.05720](https://arxiv.org/abs/1812.05720)
- <a id="ref-ho-1998"></a>Ho, T. K. (1998). The random subspace method for constructing decision forests. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 20(8), 832–844. [doi:10.1109/34.709601](https://doi.org/10.1109/34.709601)
- <a id="ref-hothorn-2010"></a>Hothorn, T., Bühlmann, P., Kneib, T., Schmid, M., & Hofner, B. (2010). Model-based boosting 2.0. *Journal of Machine Learning Research*, 11, 2109–2113. [JMLR](https://www.jmlr.org/papers/v11/hothorn10a.html)
- <a id="ref-huang-2017"></a>Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J. E., & Weinberger, K. Q. (2017). Snapshot ensembles: train 1, get M for free. *ICLR 2017*. [arXiv:1704.00109](https://arxiv.org/abs/1704.00109)
- <a id="ref-huang-2018"></a>Huang, F., Ash, J., Langford, J., & Schapire, R. (2018). Learning deep ResNet blocks sequentially using boosting theory. *ICML 2018*. [arXiv:1706.04964](https://arxiv.org/abs/1706.04964)
- <a id="ref-jain-2019"></a>Jain, S., Liu, G., Mueller, J., & Gifford, D. (2019). Maximizing overall diversity for improved uncertainty estimates in deep ensembles. *arXiv preprint*. [arXiv:1906.07380](https://arxiv.org/abs/1906.07380)
- <a id="ref-kondratyuk-2020"></a>Kondratyuk, D., Tan, M., Brown, M., & Gong, B. (2020). When ensembling smaller models is more efficient than single large models. *arXiv preprint*. [arXiv:2005.00570](https://arxiv.org/abs/2005.00570)
- <a id="ref-krogh-1995"></a>Krogh, A., & Vedelsby, J. (1995). Neural network ensembles, cross validation, and active learning. *Advances in Neural Information Processing Systems 7*, 231–238. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/1994/hash/b8c37e33defde51cf91e1e03e51657da-Abstract.html)
- <a id="ref-lakshminarayanan-2017"></a>Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NIPS 2017*. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)
- <a id="ref-lee-2015"></a>Lee, S., Purushwalkam, S., Cogswell, M., Crandall, D., & Batra, D. (2015). Why M heads are better than one: training a diverse ensemble of deep networks. *arXiv preprint*. [arXiv:1511.06314](https://arxiv.org/abs/1511.06314)
- <a id="ref-li-2008"></a>Li, X., Wang, L., & Sung, E. (2008). AdaBoost with SVM-based component classifiers. *Engineering Applications of Artificial Intelligence*, 21, 785–795. [doi:10.1016/j.engappai.2007.07.001](https://doi.org/10.1016/j.engappai.2007.07.001)
- <a id="ref-liu-1999"></a>Liu, Y., & Yao, X. (1999). Ensemble learning via negative correlation. *Neural Networks*, 12(10), 1399–1404. [doi:10.1016/S0893-6080(99)00073-8](https://doi.org/10.1016/S0893-6080(99)00073-8)
- <a id="ref-nixon-2020"></a>Nixon, J., Lakshminarayanan, B., & Tran, D. (2020). Why are bootstrapped deep ensembles not better? *NeurIPS 2020 Workshop "I Can't Believe It's Not Better!"*. [PDF](https://www.gatsby.ucl.ac.uk/~balaji/why_arent_bootstrapped_neural_networks_better.pdf)
- <a id="ref-opitz-1999"></a>Opitz, D., & Maclin, R. (1999). Popular ensemble methods: an empirical study. *Journal of Artificial Intelligence Research*, 11, 169–198. [arXiv:1106.0257](https://arxiv.org/abs/1106.0257)
- <a id="ref-ovadia-2019"></a>Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J. V., Lakshminarayanan, B., & Snoek, J. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. *NeurIPS 2019*. [arXiv:1906.02530](https://arxiv.org/abs/1906.02530)
- <a id="ref-perrone-1993"></a>Perrone, M. P., & Cooper, L. N. (1993). When networks disagree: ensemble methods for hybrid neural networks. Book chapter (1993); reprinted in *World Scientific Series in 20th Century Physics* (1995), 342–358. [doi:10.1142/9789812795885_0025](https://doi.org/10.1142/9789812795885_0025)
- <a id="ref-rosset-2004"></a>Rosset, S., Zhu, J., & Hastie, T. (2004). Boosting as a regularized path to a maximum margin classifier. *Journal of Machine Learning Research*, 5, 941–973. [JMLR PDF](https://www.jmlr.org/papers/volume5/rosset04a/rosset04a.pdf)
- <a id="ref-schwenk-2000"></a>Schwenk, H., & Bengio, Y. (2000). Boosting neural networks. *Neural Computation*, 12(8), 1869–1887. [doi:10.1162/089976600300015178](https://doi.org/10.1162/089976600300015178)
- <a id="ref-skurichina-2002"></a>Skurichina, M., & Duin, R. P. W. (2002). Bagging, boosting and the random subspace method for linear classifiers. *Pattern Analysis and Applications*, 5(2), 121–135. [doi:10.1007/s100440200011](https://doi.org/10.1007/s100440200011)
- <a id="ref-valentini-2004"></a>Valentini, G., & Dietterich, T. G. (2004). Bias-variance analysis of support vector machines for the development of SVM-based ensemble methods. *Journal of Machine Learning Research*, 5, 725–775. [JMLR PDF](https://www.jmlr.org/papers/volume5/valentini04a/valentini04a.pdf)
- <a id="ref-wang-2022"></a>Wang, X., Kondratyuk, D., Christiansen, E., Kitani, K. M., Alon, Y., & Eban, E. (2022). Wisdom of committees: an overlooked approach to faster and more accurate models. *ICLR 2022*. [arXiv:2012.01988](https://arxiv.org/abs/2012.01988)
- <a id="ref-wen-2020"></a>Wen, Y., Tran, D., & Ba, J. (2020). BatchEnsemble: an alternative approach to efficient ensemble and lifelong learning. *ICLR 2020*. [arXiv:2002.06715](https://arxiv.org/abs/2002.06715)
- <a id="ref-wickramaratna-2001"></a>Wickramaratna, J., Holden, S., & Buxton, B. (2001). Performance degradation in boosting. *Multiple Classifier Systems (MCS 2001)*, LNCS, 11–21. [doi:10.1007/3-540-48219-9_2](https://doi.org/10.1007/3-540-48219-9_2) (full text not accessed)
- <a id="ref-wolpert-1992"></a>Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259. [doi:10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1)
- <a id="ref-xu-2021"></a>Xu, K., Zhang, M., Li, J., Du, S. S., Kawarabayashi, K., & Jegelka, S. (2021). How neural networks extrapolate: from feedforward to graph neural networks. *ICLR 2021*. [arXiv:2009.11848](https://arxiv.org/abs/2009.11848)
- <a id="ref-zhou-2002"></a>Zhou, Z.-H., Wu, J., & Tang, W. (2002). Ensembling neural networks: many could be better than all. *Artificial Intelligence*, 137, 239–263. [doi:10.1016/S0004-3702(02)00190-X](https://doi.org/10.1016/S0004-3702(02)00190-X)
- <a id="ref-ziyin-2020"></a>Ziyin, L., Hartwig, T., & Ueda, M. (2020). Neural networks fail to learn periodic functions and how to fix it. *NeurIPS 2020*. [arXiv:2006.08195](https://arxiv.org/abs/2006.08195)
