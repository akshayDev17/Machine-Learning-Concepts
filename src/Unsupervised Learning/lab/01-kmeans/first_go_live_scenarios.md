# The first-ever training window

How to define the training data of a company's first k-means segmentation of an online store's customers, built on
the notes in [`kmeans_in_production.ipynb`](kmeans_in_production.ipynb). The feature set is taken as given. Every
element the notes don't contain is tagged $\color{green}{\text{[NOT IN THE NOTES]}}$; untagged elements cite the part
of the notes they come from.

---

## Quantities

| symbol | meaning | where it comes from |
|---|---|---|
| <a id="q-order-log"></a>order log | one row per order: the raw data, recorded from day one | $\color{green}{\text{[NOT IN THE NOTES]}}$ |
| <a id="q-customer-table"></a>customer table | one row per customer, each customer's orders compressed into one feature vector | part 1: "one row per customer" |
| <a id="q-k"></a>K | the minimum number of orders a customer needs to be eligible for training and monitoring; scenario 1: set once per company, from how much it sells (here 20); scenario 2: set from the data (step 2) | $\color{green}{\text{[NOT IN THE NOTES]}}$ |
| <a id="q-m"></a>M | scenario 1 only: the number of eligible customers at which the first legitimate model is trained; here 100 | $\color{green}{\text{[NOT IN THE NOTES]}}$ |
| <a id="q-stages"></a>warm-up stages | scenario 1 only: the eligible-customer counts at which a weak model is retrained before [M](#q-m); here 20, 40, 50, 75 | $\color{green}{\text{[NOT IN THE NOTES]}}$ |
| <a id="q-batch"></a>batch | the customers who ordered that week, one row each: what the monitoring scores | part 5: "weekly batches ... one row per customer" |
| <a id="q-rolling-period"></a>N | how many days back each customer's features look | part 3, step 1, *Features* (no value given); set in [Setting N](#setting-n) $\color{green}{\text{[NOT IN THE NOTES]}}$ |

---

## Scenario 1 · A new business, no data yet

1. **Orders are the only contact with the outside world.** Everything the shop learns about a customer arrives as an
   order, so the raw data is the [order log](#q-order-log). $\color{green}{\text{[NOT IN THE NOTES]}}$
2. **k-means needs a customer table, built from the order log.** k-means clusters customers, not orders, so each
   customer's orders are compressed into one row of the [customer table](#q-customer-table) (part 1: "one row per
   customer").
3. **Record from day one.** Every order goes into the [order log](#q-order-log) from the first day, with a customer ID
   and a timestamp; there is nothing to choose here. $\color{green}{\text{[NOT IN THE NOTES]}}$
4. **A customer is eligible once they have at least [K](#q-k) orders.** One point has to summarise a customer's
   ordering behaviour, and a vector built from a handful of orders summarises noise; [K](#q-k) = 20 is the hard limit.
   $\color{green}{\text{[NOT IN THE NOTES]}}$
5. **Build the features** for every eligible customer from their orders: one vector per customer in the
   [customer table](#q-customer-table). The feature set itself is out of scope here.
6. **Train the first legitimate model the day [M](#q-m) customers are eligible:** at least 100 customers with at least
   20 orders each. Its training data is every eligible customer on that day. $\color{green}{\text{[NOT IN THE NOTES]}}$
7. **Until then, weak k-means models, retrained on a schedule, not on alarms.** At each
   [warm-up stage](#q-stages) the model is refitted on all eligible customers: $\color{green}{\text{[NOT IN THE NOTES]}}$

   | eligible customers (≥ [K](#q-k) orders each) | model | alarms |
   |---|---|---|
   | 20 | weak model, fitted on these 20 | off |
   | 40 | refitted on all 40 | off |
   | 50 | refitted on all 50 | off |
   | 75 | refitted on all 75 | off |
   | 100 = [M](#q-m) | the first legitimate model | on |

   * These refits are plain retraining: no alarm triggers them, so the monitoring's alarm engine stays off until
     [M](#q-m) is reached, via a config switch:

     ```yaml
     training:
       min_orders_per_customer: 20       # K
       first_model_customers: 100        # M
       warmup_stages: [20, 40, 50, 75]   # eligible-customer counts that trigger a refit
     monitoring:
       alarms_enabled: false             # switched to true once the first legitimate model is live
     ```

   * Once alarms are on, retraining follows the notes: it waits for an alarm (part 1: "only *retraining* waits for
     evidence").

8. **From [M](#q-m) on, monitoring runs** (part 1, Timeline). The day [M](#q-m) is reached is `t_golive`; a weekly
   [batch](#q-batch) outside a band is a breach; `t_alarm` is the first batch at which m of the last n are breaches
   (2 of 5 in the shop), and `t_alarm_window_earliest` the earliest breach in that window.

### The generic form $\color{green}{\text{[NOT IN THE NOTES]}}$

**The training data opens the day at least [M](#q-m) customers each have at least [K](#q-k) orders; before that,
refit a weak model at each [warm-up stage](#q-stages), with alarms off.**

* [K](#q-k) sets how much history one customer vector needs; [M](#q-m) sets how many vectors the first legitimate
  model needs.
* Both are parameters, so the rule carries over to any order rate: a faster-growing shop reaches [M](#q-m) sooner, a
  slower one later, and neither needs a calendar date.

---

## Scenario 2 · Two years of data at 400–500 orders a week

The history already exists, so nothing is waited for and nothing is staged: a snapshot of today's customers is taken,
the first model trained on it is the legitimate one, and alarms are on from go-live. $\color{green}{\text{[NOT IN THE NOTES]}}$

**1. Build the [customer table](#q-customer-table) from the existing [order log](#q-order-log):** one row per customer (part 1: "one row per customer").

**2. Set [K](#q-k)** $\color{green}{\text{[NOT IN THE NOTES]}}$

**2a. Fit a provisional k-means, used only to measure switches.**

* Who it is fitted on: customers with at least 20 orders, each with a vector built from their last 10 orders.
* Its number of clusters: chosen by the full k-scan, elbow, silhouette, stability and gap together (part 3, step 2).
* How it is used: fitted once, before the scan starts; every step of the scan only calls its `predict`.
* Afterwards: discarded once [K](#q-k) is set; the model that goes live is fitted fresh in step 6.

**2b. Scan k = 10, 15, 20, …** At each k:

* take the customers with at least 2 × k orders;
* build each one's vector twice, from their latest k orders and from the k orders before those;
* assign both vectors with the provisional model from 2a;
* the switch share is the fraction of these customers whose two vectors land in different segments.

**2c. Record four answers:** the smallest k at which the switch share is at most 20%, 15%, 10% and 5%.

**2d. Stop raising k** at whichever comes first:

* the one-year line: the median test customer's latest k orders span more than a year, so two blocks would reach past
  the two years of data;
* the 100-customer line: fewer than 100 customers have 2 × k orders, so a share would rest on too few of them.

**2e. Choose [K](#q-k): the strictest level that still leaves at least 80% of customers eligible.**

* Try the 5% answer first, then the 10%, 15% and 20% answers: each step down needs fewer orders per customer, so more
  customers reach [K](#q-k).
* The first level that passes gives [K](#q-k): at least 80% of the customers in the [customer table](#q-customer-table) have at least [K](#q-k) orders.
* If no level was reached in 2c: each customer's whole order history is their vector, and a customer needs at least
  10 orders; the 80% rule applies here too.
* If no option leaves 80% eligible: rank customers by their total number of orders and keep the top 95%.
  * [K](#q-k) is the order count at the 5th percentile, so everyone at or above it is eligible.
  * Each eligible customer's vector is built from their whole order history, the same orders the ranking counts.
* Record what was chosen, e.g. "K = 20, at most 15% switch, 83% eligible", or "fallback: top 95% by order count,
  K = 2".

**3. Eligible customers:** those with at least [K](#q-k) orders, by step 2e; the rest go into a "not yet eligible" group
outside the model. $\color{green}{\text{[NOT IN THE NOTES]}}$

**4. The snapshot** $\color{green}{\text{[NOT IN THE NOTES]}}$

* The vector: each eligible customer's vector is built from their last [K](#q-k) orders as of today (their whole order
  history, where step 2e says so).
* How time enters: through those orders, as the days they span (so orders per week = [K](#q-k) ÷ span) and the days since
  the latest one.
* No rolling period and no time split: there is nothing left to split in time.
* The dates: today is `t_golive`; the training window is the stretch the eligible customers' orders in their vectors
  cover.

**5. Split by customer:** 80% to train, 20% held out and never fitted (part 1).

* Band draws match a real week: a weekly [batch](#q-batch) holds the customers who ordered that week, and frequent buyers are more
  likely to be in it, so each band draw picks held-out customers with probability proportional to their orders per
  week, capped at 1. $\color{green}{\text{[NOT IN THE NOTES]}}$
* A small held-out set: if it is under about 10 weekly [batches](#q-batch)' worth, draw with replacement (part 1).

**6. Fit, check, go live**

* Fit: a fresh k-means with the full k-scan, on the training customers' vectors (part 3, step 2).
* Check: confirm the chosen model on the held-out customers by predicting them; that doesn't fit them, so they stay
  valid for the bands. $\color{green}{\text{[NOT IN THE NOTES]}}$
* Profile: profile and name the clusters (part 4).
* Store: everything part 1's "Store at go-live" lists.

**7. Monitoring starts at `t_golive`.** Each week's [batch](#q-batch) is the customers who ordered that week and are eligible, each
with their vector as of that week; breaches, `t_alarm` and `t_alarm_window_earliest` follow part 1's Timeline. $\color{green}{\text{[NOT IN THE NOTES]}}$

**8. [K](#q-k) stays fixed until the next retrain.** Changing it changes who is scored and how, the same kind of change as a
redefined feature, and part 2 says that forces a retrain. At that retrain, other candidate models compete with k-means
(part 3, step 2). $\color{green}{\text{[NOT IN THE NOTES]}}$

### Why a K exists, and when it doesn't $\color{green}{\text{[NOT IN THE NOTES]}}$

* A vector is an estimate: a customer's last k orders estimate their true profile, with an error of about σ/√k.
* The switch share has two parts:

  | part | what it is | as k grows |
  |---|---|---|
  | noise | two samples of the same profile landing on opposite sides of a segment boundary | falls like c/√k, towards 0 |
  | real change | customers whose profile really crossed a boundary between the two blocks | rises, because the blocks reach further back |

* With little real change, as under "no bumps", the noise part dominates, so the share falls below 5% at roughly
  [K](#q-k) ≈ (c / 0.05)².
* With a lot of real change, the sum bottoms out above 5%: the levels in 2c then pick the best stability the data
  allows, and the one-year line in 2d stops the test before it mostly measures change.
* The test can't tell the two parts apart: a customer who really changed counts against [K](#q-k) just like a noisy one, so a
  high [K](#q-k) can also mean behaviour is moving.

---

## Retraining after the first legitimate model $\color{green}{\text{[NOT IN THE NOTES]}}$

At an alarm, the training data is the eligible customers' rows since the change point:

1. **Start:** the latest of the change point the monitoring flagged or an upstream feature change (part 2), plus
   [N](#q-rolling-period) days if the change redefined a feature.
2. **One homogeneity check:** split that stretch in time into an older and a newer half; for every feature, PSI
   between the halves must be below 0.1, the notes' "practical minimum" for a real shift (part 1). If a feature fails,
   keep the newer half.
3. **Same customer definition as the live model:** customers with at least [K](#q-k) orders, with [K](#q-k) unchanged.

---

<a id="setting-n"></a>

## Setting N $\color{green}{\text{[NOT IN THE NOTES]}}$

[N](#q-rolling-period) has to hold enough of one customer's orders to describe them, so it comes from the gaps between
a customer's own orders.

1. **Check the regime.** From the [order log](#q-order-log), estimate the time from an order to the same customer's
   next order as a survival curve (Kaplan–Meier), so customers who haven't reordered yet count as "at least this long".
   * The curve passes 50% reordered: **repeat-driven**.
   * It flattens out well below 50%: **acquisition-driven**, since most customers never place a second order.
2. **Repeat-driven:** [N](#q-rolling-period) = 3 × the median time to the next order, rounded up to whole weeks.
3. **Acquisition-driven:** most customers have a single order, so every rolling feature describes that one order;
   cluster one-time buyers on their first order's features, and give repeat buyers [N](#q-rolling-period) =
   marketing's action horizon.
4. **No history yet (scenario 1):** start with marketing's action horizon; re-estimate monthly, and switch once two
   monthly estimates agree within 10%. After go-live, changing [N](#q-rolling-period) redefines every feature, which
   forces a retrain (part 2: "an upstream feature is added, removed, redefined or rescaled").

---

## After the training data is fixed

1. Fit k-means with the full k-scan (part 3, step 2); profile and name the clusters (part 4).
2. Store everything part 1's "Store at go-live" lists, and start the weekly monitoring job (part 1).
