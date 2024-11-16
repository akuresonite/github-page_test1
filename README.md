---
layout: default
title: Machine Learning
---


# Machine Learning

<!-- # [1] AdaBoost -->
<div style="border-radius: 30px 0 30px 0px; border: 2px solid #00ea98; padding: 20px; background-color: #000000; text-align: center; box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);">
    <h1 style="color: #87CEEB; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); font-weight: bold; margin-bottom: 10px; font-size: 36px;">[1] 🐱🐶🚀 Adapative Boosting - AdaBoost 🔥!</h1>
</div>

NOTE: Here i will shows how to combine AdaBoost with Decision Trees, because that is the most common way to use AdaBoost.

So let's start by using Decision Trees and Random Forests to explain the three main concepts behind AdaBoost!

## Algorithm

Of **AdaBoost** for both regression and classification:

### **Algorithm for AdaBoost (General Framework)**

1. **Initialize Weights**:
   - Assign equal weights to all training samples:
   - $w_i = \frac{1}{N}, \, \forall i = 1, 2, \dots, N$
   - where $N$ is the total number of samples.

2. **Repeat for Each Weak Learner** ($m = 1$ to $M$):
   - Train a weak learner $h_m(x)$ using the weighted training data.

3. **Compute Error**:
   - Calculate the weighted error for the weak learner:
   - $\epsilon_m = \frac{\sum_{i=1}^{N} w_i \cdot \mathbb{I}(h_m(x_i) \neq y_i)}{\sum_{i=1}^{N} w_i}$
   - For regression, use:
   - $\epsilon_m = \frac{\sum_{i=1}^{N} w_i \cdot |h_m(x_i) - y_i|}{\sum_{i=1}^{N} w_i}$

4. **Calculate Alpha** (Weight of the Weak Learner):
   - For classification:
   - $\alpha_m = \frac{1}{2} \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)$
   - For regression:
   - $\alpha_m = \text{minimize a loss function (e.g., squared loss or absolute loss)}.$

5. **Update Weights**:
   - For classification:
   - $w_i \leftarrow w_i \cdot \exp\left(-\alpha_m \cdot y_i \cdot h_m(x_i)\right)$
   - For regression:
   - $w_i \leftarrow w_i \cdot \exp\left(-\alpha_m \cdot |y_i - h_m(x_i)|\right)$
   - Normalize weights so that $\sum_{i=1}^{N} w_i = 1$.

5. **Resample the Dataset** (Optional, depending on implementation):
   - Create a new dataset by resampling from the original dataset using the updated weights as probabilities.
   - Samples with higher weights are more likely to appear multiple times in the new dataset, while samples with lower weights may appear less frequently or not at all.

6. **Train the Next Weak Learner**:
   - Train the next weak learner on this newly weighted or resampled dataset.
   - This iterative process ensures that each new weak learner focuses on the "harder" samples, improving the overall model's performance.

7. **Aggregate Weak Learners**:
   - For classification, combine predictions using a weighted vote:
   - $H(x) = \text{sign}\left(\sum_{m=1}^{M} \alpha_m \cdot h_m(x)\right)$
   - For regression, combine predictions as a weighted sum:
   - $H(x) = \sum_{m=1}^{M} \alpha_m \cdot h_m(x)$

---

### Differences for Regression vs. Classification:
1. **Error Metric**:
   - Classification: Weighted misclassification error.
   - Regression: Weighted loss (e.g., squared or absolute error).

2. **Final Prediction**:
   - Classification: Majority vote using sign of weighted sum.
   - Regression: Weighted average of predictions.

---

This framework ensures that the algorithm iteratively focuses on harder-to-predict samples by adjusting their weights and combining weak learners to build a strong model.
