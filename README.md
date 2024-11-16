<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Machine Learning

<!-- # [1] AdaBoost -->
<div style="border-radius: 30px 0 30px 0px; border: 2px solid #00ea98; padding: 20px; background-color: #000000; text-align: center; box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);">
    <h1 style="color: #87CEEB; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); font-weight: bold; margin-bottom: 10px; font-size: 36px;">[1] 🐱🐶🚀 Adapative Boosting - AdaBoost 🔥!</h1>
</div>

NOTE: Here i will shows how to combine AdaBoost with Decision Trees, because that is the most common way to use AdaBoost.

So let's start by using Decision Trees and Random Forests to explain the three main concepts behind AdaBoost!

1. **Use of Stumps**
    - In a Random Forest, each time you make a tree, you make a full sized tree.
    - Some trees might be bigger than others, but there is no predetermined maximum depth.
    - In contrast, in a Forest of Trees made with AdaBoost, the trees are usually just a node and two leaves.
    - A tree with just one node and two leaves is called a stump.
    - So this is really a Forest of Stumps rather than trees.
    - Stumps are not great at making accurate classifications.
    - A full sized Decision Tree would take advantage of all features to make a decision.
    - But a Stump can only use one variable to make a decision.
    - Thus, Stumps are technically "weak learners".
   > AdaBoost combines a lot of "weak learners" to make classifications. The weak learners are almost aways stumps.
2. **Say of each Stump**
    - In a Random Forest, each tree has an equal vote on the final classification.
    - In contrast, in a Forest of Stumps made with AdaBoost, some stumps get more say in the final classification than others.
   > Some stumps get more say in the classification than others.
3. **Influence of Previous Stump**
   - Lastly, in a Random Forest, each decision tree is made independently of the others.
   - In other words, it doesn't matter if this tree was made first.
   - In contrast, in a Forest of Stumps made with AdaBoost, order is important.
   - The errors that the first stump makes influence how the second stump is made and the errors that the second stump makes influence how the third stump is made etc. etc. etc..
   > Each stump is made by taking the previous stump's mistakes into account.

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

<!-- # [2] Gradient Boost -->
<div style="border-radius: 30px 0 30px 0px; border: 2px solid #00ea98; padding: 20px; background-color: #000000; text-align: center; box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);">
    <h1 style="color: #87CEEB; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); font-weight: bold; margin-bottom: 10px; font-size: 36px;">[2] 🐱🐶🚀 Gradient Boost Regression🔥!</h1>
</div>

Let's briefly compare and contrast AdaBoost and Gradient Boost.

>**AdaBoost**
>
>1. AdaBoost starts by building a very short tree, called a Stump, from the Training Data.
>2. The amount of say that the stump has on the final output is based on how well it compensated for those previous errors.
>3. Then AdaBoost builds the next stump based on errors that the previous stump made.
>4. Then AdaBoost builds another stump based on the errors made by the previous stump.
>5. Then AdaBoost continues to make stumps in this fashion until it has made the number of stumps you asked for, or it has a perfect fit.


>**Gradient Boost**
>
>1. In contrast, Gradient Boost starts by making a single leaf, instead of a tree or stump.
>2. This leaf represents an initial guess for the Weights of all of the samples.
>3. When trying to Predict a continuous value like Weight, the first guess is the the average value.
>4. Then Gradient Boost builds a tree.
>5. Like AdaBoost, this tree is based on the errors made by the previous tree.
>6. But unlike AdaBoost, this tree is usually larger than a stump.
>7. That said, Gradient Boost still restricts the size of the tree.
>8. In practice, people often set the maximum number of leaves to be between 8 and 32.
>9. Thus, like AdaBoost, Gradient Boost builds fixed sized trees based on the previous tree's errors, but unlike AdaBoost, each tree can be larger than a stump.
>10. Also like AdaBoost, Gradient Boost scales the trees. However, Gradient Boost scales all trees by the same amount.
>11. Then Gradient Boost builds another tree based on the errors made by the previous tree and then it scales the tree.
>12. And Gradient Boost continues to build trees in this fashion until it has made the number of trees you asked for, or additional trees fail to improve the fit.


## Algorithm for Gradient Boosting Regression

1. **Initialize the Model**:
   - Start with an initial prediction for all samples, typically the mean of the target values:
     - $F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{N} \left( y_i - F(x) \right)^2= \text{mean}(y)$
   - Compute the Pseudo residuals:
     - $r_i^{(0)} = y_i - F_0(x_i), \, \forall i = 1, \dots, N$

2. **For Each Iteration (from $m = 1$ to $M$)**:
   - **Fit a Weak Learner**:
     - Train a weak learner $h_m(x)$ (e.g., a decision tree) to predict the residuals $r_i^{(m-1)}$ from the previous iteration.

   - **Compute the Step Size**:
     - Use a learning rate $\eta$ (e.g., 0.1) to scale the contribution of the weak learner.
     - Optionally, minimize the residual sum of squares (RSS) to determine the optimal step size:
       - $\gamma_m = \arg\min_{\gamma} \sum_{i=1}^{N} \left( r_i^{(m-1)} - \gamma \cdot h_m(x_i) \right)^2$

   - **Update the Model**:
     - Add the scaled weak learner to the current model:
       - $F_m(x) = F_{m-1}(x) + \eta \cdot \gamma_m \cdot h_m(x)$

   - **Update Residuals**:
     - Compute new residuals based on the updated model:
       - $r_i^{(m)} = y_i - F_m(x_i), \, \forall i = 1, \dots, N$

3. **Final Prediction**:
   - After $M$ iterations, the final model is:
     - $F_M(x) = F_0(x) + \sum_{m=1}^{M} \eta \cdot \gamma_m \cdot h_m(x)$

---
### Sumarry:

1. We start with a leaf that is the average value of the variable we want to Predict.
2. Then we add a tree based on the Residuals, the difference between the Observed values and the Predicted values.
3. And we scale the tree's contribution to the final Prediction with a Learning Rate.
4. Then we add another tree based on the new Residuals.
5. And we keep adding trees based on the errors made by the previous tree.

### Key Features of Gradient Boosting Regression:

- **Residual Focus**: Each weak learner focuses on minimizing the residuals (errors) of the previous model.
- **Learning Rate**: Controls the contribution of each weak learner to avoid overfitting.
- **Iterative Improvement**: Combines weak learners iteratively to build a strong model.

This algorithm effectively minimizes the loss function (e.g., mean squared error) over multiple iterations to improve predictive performance.
