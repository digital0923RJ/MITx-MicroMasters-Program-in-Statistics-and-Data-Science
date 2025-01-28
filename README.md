
# MIT Micro Masters in Statistics and Data Science

The **MIT MicroMasters in Statistics and Data Science** is an advanced program designed to provide learners with in-depth knowledge in probability, statistics, data analysis, and machine learning. It is ideal for professionals and students who wish to gain practical skills for tackling data-driven challenges.
---

## 📚 Program Overview

The program comprises the following core courses:

1. **[6.431x Probability - The Science of Uncertainty and Data](https://micromasters.mit.edu/ds/)**  
   - Explore the mathematical foundation of probability theory.  
   - Key topics: Random variables, distributions, law of large numbers, and central limit theorem.  

2. **[14.310x/14.310Fx - Data Analysis and Modeling](https://micromasters.mit.edu/ds/)**  
   - Learn to analyze and model data in social and scientific contexts.  
   - Key topics: Regression analysis, causal inference, and experimental design.  

3. **[18.6501x Fundamentals of Statistics](https://micromasters.mit.edu/ds/)**  
   - Dive into the principles of statistical inference and hypothesis testing.  
   - Key topics: Confidence intervals, Bayesian statistics, and Markov chains.  

4. **[6.86x Machine Learning with Python: From Linear Models to Deep Learning](https://micromasters.mit.edu/ds/)**  
   - Build a strong understanding of machine learning concepts and techniques.  
   - Key topics: Supervised and unsupervised learning, neural networks, and optimization.  

5. **Capstone Exam**  
   - A rigorous assessment that integrates the knowledge and skills from all courses


## ** Machine Learning with Python: From Linear Models to Deep Learning ** ## 

Through this machine learning course, I gained theoretical knowledge about various aspects of machine learning. 
In addition, I undertook several projects to put these theoretical concepts into practice. 
These projects included a Netflix recommendation system, sentiment analysis, regression analysis, and reinforcement learning. 
While working on these projects, I wrote code to implement the mathematical algorithms specific to each project.
First of all, You can see the Syllabus of the Machine Learning Course and find the summary of each units in this Repository. 

Secondly, you can see how I translated mathematical algorithms into code to complete these tasks. 
The related project files are also available for review, and below is a brief summary of the algorithms I implemented in code.

### Machine Learning with Python-From Linear Models to Deep Learning - Syllabus 

- Unit 1. Linear Classifiers and Generalizations
  * Lecture 1: Introduction to Machine Learning
  * Lecture 2: Linear Classifier and Perceptron
  * Lecture 3: Hinge Loss, Margin Boundaries, and Regularization
  * Lecture 4: Linear Classification and Generalization
    
- Unit 2. Nonlinear Classification, Linear regression, Collaborative Filtering
  * Lecture 5: Linear Regression
  * Lecture 6: Nonlinear Classification
  * Lecture 7: Recommender Systems
    
- Unit 3. Neural networks
  * Lecture 8: Introduction to Feedforward Neural Networks
  * Lecture 9: Feedforward Neural Networks, Back Propagation, and Stochastic Gradient Descent (SGD)
  * Lecture 10: Recurrent Neural Networks (RNNs)-part 1
  * Lecture 11: Recurrent Neural Networks (RNNs)-part 2
  * Lecture 12: Convolutional Neural Networks (CNNs)
    
- Unit 4. Unsupervised Learning
  * Lecture 13: Clustering 1
  * Lecture 14: Clustering 2
  * Lecture 15: Generative Models
  * Lecture 16: Mixture Models; EM Algorithm
    
- Unit 5. Reinforcement Learning
  * Lecture 17: Reinforcement Learning 1
  * Lecture 18: Reinforcement Learning 2
  * Lecture 19: Applications: Natural Language Processing

# **Summary of Machine Learning Concepts and Implementations**

## **1. Q-learning Algorithm and Epsilon-Greedy Exploration**
- **Q-learning Algorithm**:
  - A model-free reinforcement learning algorithm.
  - Learns the optimal action-value function \( Q(s, a) \) for each state-action pair:
    \[
    Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
    \]
    - \( \alpha \): Learning rate.
    - \( \gamma \): Discount factor.
    - \( r \): Reward observed after taking action \( a \) in state \( s \).

- **Epsilon-Greedy Exploration**:
  - Balances exploration and exploitation:
    - With probability \( \epsilon \), choose a random action.
    - With probability \( 1 - \epsilon \), choose the action with the highest \( Q(s, a) \).
  - \( \epsilon \) decays over time to reduce exploration as learning progresses.

- **Single-Step Update**:
  - Updates the \( Q \)-value for a single state-action pair after observing the reward and next state.

---

## **2. Perceptron Variants**
- **Perceptron Algorithm**:
  - Updates the weight vector \( \theta \) for misclassified points:
    \[
    \theta \leftarrow \theta + y x
    \]
    - \( y \): True label.
    - \( x \): Input feature vector.

- **Average Perceptron**:
  - Takes the average of all weight vectors across updates to improve generalization:
    \[
    \text{Average Weight} = \frac{1}{T} \sum_{t=1}^T \theta_t
    \]

- **Pegasos Algorithm**:
  - Stochastic optimization algorithm for Support Vector Machines (SVMs).
  - Minimizes the hinge loss with L2 regularization:
    \[
    \text{Loss} = \max(0, 1 - y (\theta^\top x)) + \frac{\lambda}{2} \|\theta\|^2
    \]
    - Update rule:
      \[
      \theta \leftarrow \theta - \eta \nabla J(\theta)
      \]
      - \( \eta \): Learning rate.
      - \( J(\theta) \): Objective function (hinge loss + regularization).

- **Bag of Words Implementation**:
  - Uses these algorithms for text classification tasks, representing text as feature vectors based on word frequency.

---

## **3. Mixture Models for Collaborative Filtering**
- **Problem**:
  - Predict missing entries in a sparse user-movie rating matrix extracted from the Netflix database.

- **Approach**:
  - Use **Mixtures of Gaussians** to model user preferences:
    - Assume \( K \) types of users, each associated with a Gaussian distribution.
    - Each user's rating profile is sampled from the Gaussian distribution corresponding to their type.

- **EM Algorithm**:
  - **Expectation Step (E-step)**:
    - Softly assign each user to a user type based on current Gaussian parameters.
    - Compute the posterior probability \( P(\text{Type} | \text{Data}) \) for each user.
  - **Maximization Step (M-step)**:
    - Update the parameters of the Gaussians (mean and covariance) using the weighted data from the E-step:
      \[
      \mu_k = \frac{\sum_{i} w_{ik} \cdot x_i}{\sum_{i} w_{ik}}, \quad \Sigma_k = \frac{\sum_{i} w_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_{i} w_{ik}}
      \]
      - \( w_{ik} \): Soft assignment weight of user \( i \) to type \( k \).

- **Prediction**:
  - Use the trained mixture model to estimate missing ratings:
    - Predict the rating as the expectation over all possible user types:
      \[
      \text{Prediction} = \sum_{k} P(\text{Type} = k | \text{Data}) \cdot \text{Gaussian Mean}_{k}
      \]

---

## **4. Key Results**
- **Q-learning**: Applied to reinforcement learning tasks with epsilon-greedy exploration to balance learning and exploration.
- **Perceptron Variants**: Implemented for binary classification tasks, such as Bag of Words text classification.
- **Collaborative Filtering**: Built a Gaussian Mixture Model to predict missing entries in a sparse rating matrix using the EM algorithm.

These projects demonstrate proficiency in applying mathematical algorithms to practical machine learning problems, coding them effectively for robust implementation.