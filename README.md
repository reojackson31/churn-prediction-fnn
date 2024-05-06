# Predicting Customer Churn using Feed-Forward Neural Networks

## 1. Introduction
Predicting customer churn is a critical challenge for businesses, especially in the banking sector where retaining existing customers is often more cost-effective than acquiring new customers. In this project, I am implementing a feed-forward neural network to predict whether a bank's customers will close their accounts, based on a dataset from  [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling). This dataset includes a variety of customer attributes such as demographics, account details, and transaction histories.

The goal of this project is to develop a predictive model that can accurately identify customers at high risk of churning. This involves preparing the dataset for machine learning, experimenting with different neural network architectures, and adjusting model parameters like activation functions and learning rates. The effectiveness of the model is evaluated using several metrics, including accuracy, precision, recall, and the F1 score. 

This report outlines the methodology followed in preparing the data, developing the neural network model, and evaluating its performance. It also discusses the insights gained from the model about factors that may influence customer churn, which can help the bank create customized campaigns to retain customers before they churn.

## 2. Data Description
The dataset from [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) comprises various attributes of bank customers, including demographic information (like age, gender, and geography), financial details (such as credit score, account balance, and estimated salary), and engagement metrics (tenure with the bank, number of products used, credit card ownership, and active membership status). The target variable, "Exited" indicates whether a customer has closed their account.

Refer to the Python Notebook for insights from exploratory data analysis done on the dataset

## 3. Data Preprocessing
Data preprocessing steps include encoding for categorical variables, dropping identifier columns, standardizing the dataset, handling class imbalance, and converting the data into tensors for modeling in PyTorch.

## 4. Model Experiments
I have experimented with a total of 60 different configurations, using the combination of five network architectures, four activation functions, and three learning rates. The idea behind conducting multiple experiments across a range of configurations was to systematically explore the interaction between model architecture, activation function, and learning rate. This approach helped in identifying the most effective combination that maximizes validation accuracy while minimizing overfitting, as indicated by the validation loss. 

Each of the 60 experiments was trained on a GPU use its computational power and speed of training. The models were trained using the Binary Cross-Entropy (BCE) loss function, which is appropriate for the binary nature of the churn prediction task. Stochastic Gradient Descent (SGD) was employed as the optimizer for its robustness and effectiveness in handling noisy gradients. All models were subjected to training for 100 epochs to ensure that each configuration had sufficient time to converge or exhibit its behavior in terms of learning. During the training process, key metrics such as training loss, validation loss, and validation accuracy were recorded for each epoch, which allowed to track the models’ performance and stability over each epoch. 

![Training & Validation loss over epochs](https://github.com/reojackson31/churn-prediction-fnn/assets/148725712/e7e9d72b-c5b0-4641-80cd-d5bedd0dc0a1)

This plot was created to monitor the training and validation loss over 100 epochs for the first model experiment. We can see that the training loss decreases steadily, indicating that the model is learning from the data. The validation loss initially decreases at the same rate, and then plateaus and does not show much improvement after about 50 epochs.

![Validation set accuracy over epochs](https://github.com/reojackson31/churn-prediction-fnn/assets/148725712/18883078-38a0-4379-b431-70102a0420de)

This plot shows the model accuracy on the validation set over 100 epochs for the first model experiment. We can see that there is an initial rapid increase in accuracy during the initial epochs, after which the curve begins to plateau, fluctuating around 85%. This suggests that the model is learning effectively and reaches a stable accuracy.


## 5. Results from Model Experiments
After conducting 60 experiments to find the most effective combination for predicting customer churn, the evaluation metrics for all variations are compiled, and the table below presents the best-performing combinations for each of the five model architectures with their respective average training loss, average validation loss, and average validation accuracy.

| **Sl No** | **Model Architecture**       | **Activation Function** | **Learning Rate** | **Train Loss** | **Validation Loss** | **Validation Accuracy** |
|-----------|------------------------------|-------------------------|-------------------|----------------|---------------------|-------------------------|
| 1         | Basic                        | ReLU                    | 0.01              | 0.278          | 0.334               | 85.67%                  |
| 2         | With Dropout Layers          | ReLU                    | 0.01              | 0.354          | 0.335               | 87.00%                  |
| 3         | Add L2 Regularization        | ReLU                    | 0.01              | 0.279          | 0.336               | 86.13%                  |
| 4         | Add Batch Normalization      | ReLU                    | 0.01              | 0.106          | 0.569               | 83.33%                  |
| 5         | Varying Layer size & depth   | ReLU                    | 0.01              | 0.300          | 0.329               | 86.33%                  |


- The results indicate that the architecture with dropout layers achieved the highest average validation accuracy of 87\%, despite having a slightly higher average training loss compared to the other models. This suggests that the dropout technique effectively prevents overfitting and helps the model generalize better to unseen data.

- The architecture with batch normalization exhibited a much lower average training loss, but this did not translate to lower validation loss or higher accuracy. Instead, it shows the highest validation loss and the lowest validation accuracy, which might indicate that this model configuration was not optimal for the given task, possibly due to overfitting on the training data while failing to generalize to the validation data.

- The fifth architecture with different layer sizes and depths, also showed a good balance between training and validation loss and achieved a strong average validation accuracy, making it a competitive model architecture for this task. 

- Overall, the choice of ReLU as the activation function and a learning rate of 0.01 seems to have worked well across different architectures.
