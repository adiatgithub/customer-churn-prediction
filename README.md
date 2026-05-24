📉 Customer Churn Prediction


A machine learning pipeline to predict customer churn using a Telco dataset. Three classification models were benchmarked — Random Forest delivered the best performance.

The system performs data preprocessing, exploratory data analysis, class imbalance handling, model comparison, and deployment through an interactive Streamlit dashboard.

📈 Model Performance

Random Forest (Selected Model)

Metric	Score

Accuracy	78.25%

Precision	58.76%

Recall	60.96%

F1 Score	59.84%

ROC-AUC	81.70%


## 🚀 Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Handling class imbalance using SMOTE
- Multiple model comparison:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
- Feature importance visualization
- Interactive Streamlit dashboard
- Churn probability prediction
- Risk categorization:
  - Low
  - Medium
  - High
- Customer retention recommendations

## 🛠️ Tech Stack

  Languages
    Python

  Libraries and Frameworks
   Pandas
   NumPyScikit-learn
   XGBoost
   Matplotlib
   Seaborn
   Streamlit
   Joblib
   Imbalanced-learn (SMOTE)


## 📂 Project Structure

customer-churn-prediction/
│


├── app/
│   └── app.py
│


├── data/
│   └── customer_churn.csv
│


├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── model_features.pkl
│


├── notebooks/
│   ├── eda.ipynb
│   └── model_training.ipynb
│


├── requirements.txt
├── README.md
├── .gitignore


## Live Demo

🔗 https://customer-churn-prediction-sbt2f9tjzlks7pvu5gvfhd.streamlit.app/
