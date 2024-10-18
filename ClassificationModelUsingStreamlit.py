import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,f1_score


def load_data():
    diabetes_df = pd.read_csv('diabete.csv') 
    heart_df = pd.read_csv('heart.csv')  
    return diabetes_df, heart_df

def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1=f1_score(y_test,y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, f1,conf_matrix

st.title('Heart Disease & Diabetes Dataset Model Comparison')
st.sidebar.title('Heart Disease & Diabetes Dataset Model Comparison')


diabetes_df, heart_df = load_data()

dataset_choice = st.sidebar.selectbox("Choose Dataset", ("Diabetes", "Heart"))

if dataset_choice == "Diabetes":
    df = diabetes_df
    target_column = 'Outcome'  
else:
    df = heart_df
    target_column = 'target' 
   
st.write(f"#### Selected Dataset: {dataset_choice}")
    
st.sidebar.write(f"Selected Dataset: {dataset_choice}")

if df is not None:
    st.sidebar.write("### Dataset Shape")
    st.sidebar.write(f"{df.shape[0]} Rows, {df.shape[1]} Columns") 
    st.sidebar.write("### Dataset Preview")
    st.sidebar.write(df.head(5))
    st.sidebar.write("### Dataset Information")
    buffer = pd.DataFrame(df.dtypes).rename(columns={0: 'Data Type'})
    buffer["Non-Null Count"] = df.count()
    st.sidebar.table(buffer)

X_train, X_test, y_train, y_test = split_data(df, target_column)

models = {
    "Bernoulli Naive Bayes": BernoulliNB(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = []
st.sidebar.write("### Confusion Matrices for Models")

for model_name, model in models.items():
    accuracy, precision, recall, f1, conf_matrix = train_and_evaluate(X_train, X_test, y_train, y_test, model)
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': conf_matrix
    })
    
    st.sidebar.write(f"#### {model_name} Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f'{model_name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.sidebar.pyplot(fig)

results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)

st.subheader("Model Performance Comparison")
st.dataframe(results_df[['Model', 'Accuracy', 'Precision', 'Recall','F1-Score']])

best_model = results_df.iloc[0]
st.subheader(f" Best Model: {best_model['Model']} ")
st.write(f"**Accuracy**: {best_model['Accuracy']:.2f}")
st.write(f"**Precision**: {best_model['Precision']:.2f}")
st.write(f"**Recall**: {best_model['Recall']:.2f}")

st.subheader("Accuracy Comparison")
plt.figure(figsize=(12,5))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.xticks(rotation=45)
st.pyplot(plt)

st.subheader(f"{dataset_choice} Dataset Correlation Heatmap")
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
st.pyplot(plt)

if 'age' in df.columns:  # Check if age column exists
    st.subheader(f"{dataset_choice} Dataset Age Distribution")
    plt.figure(figsize=(10, 5))
    sns.histplot(df['age'], bins=20, kde=True)
    plt.title(f"Age Distribution in {dataset_choice} Dataset")
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    st.pyplot(plt)
else:
    st.write(f"'{dataset_choice}' dataset does not have an 'age' column.")

st.markdown("Made by SUBIKSHA S ", unsafe_allow_html= True)
