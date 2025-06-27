import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_shap import st_shap

import mlflow
import mlflow.sklearn
import dagshub
import shap

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier


# Initialize DagsHub with MLflow integration

st.set_page_config(
    page_title=" Bank Crunch: Predicting Customer Churn in the Banking Sector",
    layout="centered",
    page_icon="🏦",
)

st.sidebar.title("Customer Churn in the Bank 🏦")
page = st.sidebar.selectbox(
    "Select Page",
    [
        "Presentation 📘",
        "Visualization 📈",
        "Prediction 🤖",
        "Feature Importance 📊",
        "Explainability 🔍",
        "MLflow Runs 🧪",
    ],
)
# Display header image

## still need the image
st.image("Churn.png")
df = pd.read_csv("cleaned_churn.csv")

# Introduction Page
if page == "Presentation 📘":
    st.subheader("01 Presentation 📘")

    st.markdown("###  Introduction:")
    st.write("💡 Customers are the key to a bank’s sustainable growth. Preventing churn is essential for long-term success.")
    st.write("📉 By identifying customers likely to leave, banks can take early action and reduce churn risk.")
    st.markdown("###  Project Goals:")
    st.write("🎯 Our goal: use data to understand why customers leave and how to prevent it.")
    st.write("📊 Visualize churn patterns, 🧠 predict churn probability, and 🔍 identify key drivers.")



    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows))

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)
    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ You have missing values")

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())



# Visulazation Page


elif page == "Visualization 📈":
   st.title("📈 Data Visualization")
   st.subheader("Explore how different factors impact churn rate 💸")
   st.write("")
   st.write("")
   st.write("")


   st.subheader("Looker dashboard:")
   st.components.v1.iframe("https://lookerstudio.google.com/embed/reporting/896a6c3b-6471-4c56-a69b-0e3e43de2b7d/page/cJKPF", height=500, width=870)



# Prediction Page
elif page == "Prediction 🤖":
    st.subheader(" Prediction with different models 🤖")
    
# Model choice
   
    model_option = st.radio("🔘 Select Model", ("Logistic Regression", "Decision Tree","Random Forest"))

    #data processing
    ## One-Hot Encoding for 'Geography'
    df_encoded = pd.get_dummies(df, columns=['Geography'])

    ## Binary Encoding for 'Gender'
    df_encoded['Gender'] = df_encoded['Gender'].map({'Male': 0, 'Female': 1})
    ## Separate the majority and minority classes
    majority_class = df_encoded[df_encoded['Exited'] == 0]
    minority_class = df_encoded[df_encoded['Exited'] == 1]
    ## Oversample the minority class
    minority_oversampled = resample(minority_class,replace=True,n_samples=len(majority_class),random_state=42)
    # Combine the majority class with the oversampled minority class
    data_balanced = pd.concat([majority_class, minority_oversampled])
    X = data_balanced.drop('Exited', axis=1)
    y = data_balanced['Exited']
    feature_names = X.columns.tolist()

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    


    # Scale the training and testing data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

        
    
    #Logistic Regression
    if model_option == "Logistic Regression":
        st.markdown("### 📊 Logistic Regression Evaluation")
        # train the model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

         # predict
        y_pred = model.predict(X_test_scaled)

        # evalution
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # result
        st.dataframe(report_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:.0f}"
        }))
        # confusion matrix
        st.markdown("### 🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
            xticklabels=["No Exit", "Exited"], 
            yticklabels=["No Exit", "Exited"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # predict 
        # 🔮 Users input
        st.subheader("🔮 Make a Prediction")

        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure = st.slider("Tenure (Years at Bank)", 0, 10, 5)
        balance = st.number_input("Account Balance", value=50000.0)
        num_of_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", value=50000.0)

        # Input
        user_input = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active == "Yes" else 0,
            'EstimatedSalary': estimated_salary,
            'Gender': 0 if gender == "Male" else 1,  # 注意：原来 Gender 是 0/1
            'Geography_France': 1 if geography == "France" else 0,
            'Geography_Germany': 1 if geography == "Germany" else 0,
            'Geography_Spain': 1 if geography == "Spain" else 0
        }

        # Input
        input_df = pd.DataFrame([user_input])
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        # Stand
        input_scaled = scaler.transform(input_df)

        # ✅ Pre
        if st.button("Predict Churn Status"):
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0]

            if pred == 1:
                st.error("⚠️ This customer is likely to **EXIT**.")
            else:
                st.success("✅ This customer is likely to **STAY**.")

            st.write("🧪 Probability：[STAY, EXIT] =", prob)

    elif model_option == "Decision Tree":
        st.markdown("### 🌳 Decision Tree Evaluation")
        # training
        max_depth = st.slider("Select Max Depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)  # No scalr
    

        # pre
        y_pred = model.predict(X_test)

        # Viz of decision tree
        with st.expander("🌳 Show Decision Tree Visualization"):
            dot_data = export_graphviz(
                model,
                out_file=None,
                feature_names=feature_names,
                class_names=["Stay", "Exit"],
                filled=True,
                rounded=True,
                special_characters=True
            )
        st.graphviz_chart(dot_data)

        # Evalution report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:.0f}"
        }))
        # 🔮 Pre
        st.subheader("🔮 Make a Prediction")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure = st.slider("Tenure (Years at Bank)", 0, 10, 5)
        balance = st.number_input("Account Balance", value=50000.0)
        num_of_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", value=50000.0)

        # DataFrame
        user_input = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active == "Yes" else 0,
            'EstimatedSalary': estimated_salary,
            'Gender': 0 if gender == "Male" else 1,
            'Geography_France': 1 if geography == "France" else 0,
            'Geography_Germany': 1 if geography == "Germany" else 0,
            'Geography_Spain': 1 if geography == "Spain" else 0
        }

        input_df = pd.DataFrame([user_input])
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]



        if st.button("Predict Churn Status"):
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
            if pred == 1:
                st.error("⚠️ This customer is likely to **EXIT**.")
            else:
                st.success("✅ This customer is likely to **STAY**.")

            st.write("🧪 Probability：[STAY, EXIT] =", prob)

    elif model_option == "Random Forest":
        st.markdown("### 🌲🌲 Random Forest Evaluation")
        # Parameter Choice
        n_estimators = st.slider("Number of Trees", min_value=10, max_value=200, value=100, step=10)
        max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5)
        # Model training
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # pre
        y_pred = model.predict(X_test)

        # evaluation
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:.0f}"
        }))
        # confusion matrix
        st.markdown("### 🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
            xticklabels=["No Exit", "Exited"], 
            yticklabels=["No Exit", "Exited"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

            # 🔮 Pre
        st.subheader("🔮 Make a Prediction")

        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure = st.slider("Tenure (Years at Bank)", 0, 10, 5)
        balance = st.number_input("Account Balance", value=50000.0)
        num_of_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", value=50000.0)

        user_input = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active == "Yes" else 0,
            'EstimatedSalary': estimated_salary,
            'Gender': 0 if gender == "Male" else 1,
            'Geography_France': 1 if geography == "France" else 0,
            'Geography_Germany': 1 if geography == "Germany" else 0,
            'Geography_Spain': 1 if geography == "Spain" else 0
        }

        input_df = pd.DataFrame([user_input])
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

     
        if st.button("Predict Churn Status"):
            
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]

            if pred == 1:
                st.error("⚠️ This customer is likely to **EXIT**.")
            else:
                st.success("✅ This customer is likely to **STAY**.")

            st.write(" Probability：[STAY, EXIT] =", prob)

    
elif page == "Feature Importance 📊":
    st.subheader("🔍 Feature Importance: What drives churn prediction?")
    df_encoded = pd.get_dummies(df, columns=['Geography'])
    df_encoded['Gender'] = df_encoded['Gender'].map({'Male': 0, 'Female': 1})
    majority_class = df_encoded[df_encoded['Exited'] == 0]
    minority_class = df_encoded[df_encoded['Exited'] == 1]
    minority_oversampled = minority_class.sample(n=len(majority_class), replace=True, random_state=42)
    data_balanced = pd.concat([majority_class, minority_oversampled])
    X = data_balanced.drop('Exited', axis=1)
    y = data_balanced['Exited']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    st.dataframe(importance_df)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(10), palette="viridis", ax=ax)
    st.pyplot(fig)




elif page == "Explainability 🔍":
    st.subheader("🔎 Explainability with SHAP")

    # 数据处理
    df_encoded = pd.get_dummies(df, columns=['Geography'])
    df_encoded['Gender'] = df_encoded['Gender'].map({'Male': 0, 'Female': 1})
    X = df_encoded.drop('Exited', axis=1)
    y = df_encoded['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # 模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # 3D: (samples, features, classes)

    # 构造全局解释对象
    shap_exp_global = shap.Explanation(
        values=shap_values[:100][:, :, 1],
        base_values=explainer.expected_value[1],
        data=X_test.iloc[:100],
        feature_names=X_test.columns
    )

    st.subheader("📊 Global Feature Importance (SHAP)")
    st_shap(shap.plots.bar(shap_exp_global))
    idx = st.slider("Choose a test sample index", 0, X_test.shape[0]-1, 0)
    # 局部解释
    shap_exp_local = shap.Explanation(
        values=shap_values[0][:, 1],
        base_values=explainer.expected_value[1],
        data=X_test.iloc[0],
        feature_names=X_test.columns
    )

    st.subheader("🔍 Local Explanation for First Customer")
    st_shap(shap.plots.waterfall(shap_exp_local), height=400)

# MLFlow Runs Page


elif page == "MLflow Runs 🧪":
   st.title("🛠️ Hyperparameter Tuning & MLflow Dashboard")


   # Transform the data into a numeric format
   df_encoded = pd.get_dummies(df, columns = ['Geography'])
   df_encoded['Gender'] = df_encoded['Gender'].map({
       'Male': 0,
       'Female': 1
       })
  
   # Create training and testing categories
   X = df_encoded.drop('Exited', axis=1)
   Y = df_encoded['Exited']
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
   tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Manual Grid Search", "2️⃣ Auto Grid Search", "3️⃣ Browse MLflow Runs", "4️⃣ MLflow Dashboard"])
  
   # Manual tuning with decision trees and KNN
   with tab1:
       st.markdown("### Select Model and Hyperparameters")


       model_type = st.selectbox("Choose Model", ["Decision Tree", "K Nearest Neighbors"])


       if model_type == "Decision Tree":
           param = st.selectbox("Select max_depth", [2, 4, 6, 8, 10, 12, None])
       else:
           param = st.selectbox("Select n_neighbors", [1, 3, 5, 7, 9, 11, 15])


       if st.button("Train & Log to MLflow"):
           with mlflow.start_run():
               if model_type == "Decision Tree":
                   model = DecisionTreeClassifier(max_depth=param, random_state=42)
                   mlflow.log_param("model_type", "DecisionTree")
                   mlflow.log_param("max_depth", param)
               else:
                   model = KNeighborsClassifier(n_neighbors=param)
                   mlflow.log_param("model_type", "K Nearest Neighbors")
                   mlflow.log_param("n_neighbors", param)


               model.fit(X_train, Y_train)
               Y_pred = model.predict(X_test)


               acc = accuracy_score(Y_test, Y_pred)
               prec = precision_score(Y_test, Y_pred)
               rec = recall_score(Y_test, Y_pred)
               f1 = f1_score(Y_test, Y_pred)


               mlflow.log_metric("accuracy", acc)
               mlflow.log_metric("precision", prec)
               mlflow.log_metric("recall", rec)
               mlflow.log_metric("f1_score", f1)


               st.success("✅ Logged to MLflow")
               st.write(f"**Accuracy:** {acc:.3f}  |  **Precision:** {prec:.3f}  |  **Recall:** {rec:.3f}  |  **F1:** {f1:.3f}")


   # Automatic tuning using GridSearchCV
   with tab2:
       st.subheader("Automatic Hyperparameter Tuning")
       model_type_auto = st.selectbox("Select Model for Auto Tuning", ["Decision Tree", "K Nearest Neighbors"], key="auto_model")


       with mlflow.start_run():
           if model_type_auto == "Decision Tree":
               param_grid = {
                   "max_depth": [2, 4, 6, 8, 10, 12, None]
               }
               model = DecisionTreeClassifier(random_state = 42)
               mlflow.log_param("model_type", "DecisionTree (GridSearchCV)")
           else:
               param_grid = {
                   "n_neighbors": [1, 3, 5, 7, 9, 11, 15]
               }
               model = KNeighborsClassifier()
               mlflow.log_param("model_type", "KNN (GridSearchCV)")


           grid_search = GridSearchCV(model, param_grid, scoring="f1", cv=5)
           grid_search.fit(X_train, Y_train)


           best_model = grid_search.best_estimator_
           Y_pred = best_model.predict(X_test)


           acc = accuracy_score(Y_test, Y_pred)
           prec = precision_score(Y_test, Y_pred)
           rec = recall_score(Y_test, Y_pred)
           f1 = f1_score(Y_test, Y_pred)


           for param_name, value in grid_search.best_params_.items():
               mlflow.log_param(param_name, value)


           mlflow.log_metric("accuracy", acc)
           mlflow.log_metric("precision", prec)
           mlflow.log_metric("recall", rec)
           mlflow.log_metric("f1_score", f1)


           st.success("✅ Auto-tuned model logged to MLflow")
           st.write(f"**Best Parameters:** {grid_search.best_params_}")
           st.write(f"**Accuracy:** {acc:.3f}  |  **Precision:** {prec:.3f}  |  **Recall:** {rec:.3f}  |  **F1:** {f1:.3f}")


   # View MLFlow embedded logged runs
   with tab3:
       st.info("Below you can browse logged experiments. Alternatively, you can go to your MLFlow server.")
       client = MlflowClient()
       experiment = client.get_experiment_by_name("Default")  # Change if your experiment has a custom name
       runs = client.search_runs(experiment_ids=[experiment.experiment_id])


       df_runs = pd.DataFrame([{
           "run_id": i.info.run_id,
           "model_type": i.data.params.get("model_type"),
           "max_depth": i.data.params.get("max_depth"),
           "n_neighbors": i.data.params.get("n_neighbors"),
           "accuracy": i.data.metrics.get("accuracy"),
           "precision": i.data.metrics.get("precision"),
           "recall": i.data.metrics.get("recall"),
           "f1_score": i.data.metrics.get("f1_score"),
       } for i in runs])
       st.dataframe(df_runs)
  
   # View MLFlow logged runs on DagsHub
   with tab4:
       st.markdown("### 📦 MLflow Dashboard on DagsHub")
       st.markdown("Track your full experiment history and compare model performance visually.")
       st.link_button("View MLflow Dashboard", "https://dagshub.com/alexreifel7/bank-churn-predictor.mlflow")











        





