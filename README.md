# 🩺 Breast Cancer Classification using Logistic Regression

## 📌 Project Overview
This project builds a **binary classification model** to predict whether a breast tumor is **malignant or benign** using **Logistic Regression**.

The workflow includes:
- Data preprocessing  
- Exploratory Data Analysis (EDA)  
- Model training  
- Model evaluation using multiple performance metrics  

The focus is not only on **accuracy**, but also on **recall, precision, F-beta scores, log loss, and ROC-AUC**, which are crucial in **healthcare-related machine learning problems**.

---

## 📂 Dataset
- Dataset file: **`breast_cancer_data.csv`**
- Each row represents a patient
- The dataset contains:
  - Multiple **numerical feature columns**
  - One target column: **`target`**
    - `0` → Benign  
    - `1` → Malignant  

---

## 🛠️ Technologies & Libraries Used
- **Python**
- **Pandas** – Data manipulation
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Seaborn** – Advanced visualization
- **Scikit-learn** – Machine learning models and evaluation

---

## 🔄 Project Workflow

### 1️⃣ Data Loading
- Load the dataset using Pandas
- Perform basic inspection:
  - Shape of the dataset
  - Data types
  - Summary statistics

---

### 2️⃣ Data Preprocessing
- Separate features (**X**) and target (**y**)
- Split the data into:
  - **80% training**
  - **20% testing**
- Standardize features using **StandardScaler**

---

### 3️⃣ Exploratory Data Analysis (EDA)
- Visualize class distribution using a **pie chart**
- Analyze correlations between features and the target variable
- Generate:
  - Feature-to-target correlation bar plot
  - Correlation heatmap

---

### 4️⃣ Model Training
- Train a **Logistic Regression** model
- Use a fixed **random state** for reproducibility

---

### 5️⃣ Model Evaluation
The model is evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report
- F2 Score (Recall-weighted)
- F0.5 Score (Precision-weighted)
- Log Loss
- ROC Curve
- AUC Score

---

## 📊 Evaluation Metrics Explained
- **Accuracy** – Overall correctness  
- **Precision** – Correct positive predictions  
- **Recall** – Correctly identified positives  
- **F2 Score** – Emphasizes recall  
- **F0.5 Score** – Emphasizes precision  
- **Log Loss** – Probability prediction quality  
- **ROC-AUC** – Class separation capability  

---

## 📈 Visualization Outputs
- Target distribution pie chart  
- Correlation bar plot  
- Correlation heatmap  
- ROC curve with AUC score  

---

## ✅ Key Learnings
- Importance of feature scaling
- Accuracy alone is insufficient
- Recall is critical in healthcare problems
- ROC-AUC provides strong evaluation insight

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>






