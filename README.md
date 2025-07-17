# Student_depression_Analysis
🎓 Student Depression & Mental Health Prediction
This project analyzes student depression and mental health using a machine learning pipeline built with Python. It uses a real-world dataset (CSV) to train a Gradient Boosting model and deploys it as a web application with Flask.

📊 Project Overview
The aim of this project is to predict the likelihood of depression among students based on various academic, social, and personal features. It supports early detection to help provide timely interventions.

🧰 Technologies & Libraries Used
Python (pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn)
Machine Learning: Gradient Boosting Classifier
Data Balancing: SMOTE
Feature Scaling: MinMaxScaler
Web Framework: Flask
Frontend: HTML, CSS
Deployment: Flask + API integration

🧪 ML Pipeline Steps
Data Loading: Import CSV dataset of student data.
Label Encoding: Transform categorical features like Gender, City, Profession, etc.
Correlation Analysis: Visualize relationships between features and target using heatmaps.
Feature Selection: Remove less significant columns (e.g., Degree, City, Profession).
Missing Values: Fill missing Financial Stress data with mean.
Outlier Handling: Detect and remove outliers using IQR method.
Data Balancing: Use SMOTE to handle class imbalance in depression labels.
Feature Scaling: Apply MinMaxScaler to selected numeric features.
Split Dataset: Divide data into train and test sets.
Model Training: Train a Gradient Boosting Classifier.
Model Evaluation: Compute Accuracy, ROC Curve, AUC, and F1 Score.

📈 Model Results
Accuracy Score: (add your score after running)
F1 Score: (add your score)
ROC Curve: Plotted to visualize true positive vs false positive rates.

🌐 Deployment
Built a Flask web app to take user inputs from HTML form.
Predicts if a student is depressed or not based on form data.
Displays results on a separate page.

✅ Key Features
Complete ML pipeline: preprocessing → balancing → scaling → training → evaluation.
Gradient Boosting for robust predictions.
Web deployment using Flask and API integration.
Frontend built with HTML and CSS.
Visual analysis: correlation heatmap, ROC curve.


**📢 Author
Obaid Ur Rehman**
