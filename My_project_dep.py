import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_curve,auc , f1_score

data = pd.read_csv('C:/Users/Cv/Desktop/Data Folder/student.csv')

# print(data.head())
# print(data.info())

encode = LabelEncoder()
columns=['Gender','City','Profession','Sleep Duration','Dietary Habits','Degree','Have you ever had suicidal thoughts ?','Family History of Mental Illness']

for col in columns:
    data[col]=encode.fit_transform(data[col])

# print(data.info())

# Checking dependencies of columns 

corr = data.corr()
plt.figure(dpi=100)
sns.heatmap(data.corr(), annot=True, fmt='.0f')
plt.show()

print(corr['Depression'].sort_values(ascending=False))
# Dropping Unnecessary Columns
# 'Job Satisfaction', ,'Sleep Duration',' ,'Study Satisfaction' 'Work Pressure'
data = data.drop(columns=['Degree','Profession','City'])

# print(data.head())
# Checking for Null Values

print("Null Values in the dataset:")
print(data.isnull().sum())
data['Financial Stress']= data['Financial Stress'].fillna(data['Financial Stress'].mean())

# Statically Analyze the Data .... checking Outliers
desc=data.describe()
# print(desc)

# Handling Outliers
q3=desc.loc['75%']
q1=desc.loc['25%']
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
data = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]

# Checking the proportion of dataset
# for d in data:
    # print(data[d].value_counts(normalize=True)*100)
# Handling the imbalance DataSet using SMOTE

x= data.drop(columns=['Depression'])
y = data['Depression']

# print(x.info())


sm= SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
x_resampled, y_resampled = sm.fit_resample(x, y)

# Checking the proportion of resampled dataset

print(y_resampled.value_counts(normalize=True) * 100)
# print(x_resampled)

# Scaling the Data
scale = MinMaxScaler()
scaled_columns = ['id','Age','Work/Study Hours','Financial Stress','Academic Pressure']

x_resampled[scaled_columns] = scale.fit_transform(x_resampled[scaled_columns])

# Checking the proportion of resampled dataset after scaling

# print(x_resampled.head())


# Splitting the Data into Train and Test Set

x_train,x_test,y_train,y_test=train_test_split(x_resampled,y_resampled,test_size=0.2,random_state=42)

# Training of Model

from sklearn.ensemble import GradientBoostingClassifier
rnf = GradientBoostingClassifier(n_estimators=300,learning_rate=0.05,random_state=42,max_features=5)


rnf.fit(x_train,y_train)
y_pred = rnf.predict(x_test)
score = accuracy_score(y_test,y_pred)
# print("Accuracy Score is:", score)


# ROC Curve 
tpr,fpr,thresholds = roc_curve(y_test,y_pred)
roc= auc(tpr,fpr)
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Depressed Student')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# Get predicted probabilities for positive class
y_proba = rnf.predict_proba(x_test)[:, 1]

# Compute False Positive Rate, True Positive Rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Compute AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

# Plotting the ROC Curve
plt.figure(figsize=(8, 6), dpi=100)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')

# Axis and Title
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Depression Prediction')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

f1Score = f1_score(y_test,y_pred) 
# print("F1 Score is : " , f1Score)

# Model Deployment
from flask import Flask,render_template, request ,redirect,url_for
app = Flask(__name__)

@app.route('/')
def html_connector():
    return render_template('pai.html')

@app.route('/success/<value>')
def result(value,message=None):
    if value=='0':
        message = "Student NOT Depressed"
    else:
        message= "Student is Depressed"    
    return render_template("result.html", result_message=message)
    
@app.route('/input', methods=['POST'])
def handle_input():
    # Extract form data from HTML form
    data = {
        'id': int(request.form['id_val']),
        'gender': int(request.form['gender']),
        'age': float(request.form['age']),
        'academic_pressure': float(request.form['academic_pressure']),
        'work_pressure': float(request.form['work_pressure']),
        'cgpa': float(request.form['cgpa']),
        'study_satisfaction': float(request.form['study_satisfaction']),
        'job_satisfaction': float(request.form['job_satisfaction']),
        'sleep_duration': int(request.form['sleep_duration']),
        'dietary_habits': int(request.form['dietary_habits']),
        'suicidal_thoughts': int(request.form['suicidal_thoughts']),
        'work_study_hours': float(request.form['work_study_hours']),
        'financial_stress': float(request.form['financial_stress']),
        'family_history': int(request.form['family_history']),
    }
    value = [list(data.values())]
    result = rnf.predict(value)
    result = int(result)
    return redirect (url_for('result',value=result))

if __name__ == '__main__':
    app.run(debug=True)
