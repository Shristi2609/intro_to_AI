Health Risk Prediction Using BMI, Exercise, and Eating Habits
Project Overview
This project aims to predict an individual's health risk category (low, medium, or high) based on three key factors:
1.	Body Mass Index (BMI)
2.	Exercise Habits
3.	Eating Habits
A machine learning model is built using a Random Forest Classifier to predict the health risk category of individuals. The model is evaluated using metrics such as accuracy, precision, and recall. The results are presented through a confusion matrix heatmap to visually assess the classification performance.
________________________________________
Objective
•	Predict an individual’s health risk category (low, medium, high) based on the following features:
o	BMI: A measure of body fat based on height and weight.
o	Exercise Habits: Frequency and intensity of physical activity.
o	Eating Habits: Categorized diet quality (e.g., balanced diet, poor diet).
•	Evaluate the model’s performance using classification metrics (accuracy, precision, recall).
•	Visualize the model’s performance using a confusion matrix heatmap.
________________________________________
Requirements
To run this project, the following Python libraries are required:
•	pandas (for data manipulation)
•	numpy (for numerical operations)
•	seaborn (for visualization)
•	matplotlib (for plotting)
•	scikit-learn (for machine learning)
You can install the required libraries using pip:
bash
CopyEdit
pip install pandas numpy seaborn matplotlib scikit-learn
________________________________________
Dataset
The dataset used for this project contains the following columns:
•	BMI: Numeric value representing the Body Mass Index of an individual.
•	Exercise: Numeric or categorical value representing exercise frequency (e.g., hours per week or a categorical scale).
•	Eating Habits: Categorical values indicating the quality of the individual's diet (e.g., "balanced", "poor").
•	Risk Category: Target variable (categorical: "low", "medium", "high") indicating the health risk category of the individual.
Example of the dataset structure:
BMI	Exercise	Eating Habits	Risk Category
22.5	3	Balanced	Medium
30.1	1	Poor	High
18.6	5	Balanced	Low
________________________________________
Steps to Run the Code
1.	Clone the repository (if hosted on GitHub):
bash
CopyEdit
git clone https://github.com/yourusername/health-risk-prediction.git
cd health-risk-prediction
2.	Ensure the dataset is in the correct location:
o	Place your dataset health_risk.csv in the same directory as the script or update the file path in the code.
3.	Install dependencies:
Run the following command to install the required libraries:
bash
CopyEdit
pip install -r requirements.txt
4.	Run the script:
Once the dependencies are installed and the dataset is correctly placed, run the Python script to train and evaluate the model:
bash
CopyEdit
python health_risk_prediction.py
________________________________________
Code Explanation
Data Loading
The data is loaded from a CSV file (health_risk.csv) using the pandas library.
python
CopyEdit
data = pd.read_csv("health_risk.csv")
Model Training
We use Random Forest Classifier to predict the health risk category. The model is trained on 80% of the data and evaluated on the remaining 20%.
python
CopyEdit
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
Model Evaluation
The model is evaluated using accuracy, precision, and recall metrics. Additionally, a confusion matrix is generated to assess the classification performance.
python
CopyEdit
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
Confusion Matrix Heatmap
The confusion matrix is visualized using seaborn's heatmap functionality to provide an intuitive view of classification results.
python
CopyEdit
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
________________________________________
Evaluation Metrics
•	Accuracy: Proportion of correct predictions (overall accuracy).
•	Precision: Proportion of positive predictions that were actually correct.
•	Recall: Proportion of actual positives that were correctly predicted.
These metrics help evaluate how well the model distinguishes between different health risk categories (low, medium, high).
________________________________________
Expected Results
After running the code, you should see:
•	Model Evaluation Metrics (accuracy, precision, recall).
•	Confusion Matrix Heatmap: A visual representation of the model’s performance on the test data.
________________________________________
Future Improvements
1.	Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV to optimize model parameters.
2.	Feature Engineering: Include additional features such as age, gender, and lifestyle factors for better predictions.
3.	Model Comparison: Test other algorithms (e.g., SVM, Logistic Regression) to improve performance.
________________________________________
References
1.	Cohn, R., & Smith, M. (2018). "Predicting Health Risks Using Machine Learning Techniques: A Review." Journal of Health Informatics, 25(4), 123-135.
2.	Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32. DOI:10.1023/A:1010933404324
3.	Dietrich, A., & Hoffer, T. (2020). "Predictive Modeling in Healthcare: A Practical Approach." Healthcare Analytics, 14(2), 22-30.
________________________________________
License
This project is licensed under the MIT License - see the LICENSE file for details.
________________________________________
Contact
For any questions or contributions, feel free to contact me at:
•	Email: iamshristi26@gmail.com
•	Github:https://github.com/Shristi2609/intro_to_AI

