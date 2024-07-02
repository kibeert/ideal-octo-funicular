#importing the neccesary libraries
import numpy as np
import pandas as pd
from scipy.stats import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from fuzzywuzzy import process
import streamlit as st
from sklearn.model_selection import cross_val_score


#importing the datasets
Data_path = pd.read_csv("datasets\Training.csv")
data = Data_path.dropna(axis = 1 )

#checking whether the datasets is balanced
disease_counts = data['prognosis'].value_counts()
temp_df = pd.DataFrame({'Disease' : disease_counts.index, 'Count' : disease_counts.values})

plt.figure(figsize = (18, 8))
sns.barplot(x = 'Disease', y = 'Count', data = temp_df)
plt.xticks(rotation = 90)
plt.show()

#Encoding the target which is the prognosis into numerical
encoder = LabelEncoder()
data['prognosis'] = encoder.fit_transform(data['prognosis'])

#split the data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 24)
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

#Building the model
#define the scoring metrics

def cv_scoring(estimator, X, y):
  return accuracy_score(y , estimator.predict(X))

#initialize the model
models = {
    "SVC": SVC(),
    "GaussianNB": GaussianNB(),
    "Random forest" :RandomForestClassifier( random_state = 18)
}

#producing cross_validation
for model_name in models:
  model = models[model_name]
  scores = cross_val_score(model, X, y, cv=10,
                           n_jobs= -1,
                           scoring = cv_scoring)
  print("==" * 30)
  print(model_name)
  print(f"Scores: {scores}")
  print(f"Mean: {np.mean(scores)}")
  print(f"Std: {np.std(scores)}")


#training and fitting SVM classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
preds = svm_model.predict(X_test)

print(f"Accuracy on train data by svm classifier: {accuracy_score(y_train, svm_model.predict(X_train)) *100}")
print(f"Accuracy on test data by svm classifier: {accuracy_score(y_test, preds) *100 }")

print(f"Accuracy on test data by SVM Classifier: {accuracy_score(y_test, preds)*100}")
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Classifier on Test Data")
plt.show()

#training the Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
val_pred = nb.predict(X_test)

acs = accuracy_score(y_test, val_pred )
print(acs)
print(f"Accuracy on test data by Naive Bayes Classifier: {accuracy_score(y_test, preds)*100}")
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Naive Bayes Classifier on Test Data")
plt.show()

#Training The random Forest
rf= RandomForestClassifier( random_state = 18)
rf.fit(X_train, y_train)
val_pred2 = rf.predict(X_test)
rf_acs = accuracy_score(y_test, val_pred2)
print(f"The accuracy score of random forest classifier is {rf_acs}")
cf_matrix = confusion_matrix(y_test, val_pred2)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
plt.show()


from statistics import mode
# Training the models on whole data
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier( random_state = 18)

final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

test_data = pd.read_csv("datasets\Testing.csv")

test_X = test_data.iloc[:, :-1]
test_y = encoder.transform(test_data.iloc[:, -1])

#making prediction on the test data
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)

final_preds = [mode([i, j, k]) for i ,j ,k in zip(svm_preds, nb_preds, rf_preds)]
final_preds = [int(x) for x in final_preds]


print(f"Accuracy score on test data by the combined model: {accuracy_score(test_y, final_preds)*100}")

cf_matrix = confusion_matrix(test_y, final_preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Combined Model on Test Data")
plt.show()

symptoms = X.columns.values

symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

 

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}


# Function to find the best match for a symptom
def get_best_match(symptom, symptom_index):
    best_match, score = process.extractOne(symptom, symptom_index.keys())
    return best_match if score > 75 else None  # You can adjust the threshold as needed


# Function to predict disease based on symptoms
# Defining the Function
# Input: string containing symptoms separated by commas
# Output: Generated predictions by models
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    symptoms = [symptom.strip() for symptom in symptoms]  # Remove leading/trailing whitespace


    # Find best matches for all symptoms
    matched_symptoms = {}
    for symptom in symptoms:
        best_match = get_best_match(symptom, data_dict["symptom_index"])
        if best_match:
            matched_symptoms[symptom] = best_match
        else:
            return f"Symptom '{symptom}' does not exist and no close match was found."

    # Creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for original_symptom, matched_symptom in matched_symptoms.items():
        index = data_dict["symptom_index"][matched_symptom]
        input_data[index] = 1

    

    # Reshaping the input data and converting it into suitable format for model predictions
    input_data = np.array(input_data).reshape(1, -1)

    # Generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    # Making final prediction by taking mode of all predictions
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions

# Testing the function
print(predictDisease("Itching,Skin Rash,Nodal Skin Eruptions"))


st.title('Disease Prediction Model')
symptoms_input = st.text_input('Enter symptoms separated by commas:')
if st.button('Predict'):
    result = predictDisease(symptoms_input)
    st.write(result)