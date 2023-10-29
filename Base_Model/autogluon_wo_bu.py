from autogluon.tabular import TabularPredictor
from sklearn import metrics
import pandas as pd
import numpy as np
import random
import time

# Set a random seed for reproducibility
random.seed(123)

# Define the target label
label = 'relapse'

# Load the training data from a CSV file and drop unnecessary columns
train_path = "train.csv"
train = pd.read_csv(train_path)
train = train.drop(columns=['filename', 'breslow', 'ulceration_yes', 'ulceration_no'])

# Load the testing data from a CSV file and drop unnecessary columns
test_path = "test.csv"
test = pd.read_csv(test_path)
test = test.drop(columns=['filename', 'breslow', 'ulceration_yes', 'ulceration_no'])

# Print the shapes of the training and testing data
print(f'Training data shape: {train.shape}')
print(f'Testing data shape: {test.shape}')

# Record the start time for training
tic = time.time()
print('Started Training Autogluon Model')

# Train an AutoGluon model with specified settings
predictor = TabularPredictor(label=label, 
                             problem_type='binary', 
                             eval_metric='log_loss', 
                             verbosity=0).fit(train, 
                                              excluded_model_types=['KNN'])
# Record the end time for training
toc = time.time()
print(f"AutoGluon Ran Successfully, time taken: {toc - tic}")

# Get the leaderboard of models based on performance on the test data
leaderboard = predictor.leaderboard(test, silent=True, extra_metrics=['accuracy'])

# Save the leaderboard to a CSV file
leaderboard.to_csv(f'leaderboard_wo_bu.csv', index=False)

# Make predictions on the test data
y_pred = predictor.predict(test)
y_pred_proba = predictor.predict_proba(test)
y_test = test[label]

# Save the predictions to numpy arrays
np.save('./prediction/autogluon_ypred.npy', y_pred)
np.save('./prediction/autogluon_ypred_proba.npy', y_pred_proba)
np.save('./prediction/autogluon_ytest.npy', y_test)

# Calculate and print accuracy and F1 score
print("Accuracy: ", np.sqrt(metrics.accuracy_score(y_test, y_pred)))
print("F1 Score: ", metrics.f1_score(y_test, y_pred, average='macro'))

# Generate and print classification report
classification_report_str = metrics.classification_report(y_test, y_pred)
print(classification_report_str)

# Save classification report to a text file
with open("report_wo_bu.txt", "w") as text_file:
    text_file.write(classification_report_str)
