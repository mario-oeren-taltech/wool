import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from general_functions import train_model, test_model, target_shuffling

# Read in the datasets.
ffcv1 = pandas.read_csv('FFCV1.csv', header=0)
ffcv2 = pandas.read_csv('FFCV2.csv', header=0)
ffcv3 = pandas.read_csv('FFCV3.csv', header=0)
ffcv4 = pandas.read_csv('FFCV4.csv', header=0)
ffcv5 = pandas.read_csv('FFCV5.csv', header=0)

# Start the model training process.
print('Choosing the model type')

models = [

    [
        'Random forest',
        RandomForestClassifier(
            # n_estimators=300,
            # class_weight='balanced',
            random_state=42
        )
    ],

    [
        'Support vector machine',
        SVC(
            random_state=42
        )
    ],

    [
        'K-NN',
        KNeighborsClassifier(
            n_neighbors=5
        )
    ],

    [
        'Decision trees',
        DecisionTreeClassifier(

        )
    ]
]

for name, model in models:
    print(name)
    a = train_model(model, ffcv1.iloc[:, 1:], ffcv1.iloc[:, 0], 0.1)[2]
    b = train_model(model, ffcv2.iloc[:, 1:], ffcv2.iloc[:, 0], 0.1)[2]
    c = train_model(model, ffcv3.iloc[:, 1:], ffcv3.iloc[:, 0], 0.1)[2]
    d = train_model(model, ffcv4.iloc[:, 1:], ffcv4.iloc[:, 0], 0.1)[2]
    e = train_model(model, ffcv5.iloc[:, 1:], ffcv5.iloc[:, 0], 0.1)[2]

    average = sum([a, b, c, d, e]) / len([a, b, c, d, e])
    print(f'Average kappa: {average}')

# Test for over-training.
print('Y-shuffle results')

model = RandomForestClassifier(random_state=42)
target_shuffling(model, ffcv1.iloc[:, 1:], ffcv1.iloc[:, 0], 0.1)
target_shuffling(model, ffcv2.iloc[:, 1:], ffcv2.iloc[:, 0], 0.1)
target_shuffling(model, ffcv3.iloc[:, 1:], ffcv3.iloc[:, 0], 0.1)
target_shuffling(model, ffcv4.iloc[:, 1:], ffcv4.iloc[:, 0], 0.1)
target_shuffling(model, ffcv5.iloc[:, 1:], ffcv5.iloc[:, 0], 0.1)

model = SVC(random_state=42)
target_shuffling(model, ffcv1.iloc[:, 1:], ffcv1.iloc[:, 0], 0.1)
target_shuffling(model, ffcv2.iloc[:, 1:], ffcv2.iloc[:, 0], 0.1)
target_shuffling(model, ffcv3.iloc[:, 1:], ffcv3.iloc[:, 0], 0.1)
target_shuffling(model, ffcv4.iloc[:, 1:], ffcv4.iloc[:, 0], 0.1)
target_shuffling(model, ffcv5.iloc[:, 1:], ffcv5.iloc[:, 0], 0.1)

model = KNeighborsClassifier()
target_shuffling(model, ffcv1.iloc[:, 1:], ffcv1.iloc[:, 0], 0.1)
target_shuffling(model, ffcv2.iloc[:, 1:], ffcv2.iloc[:, 0], 0.1)
target_shuffling(model, ffcv3.iloc[:, 1:], ffcv3.iloc[:, 0], 0.1)
target_shuffling(model, ffcv4.iloc[:, 1:], ffcv4.iloc[:, 0], 0.1)
target_shuffling(model, ffcv5.iloc[:, 1:], ffcv5.iloc[:, 0], 0.1)

model = DecisionTreeClassifier(random_state=42)
target_shuffling(model, ffcv1.iloc[:, 1:], ffcv1.iloc[:, 0], 0.1)
target_shuffling(model, ffcv2.iloc[:, 1:], ffcv2.iloc[:, 0], 0.1)
target_shuffling(model, ffcv3.iloc[:, 1:], ffcv3.iloc[:, 0], 0.1)
target_shuffling(model, ffcv4.iloc[:, 1:], ffcv4.iloc[:, 0], 0.1)
target_shuffling(model, ffcv5.iloc[:, 1:], ffcv5.iloc[:, 0], 0.1)


# Continue with the Random forest classifier.
model = RandomForestClassifier(random_state=42)

# Train the five models.
print('Training the models')
model_1 = train_model(model, ffcv1.iloc[:, 1:], ffcv1.iloc[:, 0], 0.1)[0]
model_2 = train_model(model, ffcv2.iloc[:, 1:], ffcv2.iloc[:, 0], 0.1)[0]
model_3 = train_model(model, ffcv3.iloc[:, 1:], ffcv3.iloc[:, 0], 0.1)[0]
model_4 = train_model(model, ffcv4.iloc[:, 1:], ffcv4.iloc[:, 0], 0.1)[0]
model_5 = train_model(model, ffcv5.iloc[:, 1:], ffcv5.iloc[:, 0], 0.1)[0]

# Test the models.
print('Testing the models')
test_set = pandas.read_csv('TEST.csv', header=0)

predictions_1, probabilities_1 = test_model(model_1, test_set.iloc[:, 1:], test_set.iloc[:, 0])[3:]
predictions_2, probabilities_2 = test_model(model_2, test_set.iloc[:, 1:], test_set.iloc[:, 0])[3:]
predictions_3, probabilities_3 = test_model(model_3, test_set.iloc[:, 1:], test_set.iloc[:, 0])[3:]
predictions_4, probabilities_4 = test_model(model_4, test_set.iloc[:, 1:], test_set.iloc[:, 0])[3:]
predictions_5, probabilities_5 = test_model(model_5, test_set.iloc[:, 1:], test_set.iloc[:, 0])[3:]

# Print out the consensus predictions.
predictions = numpy.mean([predictions_1, predictions_2, predictions_3, predictions_4, predictions_5], axis=0)
for a, b in zip(test_set.iloc[:, 0], predictions):
    print(a, round(b))

predictions = numpy.mean([predictions_1, predictions_2, predictions_3, predictions_4, predictions_5], axis=0)
probabilities = numpy.mean([probabilities_1, probabilities_2, probabilities_3, probabilities_4, probabilities_5], axis=0)
for a, b, c, d in zip(test_set.iloc[:, 0], predictions, probabilities, test_set.iloc[:, 1]):
    print(a, round(b), c, d)

importances = model_1.feature_importances_

feature_names = ffcv1.iloc[:, 1:].columns

importance_df = pandas.DataFrame({
    "feature": feature_names,
    "importance": importances
})
importance_df = importance_df.sort_values(by="importance", ascending=False)
print(importance_df.head(5000))

importances = model_2.feature_importances_

feature_names = ffcv2.iloc[:, 1:].columns

importance_df = pandas.DataFrame({
    "feature": feature_names,
    "importance": importances
})
# importance_df = importance_df.sort_values(by="importance", ascending=False)
print(importance_df.head(5000))
importances = model_3.feature_importances_

feature_names = ffcv3.iloc[:, 1:].columns

importance_df = pandas.DataFrame({
    "feature": feature_names,
    "importance": importances
})
# importance_df = importance_df.sort_values(by="importance", ascending=False)
print(importance_df.head(5000))
importances = model_4.feature_importances_

feature_names = ffcv4.iloc[:, 1:].columns

importance_df = pandas.DataFrame({
    "feature": feature_names,
    "importance": importances
})
# importance_df = importance_df.sort_values(by="importance", ascending=False)
print(importance_df.head(5000))
importances = model_5.feature_importances_

feature_names = ffcv5.iloc[:, 1:].columns

importance_df = pandas.DataFrame({
    "feature": feature_names,
    "importance": importances
})
# importance_df = importance_df.sort_values(by="importance", ascending=False)
print(importance_df.head(5000))
