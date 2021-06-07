import pandas as pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


attributes = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"]
data = pandas.read_csv("iris.data", sep="[;,]", engine='python', names=attributes)
parameters = data[data.columns.difference(["Class"])].values
classifications = data['Class'].values

irisClassifier = DecisionTreeClassifier(random_state=1234, criterion='entropy', max_depth=3)
irisClassifier.fit(parameters, classifications)

newExamples = [[4.2, 1.2, 5.8, 2.7], [5.2, 2.4, 7.0, 3.2]]
print(irisClassifier.predict(newExamples))

accuracy = cross_val_score(irisClassifier, parameters, classifications, scoring='accuracy', cv=5)
print(accuracy.mean())
