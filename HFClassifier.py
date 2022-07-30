# -*- coding: utf-8 -*-
import graphviz
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier



col_names = ['Taille', 'Poids', 'Longueur_cheveux', 'Tonalite_voix', 'Classe']
# load dataset
learning_set = pd.read_csv("learning_set.csv", header=None, names=col_names)

# 0 est pour 'Basse' et 1 pour 'Haute'
learning_set.head()
#split dataset in features and target variable
feature_cols = ['Taille', 'Poids', 'Longueur_cheveux', 'Tonalite_voix']

X = learning_set[feature_cols].values[1:7] # Features
y = learning_set.Classe.values[1:7]# Target variable

print(X,'\n',y)
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X,y)

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_cols,
                                class_names=['Femme','Homme'],
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png")

graph.render("decision_tree")

X_test = [[176, 65, 12, 0],[145, 70, 8, 1]]

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print(y_pred)
