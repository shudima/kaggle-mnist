import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

def get_train_data():
    df_train = pd.read_csv('train.csv')
    X = df_train.drop('label', 1)
    Y = df_train['label']

    return X, Y


def train_with_classifier(cls, X, Y):

    cls.fit(X, Y)


def create_submission(cls, X, Y):

    X_test = pd.read_csv('test.csv')
    Y_pred = cls.predict(X_test)

    X_test['label'] = Y_pred

    X_test = X_test[['label']]
    X_test['imageId'] = X_test.index + 1

    X_test.to_csv('scikit-learn-sub.csv', index=False)


def main():
    X, Y = get_train_data()

    cv_scores = []
    classifiers_names = ['RandomForest', 'GaussianNB', 'LogisticRegression']
    classifiers = [RandomForestClassifier(n_estimators=200), GaussianNB(), LogisticRegression()]

    for cls in classifiers:
        scores = cross_val_score(cls, X, Y)
        cv_scores.append(np.mean(scores))

    plt.bar(range(len(classifiers)), cv_scores)
    plt.xticks(range(len(classifiers)), classifiers_names)


if __name__ == '__main__':
    main()






