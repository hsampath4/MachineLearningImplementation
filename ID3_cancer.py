import pandas as pd, numpy as np
from collections import Counter
from math import log
from sklearn.base import BaseEstimator as estimator, ClassifierMixin as mix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def entropy(class1=0, class2=0):
    class_list = [class1, class2]
    final_entropy = 0
    for c in class_list:
        if c != 0:
            final_entropy += -((c / sum(class_list)) * log(c / sum(class_list), 4))
    return final_entropy


class ID3(estimator, mix):

    def __init__(self, class_col="labels"):
        self.class_col = class_col

    @staticmethod
    def score(split_s, entro, total):
        entro_set = [entropy(*i) for i in split_s]
        f = lambda x, y: (sum(x) / total) * y
        result = [f(i, j) for i, j in zip(split_s, entro_set)]
        return entro - sum(result)

    @staticmethod
    def split_set(header, dataset, class_col):
        df = pd.DataFrame(dataset.groupby([header, class_col])[class_col].count())
        result = []
        for i in Counter(dataset[header]).keys():
            result.append(df.loc[i].values)

        return result

    @classmethod
    def node(cls, dataset, class_col):
        entro = entropy(*[i for i in Counter(dataset[class_col]).values()])
        result = {}
        for i in dataset.columns:
            if i != class_col:
                split_s = cls.split_set(i, dataset, class_col)
                g_score = cls.score(split_s, entro, total=len(dataset))  # total gain of an attribute
                result[i] = g_score
        return max(result, key=result.__getitem__)

    @classmethod
    def recursion(cls, dataset, tree, class_col):
        n = cls.node(dataset, class_col)
        branchs = [i for i in Counter(dataset[n])]
        tree[n] = {}
        for j in branchs:
            br_data = dataset[dataset[n] == j]
            if entropy(*[i for i in Counter(br_data[class_col]).values()]) != 0:
                tree[n][j] = {}
                cls.recursion(br_data, tree[n][j], class_col)
            else:
                r = Counter(br_data[class_col])
                tree[n][j] = max(r, key=r.__getitem__)
        return

    @classmethod
    def pred_recur(cls, tupl, t):
        if type(t) is not dict:
            return t
        index = {'diagnosis': 1, 'radius': 2, 'texture': 3, 'perimeter': 4, 'area': 5, 'smoothness': 6,
                 'compactness': 7, 'concavity': 8, 'concave points': 9}
        for i in t.keys():
            if i in index.keys():
                td = tupl[index[i]]
                s = t[i].get(tupl[index[i]], 0)
                r = cls.pred_recur(tupl, t[i].get(tupl[index[i]], 0))
        return r

    def predict(self, test):
        result = []
        for i in test.itertuples():
            result.append(ID3.pred_recur(i, self.tree_))
        return pd.Series(result)

    def fit(self, X, y):
        class_col = self.class_col
        dataset = X.assign(labels=y)
        self.tree_ = {}
        ID3.recursion(dataset, self.tree_, class_col)
        return self


if __name__ == '__main__':

    acc = []
    # Preprocessing of the dataset
    cancer_dataset = pd.read_csv("./data/breast-cancer-wisconsin.data",
                                 names=["id", "diagnosis", "radius", "texture", "perimeter", "area", "smoothness",
                                        "compactness", "concavity", "concave points", "decision"])
    cancer_dataset["smoothness"].replace(["?"], ['0'], inplace=True)
    cancer_dataset['smoothness'] = cancer_dataset['smoothness'].astype(int)
    average = cancer_dataset["smoothness"].mean()
    cancer_dataset["smoothness"].replace(["0"], [average], inplace=True)

    cancer_dataset['diagnosis'] = cancer_dataset['diagnosis'].astype(int)
    cancer_dataset['radius'] = cancer_dataset['radius'].astype(int)
    cancer_dataset['texture'] = cancer_dataset['texture'].astype(int)
    cancer_dataset['perimeter'] = cancer_dataset['perimeter'].astype(int)
    cancer_dataset['area'] = cancer_dataset['area'].astype(int)
    cancer_dataset['smoothness'] = cancer_dataset['smoothness'].astype(int)
    cancer_dataset['compactness'] = cancer_dataset['compactness'].astype(int)
    cancer_dataset['concavity'] = cancer_dataset['concavity'].astype(int)
    cancer_dataset['concave points'] = cancer_dataset['concave points'].astype(int)
    cancer_dataset['decision'] = cancer_dataset['decision'].astype(int)

    # Looping to perform k-fold 10 times
    for i in range(10):
        # Shuffling the dataset
        cancer_dataset = cancer_dataset.sample(frac=1)
        model = ID3()
        X = cancer_dataset.drop(["id", "decision"], axis=1)
        y = cancer_dataset.decision
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45)
        entro_set = entropy(*[i for i in Counter(y_train).values()])
        model.fit(X_train, y_train)
        accuracy_score(y_test, model.predict(X_test))
        cross_val = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        for i in range(0, len(cross_val)):
            acc.append(cross_val[i])

    average_acc = np.sum(acc) / len(acc)
    standardDeviation = np.std(acc)
    print("Average Accuracy:", average_acc)
    print("Standard Deviation: ", standardDeviation)