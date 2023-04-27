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


# This is our main class
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
                g_score = cls.score(split_s, entro, total=len(dataset))
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
        index = {'cap-shape': 1, 'cap-surface': 2, 'cap-color': 3, 'bruises': 4, 'odor': 5, 'gill-attachment': 6,
                 'gill-spacing': 7, 'gill-size': 8, 'gill-color': 9, 'stalk-shape': 10, 'stalk-root': 11,
                 'stalk-surface-above-ring': 12, 'stalk-surface-below-ring': 13,
                 'stalk-color-above-ring': 14, 'stalk-color-below-ring': 15, 'veil-type': 16, 'veil-color': 17,
                 'ring-number': 18, 'ring-type': 19, 'spore-print-color': 20,
                 'population': 21, 'habitat': 22}
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
    mushroom_dataset = pd.read_csv("./data/mushroom.data",
                                   names=["decision", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
                                          "gill-attachment",
                                          "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root",
                                          "stalk-surface-above-ring", "stalk-surface-below-ring",
                                          "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
                                          "ring-number", "ring-type", "spore-print-color",
                                          "population", "habitat"])

    mushroom_dataset.replace(
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w', 'x', 'y', 'z'],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], inplace=True)

    mushroom_dataset["stalk-root"].replace(["?"], ['0'], inplace=True)
    mushroom_dataset["stalk-root"] = mushroom_dataset['stalk-root'].astype(int)
    average = mushroom_dataset["stalk-root"].mean()
    mushroom_dataset["stalk-root"].replace(["?"], [average], inplace=True)

    mushroom_dataset["decision"] = mushroom_dataset['decision'].astype(int)
    mushroom_dataset["cap-shape"] = mushroom_dataset['cap-shape'].astype(int)
    mushroom_dataset["cap-surface"] = mushroom_dataset['cap-surface'].astype(int)
    mushroom_dataset["cap-color"] = mushroom_dataset['cap-color'].astype(int)
    mushroom_dataset["bruises"] = mushroom_dataset['bruises'].astype(int)
    mushroom_dataset["odor"] = mushroom_dataset['odor'].astype(int)
    mushroom_dataset["gill-attachment"] = mushroom_dataset['gill-attachment'].astype(int)
    mushroom_dataset["gill-spacing"] = mushroom_dataset['gill-spacing'].astype(int)
    mushroom_dataset["gill-size"] = mushroom_dataset['gill-size'].astype(int)
    mushroom_dataset["gill-color"] = mushroom_dataset['gill-color'].astype(int)
    mushroom_dataset["stalk-shape"] = mushroom_dataset['stalk-shape'].astype(int)
    mushroom_dataset["stalk-surface-above-ring"] = mushroom_dataset['stalk-surface-above-ring'].astype(int)
    mushroom_dataset["stalk-surface-below-ring"] = mushroom_dataset['stalk-surface-below-ring'].astype(int)
    mushroom_dataset["stalk-color-above-ring"] = mushroom_dataset['stalk-color-above-ring'].astype(int)
    mushroom_dataset["stalk-color-below-ring"] = mushroom_dataset['stalk-color-below-ring'].astype(int)
    mushroom_dataset["veil-type"] = mushroom_dataset['veil-type'].astype(int)
    mushroom_dataset["veil-color"] = mushroom_dataset['veil-color'].astype(int)
    mushroom_dataset["ring-number"] = mushroom_dataset['ring-number'].astype(int)
    mushroom_dataset["ring-type"] = mushroom_dataset['ring-type'].astype(int)
    mushroom_dataset["spore-print-color"] = mushroom_dataset['spore-print-color'].astype(int)
    mushroom_dataset["population"] = mushroom_dataset['population'].astype(int)
    mushroom_dataset["habitat"] = mushroom_dataset['habitat'].astype(int)

    # Looping to perform k-fold 10 times
    for i in range(10):
        # Shuffling the dataset
        mushroom_dataset = mushroom_dataset.sample(frac=1)
        model = ID3()
        X = mushroom_dataset.drop(["decision"], axis=1)
        y = mushroom_dataset.decision
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.55)
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
