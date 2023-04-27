import pandas as pd, numpy as np
from collections import Counter
from math import log
from sklearn.base import BaseEstimator as estimator, ClassifierMixin as mix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def entropy(class1=0, class2=0, class3=0, class4=0):
    classList = [class1, class2, class3, class4]
    entrpy = 0
    for c in classList:
        if c != 0:
            entrpy += -((c / sum(classList)) * log(c / sum(classList), 4))
    return entrpy


class ID3(estimator, mix):

    def __init__(self, columns="labels"):
        self.columns = columns

    @staticmethod
    def split(header, dataset, columns):
        df = pd.DataFrame(dataset.groupby([header, columns])[columns].count())
        ans = []
        for i in Counter(dataset[header]).keys():
            ans.append(df.loc[i].values)
        return ans

    @staticmethod
    def score(split_s, entro, total):
        entro = [entropy(*i) for i in split_s]
        f = lambda x, y: (sum(x) / total) * y
        ans = [f(i, j) for i, j in zip(split_s, entro)]
        return entro - sum(ans)

    @classmethod
    def node(cls, dataset, columns):
        entro = entropy(*[i for i in Counter(dataset[columns]).values()])
        ans = {}
        for i in dataset.columns:
            if i != columns:
                split_s = cls.split(i, dataset, columns)
                g_score = cls.score(split_s, entro, total=len(dataset))
                ans[i] = g_score
        return max(ans, key=ans.__getitem__)

    @classmethod
    def recursion(cls, dataset, tree, columns):
        n = cls.node(dataset, columns)
        branchs = [i for i in Counter(dataset[n])]
        tree[n] = {}
        for j in branchs:
            br_data = dataset[dataset[n] == j]
            if entropy(*[i for i in Counter(br_data[columns]).values()]) != 0:
                tree[n][j] = {}
                cls.recursion(br_data, tree[n][j], columns)
            else:
                r = Counter(br_data[columns])
                tree[n][j] = max(r, key=r.__getitem__)
        return
    #     def recursion(cls, dataset, tree, class_col):
    #         n = cls.node(dataset, class_col)
    #         branchs = [i for i in Counter(dataset[n])]
    #         tree[n] = {}
    #         for j in branchs:
    #             br_data = dataset[dataset[n] == j]
    #             if entropy(*[i for i in Counter(br_data[class_col]).values()]) != 0:
    #                 tree[n][j] = {}
    #                 cls.recursion(br_data, tree[n][j], class_col)

    #             else:
    #                 r = Counter(br_data[class_col])
    #                 tree[n][j] = max(r, key=r.__getitem__)
    #         return
    @classmethod
    def pred_recur(cls, tupl, t):
        # if type(t) is int:
        #    return "NaN"
        # elif type(t) is not dict:
        if type(t) is not dict:
            return t
        index = {'buying': 1, 'maint': 2, 'doors': 3, 'persons': 4, 'lug_boot': 5, 'safety': 6}
        for i in t.keys():
            if i in index.keys():
                s = t[i].get(tupl[index[i]])
                r = cls.pred_recur(tupl, t[i].get(tupl[index[i]], 0))
        return r

    def fit(self, X, y):
        columns = self.columns
        dataset = X.assign(labels=y)
        self.tree_ = {}
        ID3.recursion(dataset, self.tree_, columns)
        return self

    def predict(self, test):
        ans = []
        for i in test.itertuples():
            ans.append(ID3.pred_recur(i, self.tree_))
        return pd.Series(ans)


if __name__ == '__main__':
    occur = 0
    avg_acc = 0.0
    accu = []
    std_dev = 0.0
    #sys.setrecursionlimit(2000)
    car_dataset = pd.read_csv("./data/car.data",
                              names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "decision"])
    car_dataset["buying"].replace(["vhigh", "high", "med", "low"], [3, 2, 1, 0], inplace=True)
    car_dataset["maint"].replace(["vhigh", "high", "med", "low"], [3, 2, 1, 0], inplace=True)
    car_dataset["doors"].replace(["5more"], [5], inplace=True)
    car_dataset["persons"].replace(["more"], [4], inplace=True)
    car_dataset["lug_boot"].replace(["small", "med", "big"], [0, 1, 2], inplace=True)
    car_dataset["safety"].replace(["low", "med", "high"], [0, 1, 2], inplace=True)
    car_dataset["decision"].replace(["unacc", "acc", "good", "vgood"], [1, 0, 2, 3], inplace=True)

    car_dataset['buying'] = car_dataset['buying'].astype(int)
    car_dataset['maint'] = car_dataset['maint'].astype(int)
    car_dataset['doors'] = car_dataset['doors'].astype(int)
    car_dataset['persons'] = car_dataset['persons'].astype(int)
    car_dataset['lug_boot'] = car_dataset['lug_boot'].astype(int)
    car_dataset['safety'] = car_dataset['safety'].astype(int)
    car_dataset['decision'] = car_dataset['decision'].astype(int)

    for i in range(10):
        # Shuffle the dataset
        car_dataset = car_dataset.sample(frac=1)
        model = ID3()
        X = car_dataset.drop(["decision"], axis=1)
        y = car_dataset.decision
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45)
        entr = entropy(*[i for i in Counter(y_train).values()])
        model.fit(X_train, y_train)
        accuracy_score(y_test, model.predict(X_test))
        a = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        for i in range(0, len(a)):
            accu.append(a[i])

    avg_acc = np.sum(accu) / len(accu)
    std = np.std(accu)
    print("Average Accuracy:", avg_acc)
    print("Standard Deviation: ", std)
