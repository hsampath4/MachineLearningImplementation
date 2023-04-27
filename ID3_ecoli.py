import pandas as pd, numpy as np
from collections import Counter
from math import log
from sklearn.base import BaseEstimator as estimator, ClassifierMixin as mix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import warnings


def entropy(class1=0, class2=0, class3=0, class4=0, class5=0, class6=0, class7=0, class8=0):
    class_list = [class1, class2, class3, class4, class5, class6, class7, class8]
    final_entropy = 0
    for c in class_list:
        if c != 0:
            final_entropy += -((c / sum(class_list)) * log(c / sum(class_list), 4))
    return final_entropy


class ID3(estimator, mix):

    def __init__(self, columns="labels"):
        self.columns = columns

    @staticmethod
    def score(spl, entro, total):
        entro_set = [entropy(*i) for i in spl]
        f = lambda x, y: (sum(x) / total) * y
        ans = [f(i, j) for i, j in zip(spl, entro_set)]
        return entro - sum(ans)

    #     def score(split_s, entro, total):
    #         entro_set = [entropy(*i) for i in split_s]
    #         f = lambda x, y: (sum(x) / total) * y
    #         result = [f(i, j) for i, j in zip(split_s, entro_set)]
    #         return entro - sum(result)

    @staticmethod
    def splits(header, dataset, columns):
        df = pd.DataFrame(dataset.groupby([header, columns])[columns].count())
        ans = []
        for i in Counter(dataset[header]).keys():
            ans.append(df.loc[i].values)

        return ans

    @classmethod
    def node(cls, dataset, columns):
        entro = entropy(*[i for i in Counter(dataset[columns]).values()])
        ans = {}
        for i in dataset.columns:
            if i != columns:
                spl = cls.splits(i, dataset, columns)
                scr = cls.score(spl, entro, total=len(dataset))
                ans[i] = scr
        return max(ans, key=ans.__getitem__)

    @classmethod
    def recursion(cls, dataset, tree, columns):
        num = cls.node(dataset, columns)
        brch = [i for i in Counter(dataset[num])]
        tree[num] = {}
        for j in brch:
            br_data = dataset[dataset[num] == j]
            if entropy(*[i for i in Counter(br_data[columns]).values()]) != 0:
                tree[num][j] = {}
                cls.recursion(br_data, tree[num][j], columns)
            else:
                r = Counter(br_data[columns])
                tree[num][j] = max(r, key=r.__getitem__)
        return

    @classmethod
    def pred_recur(cls, tupl, t):
        if type(t) is not dict:
            return t
        r = 0
        index = {'mcg': 1, 'gvh': 2, 'lip': 3, 'chg': 4, 'aac': 5, 'alm1': 6, 'alm2': 7}
        for i in t.keys():
            if i in index.keys():
                td = tupl[index[i]]
                s = t[i].get(tupl[index[i]], 0)
                r = cls.pred_recur(tupl, t[i].get(tupl[index[i]], 0))
        return r

    def predict(self, test):
        ans = []
        for i in test.itertuples():
            ans.append(ID3.pred_recur(i, self.tree_))
        return pd.Series(ans)

    def fit(self, X, y):
        columns = self.columns
        dataset = X.assign(labels=y)
        self.tree_ = {}
        ID3.recursion(dataset, self.tree_, columns)
        return self


if __name__ == '__main__':
    acc = []
    ecoli_dataset = pd.read_csv("./data/ecoli.data", names=["sequence names", "mcg", "gvh", "lip", "chg",
                                                            "aac", "alm1", "alm2", "decision"], delim_whitespace=True)
    ecoli_dataset["decision"].replace(["cp", "im", "imU", "imS", "imL", "om", "omL", "pp"], [0, 1, 2, 3, 4, 5, 6, 7],
                                      inplace=True)

    ecoli_dataset['mcg'] = ecoli_dataset['mcg'].astype(float)
    ecoli_dataset['gvh'] = ecoli_dataset['gvh'].astype(float)
    ecoli_dataset['lip'] = ecoli_dataset['lip'].astype(float)
    ecoli_dataset['chg'] = ecoli_dataset['chg'].astype(float)
    ecoli_dataset['aac'] = ecoli_dataset['aac'].astype(float)
    ecoli_dataset['alm1'] = ecoli_dataset['alm1'].astype(float)
    ecoli_dataset['alm2'] = ecoli_dataset['alm2'].astype(float)
    ecoli_dataset['decision'] = ecoli_dataset['decision'].astype(float)

    # Looping to perform k-fold 10 times
    for i in range(10):
        # Shuffling the dataset
        ecoli_dataset = ecoli_dataset.sample(frac=1)
        model = ID3()
        X = ecoli_dataset.drop(["decision"], axis=1)
        y = ecoli_dataset.decision
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=10)
        warnings.filterwarnings("ignore")
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
