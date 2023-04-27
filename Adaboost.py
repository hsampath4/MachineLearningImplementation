import numpy as np, pandas as pd
import sys

class dec_stump:
    def __init__(self):
        self.polarity = 1
        self.f_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        s1 = X.shape[0]
        x_col = X[:, self.f_idx]
        predictions = np.ones(s1)
        if self.polarity == 1:
            predictions[x_col < self.threshold] = -1
        else:
            predictions[x_col > self.threshold] = -1

        return predictions


class Adaboost:
    def __init__(self, k):
        self.k = k
        self.clfs = []

    def fit(self, X, y):
        s1, n_features = X.shape

        w = np.full(s1, (1 / s1))

        self.clfs = []

        for x in range(self.k):
            clf = dec_stump()
            min_error = float("inf")

            for feature_i in range(n_features):
                x_col = X[:, feature_i]
                thresholds = np.unique(x_col)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(s1)
                    predictions[x_col < threshold] = -1

                    misc = w[y != predictions]
                    e1 = sum(misc)

                    if e1 > 0.5:
                        e1 = 1 - e1
                        p = -1

                    if e1 < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.f_idx = feature_i
                        min_error = e1

            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)
            self.clfs.append(clf)

    def predict(self, X):
        clf = [clf.alpha * clf.predict(X) for clf in self.clfs]
        ypred = np.sum(clf, axis=0)
        ypred = np.sign(ypred)

        return ypred


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold

    dataset = sys.argv[1]
    k = 2
    acc =[]
    def accuracy(y_true, ypred):
        accuracy = np.sum(y_true == ypred) / len(y_true)
        return accuracy

    if dataset == 'breast_cancer':
        #Preprocessing dataset
        cancer_dataset = pd.read_csv("./data/breast-cancer-wisconsin.data", names=["id","diagnosis","radius","texture","perimeter","area","smoothness","compactness","concavity","concave points","decision"])
        cancer_dataset["smoothness"].replace(["?"], ['0'], inplace = True)
        cancer_dataset['smoothness'] = cancer_dataset['smoothness'].astype(int)
        average = cancer_dataset["smoothness"].mean()
        cancer_dataset["smoothness"].replace(["?"], [average], inplace = True)

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
        
        #Repeating 10 times
        for i in range(10):
            #Shuffle the dataset
            cancer_dataset = cancer_dataset.sample(frac=1)
            model = Adaboost(k)
            X, y = cancer_dataset.iloc[:, 0:6].to_numpy(), cancer_dataset.iloc[:,6].to_numpy()
            y[y == 0] = -1

            #5-fold cross validation
            kf = KFold(n_splits=5)
            for train_index , test_index in kf.split(X):
                X_train , X_test = X[train_index,:],X[test_index,:]
                y_train , y_test = y[train_index] , y[test_index]
        
                model.fit(X_train,y_train)
                pred_values = model.predict(X_test)
        
                acc.append(accuracy(pred_values, y_test))
        print('Adaboost classification accuracy %.3f%%' % ((sum(acc)/float(len(acc)))*100))
    
           
    elif dataset == 'car':
        #Preprocessing dataset
        car_dataset = pd.read_csv("./data/car.data", names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "decision"])
        car_dataset["buying"].replace(["vhigh", "high", "med", "low"], [3, 2, 1, 0], inplace = True)
        car_dataset["maint"].replace(["vhigh", "high", "med", "low"], [3, 2, 1, 0], inplace = True)
        car_dataset["doors"].replace(["5more"], [5], inplace = True)
        car_dataset["persons"].replace(["more"], [4], inplace = True)
        car_dataset["lug_boot"].replace(["small", "med", "big"], [0, 1, 2], inplace = True)
        car_dataset["safety"].replace(["low", "med", "high"], [0, 1, 2], inplace = True)
        car_dataset["decision"].replace(["unacc", "acc", "good", "vgood"], [1, 0, 2, 3], inplace = True)
            
        car_dataset['buying'] = car_dataset['buying'].astype(int)
        car_dataset['maint'] = car_dataset['maint'].astype(int)
        car_dataset['doors'] = car_dataset['doors'].astype(int)
        car_dataset['persons'] = car_dataset['persons'].astype(int)
        car_dataset['lug_boot'] = car_dataset['lug_boot'].astype(int)
        car_dataset['safety'] = car_dataset['safety'].astype(int)
        car_dataset['decision'] = car_dataset['decision'].astype(int)

        #Repeating 10 times
        for i in range(10):
            #Shuffle the dataset
            car_dataset = car_dataset.sample(frac=1)
            
            model = Adaboost(k)
            X, y = car_dataset.iloc[:, 0:6].to_numpy(), car_dataset.iloc[:,6].to_numpy()
            y[y == 0] = -1

            #5-fold cross validation
            kf = KFold(n_splits=5)
            for train_index , test_index in kf.split(X):
                X_train , X_test = X[train_index,:],X[test_index,:]
                y_train , y_test = y[train_index] , y[test_index]
        
                model.fit(X_train,y_train)
                pred_values = model.predict(X_test)
        
                acc.append(accuracy(pred_values, y_test))
        print('Adaboost classification accuracy %.3f%%' % ((sum(acc)/float(len(acc)))*100))
    
    elif dataset == 'ecoli':
        #Preprocessing dataset
        ecoli_dataset = pd.read_csv("./data/ecoli.data", names=["sequence names", "mcg", "gvh", "lip", "chg",
                                "aac", "alm1", "alm2", "decision"], delim_whitespace=True)    
        ecoli_dataset["decision"].replace(["cp","im","imU","imS","imL","om","omL","pp"], [0,1,2,3,4,5,6,7], inplace = True)

        ecoli_dataset['mcg'] = ecoli_dataset['mcg'].astype(float)
        ecoli_dataset['gvh'] = ecoli_dataset['gvh'].astype(float)
        ecoli_dataset['lip'] = ecoli_dataset['lip'].astype(float)
        ecoli_dataset['chg'] = ecoli_dataset['chg'].astype(float)
        ecoli_dataset['aac'] = ecoli_dataset['aac'].astype(float)
        ecoli_dataset['alm1'] = ecoli_dataset['alm1'].astype(float)
        ecoli_dataset['alm2'] = ecoli_dataset['alm2'].astype(float)
        ecoli_dataset['decision'] = ecoli_dataset['decision'].astype(int)
        
        for i in range(10):
            #Shuffle the dataset
            ecoli_dataset = ecoli_dataset.sample(frac=1)
            model = Adaboost(k)
            X, y = ecoli_dataset.iloc[:, 0:6].to_numpy(), ecoli_dataset.iloc[:,6].to_numpy()
            y[y == 0] = -1

            #5-fold cross validation
            kf = KFold(n_splits=5)
            for train_index , test_index in kf.split(X):
                X_train , X_test = X[train_index,:],X[test_index,:]
                y_train , y_test = y[train_index] , y[test_index]
        
                model.fit(X_train,y_train)
                pred_values = model.predict(X_test)
        
                acc.append(accuracy(pred_values, y_test))
        print('Adaboost classification accuracy %.3f%%' % ((sum(acc)/float(len(acc)))*100))


    elif dataset == 'letter':
        #Preprocessing dataset
        letter_dataset = pd.read_csv('./data/letter-recognition.data', names=["lettr", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"])
        ObjectColumns = letter_dataset.select_dtypes(include=np.object).columns.tolist()
        letter_dataset['lettr'] = [ord(item)-64 for item in letter_dataset['lettr']]

        for i in range(10):
            #Shuffle the dataset
            letter_dataset = letter_dataset.sample(frac=1)
            
            model = Adaboost(k)
            X, y = letter_dataset.iloc[:, 0:6].to_numpy(), letter_dataset.iloc[:,6].to_numpy()
            y[y == 0] = -1

            #5-fold cross validation
            kf = KFold(n_splits=5)
            for train_index , test_index in kf.split(X):
                X_train , X_test = X[train_index,:],X[test_index,:]
                y_train , y_test = y[train_index] , y[test_index]
        
                model.fit(X_train,y_train)
                pred_values = model.predict(X_test)
        
                acc.append(accuracy(pred_values, y_test))
        print('Adaboost classification accuracy %.3f%%' % ((sum(acc)/float(len(acc)))*100))

    elif dataset == 'mushroom':
        #Preprocessing dataset
        mushroom_dataset = pd.read_csv("./data/mushroom.data", names=["decision", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",  "gill-attachment", 
    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", 
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", 
    "population", "habitat"])

        mushroom_dataset.replace(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],inplace = True)
        
        mushroom_dataset["stalk-root"].replace(["?"], ['0'], inplace = True)
        mushroom_dataset["stalk-root"] = mushroom_dataset['stalk-root'].astype(int)
        average = mushroom_dataset["stalk-root"].mean()
        mushroom_dataset["stalk-root"].replace(["?"], [average], inplace = True)

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

        for i in range(10):
            #Shuffle the dataset
            mushroom_dataset = mushroom_dataset.sample(frac=1)
            
            model = Adaboost(k)
            X, y = mushroom_dataset.iloc[:, 0:6].to_numpy(), mushroom_dataset.iloc[:,6].to_numpy()
            y[y == 0] = -1

            #5-fold cross validation
            kf = KFold(n_splits=5)
            for train_index , test_index in kf.split(X):
                X_train , X_test = X[train_index,:],X[test_index,:]
                y_train , y_test = y[train_index] , y[test_index]
        
                model.fit(X_train,y_train)
                pred_values = model.predict(X_test)
                acc.append(accuracy(pred_values, y_test))
        print('Adaboost classification accuracy %.3f%%' % ((sum(acc)/float(len(acc)))*100))
    std = np.std(acc)

    print("Adaboost classification standard deviation", std)
