from random import randrange
from math import sqrt,exp, pi
import pandas as pd, numpy as np, sys


def fit(dataset, nb, decision, *args):
    data = cross_validation(dataset)
    acc_list = list()
    for index, unf_data in enumerate(data):
        X_train = np.array(data)
        X_train = np.delete(X_train, index, axis=0)
        X_train = np.concatenate((X_train))
        X_test = list()
        for record in unf_data:
            r = list(record)
            X_test.append(r)
            r[decision] = None
        pred = nb(X_train, X_test, decision, *args)
        y_test = [record[decision] for record in unf_data]
        accuracy = acc_per(y_test, pred)
        acc_list.append(accuracy)
    return acc_list

def naive_bayes_classifier(X_train, X_test, decision):
    sort_data = cls_std(X_train, decision)
    predictions = list()
    for record in X_test:
        res_prob = t_prob(
            sort_data, record)
        best_target_class, high_prob = None, -1
        for target_value, probability in res_prob.items():
            if best_target_class is None or probability > high_prob:
                high_prob = probability
                best_target_class = target_value
        predictions.append(best_target_class)
    return predictions


def cross_validation(dataset):
    num_of_folds = 5
    aftr_crss = list()
    temp_dataset = list(dataset)
    datasize = int(len(dataset) / num_of_folds)
    for f in range(num_of_folds):
        fnl_fld_data = list()
        while len(fnl_fld_data) < datasize:
            index = randrange(len(temp_dataset))
            fnl_fld_data.append(temp_dataset.pop(index))
        aftr_crss.append(fnl_fld_data)
    return aftr_crss


def cls_std(dataset, decision):
    dd_cls = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        target_value = vector[decision]
        if (target_value not in dd_cls):
            dd_cls[target_value] = list()
        dd_cls[target_value].append(vector)
    result = dict()
    for target_value, record in dd_cls.items():
        value = [(mean(column), stdev(column), len(column))
                           for column in zip(*record)]
        del(value[decision])
        result[target_value] = value
    return result


def t_prob(sort_data, row):
    t_rows = sum([sort_data[label][0][2] for label in sort_data])
    res_prob = dict()
    for target_value, sorted_data in sort_data.items():
        res_prob[target_value] = sort_data[target_value][0][2] / \
            float(t_rows)
        for i in range(len(sorted_data)):
            mean, stdev, _ = sorted_data[i]
            if stdev == 0.0:
                stdev = 1
            exponent = exp(-((row[i]-mean)**2 / (2 * stdev**2)))
            prob = (1 / (sqrt(2 * pi) * stdev)) * exponent
            res_prob[target_value] *= prob
    return res_prob

def acc_per(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)

if __name__ == "__main__":

    dataset = sys.argv[1]

    acc = []

    if dataset == "car":
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


        for i in range(10):
            decision = 6 
            accuracy = fit(car_dataset.values,
                           naive_bayes_classifier, decision)            
            acc.append(accuracy)
        

    elif dataset == "breast_cancer":

        cancer_dataset = pd.read_csv("./data/breast-cancer-wisconsin.data", names=["id","diagnosis","radius","texture","perimeter","area","smoothness","compactness","concavity","concave points","decision"])

        # Shuffle the dataset
        cancer_dataset = cancer_dataset.sample(frac=1)

        cancer_dataset = cancer_dataset.replace("?", np.NaN)
        cancer_dataset = cancer_dataset.dropna()

        cancer_dataset = cancer_dataset.drop("id", axis="columns")

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

        
        for i in range(10):
            decision = 9 
            accuracy = fit(cancer_dataset.values,
                           naive_bayes_classifier, decision)

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
            decision = 22  
            accuracy = fit(mushroom_dataset.values,
                           naive_bayes_classifier, decision)
            acc.append(accuracy)

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
            decision = 7 
            accuracy = fit(ecoli_dataset.values,naive_bayes_classifier, decision)
            acc.append(accuracy)


    elif dataset == 'letter':
        #Preprocessing dataset
        letter_dataset = pd.read_csv('./data/letter-recognition.data', names=["lettr", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"])
        ObjectColumns = letter_dataset.select_dtypes(include=np.object).columns.tolist()
        letter_dataset['lettr'] = [ord(item)-64 for item in letter_dataset['lettr']]

        for i in range(10):
            decision = 16  
            accuracy = fit(letter_dataset.values,
                           naive_bayes_classifier, decision)
            acc.append(accuracy)

    std = np.std(accuracy)
    print('Naive Bayes Classification accuracy:',sum(accuracy)/len(accuracy))
    print("Naive Bayes Classification standard deviation", std)