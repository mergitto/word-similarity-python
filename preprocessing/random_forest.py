import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from calc import Calc

class SupervisedLearning():
    def __init__(self, pickle_data):
        self.df = pd.DataFrame.from_dict(pickle_data).T
        self.pickle_data = pickle_data
        self.X = pd.DataFrame()
        self.y = pd.DataFrame()
        self.class_names = []
        self.clf = ""
        self.pluck_list = []

    def drop_na(self, drop_na_list=[]):
        self.df = self.df.dropna(subset=drop_na_list)

    def set_X_and_y(self, objective_key=""):
        self.X = self.df.drop(objective_key, axis=1)
        self.y = self.df[objective_key].astype(int)
        self.pluck_list = list(self.X)

    def train_test_data_split(self, random_state=1, test_size=0.3):
        from sklearn.model_selection import train_test_split
        return train_test_split(self.X, self.y,random_state=random_state, test_size=test_size)

    def std_X(self, X_train, X_test, with_mean=True):
        # データの標準化処理
        sc = StandardScaler(with_mean=with_mean)
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        return X_train_std, X_test_std

    def add_predicted(self, clf=None, pickle_data=None, is_high_predicted_name="", predicted_high_rate_name=""):
        df = pd.DataFrame.from_dict(pickle_data).T
        X_list = df[self.pluck_list]
        X_std,X_dummy_std = self.std_X(X_list, X_list)
        predicted = clf.predict(X_std)
        df[is_high_predicted_name] = [int(i) for i in predicted]
        df[predicted_high_rate_name] = [high_rate[1] if high_rate[1] > 0.5 else 0 for high_rate in clf.predict_proba(X_std)]
        return df.T.to_dict()

    def exist_key(self, dictionary, key_name=""):
        return True if key_name in dictionary else False

    def save_model(self, save_model_name):
        import pickle
        with open(save_model_name, 'wb') as f:
            pickle.dump(self.clf, f)

    def load_model(self, load_model_name):
        import pickle
        with open(load_model_name, 'rb') as f:
            return pickle.load(f)

    def clf_importance(self, X, clf):
        # 各項目における重要度を辞書型で取得
        importance_dict = {}
        from pprint import pprint
        for index, importance in enumerate(clf.feature_importances_):
            importance_dict[X.columns[index]] = importance
        return importance_dict

    def precision_recall_curve(self, clf, y_test, X_test):
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_test, clf.predict_proba(X_test)[:, 1])
        plt.subplot(1, 2, 1)
        plt.step(recall, precision, color='black', where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])

    def auc_curve(self, clf, y_test, X_test):
        from sklearn.metrics import roc_curve
        from sklearn.metrics import auc
        fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
        print("auc :", auc(fpr, tpr))
        plt.subplot(1, 2, 2)
        plt.step(fpr, tpr, color='b', alpha=0.2, where='post')
        plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        #plt.savefig("Write save file name")

    def cross_validation(self, max_depth=2):
        from sklearn.model_selection import cross_val_score
        print("======= 交差検証 ======")
        clf = self.get_model(max_depth = max_depth, n_estimators=10)
        score = cross_val_score(estimator = clf, X = self.X, y = self.y, cv = 5)

        X_train,X_test,y_train,y_test = self.train_test_data_split(random_state=max_depth, test_size=0.3)
        X_train_std, X_test_std = self.std_X(X_train, X_test)
        clf.fit(X_train_std, y_train)
        print("[max_depth, score_mean, train_predict, test_predict]")
        print("", [max_depth, score.mean(), metrics.accuracy_score(y_train, clf.predict(X_train_std)), metrics.accuracy_score(y_test, clf.predict(X_test_std))])
        print("============ end =============")

    def get_model(self, clf_name="random_forest", max_depth=2, n_estimators=100):
        clf = RandomForestClassifier(
                bootstrap=True, class_weight=None, criterion='gini',
                max_depth=max_depth, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=n_estimators, n_jobs=1, oob_score=False, random_state=None,
                verbose=0, warm_start=False)
        return clf

    def drop_columns(self, drop_list):
        self.df =  self.df.drop(drop_list, axis=1)

    def add_dummy_score(self, high_report_rate=0.4):
        tmp_df = self.df.sort_values("score_std", ascending=False)
        df_size = len(self.df)
        high_rate = int(df_size * high_report_rate)
        threshold = tmp_df[:high_rate].iloc[-1].score_std
        print("正規化後の閾値: ", threshold)
        self.df.loc[self.df["score_std"] >= threshold, "score_dummy"] = 1 # High
        self.df.loc[self.df["score_std"] < threshold, "score_dummy"] = 0 # Low
        print("高評価の報告書数: {}, 低評価の報告書数: {}".format(len(self.df[self.df["score_dummy"] == 1]), len(self.df[self.df["score_dummy"] == 0])))
        self.class_names = ["low", "high"]

    def f1_value(self, true_score, predicted_score):
        from sklearn.metrics import classification_report
        y_true = true_score
        y_pred = predicted_score
        print("=============== f1-値 =============")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

    def neural(self, random_state=0):
        print("======= MLPClassifier ======")
        X_train, X_test, y_train, y_test = self.train_test_data_split(random_state=random_state, test_size=0.3)
        X_train_std, X_test_std = self.std_X(X_train, X_test)
        tuned_parameters = [{
            'hidden_layer_sizes': [(100,),(200,),(100,100),(200,200)],
            'learning_rate': ['invscaling', 'adaptive', 'constant'],
            'activation': ['logistic', 'identity', 'relu', 'tanh']
            },]
        from sklearn.neural_network import MLPClassifier
        gsearch = GridSearchCV(MLPClassifier(max_iter=1000), tuned_parameters, cv=5, scoring='accuracy', n_jobs=-1)
        gsearch.fit(X_train_std, y_train)
        print("ベストパラメータ:")
        print(gsearch.best_estimator_)
        print("各パラメータの平均スコア")
        for params, mean_score, all_scores in sorted(gsearch.grid_scores_, key=lambda k: k[1],reverse=True) :
            print("{:.3f} std:{:.3f} param: {}".format(mean_score, all_scores.std() , params))
        mlp = gsearch.best_estimator_
        mlp.fit(X_train_std, y_train)
        self.clf = mlp
        self.f1_value(y_test, self.clf.predict(X_test_std))
        self.precision_recall_curve(self.clf, y_test, X_test_std)
        self.auc_curve(self.clf, y_test, X_test_std)
        print("======= END ======")

    def svm(self):
        print("======= SVM ======")
        X_train, X_test, y_train, y_test = self.train_test_data_split(test_size=0.3)
        X_train_std, X_test_std = self.std_X(X_train, X_test)
        params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
        }
        from sklearn.svm import SVC
        gsearch = GridSearchCV(SVC(max_iter=-1, probability=True), params, cv=5, scoring='accuracy', n_jobs=-1)
        gsearch.fit(X_train_std, y_train)
        print("ベストパラメータ:")
        print(gsearch.best_estimator_)
        print("各パラメータの平均スコア")
        for params, mean_score, all_scores in sorted(gsearch.grid_scores_, key=lambda k: k[1],reverse=True) :
            print("{:.3f} std:{:.3f} param: {}".format(mean_score, all_scores.std() , params))
        clf = gsearch.best_estimator_
        clf.fit(X_train_std, y_train)
        self.clf = clf
        self.f1_value(y_test, self.clf.predict(X_test_std))
        self.precision_recall_curve(self.clf, y_test, X_test_std)
        self.auc_curve(self.clf, y_test, X_test_std)
        print("======= END ======")

    def random_forest(self, random_state=0, max_depth=2):
        print("======= RandomForestClassifier ======")
        X_train, X_test, y_train, y_test = self.train_test_data_split(test_size=0.3)
        X_train_std, X_test_std = self.std_X(X_train, X_test)
        params = {
            'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
            'n_estimators': [10, 100, 1000]
        }
        gsearch = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', n_jobs=-1)
        gsearch.fit(X_train_std, y_train)
        print("ベストパラメータ:")
        print(gsearch.best_estimator_)
        print("各パラメータの平均スコア")
        for params, mean_score, all_scores in sorted(gsearch.grid_scores_, key=lambda k: k[1],reverse=True) :
            print("{:.3f} std:{:.3f} param: {}".format(mean_score, all_scores.std() , params))
        clf = gsearch.best_estimator_
        print("説明変数の重要度: ",self.clf_importance(self.X, clf))
        self.clf = clf
        print('テストデータ：Confusion matrix:\n{}'.format(confusion_matrix(y_test, clf.predict(X_test_std))))
        self.f1_value(y_test, self.clf.predict(X_test_std))
        score = {
                'train': metrics.accuracy_score(y_train, clf.predict(X_train_std)) ,
                'test': metrics.accuracy_score(y_test, clf.predict(X_test_std))
            }
        print(score)
        self.cross_validation(max_depth=4)
        self.precision_recall_curve(self.clf, y_test, X_test_std)
        self.auc_curve(self.clf, y_test, X_test_std)
        print("============ end =============")

