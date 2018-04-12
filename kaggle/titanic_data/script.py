# -*- encoding=utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

data_train = pd.read_csv('train.csv', delimiter=',')
data_test = pd.read_csv("test.csv", delimiter=',')
# data_train.info()
# data_test.info()
# assert 1 == False
"""
plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')# 柱状图
plt.title("获救情况 (1为获救)") # 标题
plt.ylabel("人数")

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel("人数")
plt.title("乘客等级分布")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel("年龄")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title("按年龄看获救分布 (1为获救)")


plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("年龄")# plots an axis lable
plt.ylabel("密度")
plt.title("各等级的乘客年龄分布")
plt.legend(('头等舱', '2等舱','3等舱'),loc='best') # sets our legend for our graph.

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("各登船口岸上船人数")
plt.ylabel("人数")
plt.show()
"""
"""
fig = plt.figure()
fig.set(alpha=0.2)
Survived_0 = data_train.Parch[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Parch[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'获救': Survived_1, '未获救': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title('有父母的获救情况')
plt.xlabel('')
plt.ylabel('人数')
plt.show()
"""
"""
# 各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'获救': Survived_1, '未获救': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title("各乘客等级的获救情况")
plt.xlabel("乘客等级")
plt.ylabel("人数")
plt.show()
"""
"""
# 各性别的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({'男性': Survived_m, '女性': Survived_f})
df.plot(kind='bar', stacked=True)
plt.title("按性别看获救情况")
plt.xlabel("性别")
plt.ylabel("人数")
"""
"""
# 登陆港口和获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'获救':Survived_1, '未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title("各登录港口乘客的获救情况")
plt.xlabel("登录港口")
plt.ylabel("人数")

plt.show()
"""

"""
# 甲板和获救情况
# print(set([i[:1] for i in data_train.Cabin if type(i) is str]))

cabin_A = pd.Series([i for i in data_train.Cabin if type(i) is str and i[:1] == 'A'])
cabin_B = pd.Series([i for i in data_train.Cabin if type(i) is str and i[:1] == 'B'])
cabin_C = pd.Series([i for i in data_train.Cabin if type(i) is str and i[:1] == 'C'])
cabin_D = pd.Series([i for i in data_train.Cabin if type(i) is str and i[:1] == 'D'])
cabin_E = pd.Series([i for i in data_train.Cabin if type(i) is str and i[:1] == 'E'])
cabin_F = pd.Series([i for i in data_train.Cabin if type(i) is str and i[:1] == 'F'])
cabin_G = pd.Series([i for i in data_train.Cabin if type(i) is str and i[:1] == 'G'])
cabin_T = pd.Series([i for i in data_train.Cabin if type(i) is str and i[:1] == 'T'])

# print(data_train.Cabin.isnull())
fig = plt.figure()
fig.set(alpha=0.2)
survived_A = data_train.Survived[data_train.Cabin.isin(cabin_A)].value_counts()
survived_B = data_train.Survived[data_train.Cabin.isin(cabin_B)].value_counts()
survived_C = data_train.Survived[data_train.Cabin.isin(cabin_C)].value_counts()
survived_D = data_train.Survived[data_train.Cabin.isin(cabin_D)].value_counts()
survived_E = data_train.Survived[data_train.Cabin.isin(cabin_E)].value_counts()
survived_F = data_train.Survived[data_train.Cabin.isin(cabin_F)].value_counts()
survived_G = data_train.Survived[data_train.Cabin.isin(cabin_G)].value_counts()
survived_T = data_train.Survived[data_train.Cabin.isin(cabin_T)].value_counts()
survived_nocabin = data_train.Survived[data_train.Cabin.isnull()].value_counts()
df = pd.DataFrame({'A':survived_A
                  ,'B':survived_B,'C':survived_C,'D':survived_D,'E':survived_E,'F':survived_F,'G':survived_G,'T':survived_T,'Nocabin': survived_nocabin
                   })
print(df)
df.plot(kind='bar', stacked=True)
plt.xlabel('获救情况')
plt.ylabel('人数')
plt.show()
"""
"""
查看title获救情况
import re
names = list(data_train.Name.values)
titles = []
for name in names:
    # print(name)
    title = re.search(r'M[a-z]+\.|Don\.|Rev\.|Dr\.|Lady\.|Sir\.|Col\.|Capt\.|Countess\.|Jonkheer\.', name).group()
    titles.append(title)
print(titles)
data_train['Title'] = pd.Series(titles)
data_train["Title"]=data_train["Title"].replace(['Mr.','Don.'],'Mr')
data_train["Title"]=data_train["Title"].replace(['Mrs.','Miss.','Mme.','Ms.','Lady.','Dona.','Mlle.'],'Ms')
data_train["Title"]=data_train["Title"].replace(['Sir.','Major.','Col.','Capt.'],'Major.')
data_train["Title"]=data_train["Title"].replace(['Master.','Jonkheer.','Countess.'],'Jonkheer.')
data_train["Title"]=data_train["Title"].replace('Rev.','Rev')
data_train["Title"]=data_train["Title"].replace('Dr.','Dr')

# 各title的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Title[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Title[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'获救': Survived_1, '未获救': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title("按title看获救情况,")
plt.xlabel("获救")
plt.ylabel("人数")
plt.show()
"""
# 根据姓名，新建title字段
import re
def set_title(df):
    names = list(data_train.Name.values)
    titles = []
    for name in names:
        # print(name)
        title = re.search(r'M[a-z]+\.|Don\.|Rev\.|Dr\.|Lady\.|Sir\.|Col\.|Capt\.|Countess\.|Jonkheer\.', name).group()
        titles.append(title)
    # print(titles)
    df['Title'] = pd.Series(titles)
    df["Title"]=df["Title"].replace(['Mr.','Don.'],'Mr')
    df["Title"]=df["Title"].replace(['Mrs.','Miss.','Mme.','Ms.','Lady.','Dona.','Mlle.'],'Ms')
    df["Title"]=df["Title"].replace(['Sir.','Major.','Col.','Capt.'],'Major')
    df["Title"]=df["Title"].replace(['Master.','Jonkheer.','Countess.'],'Jonkheer')
    df["Title"]=df["Title"].replace('Rev.','Rev')
    df["Title"]=df["Title"].replace('Dr.','Dr')
    df['Title'] = df['Title'].map({'Mr': 0, 'Ms': 1, 'Major': 2, 'Jonkheer': 2, 'Rev': 3, 'Dr': 4}).astype(int)
    return df

# 拟合缺失的age
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
def set_miss_age(df, model=None):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass', 'Title']].astype(float)
    know_age = age_df[age_df.Age.notnull()].as_matrix()
    unknow_age = age_df[age_df.Age.isnull()].as_matrix()

    y = know_age[:, 0]
    X = know_age[:, 1:]

    if model is None:
        model = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
        model.fit(X, y)
    y_pred = model.predict(unknow_age[:, 1:])
    df.loc[(df.Age.isnull()), 'Age'] = y_pred
    return df, model

# 设置cabin
def set_cabin(df):
    df.Cabin = df.Cabin.replace(['A.*', 'C.*', 'F.*', 'G.*'], 'A', regex=True)
    df.Cabin = df.Cabin.replace(['B.*', 'D.*', 'E.*', 'T.*'], 'B', regex=True)
    df.Cabin = df.Cabin.fillna('C')
    df.Cabin = df.Cabin.map({'A': 2, 'B': 1, 'C': 0}).astype(int)
    return df

from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
# 预处理
def preprocessing_data(df):
    df.loc[ (df.Fare.isnull()), 'Fare' ] = 0
    # 标准化和归一化
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    # scaler = Normalizer()
    s1 = scaler.fit(df.Age.values.reshape((-1,1)))
    s2 = scaler.fit(df.Fare.values.reshape((-1,1)))
    df['Age_Scaled'] = scaler.fit_transform(df.Age.values.reshape((-1,1)), s1)
    df['Fare_Scaled'] = scaler.fit_transform(df.Fare.values.reshape((-1,1)), s2)
    # print(df.Age_Scaled)
    # print(df.Fare_Scaled)

    # onehot
    dummies_Cabin = pd.get_dummies(df.Cabin, prefix='Cabin')
    dummies_Embarked = pd.get_dummies(df.Embarked, prefix='Embarked')
    dummies_Sex = pd.get_dummies(df.Sex, prefix='Sex')
    dummies_Pclass = pd.get_dummies(df.Pclass, prefix='Pclass')
    dummies_Title = pd.get_dummies(df.Title, prefix='Title')
    df = pd.concat([df, dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_Title, dummies_Cabin], axis=1)
    df.drop(['Embarked', 'Pclass', 'Sex', 'Ticket', 'Name', 'Cabin'], axis=1, inplace=True)
    return df

# learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.1, 1., 5), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

data_train = set_title(data_train)
data_test = set_title(data_test)
data_train = set_cabin(data_train)
data_test = set_cabin(data_test)
data_train, age_model = set_miss_age(data_train)
data_test, age_model = set_miss_age(data_test, age_model)
data_train = preprocessing_data(data_train)
data_test = preprocessing_data(data_test)

# 训练模型
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier

if __name__ == '__main__':

    # 用正则取出我们要的属性值
    train_df = data_train.filter(regex='Survived|Age_.*|Carbin_.*|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')

    test = data_test.filter(regex='Age_.*|Carbin_.*|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
    train_np = train_df.as_matrix()
    test = test.as_matrix()

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # cross validation
    train_data, test_data = train_test_split(train_np, test_size=0.3, random_state=0)

    rfc = RandomForestClassifier(
                            n_estimators=20,
                            max_features=9,
                            max_depth=16,
                            min_samples_split=7,
                            min_samples_leaf=1,
                            criterion='entropy',
                            bootstrap=True,
                            oob_score=False,
                            n_jobs=1,
                            verbose=0)
    rfc.fit(train_data[:, 1:], train_data[:, 0])
    y_pred = rfc.predict(test_data[:, 1:])
    print(metrics.roc_auc_score(test_data[:, 0], y_pred))
    print(cross_val_score(rfc, test_data[:, 1:], test_data[:, 0], cv=10).mean())
    print(metrics.confusion_matrix(test_data[:, 0], y_pred))
    # param_grid = {
    #             # 'n_estimators':[i for i in range(10, 301, 10)],
    #             # 'max_depth': [i for i in range(1,20)],
    #             # 'min_samples_split': [i for i in range(2, 18)],
    #             # 'min_samples_leaf': [i for i in range(1, 3)],
    #             'max_features': [i for i in range(1, 13)],
    # }
    # gsearch = GridSearchCV(rfc, param_grid=param_grid, scoring='roc_auc', cv=5)
    # gsearch.fit(train_data[:, 1:], train_data[:, 0])
    # print(gsearch.cv_results_)
    # print(gsearch.best_params_)
    # print(gsearch.best_score_)

    # param_grid = {
    #         'C':[0.1, 1, 10],
    #         'gamma': [1, 0.1, 0.01],
    # }
    # svc = SVC(C=10, gamma=0.01)
    # gsearch = GridSearchCV(svc, param_grid=param_grid, scoring='roc_auc', cv=5)
    # gsearch.fit(train_data[:, 1:], train_data[:, 0])
    # print(gsearch.best_params_)
    # print(gsearch.best_score_)

    # param_grid = {
    #             # 'n_estimators':[i for i in range(10, 301, 10)],
    #             # 'max_depth': [i for i in range(1,20)],
    #             # 'min_samples_split': [i for i in range(2, 18)],
    #             # 'min_samples_leaf': [i for i in range(1, 10)],
    #             # 'max_features': [i for i igbdt_modelange(1, 13)],
    #             'subsample': [0.6, 0.7, 0.8, 0.9],
    # }
    # gbdt_model = GradientBoostingClassifier(
    #                 learning_rate=0.1,
    #                 max_features=9,
    #                 max_depth=3,
    #                 min_samples_split=14,
    #                 min_samples_leaf=1,
    #                 subsample=0.8,
    #                 n_estimators=90,
    #                 random_state=0)
    # gbdt_model.fit(train_np[:, 1:], train_np[:, 0])
    # y_pred = gbdt_model.predict(test_data[:, 1:])
    # print(metrics.roc_auc_score(test_data[:, 0], y_pred))
    # print(cross_val_score(gbdt_model, test_data[:, 1:], test_data[:, 0], cv=10).mean())
    # print(metrics.confusion_matrix(test_data[:, 0], y_pred))

    # # gsearch = GridSearchCV(gbdt_model, param_grid=param_grid, scoring='roc_auc', cv=5)
    # # gsearch.fit(train_data[:, 1:], train_data[:, 0])
    # # print(gsearch.best_params_)
    # # print(gsearch.best_score_)
    # param_grid={
    #         'learning_rate': [0.1, 0.05, 0.01],
    #         'tree_depth': [i for i in range(2, 9)],
    #         'subsample': [0.6, 0.7, 0.8, 0.9],
    # }
    # xgbc = XGBClassifier(learning_rate=0.1, tree_depth=2, subsample=0.6)
    # gsearch = GridSearchCV(xgbc, param_grid=param_grid, scoring='roc_auc', cv=5)
    # gsearch.fit(train_data[:, 1:], train_data[:, 0])
    # print(gsearch.best_params_)

    # learning curve
    plot_learning_curve(rfc, "学习曲线", train_np[:, 1:], train_np[:, 0], cv=10)

    # # fit到RandomForest之中
    # rfc = RandomForestClassifier(
    #                         n_estimators=20,
    #                         max_features=9,
    #                         max_depth=16,
    #                         min_samples_split=7,
    #                         min_samples_leaf=1,
    #                         criterion='entropy',
    #                         bootstrap=True,
    #                         oob_score=False,
    #                         n_jobs=1,
    #                         random_state=0,
    #                         verbose=0)
    # rfc.fit(X, y)
    # predictions = rfc.predict(test)

    """
    gbdt_model = GradientBoostingClassifier(
                    n_estimators=1000,
                    random_state=0)
    gbdt_model.fit(X, y)
    predictions = gbdt_model.predict(test)
    """
    """
    # SVC
    svc = SVC(C=10, gamma=0.01)
    svc.fit(X, y)
    predictions = svc.predict(test)
    """
    """
    gbdt_model = GradientBoostingClassifier(
                    learning_rate=0.1,
                    max_features=9,
                    max_depth=3,
                    min_samples_split=14,
                    min_samples_leaf=1,
                    subsample=0.8,
                    n_estimators=90,
                    random_state=0)

    gbdt_model.fit(X, y)
    predictions = gbdt_model.predict(test)
    """
    """
    lr = LogisticRegression()
    lr.fit(X, y)
    predictions =lr.predict(test)
    """
    # xgbc.fit(X, y)
    # predictions = xgbc.predict(test)
    # save result
    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
    result.to_csv("XGBC_predictions.csv", index=False)

