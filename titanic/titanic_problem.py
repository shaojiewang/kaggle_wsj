import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model

data_train = pd.read_csv("./input/Train.csv")
data_train.info()

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

plt.subplot2grid((2, 3), (0, 0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title("获救情况（1为获救）")

plt.subplot2grid((2, 3), (0 ,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title(u"乘客等级分布")
plt.ylabel(u"人数")

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄")
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布")

plt.subplot2grid((2, 3), (1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u"头等舱", u"2等舱", u"3等舱"), loc='best')

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数") 

#plt.show()

fig = plt.figure()
fig.set(alpha=0.2)
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u'各乘客等级的获救情况')
plt.xlabel(u'乘客登机')
plt.ylabel(u'人数')

#lt.show()

fig = plt.figure()
fig.set(alpha=0.2)
survived_m = data_train.Survived[data_train.Sex == "male"]\
    .value_counts()
survived_f = data_train.Survived[data_train.Sex == "female"]\
    .value_counts()
df = pd.DataFrame({u'男性': survived_m, u'女性': survived_f})
df.plot(kind="bar", stacked=True)
plt.title(u'按性别看获救情况')
plt.xlabel(u'性别')
plt.ylabel(u'人数')

#plt.show()

#然后我们再来看看各种舱级别情况下各性别的获救情况
fig=plt.figure()
fig.set(alpha=0.65) # 设置图像透明度，无所谓
plt.title(u"根据舱等级和性别的获救情况")

ax1=fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
ax1.legend([u"女性/高级舱"], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/高级舱"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/低级舱"], loc='best')

#plt.show()

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各登录港口乘客的获救情况")
plt.xlabel(u"登录港口") 
plt.ylabel(u"人数") 

#plt.show()

g = data_train.groupby(['SibSp', 'Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
#print(df)

g = data_train.groupby(['Parch', 'Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
#print(df)

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"按Cabin有无看获救情况")
plt.xlabel(u"Cabin有无") 
plt.ylabel(u"人数")
#plt.show()

def set_missing_ages(df):
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    y = known_age[:, 0]
    X = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000,\
        n_jobs=-1)
    rfr.fit(X, y)
    predictedAges = rfr.predict(unknown_age[:, 1::])
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr

def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'no'
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_cabin_type(data_train)
#print(data_train)

#把yes no等文字属性转换为one hot属性
dummies_cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
dummies_embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')

df = pd.concat([data_train, dummies_cabin, dummies_embarked, dummies_pclass, dummies_sex], axis=1)
df.drop(['Name', 'Cabin', 'Sex', 'Pclass', 'Ticket', 'Embarked'], axis=1, inplace=True)
#print(df)

scaler = preprocessing.StandardScaler()
age_scaled_param = scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scaled_param)

age_scaled_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), age_scaled_param)
#print(df)

#取出有用数据
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Pclass_.*|Sex_.*')
train_np = train_df.values

y = train_np[:,0]
x = train_np[:,1:]

clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
clf.fit(x, y)

#处理测试数据
data_test = pd.read_csv("./input/test.csv")
data_test.info()

data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0

tmp_df = data_test[['Age', 'Parch', 'Fare', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values
x = null_age[:, 1:]
predicted_age = rfr.predict(x)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predicted_age
data_test = set_cabin_type(data_test)
dummies_cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
dummies_embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
df_test = pd.concat([data_test, dummies_cabin, dummies_embarked, dummies_pclass, dummies_sex], axis=1)
df_test.drop(['Name', 'Cabin', 'Sex', 'Pclass', 'Ticket', 'Embarked'], axis=1, inplace=True)

age_scaled_param = scaler.fit(df_test['Age'].values.reshape(-1, 1))
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1, 1), age_scaled_param)

fare_scaled_param = scaler.fit(df_test['Fare'].values.reshape(-1, 1))
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1, 1), fare_scaled_param)
#print(df_test)

test_df = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Pclass_.*|Sex_.*')
test_np = test_df.values
predictions = clf.predict(test_np)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].values, 'Survived': predictions.astype(np.int32)})

result.to_csv('./input/logistic_regression_predictions.csv', index=False)

data_result = pd.read_csv('./input/logistic_regression_predictions.csv')
#print(data_result)

model_coef = pd.DataFrame({'columns':list(train_df.columns)[1:], 'coef':list(clf.coef_.T)})
print(model_coef)

# to run a cross validation
from sklearn.model_selection import cross_val_score, train_test_split

clf = linear_model.LogisticRegression(solver='liblinear', C=1.0, penalty='l2', tol=1e-6)
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X = all_data.values[:,1:]
y = all_data.values[:,0]
print(cross_val_score(clf, X, y, cv=5))

# see some of the bad cases
split_train, split_cv = train_test_split(df, test_size=0.3, random_state=42)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
clf = linear_model.LogisticRegression(solver='liblinear', C=1.0, penalty='l2', tol=1e-6)
clf.fit(train_df.values[:, 1:], train_df.values[:, 0])
cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(cv_df.values[:, 1:])

original_data_train = pd.read_csv('./input/train.csv')
bad_cases = original_data_train.loc[original_data_train['PassengerId'].isin(split_cv[predictions != cv_df.values[:, 0]]['PassengerId'].values)]
print(bad_cases.head(10))

# check learning curve
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=False):
    """
    plot learning curve of an estimator

    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # to display chinese
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u'训练样本数')
        plt.ylabel(u'得分')
        plt.gca().invert_yaxis()
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='b')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='b')

        plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label=u'训练集得分')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label=u'交叉验证集上得分')
        plt.legend(loc='best')

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(clf, u"学习曲线", X, y)

# ensemble model method
from sklearn.ensemble import BaggingRegressor

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到BaggingRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("./input/logistic_regression_bagging_predictions.csv", index=False)