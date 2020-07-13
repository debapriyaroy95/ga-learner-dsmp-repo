# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df = pd.read_csv(path)
df.head(5)
 
X = df.copy()
X.drop(columns=['insuranceclaim'])

y = df['insuranceclaim']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 6)

# Code ends here


# --------------
import matplotlib.pyplot as plt


# Code starts here

X_train.boxplot(column=['bmi'])
plt.show()

q_value = X_train['bmi'].quantile(.95)

#q_value = 0.95

print(q_value)

print(y_train.value_counts(normalize=True))

# Code ends here


# --------------
# Code starts here

print(X_train.columns)


relation = pd.DataFrame(X_train.corr())
print(relation)

sns.pairplot(X_train)


# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# code starts here

# store categorical variable
cols = ['children','sex','region','smoker']
        


# create subplot 

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,20))

# create loop for plotting countplot

for i in range(0,2):
    for j in range(0,2):
             col=cols[i*2 + j]
             sns.countplot(x=X_train[col], hue=y_train, ax=axes[i,j])

# Code ends here




# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

lr = LogisticRegression(random_state=9)
grid= GridSearchCV(estimator=lr,param_grid=parameters)
grid.fit(X_train,y_train)
y_true, y_pred = y_test, grid.predict(X_test)
accuracy = accuracy_score(y_true, y_pred)

# Code starts here



# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Code starts here
score = roc_auc_score( y_test, y_pred)
print(score)
y_pred_proba = grid.predict_proba(X_test)[:,1]
print(y_pred_proba[0])
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(roc_auc)
plt.plot(fpr,tpr,label="Logistic model, auc="+str(roc_auc))
plt.legend(loc=4)
plt.show()
# Code ends here
print(round(score,2))
print(round(y_pred_proba[0],2))
print(round(roc_auc,2))


