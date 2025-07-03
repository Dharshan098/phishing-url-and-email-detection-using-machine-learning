#%%
#pip install tldextract
#%%
#!pip install pandas
#%%
#pip install xgboost
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.stats import randint, uniform
import random
from sklearn.model_selection import KFold, cross_val_score
import re
import tldextract
from urllib.parse import urlparse
from collections import Counter
from scipy.stats import entropy
warnings.filterwarnings("ignore")
#%%
X = pd.read_csv("C:/Users/Dharshan/OneDrive/Desktop/url folder/phishing_site_urls.csv")
X['URL'].str.strip()
X
#%%
special_df = pd.DataFrame()

special_chars = set() 

def find_special_char(x):
    special_chars_in_x= re.findall(r'[^a-zA-Z0-9]', x)
    special_chars.update(special_chars_in_x)
    return None

X_bad = X[X['Label'] == 'bad']
X_bad['URL'].apply(find_special_char)

special_chars = list(special_chars)

special_chars.remove('.')
special_chars.remove('/')

special_df['Special Character'] = special_chars
special_df['Frequency in bad URLs'] = special_df['Special Character'].apply(lambda x: X_bad[X_bad['URL'].str.contains(re.escape(x), regex=True)].shape[0])
special_df['Bad probability'] = special_df['Frequency in bad URLs']/special_df['Special Character'].apply(lambda x: X[X['URL'].str.contains(re.escape(x), regex=True)].shape[0])
special_df['Score'] = special_df['Bad probability']*special_df['Frequency in bad URLs'].apply(math.log)

special_df.sort_values(by='Score', ignore_index=True, ascending=False, inplace=True)
special_df
#%%
dangerous_chars = list(special_df['Special Character'].head(10))
print(dangerous_chars)
plt.bar(special_df['Special Character'].head(10), special_df['Score'].head(10), color = 'green')
plt.xlabel('Special Character')
plt.ylabel('Score')
plt.show()
#%%
TLD_df = pd.DataFrame()

TLD_list = pd.Series(X_bad['URL'].apply(lambda x: tldextract.extract(x).suffix)).unique()

TLD_df['TLD'] = TLD_list

TLD_df['Frequency in bad URLs'] = TLD_df['TLD'].apply(lambda x: X_bad[X_bad['URL'].str.contains(re.escape(x), regex=True)].shape[0])
TLD_df['Bad probability'] = TLD_df['Frequency in bad URLs']/TLD_df['TLD'].apply(lambda x: X[X['URL'].str.contains(re.escape(x), regex=True)].shape[0])
TLD_df['Score'] = TLD_df['Bad probability']*TLD_df['Frequency in bad URLs'].apply(math.log)

TLD_df.sort_values(by='Score', ignore_index=True, ascending=False, inplace=True)
TLD_df
#%%
dangerous_TLDs = list(TLD_df['TLD'].head(10))
print(dangerous_TLDs)
plt.bar(TLD_df['TLD'].head(10), TLD_df['Score'].head(10), color = 'green')
plt.xlabel('Dangerous TLD')
plt.ylabel('Score')
plt.show()
#%%
X['URL length'] = X['URL'].apply(len)

#2 Numbers of dots

X['Number of dots'] = X['URL'].apply(lambda x: x.count('.'))

#3 Number of slashes

X['Number of slashes'] = X['URL'].apply(lambda x: x.count('/'))

#4 Percentage of numerical characters

X['Percentage of numerical characters'] = X['URL'].apply(lambda x: sum(c.isdigit() for c in x))/X['URL length']

#5 Dangerous characters

X['Dangerous characters'] = X['URL'].apply(lambda x: any(char in x for char in dangerous_chars))

#6 Dangerous TLD

X['Dangerous TLD'] = X['URL'].apply(lambda x: tldextract.extract(x).suffix in dangerous_TLDs)

#7 Entropy

def urlentropy(url):
    frequencies = Counter(url)
    prob = [frequencies[char] / len(url) for char in url]
    return entropy(prob, base=2)


X['Entropy'] = X['URL'].apply(urlentropy)

#8 IP Address

ip_pattern = r'[0-9]+(?:\.[0-9]+){3}'
X['IP Address'] = X['URL'].apply(lambda x: bool(re.search(ip_pattern, x)))

#9 Domain name length

X['Domain name length'] = X['URL'].apply(lambda x: len(tldextract.extract(x).domain))

#10 Suspicious keywords

sus_words = ['secure', 'account', 'update', 'login', 'verify' ,'signin', 'bank',
            'notify', 'click', 'inconvenient']

X['Suspicious keywords'] = X['URL'].apply(lambda x: sum([word in x for word in sus_words]) != 0)


#11 Repetitions

X['Repetitions'] = X['URL'].apply(lambda x: True if re.search(r'(.)\1{2,}', tldextract.extract(x).domain) else False)

#12 Redirections

def redirection(url):
  pos = url.rfind('//') #If the // is not found, it returns -1
  return pos>7

X['Redirections'] = X['URL'].apply(redirection)

#We print the new dataset

X
#%%
scaler = StandardScaler()
num_columns = ['URL length', 'Number of dots', 'Number of slashes', 'Domain name length', 'Entropy']
X[num_columns] = scaler.fit_transform(X[num_columns])


#%%
X['IP Address'] = X['IP Address'].astype(int)
X['Suspicious keywords'] = X['Suspicious keywords'].astype(int)
X['Repetitions'] = X['Repetitions'].astype(int)
X['Redirections'] = X['Redirections'].astype(int)
X['Dangerous characters'] = X['Dangerous characters'].astype(int)
X['Dangerous TLD'] = X['Dangerous TLD'].astype(int)
X['Label'] = (X['Label'] == 'good').astype(int)

X.drop(columns=['URL'], inplace=True)

X
#%%
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 6})
plt.show()
sns.heatmap(corr_matrix[['Label']].sort_values(by='Label').T, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 8})
plt.show()
#print(corr_matrix[['Label']].sort_values(by='Label'))
#%%
pca = PCA(n_components=1)
X['Entropy and length (PCA)'] = pca.fit_transform(X[['Entropy', 'URL length']])
X.drop(columns=['Entropy', 'URL length'], inplace=True)

X
#%%
X['Label'].value_counts(normalize=True)
#%%
n_samples = X['Label'].value_counts()[0]
X_good = X[X['Label'] == 1]
X_bad = X[X['Label'] == 0]
X_goodsample = X_good.sample(n=n_samples, random_state=22)
X_goodmissing = X_good.drop(X_goodsample.index)

X = pd.concat([X_bad, X_goodsample], ignore_index=True)

X
#%%
y = X['Label']
X.drop(columns=['Label'], inplace=True)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

y_goodmissing = X_goodmissing['Label']
X_goodmissing.drop(columns=['Label'], inplace=True)

# Merging X_test and X_goodmissing

X_test = pd.concat([X_test, X_goodmissing], axis=0)

# Merging y_test and y_goodmissing

y_test = pd.concat([y_test, y_goodmissing], axis=0)
#%%
kf = KFold(n_splits=3, shuffle=True, random_state=22)

xgb_model = XGBClassifier(random_state=22)
print(cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring='accuracy').mean())

rf_model = RandomForestClassifier(random_state=22)
print(cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy').mean())
#%%
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
feature_names = X.columns 
indices = np.argsort(importances)[::-1]

plt.title('Feature Importance (RandomForestClassifier)')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()
#%%
xgb_model.fit(X_train, y_train)
importances = xgb_model.feature_importances_
feature_names = X.columns 
indices = np.argsort(importances)[::-1]

plt.title('Feature Importance (XGBClassifier)')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()
#%%
from sklearn.metrics import accuracy_score

rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

print(accuracy_score(y_test, rf_pred))
print(accuracy_score(y_test, xgb_pred))
#%%
import sys
import optuna



def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 400)
    max_depth = trial.suggest_int('max_depth', 3, 7)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.3)
    subsample = trial.suggest_uniform('subsample', 0.6, 1.0)
    reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-3, 10.0)

    
    model = XGBClassifier(
        random_state=22,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        reg_alpha=reg_alpha,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    
    mean_score = cross_val_score(model, X, y, cv=kf, scoring='accuracy').mean()
    return mean_score 


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50) 


print("Best hyperparameters:", study.best_params)
print("Best accuracy:", study.best_value)

best_xgb_model =XGBClassifier(random_state=22, **study.best_params)
#%%
best_xgb_model.fit(X_train, y_train)

best_xgb_pred = best_xgb_model.predict(X_test)

print(accuracy_score(y_test, best_xgb_pred))

import pickle

# Assuming your model is named `model`
with open('model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)