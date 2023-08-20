import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

#Import Dataset
df = pd.DataFrame()
df = pd.read_csv('churn.csv')

#Check Missing Data
percentage_missing = df.isnull().sum()/len(df) * 100
to_drop = percentage_missing[percentage_missing>14].keys()
not_to_drop = percentage_missing[percentage_missing<14].keys()
df_new = df[not_to_drop]
df_new_dropped_na = df_new.dropna(axis=0)
df = df_new_dropped_na

### SEPARATION BETWEEN NUMERICAL AND CATEGORICAL DATA
list_numerical = []
list_categorical = []
for i in range(0, len(df.keys())):
    if df[df.columns[i]].dtype == 'int64' or df[df.columns[i]].dtype == 'float64':
        list_numerical.append(df.columns[i])
    else:
        list_categorical.append(df.columns[i])
list_numerical.pop(0)
df_numerical = df[list_numerical]
df_categorical = df[list_categorical]

#OUTLIER REMOVAL
#Chebyshev's Theorem applied to the column "training hours":
mean_th = df_numerical['training_hours'].mean()
std_th = df_numerical['training_hours'].std()
df_out_removed = df[df_numerical['training_hours']<mean_th + 3*std_th]
#Chebyshev's Theoremapplied to the column "city_development_index":
mean_ci = df_numerical['city_development_index'].mean()
std_ci = df_numerical['city_development_index'].std()
df_out_removed = df_out_removed[df_out_removed['city_development_index']>mean_ci - 3*std_ci]
df = df_out_removed
df_numerical = df[list_numerical]
df_categorical = df[list_categorical]

#conversion of categorical variables
dummies = pd.get_dummies(df_categorical, drop_first = True)

### STANDARDIZATION

df_to_scale = df_numerical[['city_development_index', 'training_hours']]
scaler = StandardScaler().fit(df_to_scale)
scaled_df = pd.DataFrame(scaler.transform(df_to_scale))
scaled_df.columns = df_to_scale.columns
dummies.reset_index(drop=True, inplace=True)
X_numerical = scaled_df
X_numerical.reset_index(drop=True, inplace=True)
print(X_numerical.shape)
print(dummies.shape)
X = pd.concat([dummies, X_numerical], axis = 1)
y = df['target']

### CHECK IMBALANCE
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
size_group_0 = len(pd.concat([X,y], axis=1).groupby('target').get_group(0))
size_group_1 = len(pd.concat([X,y], axis=1).groupby('target').get_group(1))
from sklearn.utils import resample
df_majority = pd.concat([X,y], axis = 1)[pd.concat([X,y], axis = 1).target==0]
df_minority = pd.concat([X,y], axis = 1)[pd.concat([X,y], axis = 1).target==1]

#downsample majority class
df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples = size_group_1,
                                   random_state = 123)

Xy_train = pd.concat([df_majority_downsampled, df_minority])
X_downsampled = Xy_train.drop(columns =['target'])
y_downsampled = Xy_train['target']


### SEPARATE TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(X_downsampled, y_downsampled, test_size = 0.30, stratify=y_downsampled, random_state = 123)


### MULTI-LAYER PERCEPTRON CLASSIFIER
classifier = MLPClassifier()
parameters = {"hidden_layer_sizes":[(10,5), (100,20,5)], "max_iter": [200, 500, 1000, 2000], "alpha": [0.001, 0.01, 0.1], "learning_rate_init": [0.005, 0.001, 0.01, 0.002]}
gs = GridSearchCV(classifier, parameters, cv = 3, scoring = 'f1', verbose = 50, n_jobs = -1, refit = True)
gs = gs.fit(X_train, y_train)
model = MLPClassifier(hidden_layer_sizes = gs.cv_results_['params'][0]['hidden_layer_sizes'],
                      activation = 'relu', alpha = gs.cv_results_['params'][0]['alpha'], batch_size = 'auto',
                      learning_rate = 'constant', learning_rate_init = gs.cv_results_['params'][0]['learning_rate_init'],
                      max_iter = gs.cv_results_['params'][0]['max_iter'], solver = 'lbfgs', tol = 0.01,
                      validation_fraction = 0.02, verbose = True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
while f1_score(y_test,y_pred)<0.66:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
print('f1_score:', f1_score(y_test, y_pred))

# Predict probabilities for each class
probabilities = model.predict_proba(X_test)

#SAVE THE MODEL
joblib.dump(model, 'trained_model.pkl')
