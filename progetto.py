import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.svm import SVC
from xgboost import XGBClassifier


#remove future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#read the dataset
df = pd.read_csv("emails.csv")
#remove the features "Email No." because it is only an ID
df.pop("Email No.")
#remove duplicate rows
df.drop_duplicates(inplace=True)

#number of features
print("Number of features: ", len(df.columns) - 1)
#number of rows
print("Number of rows: ", len(df))
#number of negative values
print("Number of negative values: ", np.sum((df < 0).values.ravel()))
#check duplicated rows
print("There are duplicated rows? -> ", df.duplicated().values.any())
#check null values
print("There are null values? -> ", df.isnull().values.any())


#description of the dataframe
#commented because having many features it isn't very useful to view them all
#print(df.describe())


#get X and y from the dataframe
y = df.pop("Prediction")
X = df

#print the percentage of target label
labelsCount = y.value_counts()
print("Label 0: ", "{:.2f}".format((labelsCount[0] * 100) / (labelsCount[0] + labelsCount[1])), "%")
print("Label 1: ", "{:.2f}".format((labelsCount[1] * 100) / (labelsCount[0] + labelsCount[1])), "%")
print(labelsCount)


#split in train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


########################################################################
                    #FEATURE SELECTION#
########################################################################
#I USE CHI * 2 BECAUSE I HAVE ONLY VALUES >=0, SO IT IS CONVENIENT
#ALSO BECAUSE IT IS MUCH FASTER 
#apply SelectKBest class to extract the k best features
bestfeatures = SelectKBest(score_func=chi2, k=100)
fit = bestfeatures.fit(X_train,y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_train.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']

bins = (max(featureScores['Score'])-min(featureScores['Score']))/20
plt.hist(featureScores['Score'], int(bins))
plt.ylabel("Number of fetures")
plt.xlabel("CHI^2 values")
plt.show()

#print(featureScores)
print("\n\nCHI^2 RESULTS\n")
print("THE MAXIMUM IS: \n", featureScores.max())
print("\nTHE MINIMUM IS: \n", featureScores.min())
print("\nTHE AVG IS: ", featureScores['Score'].mean())

cnt1=0
cnt2=0
cnt3=0
cnt4=0
for x in featureScores['Score']:
    if(x <= 100):
        cnt1 = cnt1 + 1
    elif(100 < x <= 500):
        cnt2 = cnt2 + 1
    elif(500 < x <= 1000):
        cnt3 = cnt3 + 1
    if(x > 1000):
        cnt4 = cnt4 + 1

print("The features with chi2 <100 are: ", cnt1)
print("The features with chi2 between 101 e 500 are: ", cnt2)
print("The features with chi2 between 501 e 1000 are: ", cnt3)
print("The features with chi2 >1000 are: ", cnt4)
print("\n\n")

#K>50 -> OVERFITTING
X_trainFS = SelectKBest(chi2, k=50).fit_transform(X_train, y_train)
X_testFS = SelectKBest(chi2, k=50).fit_transform(X_test, y_test)


#PCA
#99% variance
pca = PCA(n_components=0.99)
#pca = PCA(n_components=10)
pca.fit(X_trainFS)
X_train_reduced = pd.DataFrame(pca.transform(X_trainFS))
#same pcs on x test
X_test_reduced = pd.DataFrame(pca.transform(X_testFS))

#plot of the features after PCA
pd.DataFrame(X_train_reduced).hist(figsize=(30, 30), bins=25)
plt.show()

########################################################################
                    #FEATURE SCALING#
########################################################################
#avg = np.mean(X_train_reduced, axis=0)
#std = np.std(X_train_reduced,axis=0)
#X_train_reduced = (X_train_reduced-avg)/std
#X_test= (X_test-avg)/std


########################################################################
                    #CROSS VALIDATION#
########################################################################
#10-FOLD CROSS VALIDATION
#initialize K-fold
kf = KFold(n_splits=10, random_state=None, shuffle=False)

modelli = [RandomForestClassifier(),
           SVC(),
           LogisticRegression(),
           AdaBoostClassifier(),
           GradientBoostingClassifier(),
           XGBClassifier(eval_metric='logloss')] #to remove warnings

mean_recall = []
mean_precision = []
mean_f1 = []

for model in modelli:
    recall = []
    precision = []
    f1 = []
    for train_index, validation_index in kf.split(X_train_reduced):
        #for every split, different training and validation
        X_trainKF = X_train_reduced[train_index[0]:train_index[-1]]
        X_validationKF = X_train_reduced[validation_index[0]:validation_index[-1]]
        y_trainKF = y_train[train_index[0]:train_index[-1]]
        y_validationKF = y_train[validation_index[0]:validation_index[-1]]

        #Train the models and calculate the RECALL, PRECISION and F1
        model.fit(X_trainKF, y_trainKF)
        y_pred = model.predict(X_validationKF)
        recall.append(recall_score(y_validationKF, y_pred))
        precision.append(precision_score(y_validationKF, y_pred))
        f1.append(f1_score(y_validationKF, y_pred))

    print(model, ":\nRecall->", np.mean(recall), "\nPrecision->", np.mean(precision), "\nF1->", np.mean(f1), "\n")
    mean_recall.append(np.mean(recall))
    mean_precision.append(np.mean(precision))
    mean_f1.append(np.mean(f1))

modelli = ["RANDOM\nFOREST", "SVC", "LOGISTIC\nREGRESSION",
           "ADA BOOST", "GRADIENT\nBOOSTING", "XGBOOST"]

#graphic for recall, precision and F1
plt.bar(modelli, mean_recall, align='center')
plt.title('MEDIA RECALL')
plt.show()

plt.bar(modelli, mean_precision, align='center')
plt.title('MEDIA PRECISION')
plt.show()

plt.bar(modelli, mean_f1, align='center')
plt.title('MEDIA F1')
plt.show()



########################################################################
                        #FINE TUNING#
########################################################################
#XGBOOST IS THE MODEL THAT HAS THE BEST RESULTS
#RANDOM FOREST ALSO HAS GOOD RESULTS, BUT ON AN AVERAGE OF 10, XGBOOST is
#LIGHTLY BETTER 

print("\nRANDOM SEARCH...")

#RANDOM SEARCH FOR XGBOOST
#initialize the model
xgb = XGBClassifier(eval_metric='logloss')

#n_estimators -> 100, 150, 200, 250, 300, 350, 400, 450, 500
#max_depth -> 3, 4, 5, 6, 7, 8, 9, 10
#gamma -> 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
#subsample -> 0.5, 0.6, 0.7, 0.8, 0.9, 1
#scale_pos_weight -> 2 (neg/pos)
#learning_rate -> 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1
#random grid
random_grid = {'n_estimators': np.arange(100, 550, 50),
               'max_depth': [x for x in range(5, 11)],
               'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
               'subsample': [x/10.0 for x in range(5, 11)],
               'scale_pos_weight': [2],
               'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3]}

#randomized search:
#   estimator -> XGBClasssifier
#   param_distributions -> random_grid (previously declared)
#   cv -> 10 (cross validation, 10 split)
#   return_train_score -> true
#   n_jobs -> -1 (parallel execution)
#   scoring -> recall
xgb_rs = RandomizedSearchCV(estimator = xgb, param_distributions = random_grid,
                            cv = 10, return_train_score = True, n_jobs=-1,
                            scoring = "recall")

xgb_rs.fit(X_train_reduced, y_train)

best_params = xgb_rs.best_params_

#print the best parameters
print("BEST PARAMETERS AFTER RANDOM SEARCH:")
print("Best Score: ", xgb_rs.best_score_)
print("Best Hyperparameters: \n", best_params)


#GRID SEARCH FOR XGBOOST

print("\nGRID SEARCH...")

xgb = XGBClassifier(eval_metric='logloss')

#set the parameters of the random search to the model 
xgb.set_params(**best_params)

# search space as a dictionary
value_reg_alpha = dict()
value_reg_alpha['reg_alpha'] = [1e-5, 1e-2, 0.1, 1, 100, 200]

#grid search
xgb_gs = GridSearchCV(xgb, value_reg_alpha, n_jobs=-1, cv=10, scoring="recall",
                      return_train_score=True)

xgb_gs.fit(X_train_reduced, y_train)

#print the best one
print("BEST PARAMETERS FOR REG_ALPHA AFTER GRID SEARCH:")
print("Best Score: ", xgb_gs.best_score_)
print("Best Hyperparameters: ", xgb_gs.best_params_)

best_params['reg_alpha'] = xgb_gs.best_params_['reg_alpha']


########################################################################
                                #TEST#
########################################################################
print("\n\nTEST")
XGBtest = XGBClassifier(eval_metric='logloss')
#set the best params to the model
XGBtest.set_params(**best_params)
#fit
XGBtest.fit(X_train_reduced, y_train)
#predict on X_Test
y_pred = XGBtest.predict(X_test_reduced)
#print of Recall, Precision and F-1
print("RECALL ->",recall_score(y_test, y_pred))
print("PRECISION ->",precision_score(y_test, y_pred))
print("F1 ->", f1_score(y_test, y_pred))
