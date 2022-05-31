#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 06:35:46 2021

@author: phishing
"""

import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, roc_curve, roc_auc_score, accuracy_score, f1_score
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def fit_multiple_estimators(classifiers, X_list, y, sample_weights = None):

    # Convert the labels `y` using LabelEncoder, because the predict method is using index-based pointers
    # which will be converted back to original data later.
    le_ = LabelEncoder()
    le_.fit(y)
    transformed_y = le_.transform(y)

    # Fit all estimators with their respective feature arrays
    estimators_ = [clf.fit(X, y) if sample_weights is None else clf.fit(X, y, sample_weights) for clf, X in zip([clf for _, clf in classifiers], X_list)]

    return estimators_, le_


def predict_from_multiple_estimator(estimators, label_encoder, X_list, weights = None):

    # Predict 'soft' voting with probabilities

    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
    pred2 = np.average(pred1, axis=0, weights=weights)
    pred = np.argmax(pred2, axis=1)

    # Convert integer predictions to original labels:
    return label_encoder.inverse_transform(pred)


# info gain feature selection
def ig_select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_classif, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

##########begin training dataset#########################
#load dataset
#df=pq.read_table('/home/phishing/Text_Phishing_Email_ML_Classification/Train_Test_datasets_AdditionalColumnDataset.parquet').to_pandas()
df=pq.read_table('/home/phishing/Text_Phishing_Email_ML_Classification/Train_Test_merged_sameWord2VecModel_noDuplicates.parquet').to_pandas()

#print(df.dtypes)

#Word2Vec features
count=0
#build the training set 
w2v=df['Word2vec']
#array that keeps the word2vec features
vec=[]
c=0
for e in w2v: #edw pairnei 1-1 ta emails
   #isolate the word2vec vectors
    #pass the vectors of each email in a dict
    vec.append(e['values'])
    
#convert the array that contains the w2v features to numpy      
vec=np.array(vec)
#vec = np.vstack(vec).astype(np.float)
df_new = pd.DataFrame(vec)    
#############################################

#training set #remove label
X=df.drop(['label'], axis=1)
#testing set #label only
y=df['label']
#split the set into training 70% and testing 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=3) 

#create a pandas DF
label=df['label']
label=np.array(label, dtype=int)
#add a new column with the label in the new df
df_new['label'] = label

#training set only w2v features
X=df_new.drop(['label'], axis=1)
#testing set #label only
y=df_new['label']

#split the set into training 70% and testing 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=3) 
#number of samples and features training set 
#print(X_train.shape)
#print(y_train.shape)

#print(df_new.info())

#############Word2Vec Feature selection###################
#keep features names
features= []
for i in X:
    features.append(i)
FS=[]
df2=[]

# feature selection with mutual information
X_train_fs, X_test_fs, fs = ig_select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
    if fs.scores_[i]>0.01:
        count+=1
        FS.append(features[i])
        #print('Feature %s: %f' % (features[i], fs.scores_[i]))
#print(count) #number of word features after MI
#print(FS)
for i in FS:
    df2.append(df_new[i])
    
#convert the array that contains the selected w2v features to numpy      
df2=np.array(df2)
#vec = np.vstack(vec).astype(np.float)
df2 = pd.DataFrame(df2)   
df2 = df2.transpose() 
########################################

########################content-based features training
#remove text-based features and unwanted content-based features
df = df.drop(['script_parts', 'scripts', 'forms', 'nports', 'link_images', 'dataset', 'lemmatized', 'stemmed', 'joinedLem', 'Word2vec'], axis =1)#Converting the encoding column to categorical - it assigns an int on each encoding-name
df['encoding']=df['encoding'].astype('category')   
#Integer Encoding the 'encoding' column
enc_encode = LabelEncoder()
#Integer encoding the 'encoding' column
df['encoding'] = enc_encode.fit_transform(df.encoding)
#########################################

#Concat word2vec with content-based features TRAINING 
df2 = pd.concat([df, df2], axis=1)
X=df2.drop(['label'], axis=1)
y=df2['label']
########################################
##########end training dataset#########################

#######classification
#split the set into training 70% and testing 30% using the selected features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=3) 

### LR
#Create an LR Classifier 
clf_lr = LogisticRegression(C=1, class_weight='balanced', max_iter=1000, penalty='l1', solver='saga')

btime_lr = time.time()
#Train the model using the training sets
clf_lr.fit(X_train, y_train)
etime_lr = time.time()

#Predict the response for test dataset
y_pred_lr = clf_lr.predict(X_test)

#print results
print("LR results...")
print(confusion_matrix(y_test,y_pred_lr))
print(classification_report(y_test, y_pred_lr, digits=4))
print('Accuracy Score: %.4f' %accuracy_score(y_test,y_pred_lr))
print('F1 Score: %.4f' %f1_score(y_test,y_pred_lr, average='micro'))        
print ('AUC: %.4f' %roc_auc_score(y_test, y_pred_lr)) 
print("Training time Sec: %.4f" %(etime_lr-btime_lr))

### GNB
#Create an Gaussian NB Classifier 
clf_gnb = GaussianNB(var_smoothing=0.01)

btime_gnb = time.time()
#Train the model using the training sets
clf_gnb.fit(X_train, y_train)
etime_gnb = time.time()

#Predict the response for test dataset
y_pred_gnb = clf_gnb.predict(X_test)
#print results
print("GNB results...")
print(confusion_matrix(y_test,y_pred_gnb))
print(classification_report(y_test, y_pred_gnb, digits=4))
print('F1 Score: %.4f' %f1_score(y_test,y_pred_gnb, average='micro'))
print('Accuracy Score: %.4f' %accuracy_score(y_test,y_pred_gnb))
print ('AUC: %.4f' %roc_auc_score(y_test, y_pred_gnb)) 
print("Training time Sec: %.4f" %(etime_gnb-btime_gnb))

###kNN
clf_knn = KNeighborsClassifier(n_neighbors=3)

#Train the model using the training sets
btime_knn = time.time()
clf_knn.fit(X_train, y_train)
etime_knn = time.time()

#Predict the response for test dataset
y_pred_knn = clf_knn.predict(X_test)

#print results
print("kNN results...")
print(confusion_matrix(y_test,y_pred_knn))
print(classification_report(y_test,y_pred_knn, digits=4))
print('Weighted F1 Score: %.4f' %f1_score(y_test,y_pred_knn, average='micro'))
print('Accuracy Score: %.4f' %accuracy_score(y_test,y_pred_knn))
print('AUC: %.4f' %roc_auc_score(y_test, y_pred_knn))
print("Training time Sec: %.4f" %(etime_knn-btime_knn))

###RF
clf_rf = RandomForestClassifier(max_depth=20, max_features=3, min_samples_leaf=3, min_samples_split=10, random_state=0)

btime_rf = time.time()
#Train the model using the training sets
clf_rf.fit(X_train, y_train)
etime_rf = time.time()
#Predict the response for test dataset
y_pred_rf = clf_rf.predict(X_test)

#print results
print("RF results...")
print(confusion_matrix(y_test,y_pred_rf))
print(classification_report(y_test, y_pred_rf, digits=4))
print('Accuracy Score: %.4f' %accuracy_score(y_test,y_pred_rf))
print('F1 Score: %.4f' %f1_score(y_test,y_pred_rf, average='micro'))        
print ('AUC: %.4f' %roc_auc_score(y_test, y_pred_rf))
print("Training time Sec: %.4f" %(etime_rf-btime_rf))
#############################################################################################

### DT
#Create an dt Classifier 
clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=20, min_samples_leaf=3, min_samples_split=10, max_features='auto')

btime_dt = time.time()
#Train the model using the training sets
clf_dt.fit(X_train, y_train)
etime_dt = time.time()
#Predict the response for test dataset
y_pred_dt = clf_dt.predict(X_test)

#print results
print("DT results...")
print(confusion_matrix(y_test,y_pred_dt))
print(classification_report(y_test, y_pred_dt, digits=4))
print('Accuracy Score: %.4f' %accuracy_score(y_test,y_pred_dt))
print('Weighted F1 Score: %.4f' %f1_score(y_test,y_pred_dt, average='micro'))        
print ('AUC: %.4f' %roc_auc_score(y_test, y_pred_dt))
print("Training time: %.4f" %(etime_dt-btime_dt))    
############################################################################################# 
#clf_mlp = MLPClassifier(alpha=0.5, hidden_layer_sizes=(50, 100, 50), max_iter=1000)

### MLP
#Create an MLP Classifier 
clf_mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 100, 50), solver='adam', alpha=0.5, learning_rate='adaptive', max_iter=1000)

btime_mlp = time.time()
#Train the model using the training sets
clf_mlp.fit(X_train, y_train)
etime_mlp = time.time()

#Predict the response for test dataset
y_pred_mlp = clf_mlp.predict(X_test)
#print results
print("MLP results...")
print(confusion_matrix(y_test,y_pred_mlp))
print(classification_report(y_test, y_pred_mlp, digits=4))
print('F1 Score: %.4f' %f1_score(y_test,y_pred_mlp, average='micro'))
print('Accuracy Score: %.4f' %accuracy_score(y_test,y_pred_mlp))
print('AUC: %.4f' %roc_auc_score(y_test, y_pred_mlp))    
print("Training time Sec: %.4f" %(etime_mlp-btime_mlp))


############soft voting classification

#Divide the dataset between content and word based
X_train1, X_train2 = X_train.iloc[:,0:18], X_train.iloc[:,18:259]
X_test1, X_test2 = X_test.iloc[:,0:18], X_test.iloc[:,18:259]

X_train_list = [X_train1, X_train2]
X_test_list = [X_test1, X_test2]


# Make sure the number of estimators here are equal to number of different feature datas
#classifiers = [('dt',  clf_dt), ('mlp', clf_mlp)]
classifiers = [('dt',  clf_dt), ('knn', clf_knn)]

btime_vote = time.time()
fitted_estimators, label_encoder = fit_multiple_estimators(classifiers, X_train_list, y_train)
etime_vote = time.time()

y_pred_voting = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list)

#print results
print("Voting results...")
print(confusion_matrix(y_test,y_pred_voting))
print(classification_report(y_test, y_pred_voting, digits=4))
print('F1 Score: %.4f' %f1_score(y_test,y_pred_voting, average='micro')) 
print('Accuracy Score: %.4f' %accuracy_score(y_test,y_pred_voting))
print ('AUC: %.4f' %roc_auc_score(y_test, y_pred_voting))
print("Training time Sec: %.4f" %(etime_vote-btime_vote))

##############stacking classification
'''
estimators = [('rf', RandomForestClassifier(max_depth=100, max_features=3, min_samples_leaf=3, min_samples_split=10, n_estimators=200)), ('mlp', MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 100, 50), solver='adam', alpha=0.5, learning_rate='adaptive', max_iter=1000))]
clf_stacked = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier())

btime_stack = time.time()
clf_stacked = clf_stacked.fit(X_train, y_train)
etime_stack = time.time()

y_pred_stacked = clf_stacked.predict(X_test)
#print results
print("Stacked results...")
print(confusion_matrix(y_test,y_pred_stacked))
print(classification_report(y_test, y_pred_stacked, digits=4))
print('Weighted F1 Score: %.4f' %f1_score(y_test,y_pred_stacked, average='weighted')) 
print('Accuracy Score: %.4f' %accuracy_score(y_test,y_pred_stacked))
print ('AUC: %.4f' %roc_auc_score(y_test, y_pred_stacked))
print("Training time Sec: %.4f" %(etime_stack-btime_stack))
'''
####stacking classification different 
'''
pipe1 = make_pipeline(ColumnSelector(cols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18)),
                      RandomForestClassifier(max_depth=100, max_features=3, min_samples_leaf=3, min_samples_split=10, n_estimators=200))
'''
pipe1 = make_pipeline(ColumnSelector(cols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18)),
                      DecisionTreeClassifier(criterion='entropy', max_depth=100, min_samples_leaf=3, min_samples_split=2, max_features='auto'))

pipe2 = make_pipeline(ColumnSelector(cols=(19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259)),
                      KNeighborsClassifier(n_neighbors=3))

sclf = StackingClassifier(classifiers=[pipe1, pipe2], 
                          meta_classifier=MLPClassifier(activation='tanh', hidden_layer_sizes=(20, 40, 20), solver='adam', alpha=0.5, learning_rate='adaptive', max_iter=1000))

#sclf = StackingClassifier(classifiers=[pipe1, pipe2], 
#                          meta_classifier=RandomForestClassifier(max_depth=100, max_features=5, min_samples_leaf=3, min_samples_split=10, n_estimators=200))

btime_stack = time.time()
sclf = sclf.fit(X_train, y_train)
etime_stack = time.time()
y_pred_stacked = sclf.predict(X_test)

print("Stacked results...")
print(confusion_matrix(y_test,y_pred_stacked)) 
print(classification_report(y_test, y_pred_stacked, digits=4))
print('F1 Score: %.4f' %f1_score(y_test,y_pred_stacked, average='micro'))
print('Accuracy Score: %.4f' %accuracy_score(y_test,y_pred_stacked))
print ('AUC: %.4f' %roc_auc_score(y_test, y_pred_stacked))
print("Training time Sec: %.4f" %(etime_stack-btime_stack))

############# PLOTS ###########################################

#plot roc curves of all classifiers
#plot knn
fpr_knn, tpr_knn, threshold = roc_curve(y_test, y_pred_knn)
pyplot.plot(fpr_knn, tpr_knn, marker='4', label='KNN')
#plot lr
fpr_lr, tpr_lr, threshold = roc_curve(y_test, y_pred_lr)
pyplot.plot(fpr_lr, tpr_lr, marker='D', label='LR')
#plot GNB
fpr_gnb, tpr_gnb, threshold = roc_curve(y_test, y_pred_gnb)
pyplot.plot(fpr_gnb, tpr_gnb, marker='X', label='GNB')
#plot dt
fpr_dt, tpr_dt, threshold = roc_curve(y_test, y_pred_dt)
pyplot.plot(fpr_dt, tpr_dt, marker='>', label='DT')
#plot MLP
fpr_mlp, tpr_mlp, threshold = roc_curve(y_test, y_pred_mlp)
pyplot.plot(fpr_mlp, tpr_mlp, marker='*', label='MLP')
#plot rf
fpr_rf, tpr_rf, threshold = roc_curve(y_test, y_pred_rf)
pyplot.plot(fpr_rf, tpr_rf, marker='.', label='RF')
#plot voting
fpr_voting, tpr_voting, threshold = roc_curve(y_test, y_pred_voting)
pyplot.plot(fpr_voting, tpr_voting, marker='s', label='Soft Voting')
#plot stacking
fpr_stacked, tpr_stacked, threshold = roc_curve(y_test, y_pred_stacked)
pyplot.plot(fpr_stacked, tpr_stacked, marker='8', label='Stacking')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

'''
#plot confusion matrix
#plot KNN
plot_confusion_matrix(clf_knn, X_test, y_test, display_labels=y)
pyplot.show()
'''