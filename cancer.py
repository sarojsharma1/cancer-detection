import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import pickle

#--->1.Load Data from CSV file
path = "C:\\Users\\saroj\\OneDrive\\Desktop\\svm model\\"
cell_df = pd.read_csv(path+"cell_samples.csv")
#print(cell_df.head())
#print(cell_df.shape)
#print(cell_df.size)
#print(cell_df.count())
#print(cell_df['Class'].value_counts())

print(cell_df.loc[:,['UnifSize','MargAdh']])

'''
#--->2.Distribution of the classes
benign_df = cell_df[cell_df['Class']==2][0:200]
#print(benign_df)
malignant_df = cell_df[cell_df['Class']==4][0:200]
#help(benign_df.plot)
#axes = benign_df.plot(kind='scatter',x="Clump", y="UnifSize",color='blue',label="Benign")
#malignant_df.plot(kind='scatter',x="Clump", y="UnifSize",color='red',label="malignant",ax=axes)
#plt.show()


#--->3.identifying unwanted rows
#print(cell_df.dtypes)
cell_df = cell_df[pd.to_numeric(cell_df["BareNuc"],errors="coerce").notnull()]
cell_df["BareNuc"] = cell_df["BareNuc"].astype("int")
#print(cell_df["BareNuc"])


#--->4.identifying unwanted columns
#print(cell_df.columns)
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
       'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
y = np.asarray(cell_df["Class"])
#print(X[0:10])


#--->5.Divide the data as Train/Test dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
#print(X_test)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)


#--->6.Modeling(SVM and Scikit-learn)
classifier = svm.SVC(kernel='linear',gamma='auto',C=2)
classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test)


#--->7.Evaluation(Results)
print(classification_report(y_test,y_predict))


#8.Save the model to disk
filename = 'train_file'
pickle.dump(classifier,open(filename, 'wb'))



'''



