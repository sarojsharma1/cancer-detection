import pickle
import numpy as np

classifier = pickle.load(open('train_file','rb'))

lis = []

for i in range(9):
    a = int(input("enter Test values"))
    lis.append(a)
    
y_predict = int(classifier.predict([lis]))
                
if y_predict==2:
    print("Cancer Type: Benign")
else:
    print("Cancer Type: Malignant")
    
