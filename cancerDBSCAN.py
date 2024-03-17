from sklearn.datasets import load_breast_cancer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import KernelPCA
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
cancer=load_breast_cancer()
data=pd.DataFrame(cancer.data,columns=cancer.feature_names)
data=data.rename(columns={'worst fractal dimension':'worst_fractal_dimension'})
x=data.drop('worst_fractal_dimension',axis=1)
y=data['worst_fractal_dimension']
scaler=StandardScaler()
x=scaler.fit_transform(x)
pca=KernelPCA(n_components=2)
x=pca.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scan=DBSCAN(eps=1,min_samples=3)
prediction=scan.fit_predict(x_train)
plt.scatter(x_train[:,0],x_train[:,1],c=prediction)
plt.plot()
