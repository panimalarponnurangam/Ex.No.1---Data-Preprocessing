# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
```
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train
```
## PROGRAM:
```
Dveloped by:panimalar.p
register no:212222110031
```
importing libraries
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```
Reading the dataset :
```
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df
```
Dropping the unwanted Columns :
```
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
```
Checking for null values :
```
df.isnull().sum()
```
Checking for duplicate values :
```
df.duplicated()
```
Describing the dataset :
```
df.describe()
```
Scaling the dataset :
```
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
```
Allocating X and Y attributes :
```
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
```
Splitting the data into training and testing dataset :
```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```
OUTPUT:
The dataset :
![264971809-4602a691-ce56-4599-be28-d1dbf84c6f8f](https://github.com/panimalarponnurangam/Ex.No.1---Data-Preprocessing/assets/121490826/0660b3fd-6b87-48f7-b2f9-34712ef4856e)

Dropping unwanted features :
![264971983-a1814a51-3183-47f3-a206-ac652fb26458](https://github.com/panimalarponnurangam/Ex.No.1---Data-Preprocessing/assets/121490826/6fdfc3c8-2043-4d10-a93c-9fe15f842f3f)
Checking for null values :

![264972071-c0c93b29-6465-44aa-84a8-791764903a37](https://github.com/panimalarponnurangam/Ex.No.1---Data-Preprocessing/assets/121490826/57bffd5f-b212-482b-8290-2c0de9f4c83d)
Checking for duplication :
![264972221-328c73bc-5773-42d8-9c0a-113749997b77](https://github.com/panimalarponnurangam/Ex.No.1---Data-Preprocessing/assets/121490826/5b8a9aa5-d88a-45d9-a2d9-d0bed24c97b2)
Describing the dataset :

![264972304-a5b12844-69bc-45e4-a81e-369f397e74a0](https://github.com/panimalarponnurangam/Ex.No.1---Data-Preprocessing/assets/121490826/699affbc-db6d-49f4-b4f4-d5b88a5b12d6)
Scaling the values :

![264972373-34613cc1-4229-40f4-94fe-4c40a08f725f](https://github.com/panimalarponnurangam/Ex.No.1---Data-Preprocessing/assets/121490826/9940cba7-11df-4ab5-bc2e-848a97fc179f)
X Features :
![264972437-82154363-58d3-4578-b1f3-71cefd8d1fec](https://github.com/panimalarponnurangam/Ex.No.1---Data-Preprocessing/assets/121490826/7de2666c-ab5c-4b2c-a2ce-d984de6bbcc9)
Y Features :
![264972629-95cff3d1-6446-4b7a-93aa-deed1cbb803d](https://github.com/panimalarponnurangam/Ex.No.1---Data-Preprocessing/assets/121490826/af081e1b-0f2f-4f3f-9c9a-d40949adb4f6)


Splitting the training and testing dataset :

![264972713-edbdc2c9-ce14-4efb-a342-d7dd77b9f460](https://github.com/panimalarponnurangam/Ex.No.1---Data-Preprocessing/assets/121490826/6861b340-4607-48d2-8331-83c9d8df40aa)


## RESULT
The Data preprocessing is performed over a data set successfully.
