## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/Encoding Data.csv")
df
```
![Screenshot 2024-09-25 101303](https://github.com/user-attachments/assets/32e3d1b1-8391-43a4-b259-0158eeee7259)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2024-09-25 101506](https://github.com/user-attachments/assets/6fcb7e9e-49ae-4959-a43a-89cc7c65434f)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-09-25 101548](https://github.com/user-attachments/assets/c095c0a9-225a-4224-b8c2-347dedd0a5f6)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-09-25 101635](https://github.com/user-attachments/assets/5b08be9f-52c3-4ffb-be85-8f8a15716d72)
```
dfc=df.copy()
dfc['con_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-09-25 101730](https://github.com/user-attachments/assets/7aa9d231-7947-474c-84aa-716f64214bbe)
```
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![Screenshot 2024-09-25 101813](https://github.com/user-attachments/assets/5146703b-938a-4953-ad28-167561df916b)
```
ohe=OneHotEncoder(sparse=False)
df1=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df1[["nom_0"]]))
enc
```
![Screenshot 2024-09-25 101848](https://github.com/user-attachments/assets/2cb8db8d-1654-44fd-a2d6-1aa6cc101760)
```
df1=pd.concat([df1,enc],axis=1)
df1
```
![Screenshot 2024-09-25 101920](https://github.com/user-attachments/assets/d6b4da47-efa4-40b4-a11b-73c0b666615f)
```
pip install --upgrade category_encoders
```
![Screenshot 2024-09-25 102019](https://github.com/user-attachments/assets/09d5d7fd-2962-4a92-a2ee-f1c9f444ea28)
```
from category_encoders import BinaryEncoder
import pandas as pd
df=pd.read_csv("/content/data.csv")
df
```
![Screenshot 2024-09-25 102119](https://github.com/user-attachments/assets/03825f8f-6d57-4453-8ea7-23dc3c52879e)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![Screenshot 2024-09-25 102224](https://github.com/user-attachments/assets/8a5e2bb2-4961-4fc6-b8a7-f9ccc406c691)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![Screenshot 2024-09-25 102323](https://github.com/user-attachments/assets/ec81bd9b-ff1b-4016-8983-eec44fbce01d)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![Screenshot 2024-09-25 102403](https://github.com/user-attachments/assets/ef0694e5-88e6-4804-97a7-c1fdc90a6478)
```
df.skew()
```
![Screenshot 2024-09-25 102432](https://github.com/user-attachments/assets/2c4aad78-5171-4770-8e18-876849138e73)
```
df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df["Highly Positive Skew"]
```
![Screenshot 2024-09-25 102505](https://github.com/user-attachments/assets/857e2b75-06a8-4782-8d85-6f11addfd250)
```
df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df["Moderate Positive Skew"]
```
![Screenshot 2024-09-25 102542](https://github.com/user-attachments/assets/cf98c2c9-bab1-4ac5-92d2-5fc678e08468)
```
df["Highly Positive Skew"]=np.sqrt(df["Highly Positive Skew"])
df["Highly Positive Skew"]
```
![Screenshot 2024-09-25 102616](https://github.com/user-attachments/assets/d71962cf-758b-417a-8f68-e4b30bca0271)
```
df.skew()
```
![Screenshot 2024-09-25 102648](https://github.com/user-attachments/assets/ffd9bd51-98a3-4ccb-b689-7dd84ffd6135)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2024-09-25 102730](https://github.com/user-attachments/assets/e2904259-fcb2-434f-b476-4371df9576ed)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-09-25 102804](https://github.com/user-attachments/assets/91ec0bc8-ae0c-41df-93c3-79cccc8364f4)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2024-09-25 102830](https://github.com/user-attachments/assets/4f37e1ca-cc8e-4f6a-9c55-3a3bc5908e09)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=892)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-09-25 102919](https://github.com/user-attachments/assets/18e289af-ed76-43d0-96e2-44eafd74ee92)

       
# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
