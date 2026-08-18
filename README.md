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
df=pd.read_csv("Encoding Data.csv") 
df
```
<img width="412" height="365" alt="image" src="https://github.com/user-attachments/assets/a08413ff-98fd-4c5d-ac79-5dadb62c1445" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder 
pm=['Hot','Warm','Cold'] 
e1=OrdinalEncoder(categories=[pm]) 
e1.fit_transform(df[["ord_2"]])
```

<img width="348" height="237" alt="image" src="https://github.com/user-attachments/assets/f88e01f4-dea7-4415-ba10-9aed5c0e0251" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]]) 
df
```

<img width="467" height="357" alt="image" src="https://github.com/user-attachments/assets/fa5d4d02-ab4c-4182-b73b-d5fc87f98b80" />

```
le=LabelEncoder() 
dfc=df.copy() 
dfc['ord_2']=le.fit_transform(dfc['ord_2']) 
dfc
```

<img width="461" height="357" alt="image" src="https://github.com/user-attachments/assets/504236a7-de04-4ecf-a907-81305b9eecea" />

```
from sklearn.preprocessing import OneHotEncoder 
ohe=OneHotEncoder(sparse_output=False) 
df2=df.copy() 
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) 
# Orders in Alphabetical Order Blue , Green, Red 
df2=pd.concat([df2,enc],axis=1) 
df2
```

<img width="590" height="376" alt="image" src="https://github.com/user-attachments/assets/c3227798-7e60-4fe8-9d6f-3035d0bcdc4b" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="877" height="367" alt="image" src="https://github.com/user-attachments/assets/3b2f6248-a609-4c50-ae0d-8b4bca16fbb9" />

```
import pandas as pd
from category_encoders import BinaryEncoder 
df=pd.read_csv("data.csv") 
df
```

<img width="640" height="387" alt="image" src="https://github.com/user-attachments/assets/aa757d4a-69dc-47af-a250-84ab965bf60f" />

```
be=BinaryEncoder() 
nd=be.fit_transform(df['Ord_2']) 
dfb=pd.concat([df,nd],axis=1) 
dfb
```

<img width="887" height="376" alt="image" src="https://github.com/user-attachments/assets/f7f0fbf0-c6eb-4d88-b5fc-929885488ece" />

```
from category_encoders import TargetEncoder 
te=TargetEncoder() 
CC=df.copy() 
new=te.fit_transform(X=CC["City"],y=CC["Target"]) 
CC=pd.concat([CC,new],axis=1) 
CC
```

<img width="801" height="387" alt="image" src="https://github.com/user-attachments/assets/16ca6111-c6c1-4dfd-a047-89e693db8035" />

```
import pandas as pd 
from scipy import stats 
import numpy as np 
df=pd.read_csv("Data_to_Transform.csv") 
df
```


<img width="970" height="412" alt="image" src="https://github.com/user-attachments/assets/933f3af5-69d3-4a29-bf59-44c151939485" />

```
df.skew()
```

<img width="485" height="122" alt="image" src="https://github.com/user-attachments/assets/e4f6f7cb-e9c0-46ea-a670-e82e703ee7b7" />


```
np.log(df["Highly Positive Skew"])
```


<img width="668" height="270" alt="image" src="https://github.com/user-attachments/assets/8c167f04-8536-4632-aded-3d8ccaec50a1" />


```
# 2. RECIPROCAL TRANSFORMATION 
np.reciprocal(df["Moderate Positive Skew"])
```


<img width="897" height="290" alt="image" src="https://github.com/user-attachments/assets/402b0dab-f1cb-46b1-a061-52385a913c15" />

```
# 4. SQUARE ROOT TRANSFORMATION 
np.sqrt(df["Highly Positive Skew"])
```

<img width="711" height="270" alt="image" src="https://github.com/user-attachments/assets/f3595d90-c194-43a9-a66b-31d89a5d8485" />

```
# 5. SQUARE TRANSFORMATION 
np.square(df["Highly Positive Skew"])
```
<img width="692" height="271" alt="image" src="https://github.com/user-attachments/assets/decbe2f5-2111-4975-91de-26b7058a2440" />

```
# POWER TRANSFORMATIONS 
#        BOX COX 
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
df
```
<img width="1236" height="456" alt="image" src="https://github.com/user-attachments/assets/5c69931a-219b-4815-ba84-d731ca0613c0" />

```
df.skew()
```
<img width="528" height="145" alt="image" src="https://github.com/user-attachments/assets/cfc5f5d4-e17a-4e8e-8bb7-5eaaf38e1533" />

```
# YEO_JOHNSON 
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"]) 
df.skew()
```
<img width="562" height="161" alt="image" src="https://github.com/user-attachments/assets/37a5be04-0349-4248-b0f3-5fa71c5734b0" />
```
# QUANTILE TRANSFORMATION 
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal') 
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
df
```
<img width="1403" height="483" alt="image" src="https://github.com/user-attachments/assets/5dc8db2d-9ef9-44b4-bbf9-aa4b97500e35" />

```
import seaborn as sns 
import statsmodels.api as sm 
# STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION 
import matplotlib.pyplot as plt 
sm.qqplot(df["Moderate Negative Skew"],line='45') 
# QQ - QUANTILE QUANTILE PLOT 
plt.show()
```
<img width="833" height="548" alt="image" src="https://github.com/user-attachments/assets/2a56b4f4-ba6e-4e9c-8448-3b915d5d0960" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL 
plt.show()
```

<img width="798" height="561" alt="image" src="https://github.com/user-attachments/assets/ca381e1d-c34c-402a-aca3-17354bb61f4c" />
```
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891) 
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
sm.qqplot(df["Moderate Negative Skew"],line='45') 
plt.show()
```
<img width="823" height="573" alt="image" src="https://github.com/user-attachments/assets/8e237da2-b674-45e3-8645-3848cd5499dc" />


# RESULT:
       Thus the Implementation of Feature Encoding and Feature Transformation executed successfully.
       
