<meta name="google-site-verification" content="DnK9Y6dcDXVP_jobMfbjs85sPnZxBxZnp8VrQst66r8" />

<!--- <img src=https://github.com/thadaJ/autoInsight/blob/master/logo.png width="4.5%" height="4.5%">  --->
# autoInsight 
autoInsight is a library that helps researchers and developers see patterns in data that have two classes at a glance.

## Step1:
copy autoInsight.py to your working directory

requirement: pandas, numpy, xgboost, re, IPython 

## Step2:
Import data and use the library in your jupyter notebook as follows. (XGBoost_Data_Exploration.ipynb)

```python
import autoInsight as ai
import pandas as pd
import numpy as np
dataset = pd.read_csv('framingham.csv')
dataset['male'] = dataset['male'].astype('bool')
ai.binary_auto_insight(dataset, labelCol = 'TenYearCHD', positive_class = 1)
```


## Examples of Results (from kaggle data for 10-Year Coronary Heart Disease Risk Prediction 'framingham.csv')
These are results created from autoInsight. 
Each feature is chosen automatically with a machine learning algorithm.
![result](https://github.com/thadaJ/autoInsight/blob/master/Example%20of%20result.png)

.
.
.

![result](https://github.com/thadaJ/autoInsight/blob/master/big_table.png)

Note: missing = No information appears on that feature (NA value)

## References
Data: https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset
