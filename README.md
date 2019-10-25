# autoInsight

## Step1:
copy autoInsight.py to your working directory

## Step2:
Import data and use the library as follows. (XGBoost_Data_Exploration.ipynb)

```python
import autoInsight as ai
import pandas as pd
dataset = pd.read_csv('framingham.csv')
ai.autoInsight(dataset, labelCol = 'TenYearCHD', positive_class = 1)
```


## Examples of Results (from data for cardiovascular disease (CVD) risk estimator 'framingham.csv')
This is results created from autoInsight. 
Each feature is chosen automatically with a machine learning algorithm.
![result](https://github.com/thadaJ/autoInsight/blob/master/Example%20of%20result.png)

5
![result](https://github.com/thadaJ/autoInsight/blob/master/big_table.png)
