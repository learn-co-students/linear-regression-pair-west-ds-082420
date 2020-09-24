```python

import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
```

For this exercise we will work through the different steps of a linear regression workflow. The notebook will walk you through building a first simple model and improving upon that model by stepwise iteration.

### 1. First Simple Model
- Load in the dataset: inspect the overall shape, duplicate entries, and NA's.
- Identify the continuous target variable
- Perform initial EDA: correlation plots
- Build a FSM (First Simple Model) with statsmodels
- Interpret coefficients
- Check the assumptions of linear regression  

### 2. Iterate: Build a better model - Add another numerical feature
- Add another feature, and fit the model
- Compare metrics and interpret coefficients
- Check the assumptions

### 3. Iterate: Build a better model - Add a categorical feature
- Add a categorical variable 
- Compare metrics and interpret coefficients
- Check the assumptions once-again

### 4. Conclusion
- Pick your best model, and interpret your findings
- Describe the next steps you would take if you had more time

## The Dataset
We will use a dataset from [Kaggle](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho). It contains information about **used car sale listings**. We are trying to understand the relationships between the various features of the listing and the **price of the car**.

### Features (as described on Kaggle)
 - `Car_Name`: The name of the car
 - `Year`: The year in which the car was bought
 - `Selling_Price`: The price the owner wants to sell the car at
 - `Present_Price`: The current ex-showroom price of the car
 - `Kms_Driven`: The distance completed by the car in km
 - `Fuel_Type`: The fuel type of the car (Petrol, Diesel, or Other)
 - `Seller_Type`: Whether the seller is a dealer or an individual
 - `Transmission`: Whether the car is manual or automatic
 - `Owner`: The number of owners the car has previously had

Looking at the original website, it looks like the **prices are listed in lakhs, meaning hundreds of thousands of rupees**.

The datasets is located in a file called `cars.csv` in the `data` directory.

# 1. FSM

### Load in the dataset and check the overall shape



```python
df = pd.read_csv(os.path.join("data", "cars.csv"))
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Car_Name</th>
      <th>Year</th>
      <th>Selling_Price</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Fuel_Type</th>
      <th>Seller_Type</th>
      <th>Transmission</th>
      <th>Owner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ritz</td>
      <td>2014</td>
      <td>3.35</td>
      <td>5.59</td>
      <td>27000</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sx4</td>
      <td>2013</td>
      <td>4.75</td>
      <td>9.54</td>
      <td>43000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ciaz</td>
      <td>2017</td>
      <td>7.25</td>
      <td>9.85</td>
      <td>6900</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>wagon r</td>
      <td>2011</td>
      <td>2.85</td>
      <td>4.15</td>
      <td>5200</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>swift</td>
      <td>2014</td>
      <td>4.60</td>
      <td>6.87</td>
      <td>42450</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
records, columns = df.shape

print(records, "records")
print(columns, "columns")
```

    301 records
    9 columns



```python
df.duplicated().sum()
```




    2




```python
df.isna().sum()
```




    Car_Name         0
    Year             0
    Selling_Price    0
    Present_Price    0
    Kms_Driven       0
    Fuel_Type        0
    Seller_Type      0
    Transmission     0
    Owner            0
    dtype: int64



### What does a row in the dataframe represent?



```python
"""
Each row represents a car listing for sale
"""
```

### Identify the continous target variable


```python
# Identify the continuous target variable of interest
df['Selling_Price']
```




    0       3.35
    1       4.75
    2       7.25
    3       2.85
    4       4.60
           ...  
    296     9.50
    297     4.00
    298     3.35
    299    11.50
    300     5.30
    Name: Selling_Price, Length: 301, dtype: float64




```python
cols = list(df.columns)
cols = [cols[2]] + cols[:2] + cols[3:]
df = df[cols]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Selling_Price</th>
      <th>Car_Name</th>
      <th>Year</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Fuel_Type</th>
      <th>Seller_Type</th>
      <th>Transmission</th>
      <th>Owner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.35</td>
      <td>ritz</td>
      <td>2014</td>
      <td>5.59</td>
      <td>27000</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.75</td>
      <td>sx4</td>
      <td>2013</td>
      <td>9.54</td>
      <td>43000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.25</td>
      <td>ciaz</td>
      <td>2017</td>
      <td>9.85</td>
      <td>6900</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.85</td>
      <td>wagon r</td>
      <td>2011</td>
      <td>4.15</td>
      <td>5200</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.60</td>
      <td>swift</td>
      <td>2014</td>
      <td>6.87</td>
      <td>42450</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Understanding the Target Variable


```python
# Revisit the continuous target variable.  
# Explore it a bit.  Plot its distribution and boxplot

fig, ax = plt.subplots(2,1, figsize=(10,5))
sns.distplot(df.Selling_Price, ax = ax[0])
sns.boxplot(df.Selling_Price, ax= ax[1])

ax[0].set_title('Distribution of Target Variable: Selling Price');
```


![png](index_files/index_16_0.png)



```python
df.sort_values('Selling_Price', ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Selling_Price</th>
      <th>Car_Name</th>
      <th>Year</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Fuel_Type</th>
      <th>Seller_Type</th>
      <th>Transmission</th>
      <th>Owner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86</th>
      <td>35.00</td>
      <td>land cruiser</td>
      <td>2010</td>
      <td>92.60</td>
      <td>78000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>33.00</td>
      <td>fortuner</td>
      <td>2017</td>
      <td>36.23</td>
      <td>6000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Automatic</td>
      <td>0</td>
    </tr>
    <tr>
      <th>63</th>
      <td>23.50</td>
      <td>fortuner</td>
      <td>2015</td>
      <td>35.96</td>
      <td>47000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Automatic</td>
      <td>0</td>
    </tr>
    <tr>
      <th>82</th>
      <td>23.00</td>
      <td>innova</td>
      <td>2017</td>
      <td>25.39</td>
      <td>15000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Automatic</td>
      <td>0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>23.00</td>
      <td>fortuner</td>
      <td>2015</td>
      <td>30.61</td>
      <td>40000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Automatic</td>
      <td>0</td>
    </tr>
    <tr>
      <th>93</th>
      <td>23.00</td>
      <td>fortuner</td>
      <td>2015</td>
      <td>30.61</td>
      <td>40000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Automatic</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>20.75</td>
      <td>innova</td>
      <td>2016</td>
      <td>25.39</td>
      <td>29000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Automatic</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>19.99</td>
      <td>fortuner</td>
      <td>2014</td>
      <td>35.96</td>
      <td>41000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Automatic</td>
      <td>0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>19.75</td>
      <td>innova</td>
      <td>2017</td>
      <td>23.15</td>
      <td>11000</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Automatic</td>
      <td>0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>18.75</td>
      <td>fortuner</td>
      <td>2014</td>
      <td>35.96</td>
      <td>78000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Automatic</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f'''
Significantly left skewed, with a long tail.  

Mean of {round(df.Selling_Price.mean(),2)}
Median of {round(df.Selling_Price.median(),2)}
Skew: {round(stats.skew(df.Selling_Price.dropna()), 2)}
''')
```

    
    Significantly left skewed, with a long tail.  
    
    Mean of 4.66
    Median of 3.6
    Skew: 2.48
    


## Perform Initial EDA

Let's look at a correlation matrix to see which of these variables might be the most useful.  (Here we are looking for variables that are highly correlated with the target variable, but not highly correlated with other input variables.) This only includes the numeric values.


```python
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Selling_Price</th>
      <th>Year</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Owner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Selling_Price</th>
      <td>1.000000</td>
      <td>0.236141</td>
      <td>0.878983</td>
      <td>0.029187</td>
      <td>-0.088344</td>
    </tr>
    <tr>
      <th>Year</th>
      <td>0.236141</td>
      <td>1.000000</td>
      <td>-0.047584</td>
      <td>-0.524342</td>
      <td>-0.182104</td>
    </tr>
    <tr>
      <th>Present_Price</th>
      <td>0.878983</td>
      <td>-0.047584</td>
      <td>1.000000</td>
      <td>0.203647</td>
      <td>0.008057</td>
    </tr>
    <tr>
      <th>Kms_Driven</th>
      <td>0.029187</td>
      <td>-0.524342</td>
      <td>0.203647</td>
      <td>1.000000</td>
      <td>0.089216</td>
    </tr>
    <tr>
      <th>Owner</th>
      <td>-0.088344</td>
      <td>-0.182104</td>
      <td>0.008057</td>
      <td>0.089216</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots()
sns.heatmap(df.corr(), ax=ax);
```


![png](index_files/index_21_0.png)



```python
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))

fig, ax = plt.subplots()
sns.heatmap(df.corr(), mask=mask, ax=ax);
```


![png](index_files/index_22_0.png)


### Use seaborn's pairplot function on the features above

There are only 5 numeric features, so this shouldn't be _too_ slow


```python
sns.pairplot(df);
```


![png](index_files/index_24_0.png)


Judging from this pairplot (either the top row, or the left column), the closest to a linear relationship is `Present_Price`. This also happens to be the feature with the highest correlation to `Selling_Price`.

This makes sense, that the original price of the car, and the listed price for that car when it's used, would be highly correlated.

Given these insights, let's use `Present_Price` to develop the First Simple Model (FSM), with one target and one predictor.

## FSM with Statsmodels



```python
fsm_df = df[["Selling_Price", "Present_Price"]].copy()
```


```python
formula = "Selling_Price ~ Present_Price"
```


```python
fsm = ols(formula=formula, data=fsm_df).fit()
```


```python
fsm.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>Selling_Price</td>  <th>  R-squared:         </th> <td>   0.773</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.772</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1016.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 24 Sep 2020</td> <th>  Prob (F-statistic):</th> <td>3.60e-98</td>
</tr>
<tr>
  <th>Time:</th>                 <td>01:18:12</td>     <th>  Log-Likelihood:    </th> <td> -693.08</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   301</td>      <th>  AIC:               </th> <td>   1390.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   299</td>      <th>  BIC:               </th> <td>   1398.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td>    0.7185</td> <td>    0.187</td> <td>    3.847</td> <td> 0.000</td> <td>    0.351</td> <td>    1.086</td>
</tr>
<tr>
  <th>Present_Price</th> <td>    0.5168</td> <td>    0.016</td> <td>   31.874</td> <td> 0.000</td> <td>    0.485</td> <td>    0.549</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>59.775</td> <th>  Durbin-Watson:     </th> <td>   1.533</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 926.121</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 0.084</td> <th>  Prob(JB):          </th> <td>7.86e-202</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>11.592</td> <th>  Cond. No.          </th> <td>    15.4</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python

rsquared = fsm.rsquared
params = fsm.params

print(f'Rsquared of FSM: {rsquared}')
print('----------')
print('Beta values of FSM:')
print(params)
              
```

    Rsquared of FSM: 0.7726103146985799
    ----------
    Beta values of FSM:
    Intercept        0.718527
    Present_Price    0.516849
    dtype: float64


Interpret the result of the FSM.  What does the R Squared tell you? Remember the formula for:

$\Large R^2 = 1 - \frac{SSE}{SST}$

Also, interepret the coefficients.  If we increase the value of our independent variable by 1, what does it mean for our predicted value?

What will our model predict the value of sale price to be for a car originally worth 0 lakhs? (This doesn't necessarily make sense.)


```python
'''
Our R_2 is not too bad. We are explaining about 77% of the variance
in selling price, but we only have one feature so far and it's statistically significant
at an alpha of 0.05.

We could stop right now and say that according to our model:

 - A car originally worth zero lakhs, we expect the selling price to be about 0.7 lakhs
 - For each additional lakh of Present_Price, we expect the selling price to increase
   by about 0.5 lakhs
 
'''
```

# Check the assumptions of Linear Regression

#### 1. Linearity

Linear regression assumes that the input variable linearly predicts the output variable.  We already qualitatively checked that with a scatter plot.  But it's also a good idea to use a statistical test.  This one is the [Rainbow test](https://www.tandfonline.com/doi/abs/10.1080/03610928208828423) which is available from the [diagnostic submodule of StatsModels](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.linear_rainbow.html#statsmodels.stats.diagnostic.linear_rainbow)

1a) What are the null and alternative hypotheses for the linear rainbow test?


```python
"""
Null hypothesis: the model is linearly predicted by the features
Alternative hypothesis: the model is not linearly predicted by the features
"""
```

1b) Run a statistical test for linearity (we've included the import below)


```python
from statsmodels.stats.diagnostic import linear_rainbow

rainbow_statistic, rainbow_p_value = linear_rainbow(fsm)
print("Rainbow statistic:", rainbow_statistic)
print("Rainbow p-value:", rainbow_p_value)
```

    Rainbow statistic: 1.0514089002232407
    Rainbow p-value: 0.38004596704813104


1c) Interpret the results. Can we reject the null hypothesis? (You can assume an alpha of 0.05.) What does this mean for the assumptions of linear regression?


```python
"""
Returning a p-value above .05 means that we fail to reject the null hypothesis.
The current model does not seem to violate the linearity assumption.
"""
```

#### 2. Normality

Linear regression assumes that the residuals are normally distributed.  It is possible to check this qualitatively with a Q-Q plot.  The fit model object has an attribute called `resid`, which is an array of the difference between predicted and true values.  Store the residuals in the variable below, show the qq plot, and interepret. You are looking for the theoretical quantiles and the sample quantiles to line up.


```python
# Create a qq-plot

fsm_resids = fsm.resid

sm.qqplot(fsm_resids);
```


![png](index_files/index_42_0.png)


Those qqplots don't look so good in the upper right corner. To pass a visual test, the qq should be a straight line.

The [Jarque-Bera](https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test) test is performed automatically as part of the model summary output, labeled **Jarque-Bera (JB)** and **Prob(JB)**.

The null hypothesis is that the residuals are normally distributed, alternative hypothesis is that they are not.  
What does the JB score output indicate. Does it support the qq-plot?


```python
'''
The JB score has a low p-value means that the current model violates the
normality assumption. 
That supports the qq visual with the crooked tail.
'''
```

#### 3. Homoscadasticity

Linear regression assumes that the variance of the dependent variable is homogeneous across different values of the independent variable(s).  We can visualize this by looking at the predicted life expectancy vs. the residuals.




```python

y_hat = fsm.predict()
```


```python

fig, ax = plt.subplots()
ax.set(xlabel="Predicted Selling Price",
        ylabel="Residuals (Actual - Predicted Selling Price")
ax.scatter(y_hat, fsm_resids);
```


![png](index_files/index_48_0.png)


Interepret the result. Do you see any patterns that suggest that the residuals exhibit heteroscedasticity?



```python
'''
It looks like we are worse at predicting the price as the predicted price
increases. Closer to 0, we get better results.  Then we have more outliers
as we exceed 10 lakhs
'''
```

Let's also run a statistical test.  The [Breusch-Pagan test](https://en.wikipedia.org/wiki/Breusch%E2%80%93Pagan_test) is available from the [diagnostic submodule of StatsModels](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html#statsmodels.stats.diagnostic.het_breuschpagan)


```python
from statsmodels.stats.diagnostic import het_breuschpagan
lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(fsm_resids, fsm_df[["Present_Price"]])
print("Lagrange Multiplier p-value:", lm_p_value)
print("F-statistic p-value:", f_p_value)
```

    Lagrange Multiplier p-value: nan
    F-statistic p-value: 1.3905016333457367e-45


The null hypothesis is homoscedasticity, alternative hypothesis is heteroscedasticity.  
What does the p-value returned above indicate? Can you reject the null hypothesis?


```python
'''
Returning a low p-value means that we can reject the null hypothesis,
and the current model violates the homoscedasticity assumption
'''
```

#### 4. Independence

The independence assumption means that the independent variables must not be too collinear.  Right now we have only one independent variable, so we don't need to check this yet.

## Train a model with `sklearn`
The `sklearn` interface is simpler than StatsModels, but it does not give us the super helpful StatsModels output.  We will, however, use its syntax consistently with other algorithms.

You can skip this step if you are short on time, since it is more relevant for Phase 3 than Phase 2


```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
```


```python
y = fsm_df.Selling_Price
X = fsm_df.drop('Selling_Price', axis=1)
```


```python
lr.fit(X, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# score is the r-squared. It should be the same as StatsModels r-squared
score = lr.score(X, y)
score
```




    0.7726103146985799




```python

#sklearn calculates the same coeficients and intercepts as statmodels.
beta = lr.coef_
intercept = lr.intercept_

print(beta)
print(intercept)
```

    [0.51684903]
    0.7185274709817686


# 2. Iterate: Build a better model - Add another numerical feature

## Adding Features to the Model

So far, all we have is a simple linear regression.  Let's start adding features to make it a multiple regression.

Let's try `Year`. Maybe in reality it should be a categorical variable, but it looks like there's a general trend where the later the year, the higher the price


```python
fig, ax = plt.subplots(figsize=(10, 5))

sns.catplot(x="Year", y="Selling_Price", data=df, ax=ax, kind="box")
plt.close(2); # closing the extra axis created by sns
```


![png](index_files/index_64_0.png)



```python
model_2_df = df[["Selling_Price", "Present_Price", "Year"]].copy()
```


```python
formula = 'Selling_Price ~ Present_Price + Year'
```


```python
model_2 = ols(formula=formula, data=model_2_df).fit()
```


```python
model_2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>Selling_Price</td>  <th>  R-squared:         </th> <td>   0.850</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.849</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   844.7</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 24 Sep 2020</td> <th>  Prob (F-statistic):</th> <td>1.64e-123</td>
</tr>
<tr>
  <th>Time:</th>                 <td>01:20:27</td>     <th>  Log-Likelihood:    </th> <td> -630.42</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   301</td>      <th>  AIC:               </th> <td>   1267.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   298</td>      <th>  BIC:               </th> <td>   1278.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td> -985.4594</td> <td>   79.494</td> <td>  -12.397</td> <td> 0.000</td> <td>-1141.900</td> <td> -829.019</td>
</tr>
<tr>
  <th>Present_Price</th> <td>    0.5246</td> <td>    0.013</td> <td>   39.731</td> <td> 0.000</td> <td>    0.499</td> <td>    0.551</td>
</tr>
<tr>
  <th>Year</th>          <td>    0.4897</td> <td>    0.039</td> <td>   12.406</td> <td> 0.000</td> <td>    0.412</td> <td>    0.567</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>67.412</td> <th>  Durbin-Watson:     </th> <td>   1.526</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>1300.900</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 0.186</td> <th>  Prob(JB):          </th> <td>3.26e-283</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>13.178</td> <th>  Cond. No.          </th> <td>1.41e+06</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.41e+06. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



### Did the r_2 improve? 


```python
'Adding another feature improved the r-squared from 0.773 to 0.850'
```

### Now check the assumptions like we did above.

#### Linearity


```python
rainbow_statistic, rainbow_p_value = linear_rainbow(model_2)
print("Rainbow statistic:", rainbow_statistic)
print("Rainbow p-value:", rainbow_p_value)

"""Our p-value is higher now, which means we are doing slightly 
better in terms of not violating the linearity assumption"""
```

    Rainbow statistic: 0.7530735703620426
    Rainbow p-value: 0.9578920893954007





    'Our p-value is higher now, which means we are doing slightly \nbetter in terms of not violating the linearity assumption'



#### Normality


```python
'''
The Jarque-Bera (JB) output has gotten worse. We are still violating the 
normality assumption.'''
```

#### Homoscedasticity


```python
y_hat = model_2.predict()
model_2_resids = model_2.resid

fig, ax = plt.subplots()
ax.set(xlabel="Predicted Selling Price",
        ylabel="Residuals (Actual - Predicted Selling Price")
ax.scatter(y_hat, model_2_resids);
```


![png](index_files/index_76_0.png)



```python
lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(y-y_hat, model_2_df[
    ["Present_Price", "Year"]])
print("Lagrange Multiplier p-value:", lm_p_value)
print("F-statistic p-value:", f_p_value)

'''Both visually and numerically, we can see some improvement. 
But we are still violating this assumption to a statistically significant degree.'''
```

    Lagrange Multiplier p-value: 3.1809369908285176e-34
    F-statistic p-value: 5.3435608431021095e-45





    'Both visually and numerically, we can see some improvement. \nBut we are still violating this assumption to a statistically significant degree.'



## Independence

You might have noticed in the regression output that there was a warning about the condition number being high. The condition number is a measure of stability of the matrix used for computing the regression (we'll discuss this more in the next module), and a number above 30 can indicate strong multicollinearity. Our output is way higher than that.

A different (more generous) measure of multicollinearity is the variance inflation factor. It is available from the outlier influence submodule of StatsModels.

Run the code below:


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
rows = model_2_df[["Present_Price", "Year"]].values

vif_df = pd.DataFrame()
vif_df["VIF"] = [variance_inflation_factor(rows, i) for i in range(2)]
vif_df["feature"] = ["Present_Price", "Year"]

vif_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF</th>
      <th>feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.781193</td>
      <td>Present_Price</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.781193</td>
      <td>Year</td>
    </tr>
  </tbody>
</table>
</div>



A "rule of thumb" for VIF is that 5 is too high.  Given the output above, it's reasonable to say that we are not violating the independence assumption, despite the high condition number.

### 3. Iterate: Build a better model - Add a categorical feature


Rather than adding any more numeric features (e.g. `Year`, `Owner`), let's add a categorical one. Out of `Seller_Type` and `Transmission`, which one looks better?


```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8))

sns.catplot(y="Selling_Price", x="Seller_Type", data=df, ax=ax1, kind="box")
plt.close(2)
sns.catplot(y="Selling_Price", x="Transmission", data=df, ax=ax2, kind="box")
plt.close(2);
```


![png](index_files/index_83_0.png)


It looks like `Seller Type` has more separation between the two classes, let's go with that


```python
df["Seller_Type"].value_counts()
```




    Dealer        195
    Individual    106
    Name: Seller_Type, dtype: int64




```python
# Create a dataframe with the necessary columns
model_3_df = df[["Selling_Price", "Present_Price", "Year", "Seller_Type"]].copy()
```

There are only two categories, so we only need a `LabelEncoder` that will convert the labels into 1s and 0s.  If there were more than two categories, we would use a `OneHotEncoder`, which would create multiple columns out of a single column.


```python
from sklearn.preprocessing import LabelEncoder

# instantiate and instance of LabelEncoder
label_encoder = LabelEncoder()
```


```python

# Pass the "Seller_Type" column of the model_3_df to the fit_transform()
# method of the Label Encoder
seller_labels = label_encoder.fit_transform(model_3_df["Seller_Type"])
```


```python
# Run the code below.  The category Dealer/Individual has been transformed to a binary
np.unique(seller_labels, return_counts=True)
```




    (array([0, 1]), array([195, 106]))




```python
# Run the code below to see the classes associated with 1 and 0
label_encoder.classes_
```




    array(['Dealer', 'Individual'], dtype=object)



This is telling us that "Dealer" is encoded as 0 and "Individual" is encoded as 1.  This means that "Dealer" is assumed at the intercept.


```python
# Add the seller labels array to the model_df as a column 
model_3_df["Seller_Encoded"] = seller_labels
```


```python
model_3_df.drop("Seller_Type", axis=1, inplace=True)
```

#### Fit the 3rd Model


```python
formula="Selling_Price~" + "+".join(model_3_df.iloc[:,1:].columns)
formula
```




    'Selling_Price~Present_Price+Year+Seller_Encoded'




```python
model_3 = ols(formula=formula, data=model_3_df).fit()
```


```python
model_3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>Selling_Price</td>  <th>  R-squared:         </th> <td>   0.859</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.858</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   605.1</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 24 Sep 2020</td> <th>  Prob (F-statistic):</th> <td>3.88e-126</td>
</tr>
<tr>
  <th>Time:</th>                 <td>01:22:43</td>     <th>  Log-Likelihood:    </th> <td> -620.74</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   301</td>      <th>  AIC:               </th> <td>   1249.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   297</td>      <th>  BIC:               </th> <td>   1264.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>      <td> -959.0629</td> <td>   77.338</td> <td>  -12.401</td> <td> 0.000</td> <td>-1111.264</td> <td> -806.862</td>
</tr>
<tr>
  <th>Present_Price</th>  <td>    0.4905</td> <td>    0.015</td> <td>   32.825</td> <td> 0.000</td> <td>    0.461</td> <td>    0.520</td>
</tr>
<tr>
  <th>Year</th>           <td>    0.4770</td> <td>    0.038</td> <td>   12.421</td> <td> 0.000</td> <td>    0.401</td> <td>    0.553</td>
</tr>
<tr>
  <th>Seller_Encoded</th> <td>   -1.1983</td> <td>    0.270</td> <td>   -4.440</td> <td> 0.000</td> <td>   -1.729</td> <td>   -0.667</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>99.918</td> <th>  Durbin-Watson:     </th> <td>   1.632</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>1181.252</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 0.980</td> <th>  Prob(JB):          </th> <td>3.12e-257</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>12.505</td> <th>  Cond. No.          </th> <td>1.41e+06</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.41e+06. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



### Third Model Evaluation

Did the R_squared improve?


```python
# Did the R_squared improve
"Adding another feature improved the r-squared a tiny bit from 0.850 to 0.859"
```

# Let's look at the model assumptions again

#### Linearity, Normality, Homoscedasticity, Independence

For each, are we violating the assumption? Have we improved from the previous model?


```python
#### Linearity
rainbow_statistic, rainbow_p_value = linear_rainbow(model_3)
print("Rainbow statistic:", rainbow_statistic)
print("Rainbow p-value:", rainbow_p_value)
```

    Rainbow statistic: 0.9294842106835415
    Rainbow p-value: 0.6720954478491306



```python
#### Normality
'''
The **Jarque-Bera (JB)** output has gotten slightly better.  
But we are still violating the normality assumption.
'''
```


```python
#### Homoscedasticity
model_3_resids = model_3.resid
y_hat = model_3.predict()

fig, ax = plt.subplots()

ax.scatter(y_hat, model_3_resids);
```


![png](index_files/index_105_0.png)



```python
lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(y-y_hat, model_3_df[
    ["Present_Price", "Year", "Seller_Encoded"]])
print("Lagrange Multiplier p-value:", lm_p_value)
print("F-statistic p-value:", f_p_value)
```

    Lagrange Multiplier p-value: 8.273933434285868e-33
    F-statistic p-value: 2.0311190051686055e-43



```python
'''This metric got worse, although the plot looks fairly similar'''
```


```python
#### Independence
rows = model_3_df[["Present_Price", "Year", "Seller_Encoded"]].values

vif_df = pd.DataFrame()
vif_df["VIF"] = [variance_inflation_factor(rows, i) for i in range(3)]
vif_df["feature"] = ["Present_Price", "Year", "Seller_Encoded"]

vif_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF</th>
      <th>feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.413593</td>
      <td>Present_Price</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.698835</td>
      <td>Year</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.091452</td>
      <td>Seller_Encoded</td>
    </tr>
  </tbody>
</table>
</div>




```python
"""The VIF metrics are getting higher, which means that there is stronger multicollinearity.  
But we have still not exceeded the threshold of 5."""
```

## Conclusion

Choose a model out of `fsm`, `model_2`, and `model_3` and declare it as your final model. How well does this model represent the target variable?  What can we say about car listing prices based on these variables?  What coefficients, if any, are statistically significant?


```python
final_model = model_3

model_3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>Selling_Price</td>  <th>  R-squared:         </th> <td>   0.859</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.858</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   605.1</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 24 Sep 2020</td> <th>  Prob (F-statistic):</th> <td>3.88e-126</td>
</tr>
<tr>
  <th>Time:</th>                 <td>01:23:11</td>     <th>  Log-Likelihood:    </th> <td> -620.74</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   301</td>      <th>  AIC:               </th> <td>   1249.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   297</td>      <th>  BIC:               </th> <td>   1264.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>      <td> -959.0629</td> <td>   77.338</td> <td>  -12.401</td> <td> 0.000</td> <td>-1111.264</td> <td> -806.862</td>
</tr>
<tr>
  <th>Present_Price</th>  <td>    0.4905</td> <td>    0.015</td> <td>   32.825</td> <td> 0.000</td> <td>    0.461</td> <td>    0.520</td>
</tr>
<tr>
  <th>Year</th>           <td>    0.4770</td> <td>    0.038</td> <td>   12.421</td> <td> 0.000</td> <td>    0.401</td> <td>    0.553</td>
</tr>
<tr>
  <th>Seller_Encoded</th> <td>   -1.1983</td> <td>    0.270</td> <td>   -4.440</td> <td> 0.000</td> <td>   -1.729</td> <td>   -0.667</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>99.918</td> <th>  Durbin-Watson:     </th> <td>   1.632</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>1181.252</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 0.980</td> <th>  Prob(JB):          </th> <td>3.12e-257</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>12.505</td> <th>  Cond. No.          </th> <td>1.41e+06</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.41e+06. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
"""
This model, with three features, explains about 85% of the variance
in the selling price

Those three features are Present_Price (the showroom price of the car),
Year (the year the car was made), and Seller_Encoded (a dummy variable
indicating whether the car listing was from a dealer or an individual).

The intercept (statistically significant coefficient) is around -950.
This means that a car being sold originally for 0 lakh, built in year 0,
and being sold by a dealer, would be -950 lakhs. Obviously it would be
very unlikely for this particular car to exist, so we won't read a lot
into this intercept term.

The Present_Price coefficient (statistically significant) is around 0.5,
meaning an increase of 1 lakh in present price is associated with
an increase of 0.5 lakhs in used car selling price

The Year coefficient (statistically significant) is also around 0.5,
meaning an increase of 1 year when the car was built (i.e. the car being
one year newer) is associated with an increase of 0.5 lakhs in used car
selling price

The Seller_Encoded coefficient (statistically significant) is about -1.2,
meaning that a car being sold by an individual (rather than a dealership)
is associated with a decrease of 1.2 lakhs in used car selling price
"""
```
