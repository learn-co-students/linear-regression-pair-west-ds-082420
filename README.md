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
# load in the dataset
df = None
```


```python
df.head()
```


```python
# How many records and columns are in the data set?
records, columns = None

print(records, "records")
print(columns, "columns")
```


```python
# Check for duplicate entries
# Your answer here
```


```python
# Check for na's (just look to get an idea; don't drop or impute yet)
# Your answer here
```

### What does a row in the dataframe represent?



```python
# Your answer here
```

### Identify the continous target variable


```python
# Your answer here
```


```python
# To make things easier to interpet, set the target to column index 0

```

### Understanding the Target Variable


```python
# Revisit the continuous target variable.  
# Explore it a bit.  Plot a histogram of its distribution as well as a boxplot
```


```python
# What are the 10 most expensive listings?
```


```python
# Describe the distribution of the target
```

## Perform Initial EDA

Let's look at a correlation matrix to see which of these variables might be the most useful.  (Here we are looking for variables that are highly correlated with the target variable, but not highly correlated with other input variables.) This only includes the numeric values.


```python
# create a correlation matrix
# first, just use the dataframe .corr() method to output a numerical matrix

# Your answer here
```


```python
# Then pass the above code into Seaborn's heatmap plot

# Your answer here
```


```python
# Try adding the code in this cell to the mask attribute in the heatmap to halve the plot
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))

# Your answer here
```

### Use seaborn's pairplot function on the features above

There are only 5 numeric features, so this shouldn't be _too_ slow


```python
# your code here
```

Judging from this pairplot (either the top row, or the left column), the closest to a linear relationship is `Present_Price`. This also happens to be the feature with the highest correlation to `Selling_Price`.

This makes sense, that the original price of the car, and the listed price for that car when it's used, would be highly correlated.

Given these insights, let's use `Present_Price` to develop the First Simple Model (FSM), with one target and one predictor.

## FSM with Statsmodels



```python
# Create a dataframe with only the target and the chosen
# high-positive correlation feature
fsm_df = None
```


```python
# Build the R-style formula.
# The format is "target ~ feature_1 + feature_2 + feature_3"
formula = None
```


```python
# Fit the model on the dataframe composed of the two features
fsm = ols(formula=formula, data=fsm_df).fit()
```


```python
# Use the summary() method on the fsm variable to print out the results of the fit.
fsm.summary()
```


```python
# The object also has attributes associated with the ouput, such as: rsquared, and params.
# save those values to the variables below.

rsquared = None
params = None

print(f'Rsquared of FSM: {rsquared}')
print('----------')
print('Beta values of FSM:')
print(params)
```

Interpret the result of the FSM.  What does the R Squared tell you? Remember the formula for:

$\Large R^2 = 1 - \frac{SSE}{SST}$

Also, interepret the coefficients.  If we increase the value of our independent variable by 1, what does it mean for our predicted value?

What will our model predict the value of sale price to be for a car originally worth 0 lakhs? (This doesn't necessarily make sense.)


```python
# Your answer here
```

# Check the assumptions of Linear Regression

#### 1. Linearity

Linear regression assumes that the input variable linearly predicts the output variable.  We already qualitatively checked that with a scatter plot.  But it's also a good idea to use a statistical test.  This one is the [Rainbow test](https://www.tandfonline.com/doi/abs/10.1080/03610928208828423) which is available from the [diagnostic submodule of StatsModels](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.linear_rainbow.html#statsmodels.stats.diagnostic.linear_rainbow)

1a) What are the null and alternative hypotheses for the linear rainbow test?


```python
# Your answer here
```

1b) Run a statistical test for linearity (we've included the import below)


```python
from statsmodels.stats.diagnostic import linear_rainbow

# Your code here
```

1c) Interpret the results. Can we reject the null hypothesis? (You can assume an alpha of 0.05.) What does this mean for the assumptions of linear regression?


```python
# Your answer here
```

#### 2. Normality

Linear regression assumes that the residuals are normally distributed.  It is possible to check this qualitatively with a Q-Q plot.  The fit model object has an attribute called `resid`, which is an array of the difference between predicted and true values.  Store the residuals in the variable below, show the qq plot, and interepret. You are looking for the theoretical quantiles and the sample quantiles to line up.


```python
# Create a qq-plot

fsm_resids = None

sm.qqplot(fsm_resids);
```

Those qqplots don't look so good in the upper right corner. To pass a visual test, the qq should be a straight line.

The [Jarque-Bera](https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test) test is performed automatically as part of the model summary output, labeled **Jarque-Bera (JB)** and **Prob(JB)**.

The null hypothesis is that the residuals are normally distributed, alternative hypothesis is that they are not.  
What does the JB score output indicate. Does it support the qq-plot?


```python
# Your answer here
```

#### 3. Homoscadasticity

Linear regression assumes that the variance of the dependent variable is homogeneous across different values of the independent variable(s).  We can visualize this by looking at the predicted life expectancy vs. the residuals.




```python
# Use the predict() method now available to be called from the fsm variable 
# to store the predictions
y_hat = None
```


```python
# plot y_hat against the residuals (stored in fsm_resids) in a scatter plot

# Your code here
```

Interepret the result. Do you see any patterns that suggest that the residuals exhibit heteroscedasticity?



```python
# Your answer here
```

Let's also run a statistical test.  The [Breusch-Pagan test](https://en.wikipedia.org/wiki/Breusch%E2%80%93Pagan_test) is available from the [diagnostic submodule of StatsModels](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html#statsmodels.stats.diagnostic.het_breuschpagan)


```python
from statsmodels.stats.diagnostic import het_breuschpagan
lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(fsm_resids, fsm_df[["Present_Price"]])
print("Lagrange Multiplier p-value:", lm_p_value)
print("F-statistic p-value:", f_p_value)
```

The null hypothesis is homoscedasticity, alternative hypothesis is heteroscedasticity.  
What does the p-value returned above indicate? Can you reject the null hypothesis?


```python
# Your answer here
```

#### 4. Independence

The independence assumption means that the independent variables must not be too collinear.  Right now we have only one independent variable, so we don't need to check this yet.

## Train a model with `sklearn`
The `sklearn` interface is simpler than StatsModels, but it does not give us the super helpful StatsModels output.  We will, however, use its syntax consistently with other algorithms.

You can skip this step if you are short on time, since it is more relevant for Phase 3 than Phase 2


```python
from sklearn.linear_model import LinearRegression

# instantiate a linear regression object 
lr = None
```


```python
# split the data into target and features
y = None
X = None
```


```python
# Call .fit from the linear regression object, and feed X and y in as parameters
# Your code here
```


```python
# lr has a method called score.  Again, feed in X and y, and read the output. 
# Save it in the variable score. What is that number?  Compare it to statsmodels. 
score = None
score
```


```python
# lr also has attributes coef_ and intercept_. Save and compare to statsmodels
beta = None
intercept = None

print(beta)
print(intercept)
```

# 2. Iterate: Build a better model - Add another numerical feature

## Adding Features to the Model

So far, all we have is a simple linear regression.  Let's start adding features to make it a multiple regression.

Let's try `Year`. Maybe in reality it should be a categorical variable, but it looks like there's a general trend where the later the year, the higher the price


```python
fig, ax = plt.subplots(figsize=(10, 5))

sns.catplot(x="Year", y="Selling_Price", data=df, ax=ax, kind="box")
plt.close(2); # closing the extra axis created by sns
```


```python
# Create another dataframe containing our three features of interest
model_2_df = None
```


```python
# save the R-like formula into the variable
formula = None
```


```python
# train the model like we did above
model_2 = None
```


```python
# print out the summary table
# Your code here
```

### Did the r_2 improve? 


```python
# Your answer here
```

### Now check the assumptions like we did above.

#### Linearity


```python
# Your answer here (code and interpretation)
```

#### Normality


```python
# Your answer here (interpretation of output from model summary)
```

#### Homoscedasticity


```python
# Your answer here (code and interpretation)
```

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

It looks like `Seller Type` has more separation between the two classes, let's go with that


```python
df["Seller_Type"].value_counts()
```


```python
# We have created a dataframe with the necessary columns
model_3_df = df[["Selling_Price", "Present_Price", "Year", "Seller_Type"]].copy()
```

There are only two categories, so we only need a `LabelEncoder` that will convert the labels into 1s and 0s.  If there were more than two categories, we would use a `OneHotEncoder`, which would create multiple columns out of a single column.


```python
from sklearn.preprocessing import LabelEncoder

# instantiate an instance of LabelEncoder
label_encoder = None
```


```python
# Pass the "Seller_Type" column of the model_3_df to the fit_transform() 
# method of the Label Encoder
seller_labels = None
```


```python
# Run the code below.  The category Dealer/Individual has been transformed to a binary
np.unique(seller_labels, return_counts=True)
```


```python
# Run the code below to see the classes associated with 1 and 0
label_encoder.classes_
```

This is telling us that "Dealer" is encoded as 0 and "Individual" is encoded as 1.  This means that "Dealer" is assumed at the intercept.


```python
# Add the seller labels array to the model_df as a column 
model_3_df["Seller_Encoded"] = None
```


```python
# Drop the Seller_Type column

# your code here
```

#### Fit the 3rd Model


```python
# assign the new formula

formula=None
```


```python
# fit the new model
model_3 = None
```


```python
# print the summary
model_3.summary()
```

### Third Model Evaluation

Did the R_squared improve?


```python
# Your answer here
```

# Let's look at the model assumptions again

#### Linearity, Normality, Homoscedasticity, Independence

For each, are we violating the assumption? Have we improved from the previous model?


```python

```


```python

```


```python

```


```python

```


```python

```


```python
#_SOLUTION__
"""This is slightly worse than previous, but is still not violating the assumption"""
```

## Conclusion

Choose a model out of `fsm`, `model_2`, and `model_3` and declare it as your final model. How well does this model represent the target variable?  What can we say about car listing prices based on these variables?  What coefficients, if any, are statistically significant?


```python

```


```python

```


```python

```
