# Salary Prediction Portfolio Project (Python)
---
<div>
	<img src="https://github.com/ovokpus/Salary-Prediction/blob/master/images/readmepic.jpeg">
</div>

---
Our goal in this project is to examine this dataset of job postings, and predict salaries for a new set of postings. This will involve building a model to predict the salaries given in the test dataset.

A practical use of this is for a HR Department of a large company or a Consulting Outfit that needs real-time solutions in order to make effective employment offers to potential hires.

It also finds use in getting to understand current realities in the job market and how businesses can leverage this in order to secure high quality talent, whilst keeping hiring costs low.

The primary tool used for this project is Python 3, along with an extensive array of libraries and packages available for the manipulation of data,and development of predictive modeling algorithms.

---

This Project is broken down into the following Sections:

#### Defining the Problem

Our twofold goal in this project is to:

1. Load and Explore Data
2. Develop and train a suitable prediction model


With Python, we created "Data" and "Plots" classes and objects to aid us in extracting, manipulating the given data. We also created "FeatEng" and "ModelEvaluation" Classes to help us with creating new features and selecting the best prediction model for employee salaries.

---

### Dataset Information

The Datasets provided contains 7 features which can help us determine salaries of various job roles. Salary is thus identified as the target value of our prediction.

#### Categorical Features
jobID - Unique for each job entry. Eventually excluded from building the model.
companyID - Unique for 63 companies represented in the Data
jobType - Various job roles and levels within the companies
degree - Level of educational qualifications of employees in those job desctiptions
major - Subject of study in relation to educational qualifications
industry - sector of the economy where company belongs i.e. Oil, Health, etc

#### Numerical Features
yearsExperience - years of experience of employees in each data entry
milesFromMetropolis - distance of employee residence from the nearest large urban center where his office is located

Salary - Our prediction Target 

---
## Exploratory Data Analysis

This involves looking at the Data Summaries and Visualizations in order to:

1. Examine the Data
2. Discover patterns and relationships between the features
3. Identify the types of data
4. Clean up the Data


Below is an overview of the Train Dataframe (after merging the 'train_features' and 'train_target' files):

![alt text](https://github.com/ovokpus/Salary-Prediction-Portfolio/blob/master/images/Train%20Dataframe%20head.jpg)


#### Visualizing Numerical features

We can see from the Distribution and Boxplots below that there is even distribution in the yearsExperience and milesFromMetropolis, whilst the salary feature has a distribution that is close to normal. There is also the presence of outliers in the Salary Boxplot, which requires some attention. This will be dealt with later in the project.

![alt text](https://github.com/ovokpus/Salary-Prediction-Portfolio/blob/master/images/Numerical%20Features%20Visuals.jpg)


#### Visualizing Categorical features

From the Boxplots showing the categorical features(as seen below), we see that:

1. The Senior Job types obviously earn the highest salaries (Strong positive correlation)
2. Advanced Degrees tend to attract higher salaries
3. Engineering, Business, Math and Computer Science are on the top end of the salary continuum
4. The Oil and Finance Industries are the highest paying in the job market represented by this data

![alt text](https://github.com/ovokpus/Salary-Prediction-Portfolio/blob/master/images/Categorical%20Features%20(boxplots).jpg)

The trends spelt out above are corroborated in the Heatmap (Correlation Matrix), which shows jobtype as the most strongly correlated feature with salary. Degree and Major have the strongest positive relationship, understandably:

![alt text](https://github.com/ovokpus/Salary-Prediction-Portfolio/blob/master/images/Correlation%20Matrix.jpg)

#### Examining the Target Variable (Salary)
After inspecting the Salary Distribution Boxplot, we can see the existence of outliers. The lower outlier sits exactly on zero. We used the IQR rule to identify these outliers and then made our determination as follows:

1. The entries above the upper bound (220.5) appear legitimate. Most of those roles are senior roles and therefore the values are realistic. We will leave those values in the train dataset.

2. The entries with zero salary appear faulty, as those positions are apparently not volunteer positions. We will proceed to remove those from the training set.

We'll accomplish this by defining a "clean_data" method when creating the Feature Engineering Class


---
## Feature Engineering

By applying label encoding to transform the categorical features and by grouping them. The group statistics were selected as new features: group_mean(mean salary), group_median(median salary), group_min(minimum salary), group_max(maximum salary), group_mad(mean absolute deviation)

Also we removed the zero salary values with a clean_data method.

---
## Model Selection and Evaluation

We selected three different Regression Algorithms for Evaluation:
1. Linear Regression
2. Random Forest Regression
3. Gradient Boosting Regressor

The Evaluation Metric selected is the MSE (Mean Squared Error).

For our Baseline Model, we used the calculated mean of the target, and included that amongst the models selected for Evaluation. The model with the lowest MSE was selected as the best model.

### Best Model
After passing the models through our evaluation code, we ended up with the following MSE values:

Baseline: 644.26
Linear Regression: 358.15
Random Forest Regressor: 313.27
Gradient Boosting Regressor: 313.06 (selected as best model)

Key predictors for this model is the group mean salary, followed by yearsExperience, as shown in the Feature Importances plot:

![alt text](https://github.com/ovokpus/Salary-Prediction-Portfolio/blob/master/images/Feature%20Importances.jpg)

---
## Conclusion/Next Steps

• Files Saved for further testing/deployment
• We could seek to further improve the model by creating new features around the yearsExperience, and miles from metropolis features
