# Salary Prediction Project (Python)

Our goal in this project is to examine this dataset of job postings, and predict salaries for a new set of postings. This will involve building a model to predict the salaries given in the test dataset.

A practical use of this is for a HR Department of a large company or a Consulting Outfit that needs real-time solutions in order to make effective employment offers to potential hires.

It also finds use in getting to understand current realities in the job market and how businesses can leverage this in order to secure high quality talent, whilst keeping hiring costs low.

The primary tool used for this project is Python 3, along with an extensive array of libraries and packages available for the manipulation of data,and development of predictive modeling algorithms.

This Project is broken down into the following Sections:

Defining the Problem
Creating "Data" and "Plots" classes and objects to aid us in extracting, manipulating the given data
Exploratory Data Analysis
This involves looking at the Data Summaries and Visualizations in order to:

1. Examine the Data
2. Discover patterns and relationships between the features
3. Identify the types of data
4. Clean up the Data

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

### Exploratory Data Analysis

Below is an overview of the Train Dataframe (after merging the 'train_features' and 'train_target' files):


