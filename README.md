



* [Python](python)
* [Flask](flask)


This is the list of things you need to use the software and how to install them.
* Python
```
Version python 3.8 <
```
* Pandas
* sklearn
* numpy
* matplotlib
* flask


Required Imports:

import numpy as np
import pandas as pd
import matplolib.pyplot as plt
from flask import Flask,request, url_for, redirect, render_template
import pickle

Term Project Report
Full Unit – Final Report
 
Diabetes Prediction using Machine Learning
Usama Raheem 
 

A report submitted in part fulfilment of the term project of
BSc (CS-329) Artificial Intelligence 
Supervisor: Dr. Naveed Anwar Butt
 
Department of Computer Science
Hafiz Hayat, University of Gujrat


October 05, 2022 
 
Declaration
This report has been prepared on the basis of my research work with guidance of my superior colleagues. Where other published and unpublished source materials have been used, these have been acknowledged.

Word Count: 873

Student Name: Usama Raheem
Student Roll No: 19011519-002
Student Section: CS-A
Student Batch: CS-19

Date of Submission: 27th March 2022 1:15 PM EST














Contents

Abstract	3
Project Specification.	4
Chapter 1:	Introduction	5
1.1	Overview	5
1.2	Data Description	5
1.3	Data Exploration (EDA)	6
Chapter 2:	Modeling	
Chapter 3:	Cross Validation Scores on Best Parameters	
Chapter 4:	Deployment Diabetes Prediction	



Abstract
This document contain report based on Diabetes Prediction dataset trained on well-known machine learning technique and models. The Author carefully predicts the probability of any individual’s having diabetes disease purely based on the that particular individual body features. The author successfully trains various machine learning models and decisively compare the predicted outcome and choose the best parameter to find the optimize predicted scores. At the end, the author deployed the best model based on Accuracy successfully deployed on Web by using Flask Framework.


 
Project Specification.
The main objective of this term project is to Predict the Probability of an individual having Diabetes Disease based on diagnostic measures.  In this project, Author uses dataset consists of several medical predictor (independent) variables and one target (dependent) variable, Outcome. Independent variables include the number of pregnancies the patient has, their BMI, insulin level, age etc. The Main task the author specify here is to build a machine learning model to accurately predict whether or not the patient in the dataset have the diabetes or not.


 
Chapter 1:	Introduction
The project report contains the steps and procedure of an end-to-end example of solving a real-world problem using Data Science. The author will be using Machine Learning to predict whether a person has diabetes or not, specifically based on information about the patient such as blood pressure, body mass index (BMI), age. This term project report walks through the various stages of the data science workflow. In particular the Report has following sections
•	Overview
•	Data Description
•	Data Exploration
1.1	Overview
The data was collected and made available by “National Institute of Diabetes and Digestive and Kidney Diseases”. Several constraints were placed on the selection of these instances from a larger database. The diabetes dataset was easily available on Kaggle to play with and attain some useful insights from the dataset.
1.2	Data Description
In data Description the author has some columns in dataset which were given below:
Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration 2 hours in an oral glucose tolerance test
Blood Pressure: Diastolic blood pressure (mm Hg)
Skin Thickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: It provided some data on diabetes mellitus history in relatives and the genetic relationship of those relatives to the patient.
Age: Age (years)
Outcome: Class variable (0 or 1) 268 of 768 are 1, the others are 0
1.3	Data Exploration (EDA)
 

Comparison Glucose with Outcome
 
Diabetes in function of Age and Blood Pressure
 

Diabetic and Non-Diabetic Pair Plot using Seaborn 
 
Histogram of All columns when the outcome is 1 (Diabetic)
 
Correlation Matrix Visualization
 
Dataset Features

 
Chapter 2:	 Modelling
 
Classification
 
 
 
Comparison of my Machine Learning Models
 
Accuracy Scores
Classifier	Accuracy 
Logistic Regression	81 %
KNN	77 %
Random Forest	79 %
Support vector machine (SVM)	81 %

HyperParameter Tunning using RandomizedSearchcv
 
 
HyperParameter Tunning using Grid Search CV Logistic Regression
 
Category	Classifier	Pram-Distribution	CV	n-iter	verbose	5-Fold	Scores
Hyperparameter
Tuning	Logistic Regression	Logistic_Reg
Grid	5	20	TRUE	For each 20 Candidates totaling 100 fits	0.831166
Grid Search CV	Logistic regression	GridSearch_Log
regression	5		TRUE	For each 30 candidates totaling 150 fits	0.837662
Best Model is Logistic Regression with 83% Accuracy Scores
ROC Curve
 
Confusion Matrix
 
Classification Report
	Precision	Recall	F1-score	Support
0	0.85	0.95	0.89	110
1	0.81	0.57	0.67	44
Accuracy			0.84	154
Macro Avg	0.83	0.76	0.78	154
Weighted avg	0.83	0.84	0.83	154
BEST PARAMETERS
{‘C’: 4.893900918477489, ‘solver’: ‘liblinear’}


 

Chapter 3:	 Cross Validation Classification Scores
Cross Validation	CV
Precision	CV
Recall	CV
F1-score
0.749743	0.7142036	0.5809116	0.569372
Visualize the CV Scores
 
Feature Importance
 



Chapter 4:	UI Diabetes Prediction
Deployment using Flask Framework after Saving Model with Pickle
 


Final UI of Diabetes Prediction using Machine Learning
 



Successful Prediction of Diabetes with %
 

Conclusion
Doing this Project is a fun activity, I learned so many new Machine learning and Data Science Techniques and Methods. After implementing all those concepts which were taught by my supervisor Dr. Naveed Anwar Butt, I am now Able to:
•	Pre-process any Data file
•	Perform Exploratory data Analysis on any given Data set
•	Train and Test Models using various Machine learning Algorithms and Classifiers
•	Generate Best Parameters using Hyper Parameter Tunning techniques
•	Produce valuable outcomes from a Raw Data file for better decisions
•	Deploy trained Model on Web With beautiful UI using Flask framework





