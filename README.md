# Machine Learning and Data Science

## Table of Contents

- [Machine Learning and Data Science](#machine-learning-and-data-science)
  - [Table of Contents](#table-of-contents)
  - [**1: Machine Learning**](#1-machine-learning)
    <details>
    <summary>Click to view all steps</summary>
    
    - [What Is Machine Learning?](#what-is-machine-learning)
    - [AI/Machine Learning/Data Science](#aimachine-learningdata-science)
    - [How Did We Get Here?](#how-did-we-get-here)
    - [Types of Machine Learning](#types-of-machine-learning)
    - [What Is Machine Learning? Round 2](#what-is-machine-learning-round-2)
    </details>
  - [**2: Machine Learning and Data Science Framework**](#2-machine-learning-and-data-science-framework)
    <details>
    <summary>Click to view all steps</summary>
    
    - [Introducing Our Framework](#introducing-our-framework)
    - [6 Step Machine Learning Framework](#6-step-machine-learning-framework)
    - [Types of Machine Learning Problems](#types-of-machine-learning-problems)
    - [Types of Data: What kind of data do we have?](#types-of-data-what-kind-of-data-do-we-have)
    - [Types of Evaluation: What defines success for us?](#types-of-evaluation-what-defines-success-for-us)
    - [Features In Data: What do we already know about the data?](#features-in-data-what-do-we-already-know-about-the-data)
    - [Modelling Part 1 - 3 sets](#modelling-part-1---3-sets)
    - [Modelling Part 2 - Choosing](#modelling-part-2---choosing)
    - [Modelling Part 3 - Tuning](#modelling-part-3---tuning)
    - [Modelling Part 4 - Comparison](#modelling-part-4---comparison)
    - [Experimentation](#experimentation)
    - [Tools We Will Use](#tools-we-will-use)
     </details>
  - [**3: Data Science Environment Setup**](#3-data-science-environment-setup)
    <details>
    <summary>Click to view all steps</summary>

    - [Introducing Our Tools](#introducing-our-tools)
    </details>
  - [**4: Pandas: Data Analysis**](#4-pandas-data-analysis)
    <details>
    <summary>Click to view all steps</summary>

    - [Pandas Introduction](#pandas-introduction)
    - [Series, Data Frames and CSVs](#series-data-frames-and-csvs)
    - [Data from URLs](#data-from-urls)
    - [Describing Data with Pandas](#describing-data-with-pandas)
    - [Selecting and Viewing Data with Pandas](#selecting-and-viewing-data-with-pandas)
    - [Manipulating Data](#manipulating-data)
    - [Assignment: Pandas Practice](#assignment-pandas-practice)
    </details>
  - [**5: NumPy**](#5-numpy)
    <details>
    <summary>Click to view all steps</summary>

    - [Section Overview](#section-overview)
    - [NumPy Introduction](#numpy-introduction)
    - [NumPy DataTypes and Attributes](#numpy-datatypes-and-attributes)
    - [Creating NumPy Arrays](#creating-numpy-arrays)
    - [NumPy Random Seed](#numpy-random-seed)
    - [Viewing Arrays and Matrices](#viewing-arrays-and-matrices)
    - [Manipulating Arrays](#manipulating-arrays)
    - [Standard Deviation and Variance](#standard-deviation-and-variance)
    - [Reshape and Transpose](#reshape-and-transpose)
    - [Dot Product vs Element Wise](#dot-product-vs-element-wise)
    - [Exercise: Nut Butter Store Sales](#exercise-nut-butter-store-sales)
    - [Comparison Operators](#comparison-operators)
    - [Sorting Arrays](#sorting-arrays)
    - [Turn Images Into NumPy Arrays](#turn-images-into-numpy-arrays)
    - [Optional: Extra NumPy resources](#optional-extra-numpy-resources)
    </details>
  - [**6: Matplotlib: Plotting and Data Visualization**](#6-matplotlib-plotting-and-data-visualization)
    <details>
    <summary>Click to view all steps</summary>

    - [Data Visualizations](#data-visualizations)
    - [Matplotlib Introduction](#matplotlib-introduction)
    - [Importing And Using Matplotlib](#importing-and-using-matplotlib)
    - [Anatomy Of A Matplotlib Figure](#anatomy-of-a-matplotlib-figure)
    - [Scatter Plot And Bar Plot](#scatter-plot-and-bar-plot)
    - [Histograms](#histograms)
    - [Subplots](#subplots)
    - [Plotting From Pandas DataFrames](#plotting-from-pandas-dataframes)
    - [Customizing Your Plots](#customizing-your-plots)
    </details>
  - [**7: Scikit-learn: Creating Machine Learning Models**](#7-scikit-learn-creating-machine-learning-models)
    <details>
    <summary>Click to view all steps</summary>

    - [Scikit-learn Introduction](#scikit-learn-introduction)
    - [Refresher: What Is Machine Learning?](#refresher-what-is-machine-learning)
    - [Typical scikit-learn Workflow](#typical-scikit-learn-workflow)
    - [Getting Your Data Ready: Splitting Your Data](#getting-your-data-ready-splitting-your-data)
    - [Quick Tip: Clean, Transform, Reduce](#quick-tip-clean-transform-reduce)
    - [Getting Your Data Ready: Convert Data To Numbers](#getting-your-data-ready-convert-data-to-numbers)
    - [Getting Your Data Ready: Handling Missing Values With Pandas](#getting-your-data-ready-handling-missing-values-with-pandas)
    - [Extension: Feature Scaling](#extension-feature-scaling)
    - [Getting Your Data Ready: Handling Missing Values With Scikit-learn](#getting-your-data-ready-handling-missing-values-with-scikit-learn)
    - [Choosing The Right Model For Your Data](#choosing-the-right-model-for-your-data)
    - [Choosing The Right Model For Your Data 2 (Regression)](#choosing-the-right-model-for-your-data-2-regression)
    - [Choosing The Right Model For Your Data 3 (Classification)](#choosing-the-right-model-for-your-data-3-classification)
    - [Fitting A Model To The Data](#fitting-a-model-to-the-data)
    - [Making Predictions With Our Model](#making-predictions-with-our-model)
    - [predict() vs predict_proba()](#predict-vs-predict_proba)
    - [Making Predictions With Our Model (Regression)](#making-predictions-with-our-model-regression)
    - [Evaluating A Machine Learning Model (Score)](#evaluating-a-machine-learning-model-score)
    - [Evaluating A Machine Learning Model 2 (Cross Validation)](#evaluating-a-machine-learning-model-2-cross-validation)
    - [Evaluating A Classification Model (Accuracy)](#evaluating-a-classification-model-accuracy)
    - [Evaluating A Classification Model (ROC Curve)](#evaluating-a-classification-model-roc-curve)
    - [Evaluating A Classification Model (Confusion Matrix)](#evaluating-a-classification-model-confusion-matrix)
    - [Evaluating A Classification Model 6 (Classification Report)](#evaluating-a-classification-model-6-classification-report)
    - [Evaluating A Regression Model 1 (R2 Score)](#evaluating-a-regression-model-1-r2-score)
    - [Evaluating A Regression Model 2 (MAE)](#evaluating-a-regression-model-2-mae)
    - [Evaluating A Regression Model 3 (MSE)](#evaluating-a-regression-model-3-mse)
    - [Machine Learning Model Evaluation](#machine-learning-model-evaluation)
    - [Evaluating A Model With Cross Validation and Scoring Parameter](#evaluating-a-model-with-cross-validation-and-scoring-parameter)
    - [Evaluating A Model With Scikit-learn Functions](#evaluating-a-model-with-scikit-learn-functions)
    - [Improving A Machine Learning Model](#improving-a-machine-learning-model)
    - [Tuning Hyperparameters by hand](#tuning-hyperparameters-by-hand)
    - [Tuning Hyperparameters with RandomizedSearchCV](#tuning-hyperparameters-with-randomizedsearchcv)
    - [Tuning Hyperparameters with GridSearchCV](#tuning-hyperparameters-with-gridsearchcv)
    - [Quick Tip: Correlation Analysis](#quick-tip-correlation-analysis)
    - [Saving And Loading A Model](#saving-and-loading-a-model)
    - [Putting It All Together](#putting-it-all-together)
    </details>
  - [**8: Supervised Learning: Classification + Regression**](#8-supervised-learning-classification--regression)
  - [**9: Milestone Project 1: Supervised Learning (Classification)**](#9-milestone-project-1-supervised-learning-classification)
    <details>
    <summary>Click to view all steps</summary>

    - [Step 1~4 Framework Setup](#step-14-framework-setup)
    - [Getting Our Tools Ready](#getting-our-tools-ready)
    - [Exploring Our Data](#exploring-our-data)
    - [Finding Patterns - Heart Disease Frequency according to Sex](#finding-patterns---heart-disease-frequency-according-to-sex)
    - [Finding Patterns - Age vs. Max Heart Rate for Heart Disease](#finding-patterns---age-vs-max-heart-rate-for-heart-disease)
    - [Finding Patterns - Heart Disease Frequency per Chest Pain Type](#finding-patterns---heart-disease-frequency-per-chest-pain-type)
    - [Preparing Our Data For Machine Learning](#preparing-our-data-for-machine-learning)
    - [Choosing The Right Models](#choosing-the-right-models)
    - [Experimenting With Machine Learning Models](#experimenting-with-machine-learning-models)
    - [Tuning/Improving Our Model](#tuningimproving-our-model)
    - [Tuning Hyperparameters](#tuning-hyperparameters)
    - [Evaluating Our Model](#evaluating-our-model)
    - [Finding The Most Important Features](#finding-the-most-important-features)
    </details>
  - [**10: Milestone Project 2: Supervised Learning (Time Series Data)**](#10-milestone-project-2-supervised-learning-time-series-data)
    <details>
    <summary>Click to view all steps</summary>

    - [Step 1~4 Framework Setup](#step-14-framework-setup-1)
    - [Exploring Our Data](#exploring-our-data-1)
    - [Feature Engineering](#feature-engineering)
    - [Turning Data Into Numbers](#turning-data-into-numbers)
    - [Filling Missing Numerical Values](#filling-missing-numerical-values)
    - [Filling Missing Categorical Values](#filling-missing-categorical-values)
    - [Fitting A Machine Learning Model](#fitting-a-machine-learning-model)
    - [Splitting Data](#splitting-data)
    - [Custom Evaluation Function](#custom-evaluation-function)
    - [Reducing Data](#reducing-data)
    - [RandomizedSearchCV](#randomizedsearchcv)
    - [Improving Hyperparameters](#improving-hyperparameters)
    - [Preproccessing Our Data](#preproccessing-our-data)
    - [Making Predictions](#making-predictions)
    - [Feature Importance](#feature-importance)
    </details>
  - [**11: Data Engineering**](#11-data-engineering)
    <details>
    <summary>Click to view all steps</summary>

    - [Data Engineering Introduction](#data-engineering-introduction)
    - [What Is Data?](#what-is-data)
    - [What Is A Data Engineer?](#what-is-a-data-engineer)
    - [Types Of Databases](#types-of-databases)
    - [Optional: OLTP Databases](#optional-oltp-databases)
    - [Hadoop, HDFS and MapReduce](#hadoop-hdfs-and-mapreduce)
    - [Apache Spark and Apache Flink](#apache-spark-and-apache-flink)
    - [Kafka and Stream Processing](#kafka-and-stream-processing)
    </details>
  - [**12: Neural Networks: Deep Learning, Transfer Learning and TensorFlow 2**](#12-neural-networks-deep-learning-transfer-learning-and-tensorflow-2)
    <details>
    <summary>Click to view all steps</summary>

    - [Deep Learning and Unstructured Data](#deep-learning-and-unstructured-data)
    - [Setting Up Google Colab](#setting-up-google-colab)
    - [Google Colab Workspace](#google-colab-workspace)
    - [Uploading Project Data](#uploading-project-data)
    - [Setting Up Our Data](#setting-up-our-data)
    - [Importing TensorFlow 2](#importing-tensorflow-2)
    - [Using A GPU](#using-a-gpu)
    - [Loading Our Data Labels](#loading-our-data-labels)
    - [Preparing The Images](#preparing-the-images)
    - [Turning Data Labels Into Numbers](#turning-data-labels-into-numbers)
    - [Creating Our Own Validation Set](#creating-our-own-validation-set)
    - [Preprocess Images](#preprocess-images)
    - [Turning Data Into Batches](#turning-data-into-batches)
    - [Visualizing Our Data](#visualizing-our-data)
    - [Preparing Our Inputs and Outputs](#preparing-our-inputs-and-outputs)
    - [How machines learn and what's going on behind the scenes?](#how-machines-learn-and-whats-going-on-behind-the-scenes)
    - [Building A Deep Learning Model](#building-a-deep-learning-model)
    - [Summarizing Our Model](#summarizing-our-model)
    - [Evaluating Our Model](#evaluating-our-model-1)
    - [Preventing Overfitting](#preventing-overfitting)
    - [Training Your Deep Neural Network](#training-your-deep-neural-network)
    - [Evaluating Performance With TensorBoard](#evaluating-performance-with-tensorboard)
    - [Make And Transform Predictions](#make-and-transform-predictions)
    - [Transform Predictions To Text](#transform-predictions-to-text)
    - [Visualizing Model Predictions](#visualizing-model-predictions)
    - [Saving And Loading A Trained Model](#saving-and-loading-a-trained-model)
    - [Training Model On Full Dataset](#training-model-on-full-dataset)
    - [Making Predictions On Test Images](#making-predictions-on-test-images)
    - [Submitting Model to Kaggle](#submitting-model-to-kaggle)
    - [Making Predictions On Our Images](#making-predictions-on-our-images)
    </details>

## **1: Machine Learning**

### What Is Machine Learning?

- Machines can perform tasks really fast
- We give them instructions to do tasks and they do it for us
- Computers used to mean people who do tasks that compute
- Problem: How to get to Danielle's house using Google maps?
- Imagine we had ten different routes to Danielle's house
  - Option 1: I measure each route one by one
  - Option 2: I program and tell the computer to calculate these 10 routes and find the shortest one.
- Problem: Somebody left a review on Amazon. Is this person angry?
- How can I describe to a computer what angry means?
- We let machines take care of the easier part of which things we can describe
- Things that are hard to just give instructions to, we let human do it
- The goal of machine learning is to make machines act more and more like humans because the smarter they

### [AI/Machine Learning/Data Science](https://towardsdatascience.com/a-beginners-guide-to-data-science-55edd0288973)

- AI: machine that acts like human
- Narrow AI: machine that acts like human at a specific task
- General AI: machine that acts like human with multiple abilities
- Machine Learning: a subset of AI
- Machine Learning: an approach to achieve artificial intelligence through systems that can find patterns in a set of data
- Machine Learning: the science of getting computers to act without being explicitly programmed
- Deep Learning: a subset of Machine Learning
- Deep Learning: one of the techniques for implementing machine learning
- Data Science: analyzing data and then doing something with a business goal
- [Teachable Machine](https://teachablemachine.withgoogle.com/)

### How Did We Get Here?

- Goal: Make business decisions
- Spreadsheets -> Relational DB -> Big Data (NoSQL) -> Machine Learning
  - Massive amounts of data
  - Massive improvements in computation
- Steps in a full machine learning project
  - Data collection (hardest part) -> Data modelling -> Deployment
- Data collection
  - How to clean noisy data?
  - What can we grab data from?
  - How do we find data?
  - How do we clean it so we can actually learn from it?
  - How to turn data from useless to useful?
- Data modelling
  - Problem definition: What problem are we trying to solve?
  - Data: What data do we have?
  - Evaluation: What defines success?
  - Features: What features should we model?
  - Modelling: What kind of model should we use?
  - Experiments: What have we tried / What else can we try?
- [Machine Learning Playground](https://ml-playground.com)

### [Types of Machine Learning](http://vas3k.com/blog/machine_learning/)

- Predict results based on incoming data
- Supervised: Data are labeled into categories
  - classification: is this an apple or is this a pear?
  - regression: based on input to predict stock prices
- Unsupervised: Data don't have labels
  - clustering: machine to create these groups
  - association rule learning: associate different things to predict what a customer might buy in the future
- Reinforcement: teach machines through trial and error
- Reinforcement: teach machines through rewards and punishment
  - skill acquisition
  - real time learning

### What Is Machine Learning? Round 2

- Now: Data -> machine learning algorithm -> pattern
- Future: New data -> Same algorithm (model) -> More patterns
- Normal algorithm: Starts with inputs and steps -> Makes output
- Machine learning algorithm
  - Starts with inputs and output -> Figures out the steps
- Data analysis is looking at a set of data and gain an understanding of it by comparing different examples, different features and making visualizations like graphs
- Data science is running experiments on a set of data with the hopes of finding actionable insights within it
  - One of these experiments is to build a machine learning model
- Data Science = Data analysis + Machine learning
- Section Review
  - Machine Learning lets computers make decisions about data
  - Machine Learning lets computers learn from data and they make predictions and decisions
  - Machine can learn from big data to predict future trends and make business decision

**[⬆ back to top](#table-of-contents)**

## **2: Machine Learning and Data Science Framework**

### Introducing Our Framework

- Focus on practical solutions and writing machine learning code
- Steps to learn machine learning
  - Create a framework
  - Match to data science and machine learning tools
  - Learn by doing

### [6 Step Machine Learning Framework](https://www.mrdbourke.com/a-6-step-field-guide-for-building-machine-learning-projects/)

- Problem definition: What problems are we trying to solve?
  - Supervised or Unsupervised
  - Classification or Regression
- Data: What kind of data do we have?
  - Structured or Unstructured
- Evaluation: What defines success for us?
  - Example: House data -> Machine learning model -> House price
  - Predicted price vs Actual price
- Features: What do we already know about the data?
  - Example: Heart disease? Feature: body weight
  - Turn features such as weight into patterns to make predictions whether a patient has heart disease?
- Modelling: Based on our problem and data, what model should we use?
  - Problem 1 -> Model 1
  - Problem 2 -> Model 2
- Experimentation: How could we improve/what can we try next?

### Types of Machine Learning Problems

- When shouldn't you use machine learning?
  - When a simple hand-coded instruction based system will work
- Main types of machine learning
  - Supervised Learning
  - Unsupervised Learning
  - Transfer Learning
  - Reinforcement Learning
- Supervised Learning: data and label -> make prediction
  - Classification: Is this example one thing or another?
    - Binary classification = two options
    - Example: heart disease or no heart disease?
    - Multi-class classification = more than two options
  - Regression: Predict a number
    - Example: How much will this house sell for?
    - Example: How many people will buy this app?
- Unsupervised Learning: has data but no labels
  - Existing Data: Purchase history of all customers
  - Scenario: Marketing team want to send out promotion for next summer
  - Question: Do you know who is interested in summer clothes?
  - Process: Apply labels such as Summer or Winter to data
  - Solution: Cluster 1 (Summer) and Cluster 2 (Winter)
- Transfer Learning: leverages what one machine learning model has learned in another machine learning model
  - Example: Predict what dog breed appears in a photo
  - Solution: Find an existing model which is learned to decipher different car types and fine tune it for your task
- Reinforcement Learning: a computer program perform some actions within a defined space and rewarding it for doing it well or punishing it for doing poorly
  - Example: teach a machine learning algorithm to play chess
- Matching your problem
  - Supervised Learning: I know my inputs and outputs
  - Unsupervised Learning: I am not sure of the outputs but I have inputs
  - Transfer Learning: I think my problem may be similar to something else

### Types of Data: What kind of data do we have?

- Different types of data
  - Structured data: all of the samples have similar format
  - Unstructured data: images and natural language text such as phone calls, videos and audio files
  - Static: doesn't change over time, example: csv
    - More data -> Find patterns -> Predict something in the future
  - Streaming: data which is constantly changed over time
    - Example: predict how a stock price will change based on news headlines
    - News headlines are being updated constantly you'll want to see how they change stocks
- Start on static data and then if your data analysis and machine learning efforts prove to show some insights you'll move towards streaming data when you go to deployment or in production
- A data science workflow
  - open csv file in jupyter notebook (a tool to build machine learning project)
  - perform data analysis with panda (a python library for data analysis)
  - make visualizations such as graphs and comparing different data points with Matplotlib
  - build machine learning model on the data using scikit learn to predict using these patterns

### Types of Evaluation: What defines success for us?

- Example: if your problem is to use patient medical records to classify whether someone has heart disease or not you might start by saying for this project to be valuable we need a machine learning model with over 99% accuracy
- data -> machine learning model -> predict: heart disease? -> accurancy 97.8%
- predicting whether or not a patient has heart disease is an important task so you want a highly accurate model
- Different types of metrics for different problems
  - Classification: accurancy, percision, recall
  - Regression: Mean absolute error (MAE), Mean squared error (MSE), Root mean squared error (RMSE)
  - Recommendation: Precision at K
- Example: Classifying car insurance claims
  - text from car insurance claims -> machine learning model -> predict who caused the accident (person submitting the claim or the other person involved ?) -> min 95% accuracy who caused the accident (allow to get it wrong 1 out of 20 claims)

### Features In Data: What do we already know about the data?

- Features is another word for different forms of data
- Features refers to the different forms of data within structured or unstructured data
- For example: predict heart disease problem
  - Features of the data: weight, sex, heart rate
  - They can also be referred to as feature variables
  - We use the feature variables to predict the target variable which is whether a person has heart disease or no.
- Different features of data
  - numerical features: a number like body weight
  - categorical features: sex or whether a patient is a smoker or not
  - derived features: looks at different features of data and creates a new feature / alter existing feature
    - Example: look at someone's hospital visit history timestamps and if they've had a visit in the last year you could make a categorical feature called visited in last year. If someone had visited in the last year they would get true.
    - feature engineering: process of deriving features like this out of data
- Unstructured data has features too
  - a little less obvious if you looked at enough images of dogs you'd start to figure out
  - legs: most of these creatures have four shapes coming out of their body
  - eyes: a couple of circles up the front
  - machine learning algorithm figure out what features are there on its own
- What features should you use?
  - a machine learning algorithm learns best when all samples have similar information
  - feature coverage: process of ensuring all samples have similar information

### Modelling Part 1 - 3 sets

- Based on our problem and data, what model should we use?
- 3 parts to modelling
  - Choosing and training a model
  - Tuning a model
  - Model comparison
- The most important concept in machine learning (the training, validation and test sets or 3 sets)
  - Your data is split into 3 sets
    - training set: train your model on this
    - validation set: tune your model on this
    - test set: test and compare on this
  - at university
    - training set: study course materials
    - validation set: practice exam
    - test set: final exam
  - generalisation: the ability for a machine learning model to perform well on data it has not seen before
- When things go wrong
  - Your professor accidentally sent out the final exam for everyone to practice on
  - when it came time to the actual exam, everyone would have already seen it now
  - Since people know what they should be expecting they go through the exam
  - They answer all the questions with ease and everyone ends up getting top marks
  - Now top marks might appear good but did the students really learn anything or were they just expert memorization machines
  - for your machine learning models to be valuable at predicting something in the future on unseen data you'll want to avoid them becoming memorization machines
- split 100 patient records
  - training split: 70 patient records (70-80%)
  - validation split: 15 patient records (10-15%)
  - test split: 15 patient records (10-15%)

### Modelling Part 2 - Choosing

- Based on our problem and data, what model should we use?
- 3 parts to modelling
  - Choosing and training a model: training data
  - Tuning a model: validation data
  - Model comparison: test data
- Choosing a model
  - Problem 1 -> model 1
  - Problem 2 -> model 2
  - Structured Data: [CatBoost](https://catboost.ai/), [XGBoost](https://github.com/dmlc/xgboost), [Random Forest](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)
  - Unstructured Data: Deep Learning, Transfer Learning
- Training a model
  - inputs: X(data) -> model -> predict outputs: y(label)
  - Goal: minimise time between experiments
    - Experiment 1: inputs -> model 1 -> outputs -> accurancy (87.5%) -> training time (3 min)
    - Experiment 2: inputs -> model 2 -> outputs -> accurancy (91.3%) -> training time (92 min)
    - Experiment 3: inputs -> model 3 -> outputs -> accurancy (94.7%) -> training time (176 min)
  - Things to remember
    - Some models work better than others and different problems
    - Don't be afraid to try things
    - Start small and build up (add complexity) as you need.

### Modelling Part 3 - Tuning

- Based on our problem and data, what model should we use?
- Example: Random Forest - adjust number of trees: 3, 5
- Example: Neural Networks - adjust number of layers: 2, 3
- Things to remember
  - Machine learning models have hyper parameters you can adjust
  - A model first results are not it's last
  - Tuning can take place on training or validation data sets

### Modelling Part 4 - Comparison

- How will our model perform in the real world?
- Testing a model
  - Data Set: Training -> Test
  - Performance: 98% -> 96%
- Underfitting (potential)
  - Data Set: Training -> Test
  - Performance: 64% -> 47%
- Overfitting (potential)
  - Data Set: Training -> Test
  - Performance: 93% -> 99%
- Balanced (Goldilocks zone)
- Data leakage -> Training Data overlap Test Data -> Overfitting
- Data mismatch -> Test Data is different to Training Data -> underfitting
- Fixes for underfitting
  - Try a more advanced model
  - Increase model hyperparameters
  - Reduce amount of features
  - Train longer
- Fixes for overfitting
  - Collect more data
  - Try a less advanced model
- Comparing models
  - Experiment 1: inputs -> model 1 -> outputs -> accurancy (87.5%) -> training time (3 min) -> prediction time (0.5 sec)
  - Experiment 2: inputs -> model 2 -> outputs -> accurancy (91.3%) -> training time (92 min) -> prediction time (1 sec)
  - Experiment 3: inputs -> model 3 -> outputs -> accurancy (94.7%) -> training time (176 min) -> prediction time (4 sec)
- Things to remember
  - Want to avoid overfitting and underfitting (head towards generality)
  - Keep the test set separate at all costs
  - Compare apples to apple
    - Model 1 on dataset 1
    - Model 2 on dataset 1
  - One best performance Metric does not equal the best model

### Experimentation

- How could we improve / what can we try next?
  - Start with a problem
  - Data Analysis: Data, Evaluation, Features
  - Machine learning modelling: Model 1
  - Experiments: Try model 2
- 6 Step Machine Learning Framework questions
  - Problem definition: What kind of problems you face day to day?
  - Data: What kind of data do you use?
  - Evaluation: What do you measure?
  - Features: What are features of your problems?
  - Modelling: What was the last thing you testing ability on?

**[⬆ back to top](#table-of-contents)**

### Tools We Will Use

- Data Science: 6 Step Machine Learning Framework
- Data Science: [Anaconda](https://www.anaconda.com/), [Jupyter Notebook](https://jupyter.org/)
- Data Analysis: Data, Evaluation and Features
- Data Analysis:[pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [NumPy](https://numpy.org/)
- Machine Learning: Modelling
- Machine Learning: [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/stable/), [XGBoost](https://xgboost.ai/), [CatBoost](https://catboost.ai/)
- [Elements of AI](https://www.elementsofai.com/)

**[⬆ back to top](#table-of-contents)**

## **3: Data Science Environment Setup**

### Introducing Our Tools

- Steps to learn machine learning [Recall]
  - Create a framework [Done] Refer to Section 3
  - Match to data science and machine learning tools
  - Learn by doing
- Your computer -> Setup Miniconda + Conda for Data Science
  - [Anaconda](https://www.anaconda.com/): Hardware Store = 3GB
  - [Miniconda](https://docs.conda.io/en/latest/miniconda.html): Workbench = 200 MB
  - [Anaconda vs. miniconda](https://stackoverflow.com/questions/45421163/anaconda-vs-miniconda)
  - [Conda](https://docs.conda.io/en/latest/): Personal Assistant
- Conda -> setup the rest of tools
  - Data Analysis:[pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [NumPy](https://numpy.org/)
  - Machine Learning: [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/stable/), [XGBoost](https://xgboost.ai/), [CatBoost](https://catboost.ai/)

**[⬆ back to top](#table-of-contents)**

## **4: Pandas: Data Analysis**

### Pandas Introduction

- Why pandas?
  - Simple to use
  - Integrated with many other data science & ML Python Tools
  - Helps you get your data ready for machine learning
- [What are we going to cover?](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)
  - Most useful functions
  - pandas Datatypes
  - Importing & exporting data
  - Describing data
  - Viewing & Selecting data
  - Manipulating data

### [Series, Data Frames and CSVs](pandas/introduction_to_pandas.ipynb)

### Data from URLs

### [Describing Data with Pandas](pandas/introduction_to_pandas.ipynb)

### [Selecting and Viewing Data with Pandas](pandas/introduction_to_pandas.ipynb)

### [Manipulating Data](pandas/introduction_to_pandas.ipynb)

- [Data Manipulation with Pandas](https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html)

### Assignment: Pandas Practice

- [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)
- [top questions and answers on Stack Overflow for pandas](https://stackoverflow.com/questions/tagged/pandas?sort=MostVotes&edited=true)
- [Google Colab](https://colab.research.google.com)

**[⬆ back to top](#table-of-contents)**

## **5: NumPy**

### Section Overview

- Why NumPy?
  - performance advantage as it is written in C under the hood
  - convert data into 1 or 0 so machine can understand
- [What is the difference between NumPy and Pandas?](https://www.quora.com/What-is-the-difference-between-NumPy-and-Pandas)

### NumPy Introduction

- Machine learning start with data.
  - Example: data frame
  - Numpy turn data into a series of numbers
  - A machine learning algorithm work out the patterns in those numbers
- Why NumPy?
  - It's fast
  - Behind the scenes optimizations written in C
  - [Vectorization via broadcasting (avoiding loops)](https://simpleprogrammer.com/vectorization-and-broadcasting/)
    - vector is a 1D array
    - matrix is a 2D array
    - vectorization: perform math operations on 2 vectors
    - broadcasting: extend an array to a shape that will allow it to successfully take part in a vectorized calculation
  - Backbone of other Python scientific packages
- What are we going to to cover?
  - Most useful functaions
  - NumPy datatypes & attributes (ndarray)
  - Creating arrays
  - Viewing arrays & matrices
  - Manipulating & comparing arrays
  - Sorting arrays
  - Use cases

### [NumPy DataTypes and Attributes](numpy/introduction_to_numpy.ipynb)

### [Creating NumPy Arrays](numpy/introduction_to_numpy.ipynb)

### [NumPy Random Seed](numpy/introduction_to_numpy.ipynb)

### [Viewing Arrays and Matrices](numpy/introduction_to_numpy.ipynb)

### [Manipulating Arrays](numpy/introduction_to_numpy.ipynb)

### [Standard Deviation and Variance](numpy/introduction_to_numpy.ipynb)

- [Standard Deviation and Variance](https://www.mathsisfun.com/data/standard-deviation.html)
- [Outlier Detection Methods](https://docs.oracle.com/cd/E17236_01/epm.1112/cb_statistical/frameset.htm?ch07s02s10s01.html)
  - If a value is a certain number of standard deviations away from the mean, that data point is identified as an outlier.
  - The specified number of standard deviations is called the threshold. The default value is 3.

### [Reshape and Transpose](numpy/introduction_to_numpy.ipynb)

### [Dot Product vs Element Wise](numpy/introduction_to_numpy.ipynb)

- [Matrix Multiplication](http://matrixmultiplication.xyz/)

### [Exercise: Nut Butter Store Sales](numpy/introduction_to_numpy.ipynb)

### [Comparison Operators](numpy/introduction_to_numpy.ipynb)

### [Sorting Arrays](numpy/introduction_to_numpy.ipynb)

### [Turn Images Into NumPy Arrays](numpy/introduction_to_numpy.ipynb)

### Optional: Extra NumPy resources

- [The Basics of NumPy Arrays](https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html)
- [A Visual Intro to NumPy and Data Representation](http://jalammar.github.io/visual-numpy/)
- [NumPy Quickstart tutorial](https://numpy.org/doc/1.17/user/quickstart.html)

**[⬆ back to top](#table-of-contents)**

## **6: Matplotlib: Plotting and Data Visualization**

### Data Visualizations

- [5 Essential Tips for Creative Storytelling Through Data Visualization](https://boostlabs.com/storytelling-through-data-visualization/)
- [Storytelling with Data: A Data Visualization Guide for Business Professionals](https://towardsdatascience.com/storytelling-with-data-a-data-visualization-guide-for-business-professionals-97d50512b407)

### [Matplotlib](https://matplotlib.org/3.1.1/contents.html) Introduction

- What is Matplotlib
  - Python ploting library
  - Turn date into visualisation
- Why Matplotlib?
  - BUilt on NumPy arrays (and Python)
  - Integrates directly with pandas
  - Can create basic or advanced plots
  - Simple to use interface (once you get the foundations)
- What are we going to cover?
  - A Matplotlib workflow
    - Create data
    - Create plot (figure)
    - Plot data (axes on figure)
    - Customise plot
    - Save/share plot
  - Importing Matplotlib and the 2 ways of plotting Plotting data - from NumPy arrays
  - Plotting data from pandas DataFrames Customizing plots
  - Saving and sharing plots

### [Importing And Using Matplotlib](matplotlib/introduction_to_matplotlib.ipynb)

- Which one should you use? (pyplpt vs matplotlib OO method?)
  - When plotting something quickly, okay to use pyplot method
  - When plotting something more customized and advanced, use the OO method
- [Effectively Using Matplotlib](https://pbpython.com/effective-matplotlib.html)
- [Pyplot tutorial](https://matplotlib.org/3.2.1/tutorials/introductory/pyplot.html)
- [The Lifecycle of a Plot](https://matplotlib.org/3.2.1/tutorials/introductory/lifecycle.html)

### [Anatomy Of A Matplotlib Figure](matplotlib/introduction_to_matplotlib.ipynb)

- [Anatomy of a figure](matplotlib/introduction_to_matplotlib.ipynb)

### [Scatter Plot And Bar Plot](matplotlib/introduction_to_matplotlib.ipynb)

- [A quick review of Numpy and Matplotlib](https://towardsdatascience.com/a-quick-review-of-numpy-and-matplotlib-48f455db383)

### [Histograms](matplotlib/introduction_to_matplotlib.ipynb)

### [Subplots](matplotlib/introduction_to_matplotlib.ipynb)

### [Plotting From Pandas DataFrames](matplotlib/introduction_to_matplotlib.ipynb)

- Which one should you use? (pyplpt vs matplotlib OO method?)
  - When plotting something quickly, okay to use pyplot method
  - When plotting something more advanced, use the OO method
- [Regular Expressions](https://regexone.com/)
- [Visualization](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html)

### [Customizing Your Plots](matplotlib/introduction_to_matplotlib.ipynb)

[Choosing Colormaps in Matplotlib](https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py)

**[⬆ back to top](#table-of-contents)**

## **7: Scikit-learn: Creating Machine Learning Models**

### [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) Introduction

- What is Scikit-Learn (sklearn)?
  - Scikit-Learn is a python machine learning library
  - Data -> Scikit-Learn -> machine learning model
  - machine learning model learn patterns in the data
  - machine learning model make prediction
- Why Scikit-Learn?
  - Built on NumPy and Matplotlib (and Python)
  - Has many in-built machine learning models
  - Methods to evaluate your machine learning models
  - Very well-designed API
- [What are we going to cover?](https://github.com/mrdbourke/zero-to-mastery-ml/blob/section-2-data-science-and-ml-tools/scikit-learn-what-were-covering.ipynb) An end-to-end Scikit-Learn workflow
  - Get data ready (to be used with machine learning models)
  - [Pick a machine learning model](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) (to suit your problem)
  - Fit a model to the data (learning patterns)
  - Make predictions with a model (using patterns)
  - [Evaluate the model](https://scikit-learn.org/stable/modules/model_evaluation.html)
  - Improving model predictions through experimentation
  - Saving and loading models
  
### Refresher: What Is Machine Learning?

- Programming: input -> function -> output
- Machine Learning: input (data) and desired output
  - machine figure out the function
  - a computer writing his own function
  - also know as model, alogrithm, bot
  - machine is the brain

### [Typical scikit-learn Workflow](scikit-learn/introduction_to_scikit_learn.ipynb)

- An end-to-end Scikit-Learn workflow
  - Getting the data ready -> `heart-disease.csv`
  - Choose the right estimator/algorithm for our problems -> [Random Forest Classifier](https://www.youtube.com/watch?v=eM4uJ6XGnSM)
    - [Random Forests in Python](http://blog.yhat.com/posts/random-forests-in-python.html)
    - [An Implementation and Explanation of the Random Forest in Python](https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76)
  - Fit the model/algorithm and use it to make predictions on our data
  - Evaluating a model
    - [Understanding a Classification Report For Your Machine Learning Model](https://medium.com/@kohlishivam5522/understanding-a-classification-report-for-your-machine-learning-model-88815e2ce397)
  - Improve a model
  - Save and load a trained model
  - Putting it all together!

### [Getting Your Data Ready: Splitting Your Data](scikit-learn/introduction_to_scikit_learn.ipynb)

Three main things we have to do:

- Split the data into features and labels (usually X & y)
  - Different names for X = features, features variables, data
  - Different names for y = labels, targets, target variables
- Converting non-numerical values to numerical values (also called feature encoding)
  - or one hot encoding
- Filling (also called imputing) or disregarding missing values

### Quick Tip: Clean, Transform, Reduce

Cannot assume all data you have is automatically going to be perfect

- Clean Data -> Transform data -> Reduce Data
- Clean Data: Remove a row or a column that's empty or has missing fields
- Clean Data: Calculate average to fill an empty cell
- Clean Data: Remove outliers in your data
- Transform data: Convert some of our information into numbers
- Transform data: Convert color into numbers
- Transform data is between zeros and ones
  - 0: No heart disease
  - 1: Heart disease
- Transform data: Data across the board uses the same units
- Reduce Data: More data more CPU
- Reduce Data: More energy it takes for us to run our computation
- Reduce Data: Same result on less data
- Reduce Data: Dimensionality reduction or column reduction
- Reduce Data: Remove irrelevant columns

### Getting Your Data Ready: [Convert Data To Numbers](scikit-learn/introduction_to_scikit_learn.ipynb)

### Getting Your Data Ready: [Handling Missing Values With Pandas](scikit-learn/introduction_to_scikit_learn.ipynb)

### Extension: Feature Scaling

- [Feature Scaling- Why it is required?](https://medium.com/@rahul77349/feature-scaling-why-it-is-required-8a93df1af310)
- [Feature Scaling with scikit-learn](https://benalexkeen.com/feature-scaling-with-scikit-learn/)
- [Feature Scaling for Machine Learning: Understanding the Difference Between Normalization vs. Standardization](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)
- Make sure all of your numerical data is on the same scale
- Normalization: rescales all the numerical values to between 0 and 1
  - [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- Standardization: z = (x - u) / s
  - z: standard score of a sample x
  - x: sample x
  - u: mean of the training samples
  - s: standard deviation of the training samples
  - [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
  - Feature scaling usually isn't required for your target variable
  - Feature scaling is usually not required with tree-based models (e.g. Random Forest) since they can handle varying features

### Getting Your Data Ready: [Handling Missing Values With Scikit-learn](scikit-learn/introduction_to_scikit_learn.ipynb)

The main takeaways:

- Split your data first (into train/test)
- Fill/transform the training set and test sets separately

### [Choosing The Right Model For Your Data](scikit-learn/introduction_to_scikit_learn.ipynb)

Scikit-Learn uses estimator as another term for machine learning model or algorithm

- [Choosing the right estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
- Regression - predicting a number
- Classification - predicting whether a sample is one thing or another

### [Choosing The Right Model For Your Data 2 (Regression)](scikit-learn/introduction_to_scikit_learn.ipynb)

### [Choosing The Right Model For Your Data 3 (Classification)](scikit-learn/introduction_to_scikit_learn.ipynb)

Tidbit:

- If you have structured data (heart_disease), used ensemble methods
- If you have unstructured data (image, audio), use deep learning or transfer learning

### [Fitting A Model To The Data](scikit-learn/introduction_to_scikit_learn.ipynb)

### [Making Predictions With Our Model](scikit-learn/introduction_to_scikit_learn.ipynb)

### [predict() vs predict_proba()](scikit-learn/introduction_to_scikit_learn.ipynb)

### [Making Predictions With Our Model (Regression)](scikit-learn/introduction_to_scikit_learn.ipynb)

- predict() can also be used for regression models

### [Evaluating A Machine Learning Model (Score)](scikit-learn/introduction_to_scikit_learn.ipynb)

[Three ways to evaluate Scikit-Learn models/esitmators](https://scikit-learn.org/stable/modules/model_evaluation.html)

- Estimator score method
- The scoring parameter
- Problem-specific metric functions.

### [Evaluating A Machine Learning Model 2 (Cross Validation)](scikit-learn/introduction_to_scikit_learn.ipynb)

### [Evaluating A Classification Model (Accuracy)](scikit-learn/introduction_to_scikit_learn.ipynb)

### [Evaluating A Classification Model (ROC Curve)](scikit-learn/introduction_to_scikit_learn.ipynb)

[Area under the receiver operating characteristic curve (AUC/ROC)](https://www.youtube.com/watch?v=4jRBRDbJemM)

- Area under curve (AUC)
- ROC curve

ROC curves are a comparison of a model's true postive rate (tpr) versus a models false positive rate (fpr).

- True positive = model predicts 1 when truth is 1
- False positive = model predicts 1 when truth is 0
- True negative = model predicts 0 when truth is 0
- False negative = model predicts 0 when truth is 1

### [Evaluating A Classification Model (Confusion Matrix)](scikit-learn/introduction_to_scikit_learn.ipynb)

- A confusion matrix is a quick way to compare the labels a model predicts and the actual labels it was supposed to predict.
- In essence, giving you an idea of where the model is getting confused.

### [Evaluating A Classification Model 6 (Classification Report)](scikit-learn/introduction_to_scikit_learn.ipynb)

Precision, Recall & F-Measure

- [Understanding Confusion Matrix](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)
- [Precision, Recall & F-Measure](https://www.youtube.com/watch?v=j-EB6RqqjGI)
- [Performance measure on multiclass classification](https://www.youtube.com/watch?v=HBi-P5j0Kec)
- Classification: Predict Category
- Determine if a sample shoe is Nike or not
- Confusion Matrix
  - True Positive (TP): Predict Nike shoe as Nike (Correct) Example: 0
  - False Positive (FP): Predict Non-Nike shoe as Nike (Wrong) Example: 0
  - False Negative (FN): Predict Nike shoe as Non-Nike (Wrong) Example: 10
  - True Negative (TN): Predict Non-Nike shoe as Non-Nike (Correct) Example: 9990
- Accuracy: % of correct prediction? (TP + TN) / total sample
  - Accuracy]() is a good measure to start with if all classes are balanced (e.g. same amount of samples which are labelled with 0 or 1).
- Precision and recall focus on TP, do not consider TN
- Precision: Of the shoes **classified** Nike, How many are **acutally** Nike?
  - Number of shoes **acutally** Nike = TP
  - Number of shoes **classified** Nike = TP + FP
  - Precision = TP / (TP + FP) = % of correct positive classification over total positive classification
  - When the model predicts a positive, how often is it correct?
- Recall: Of the shoes that are **actually** Nike, How many are **classified** as Nike?
  - Number of shoes **classified** Nike = TP
  - Number of shoes **acutally** Nike = TP + FN
  - Recall = TP / (TP + FN) = % of correct positive classification over total positive
  - When it is actually positive, how often does it predict a positive?
- Precision and recall become more important when classes are imbalanced.
  - If cost of false positive predictions are worse than false negatives, aim for higher precision.
    - For example, in spam detection, a false positive risks the receiver missing an important email due to it being incorrectly labelled as spam.
  - If cost of false negative predictions are worse than false positives, aim for higher recall.
    - For example, in cancer detection and terrorist detection the cost of a false negative prediction is likely to be deadly. Tell a cancer patient you have no cancer.
- F1-score is a combination of precision and recall.
  - Use F1 score if data is imbalanced
  
### [Evaluating A Regression Model 1 (R2 Score)](scikit-learn/introduction_to_scikit_learn.ipynb)

Regression model evaluation metrics

- R^2 (pronounced r-squared) or coefficient of determination.
- Mean absolute error (MAE)
- Mean squared error (MSE)

Which regression metric should you use?

- R2 is similar to accuracy. It gives you a quick indication of how well your model might be doing. Generally, the closer your R2 value is to 1.0, the better the model. But it doesn't really tell exactly how wrong your model is in terms of how far off each prediction is.
- MAE gives a better indication of how far off each of your model's predictions are on average.
- As for MAE or MSE, because of the way MSE is calculated, squaring the differences between predicted values and actual values, it amplifies larger differences. Let's say we're predicting the value of houses (which we are).
  - Pay more attention to MAE: When being $10,000 off is twice as bad as being $5,000 off.
  - Pay more attention to MSE: When being $10,000 off is more than twice as bad as being $5,000 off.

What R-squared does:

- Compares your models predictions to the mean of the targets. Values can range from negative infinity (a very poor model) to 1.
- For example, if all your model does is predict the mean of the targets, it's R^2 value would be 0.
- And if your model perfectly predicts a range of numbers it's R^2 value would be 1.

### [Evaluating A Regression Model 2 (MAE)](scikit-learn/introduction_to_scikit_learn.ipynb)

Mean absolue error (MAE)

- MAE is the average of the aboslute differences between predictions and actual values. It gives you an idea of how wrong your models predictions are.

### [Evaluating A Regression Model 3 (MSE)](scikit-learn/introduction_to_scikit_learn.ipynb)

Mean squared error (MSE)

- MSE is the average of the square value of aboslute differences between predictions and actual values.

### Machine Learning Model Evaluation

- Evaluating the results of a machine learning model is as important as building one.
- But just like how different problems have different machine learning models, different machine learning models have different evaluation metrics.
- Below are some of the most important evaluation metrics you'll want to look into for classification and regression models.

Classification Model Evaluation Metrics/Techniques

- Accuracy - The accuracy of the model in decimal form. Perfect accuracy is equal to 1.0.
- [Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score) - Indicates the proportion of positive identifications (model predicted class 1) which were actually correct. A model which produces no false positives has a precision of 1.0.
- [Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score) - Indicates the proportion of actual positives which were correctly classified. A model which produces no false negatives has a recall of 1.0.
- [F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) - A combination of precision and recall. A perfect model achieves an F1 score of 1.0.
- [Confusion matrix](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) - Compares the predicted values with the true values in a tabular way, if 100% correct, all values in the matrix will be top left to bottom right (diagonal line).
- [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) - Splits your dataset into multiple parts and train and tests your model on each part then evaluates performance as an average.
- [Classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) - Sklearn has a built-in function called classification_report() which returns some of the main classification metrics such as precision, recall and f1-score.
- ROC Curve - Also known as receiver operating characteristic is a plot of true positive rate versus false-positive rate.
- [Area Under Curve (AUC) Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) - The area underneath the ROC curve. A perfect model achieves an AUC score of 1.0.

Which classification metric should you use?

- Accuracy is a good measure to start with if all classes are balanced (e.g. same amount of samples which are labelled with 0 or 1).
- Precision and recall become more important when classes are imbalanced.
  - If false-positive predictions are worse than false-negatives, aim for higher precision.
  - If false-negative predictions are worse than false-positives, aim for higher recall.
- F1-score is a combination of precision and recall.
- A confusion matrix is always a good way to visualize how a classification model is going.

Regression Model Evaluation Metrics/Techniques

- [R^2 (pronounced r-squared)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) or the coefficient of determination - Compares your model's predictions to the mean of the targets. Values can range from negative infinity (a very poor model) to 1. For example, if all your model does is predict the mean of the targets, its R^2 value would be 0. And if your model perfectly predicts a range of numbers it's R^2 value would be 1.
- [Mean absolute error (MAE)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) - The average of the absolute differences between predictions and actual values. It gives you an idea of how wrong your predictions were.
- [Mean squared error (MSE)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) - The average squared differences between predictions and actual values. Squaring the errors removes negative errors. It also amplifies outliers (samples which have larger errors).

Which regression metric should you use?

- R2 is similar to accuracy. It gives you a quick indication of how well your model might be doing. Generally, the closer your R2 value is to 1.0, the better the model. But it doesn't really tell exactly how wrong your model is in terms of how far off each prediction is.
- MAE gives a better indication of how far off each of your model's predictions are on average.
- As for MAE or MSE, because of the way MSE is calculated, squaring the differences between predicted values and actual values, it amplifies larger differences. Let's say we're predicting the value of houses (which we are).
- Pay more attention to MAE: When being $10,000 off is twice as bad as being $5,000 off.
- Pay more attention to MSE: When being $10,000 off is more than twice as bad as being $5,000 off.

For more resources on evaluating a machine learning model, be sure to check out the following resources:

- [Scikit-Learn documentation for metrics and scoring (quantifying the quality of predictions)](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Beyond Accuracy: Precision and Recall](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c)
- [Stack Overflow answer describing MSE (mean squared error) and RSME (root mean squared error)](https://stackoverflow.com/questions/17197492/is-there-a-library-function-for-root-mean-square-error-rmse-in-python/37861832#37861832)

### [Evaluating A Model With Cross Validation and Scoring Parameter](scikit-learn/introduction_to_scikit_learn.ipynb)

- [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)

### [Evaluating A Model With Scikit-learn Functions](scikit-learn/introduction_to_scikit_learn.ipynb)

### [Improving A Machine Learning Model](scikit-learn/introduction_to_scikit_learn.ipynb)

### [Tuning Hyperparameters by hand](scikit-learn/introduction_to_scikit_learn.ipynb)

### [Tuning Hyperparameters with RandomizedSearchCV](scikit-learn/introduction_to_scikit_learn.ipynb)

### [Tuning Hyperparameters with GridSearchCV](scikit-learn/introduction_to_scikit_learn.ipynb)

- GridSearchCV goes through ALL combinations of hyperparameters in grid2
- [Metric Comparison Improvement](https://colab.research.google.com/drive/1ISey96a5Ag6z2CvVZKVqTKNWRwZbZl0m#scrollTo=b18kPvUFoh1z)

### Quick Tip: Correlation Analysis

- [Intro to Feature Selection Methods for Data Science](https://towardsdatascience.com/intro-to-feature-selection-methods-for-data-science-4cae2178a00a)
- Correlation Analysis
  - a statistical method used to evaluate the strength of relationship between two quantitative variables
  - A high correlation means that two or more variables have a strong relationship with each other
  - A weak correlation means that the variables are hardly related
- Forward Attribute Selection
  - Forward selection is an iterative method in which we start with having no feature in the model. In each iteration, we keep adding the feature which best improves our model till an addition of a new variable does not improve the performance of the model.
- Backward Attribute Selection
  - In backward elimination, we start with all the features and removes the least significant feature at each iteration which improves the performance of the model. We repeat this until no improvement is observed on removal of features.

### [Saving And Loading A Model](scikit-learn/introduction_to_scikit_learn.ipynb)

### Putting It All Together

Things to remember

- All data should be numerical
- There should be no missing values
- Manipulate the test set the same as the training set
- Never test on data you’ve trained on
- Tune hyperparameters on validation set OR use cross-validation
- One best performance metric doesn’t mean the best model

[Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

- chain multiple estimators into one
- chain a fixed sequence of steps in preprocessing and modelling

Steps we want to do (all in one cell):

- Fill missing data
- Convert data to numbers
- Build a model on the data

**[⬆ back to top](#table-of-contents)**

## **8: Supervised Learning: Classification + Regression**

[Structured Data Projects(https://github.com/mrdbourke/zero-to-mastery-ml/tree/master/section-3-structured-data-projects)

**[⬆ back to top](#table-of-contents)**

## [**9: Milestone Project 1: Supervised Learning (Classification)**](heart-disease-project/heart-disease-classification.ipynb)

### Step 1~4 Framework Setup

1. Problem Definition
   In a statement,

Given clinical parameters about a patient, can we predict whether or not they have heart disease?

2. Data
   The original data came from the Cleavland data from the UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/heart+Disease

There is also a version of it available on Kaggle. https://www.kaggle.com/ronitf/heart-disease-uci

3. Evaluation
   If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.

4. Features
   This is where you'll get different information about each of the features in your data. You can do this via doing your own research (such as looking at the links above) or by talking to a subject matter expert (someone who knows about the dataset).

**Create data dictionary**

1. age - age in years
2. sex - (1 = male; 0 = female)
3. cp - chest pain type
   - 0: Typical angina: chest pain related decrease blood supply to the heart
   - 1: Atypical angina: chest pain not related to heart
   - 2: Non-anginal pain: typically esophageal spasms (non heart related)
   - 3: Asymptomatic: chest pain not showing signs of disease
4. trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern
5. chol - serum cholestoral in mg/dl
   - serum = LDL + HDL + .2 \* triglycerides
   - above 200 is cause for concern
6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
   - '>126' mg/dL signals diabetes
7. restecg - resting electrocardiographic results
   - 0: Nothing to note
   - 1: ST-T Wave abnormality
     - can range from mild symptoms to severe problems
     - signals non-normal heart beat
   - 2: Possible or definite left ventricular hypertrophy
     - Enlarged heart's main pumping chamber
8. thalach - maximum heart rate achieved
9. exang - exercise induced angina (1 = yes; 0 = no)
10. oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more
11. slope - the slope of the peak exercise ST segment
    - 0: Upsloping: better heart rate with excercise (uncommon)
    - 1: Flatsloping: minimal change (typical healthy heart)
    - 2: Downslopins: signs of unhealthy heart
12. ca - number of major vessels (0-3) colored by flourosopy
    - colored vessel means the doctor can see the blood passing through
    - the more blood movement the better (no clots)
13. thal - thalium stress result
    - 1,3: normal
    - 6: fixed defect: used to be defect but ok now
    - 7: reversable defect: no proper blood movement when excercising
14. target - have disease or not (1=yes, 0=no) (= the predicted attribute)

### [Getting Our Tools Ready](heart-disease-project/heart-disease-classification.ipynb)

We're going to use pandas, Matplotlib and NumPy for data analysis and manipulation.

### [Exploring Our Data](heart-disease-project/heart-disease-classification.ipynb)

### [Finding Patterns - Heart Disease Frequency according to Sex](heart-disease-project/heart-disease-classification.ipynb)

### [Finding Patterns - Age vs. Max Heart Rate for Heart Disease](heart-disease-project/heart-disease-classification.ipynb)

### [Finding Patterns - Heart Disease Frequency per Chest Pain Type](heart-disease-project/heart-disease-classification.ipynb)

### [Preparing Our Data For Machine Learning](heart-disease-project/heart-disease-classification.ipynb)

### [Choosing The Right Models](heart-disease-project/heart-disease-classification.ipynb)

### [Experimenting With Machine Learning Models](heart-disease-project/heart-disease-classification.ipynb)

### [Tuning/Improving Our Model](heart-disease-project/heart-disease-classification.ipynb)

### [Tuning Hyperparameters](heart-disease-project/heart-disease-classification.ipynb)

### [Evaluating Our Model](heart-disease-project/heart-disease-classification.ipynb)

### [Finding The Most Important Features](heart-disease-project/heart-disease-classification.ipynb)

**[⬆ back to top](#table-of-contents)**

## [**10: Milestone Project 2: Supervised Learning (Time Series Data)**](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

### Step 1~4 Framework Setup

Predicting the Sale Price of Bulldozers using Machine Learning

In this notebook, we're going to go through an example machine learning project with the goal of predicting the sale price of bulldozers.

1. Problem defition

> How well can we predict the future sale price of a bulldozer, given its characteristics and previous examples of how much similar bulldozers have been sold for?

2. Data

The data is downloaded from the Kaggle Bluebook for Bulldozers competition: https://www.kaggle.com/c/bluebook-for-bulldozers/data

There are 3 main datasets:

- Train.csv is the training set, which contains data through the end of 2011.
- Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
- Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.

3. Evaluation

The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.

For more on the evaluation of this project check: https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation

**Note:** The goal for most regression evaluation metrics is to minimize the error. For example, our goal for this project will be to build a machine learning model which minimises RMSLE.

4. Features

Kaggle provides a data dictionary detailing all of the features of the dataset. You can view this data dictionary on Google Sheets: https://docs.google.com/spreadsheets/d/18ly-bLR8sbDJLITkWG7ozKm8l3RyieQ2Fpgix-beSYI/edit?usp=sharing

### [Exploring Our Data](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

### [Feature Engineering](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

### [Turning Data Into Numbers](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

### [Filling Missing Numerical Values](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

### [Filling Missing Categorical Values](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

### [Fitting A Machine Learning Model](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

### [Splitting Data](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

### [Custom Evaluation Function](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

### [Reducing Data](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

### [RandomizedSearchCV](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

### [Improving Hyperparameters](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

### [Preproccessing Our Data](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

### [Making Predictions](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

### [Feature Importance](bulldozer-price-prediction-project/end-to-end-bulldozer-price-regression.ipynb)

**[⬆ back to top](#table-of-contents)**

## **11: Data Engineering**

### Data Engineering Introduction

Data science is all about using data to make business decisions

Data science is the idea of using data and converting it into something useful for a product or business.

Data analysis is a subset of data science that allows us to analyze the data that we have.

Machine Learning is a technique to allow a computer to learn and figure out the solution to a problem that may be a little too complicated for a human to solve or maybe too tedious and takes too long of a time so we want to automate it.

A company has all these datas are coming from their users from their security cameras from their Web site from IOT devices.

A data engineer takes all this information and then produces it and maintains it in databases or a certain type of computers so that the business has access to this data in an organized fashion.

### What Is Data?

- Part of Product - eg. YouTube recommendation engine
- Are we doing ok? - monitor the company's sales
- Can we do better?

Type of data (organised -> unorganised)

1. Structured Data - from relational Database
2. Semi-Structured Data - eg. XML, CSV, JSON
3. UnStructured Data - eg. pdf, email, document
4. Binary Data - audio, image, video

So one of the tasks of a data engineer is to essentially use the fact that there's all these types of data and somehow combine them or organize them in a way that is useful to the business.

### What Is A Data Engineer?

Data Mining - pre processing and extracting some knowledge from the data

Big Data - data that's so big that you need to have it running on cloud computing or multiple computers

Data Pipeline - build a pipeline that allows us to flow from that unknown large amount of data to a pipeline that extracts data to a more useful form

A data engineer allows us to do this data collection part. They bring in all this information organize it in a way for us to do our data modelling.

And this is what a data engineer built a data engineer starts off with what we call data ingestion that

is acquiring data from various sources and we acquire all these different sources of data and ingested

into what we call a data lake a data lake is a collection.

Well all this data into one location from there we could just leave the lake as it is.

Build the following data pipeline

- Rain -> Data
- Collected into streams and rivers - data ingestion
  - acquire data from various sources and ingested into a data lake
- Lakes / Dam - Data lake (pool of raw data)
- filtration sanitary area - data transformation that is convert data from one format to another
  - data warehouse is a location for structured filtered data that has been processed and has a specific purpose
- plumbing and pipes for us to deliver water

Data Ingestion Tool

- Kafka

Data Lake Tools

- hadoop
- Azure Data lake
- Amazon S3

Data warehouse Tools

- Amazon Athena
- Amazon Redshift
- Google BigQuery

Who use Data Lake?

- Machine Learning
- Data Scientist

Who use Data Warehouse?

- Business intelligent
- business analyst
- data analyst

A software engineer, a software developer, app developer and mobile developer build programs and apps that users and customers use.

The app releases data. A data engineer would build this pipeline for us to ingest data and store it in different services like Hadoop like Google big query so that that data can be accessed by the rest of the business.

Next, data scientists use the data lake to extract information and deliver some sort of business value.

Finally we have data analysts or business intelligence to use something like a data warehouse or structured data to again derive business value.

3 main tasks of data engineer,

- Build ETL pipeline (Extract, Transform and Load into data warehouse)
- Build analysis tools
- Maintain data warehouse and data lakes

### Types Of Databases

Relational Database

- use SQL to make transaction
- [ACID transaction](https://blog.yugabyte.com/a-primer-on-acid-transactions/)

NoSQL - eg. MongoDB,

- distributed database
- Disorganised

NewSQL - eg. VoltDB, CockroachDB

- distributed
- ACID transaction

Usage

- Search - eg. ElasticSearch or solr
- Computation - eg. Apache Spark

[OLTP vs OLAP](https://techdifferences.com/difference-between-oltp-and-olap.html)
OLTP - SQL database, relational database, transactional
OLAP - use for analytical purpose

- view a financial report, or budgeting, marketing management, sales report

### Optional: OLTP Databases

What is a database?

- A database is a collection of data.

Many form of data

- numbers
- dates
- password hashes
- user information

2 types of DBMS

- Relational Database
- NoSQL / Non Relational Database (document oriented)

### Hadoop, HDFS and MapReduce

Hadoop (store a lots of data across multiple machine)

- HDFS (Hadoop distributed file system)
- MapReduce (batch processing)

Hive - makes your Hadoop cluster feel like it's a relational database

### Apache Spark and Apache Flink

Apache Spark

- run ETL jobs like extract transform load to clean and transform that data

Apache Flink

- real time processing started to happen things like spark streaming

### Kafka and Stream Processing

Batch processing

- Hadoop
- Spark
- AWS S3
- Common Databases

Real time stream processing

- Spark Streaming
- Flink
- Storm
- Kineses

Data -> Ingest data through Kafka -> Real time stream processing
Data -> Ingest data through Kafka -> Batch processing

**[⬆ back to top](#table-of-contents)**

## **12: Neural Networks: Deep Learning, Transfer Learning and TensorFlow 2**

### Deep Learning and Unstructured Data

[Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)

- alternative to jupyter notebook

[TensorFlow](https://www.tensorflow.org/)

- a deep learning or numerical computing library
- use for unstructured data - Photos, Audio waves, natural language text

Why TensorFlow ?

- Write fast deep learning code in Python (able to run on a GPU)
- Able to access many pre-built deep learning models
- Whole stack: preprocess, model, deploy
- Originally designed and used in-house by Google (now open-source)

Choosing a model (throwback)

- Problem 1 (structured data) -> Choose a Model
  - CatBoost, dmlc XGBoost, Random Forest
- Problem 2 (unstructured data) -> Choose a Model
  - Deep Learning use TensorFlow
  - Transfer Learning use TensorFlow Hub

What is deep learning?

- another form of machine learning

What are neural networks?

- type of machine learning algorithm for deep learning

What kind of deep learning problems are there?

- Classification
  - multi-classification of dog breed
  - classification of spam email
- Sequence to sequence (seq2seq)
  - audio to text translation
- Object detection

What is transfer learning? Why use transfer learning?

- Take what you know in one domain and apply it to another.
- Starting from scratch can be expensive and time consuming.
- Why not take advantage of what’s already out there?

A TensorFlow workflow

- Get the data ready (turn into Tensors)
  - An end-to-end multi-class classification workflow with TensorFlow Preprocessing image data (getting it into Tensors)
- Pick a model from TensorFlow Hub
  - Choosing a deep learning model
- Fit the model to the data and make a prediction
  - Fitting a model to the data (learning patterns)
  - Making predictions with a model (using patterns)
- Evaluate the model
  - Evaluating model predictions
- Improve through experimentation
- Save and reload your trained model
  - Saving and loading models
- Using a trained model to make predictions on custom data

**[⬆ back to top](#table-of-contents)**

### Setting Up Google Colab

- [Using Transfer Learning and TensorFlow 2.0 to Classify Different Dog Breeds](https://github.com/mrdbourke/zero-to-mastery-ml/blob/wip/section-4-unstructured-data-projects/end-to-end-dog-vision.ipynb)
- [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification/overview)
- [Kaggle API](https://github.com/Kaggle/kaggle-api)
- [What is Colaboratory?](https://colab.research.google.com/notebooks/intro.ipynb)
- [External data: Local Files, Drive, Sheets, and Cloud Storage](https://colab.research.google.com/notebooks/io.ipynb)

**[⬆ back to top](#table-of-contents)**

### Google Colab Workspace

- [Welcome To Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb)
- [Colaboratory Frequently Asked Questions](https://research.google.com/colaboratory/faq.html)

### Uploading Project Data

### Setting Up Our Data

### [Importing TensorFlow 2](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### Using A GPU

[Tensorflow with GPU](https://colab.research.google.com/notebooks/gpu.ipynb)

But we can fix this going to runtime and then changing the runtime type:

- Go to Runtime.
- Click "Change runtime type".
- Where it says "Hardware accelerator", choose "GPU" (don't worry about TPU for now but feel free to research them).
- Click save.
- The runtime will be restarted to activate the new hardware, so you'll have to rerun the above cells.
  - If the steps have worked you should see a print out saying "GPU available".

### [Loading Our Data Labels](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Preparing The Images](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Turning Data Labels Into Numbers](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Creating Our Own Validation Set](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Preprocess Images](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Turning Data Into Batches](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

[Yann LeCun Batch Size](https://twitter.com/ylecun/status/989610208497360896?s=20)

### [Visualizing Our Data](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Preparing Our Inputs and Outputs](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### How machines learn and what's going on behind the scenes?

Massive effort getting the data ready for use with a machine learning model! This is one of the most important steps in any machine learning project.

Now you've got the data ready, you're about to dive headfirst into writing deep learning code with TensorFlow 2.x.

Since we're focused on writing code first and foremost, these videos are optional but they're here for those who want to start to get an understanding of what goes on behind the scenes.

How Machines Learn

The first is a video called [How Machines Learn](https://www.youtube.com/watch?v=R9OHn5ZF4Uo) by GCP Grey on YouTube.

It's a non-technical narrative explaining how some of the biggest tech companies in the world use data to improve their businesses. In short, they're leveraging techniques like the ones you've been learning. Instead of trying to think of every possible rule to code, they collect data and then use machines to figure out the patterns for them.

What actually is a neural network?

You're going to be writing code which builds a neural network (a type of machine learning model) so you might start to wonder, what's going on when you run the code?

When you pass inputs (often data and labels) to a neural network and it figures out patterns between them, how is it doing so?

When it tries to make predictions and gets them wrong, how does it improve itself?

[The deep learning series](https://www.youtube.com/watch?v=aircAruvnKk) by 3Blue1Brown on YouTube contains a technical deep-dive into what's going on behind the code you're writing.

### [Building A Deep Learning Model](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

[TensorFlow Hub](https://www.tensorflow.org/hub)
[Papers With Code](https://paperswithcode.com/)
[PyTouch Hub](https://pytorch.org/hub/)
[Model Zoo](https://modelzoo.co/)
[TensorFlow Keras](https://www.tensorflow.org/guide/keras)

### [Summarizing Our Model](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Evaluating Our Model](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Preventing Overfitting](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Training Your Deep Neural Network](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Evaluating Performance With TensorBoard](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Make And Transform Predictions](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Transform Predictions To Text](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Visualizing Model Predictions](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Saving And Loading A Trained Model](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Visualizing Model Predictions](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Training Model On Full Dataset](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Making Predictions On Test Images](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Submitting Model To Kaggle](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

### [Making Predictions On Our Images](https://colab.research.google.com/drive/1mSRaBRNycifqlVGTop-cWLw9fqkajj3M?usp=sharing)

**[⬆ back to top](#table-of-contents)**
