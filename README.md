Prediction of Alzheimer's & Parkinson's Using EEG
This project focuses on developing a machine learning model to predict Alzheimer's and Parkinson's diseases using EEG (Electroencephalogram) signals. The primary goal is to utilize EEG data to create an accurate and reliable predictive model that can aid in the early diagnosis of these neurodegenerative diseases using flask.
Table of Contents
- Introduction
- Dataset
- Installation
- Usage
- Model
- Results
Introduction
Alzheimer's and Parkinson's diseases are progressive neurological disorders that impact millions of people worldwide. Early diagnosis is crucial for managing symptoms and improving patient outcomes. This project aims to leverage EEG signals to develop a predictive model for these diseases, using various machine learning techniques.
 Dataset
This dataset contains the EEG electrode recordings from 585 subjects in total. A total of 195 of them were diagnosed with Alzheimer’s disease (AD group), 195 were diagnosed with Parkinson’s disease (PD group), and 195 were Healthy, storing them in a ‘.csv’ file. Recordings include the EEG signal from 18 scalp electrodes (Fp1, Fp2, F3, F4, F7, F8, T3, T4, C3, C4, T5, T6, P3, P4, T6, O1, and O2) and 1 target. This dataset is trained by Convolutional Neural Networks (CNN) and uses Principal Component Analysis (PCA) for dimensionality reduction.
Installation
To run this project, you will need to install the following dependencies:
pip install NumPy pandas scikit-learn matplotlib seaborn
Execution Process:
1.Extract the Alzheimer file. 
2.Open project.py file to review the code of the project.
3.Install library’s (keras, TensorFlow, sklearn)
4.After importing those library’s run the code 
5.Interface path will shown at the bottom of complied code.
6.Select a file named  test_multiple_records file to check multiple patients records
7.To check single patient records select single_test_record . 
 Model
The predictive model is built using various machine learning algorithms, including but not limited to:
- Convolutional Neural Networks (CNN)
Feature extraction from the EEG signals is performed using methods such as:
- Principal Component Analysis (PCA)
The models are trained and validated using cross-validation techniques to ensure robustness and accuracy.
Results
The performance of the models is evaluated using metrics such as:
- Accuracy
- Precision
- Recall
- F1 Score
-Patients Graph
-FAR vs FRR
- Validation Graph
-Confusion Matrix
Hardware Components:
RAM: 4.0 GB
CPU: Core i3 processor (Minimum) Hard Disk   Space:128GB
Software Implementation. 
Python (Version 3.12.4(64 bits))
Flask
