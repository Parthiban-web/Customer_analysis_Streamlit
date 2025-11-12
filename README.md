# Customer Segmentation Analysis

## Project Description
A machine learning project that performs **customer segmentation** using **K-means clustering** on transaction data to identify distinct customer groups based on shopping behavior and demographics.

---

## Features
- **Data Analysis:** Customer transaction data exploration  
- **Clustering:** K-means implementation for customer segmentation  
- **Optimization:** Silhouette analysis to determine optimal cluster count  and Principal Component Analysis (PCA)
- **Visualization:** Data patterns and cluster evaluation  

---

## Dataset
The dataset contains customer transaction records with features including:  
- Customer demographics (age, gender)  
- Purchase details (category, quantity, price)  
- Transaction information (payment method, date, location)  

---

## Installation & Requirements
Install the required libraries using pip:

```bash
pip install pandas scikit-learn matplotlib jupyter
jupyter notebook Customer_analysis.ipynb

Results

Optimal Clusters: 12 segments

Best Silhouette Score: 0.202

Method: K-means clustering with silhouette analysis and PCA

Technologies

Python

Scikit-learn

Unsupervised Machine Learning

Pandas

Matplotlib

Jupyter Notebook
