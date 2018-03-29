# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
# Dataset contains transaction in a week
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):  # Create sparse matrix
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# # Look the first second sparse matrix of dataset
# print(transactions[0])
# print(transactions[1])

# Training Apriori on the dataset
from apyori import apriori
# support=(3*7)/7501 is at least three transactions in a week
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
