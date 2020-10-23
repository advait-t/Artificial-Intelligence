## Import Libraries
"""

import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from pandas import DataFrame as dframe

"""#### Define Test Size"""

TEST_SIZE = 0.4

"""## Define Main Function"""

def main():
    
    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(evidence, labels, test_size=TEST_SIZE)
    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

"""## Define function to load data"""

def load_data(filename):
   
    import numpy as np
    import pandas as pd
    filename = pd.read_csv('/Users/home/Downloads//shopping.csv')
    df = filename
    month = { 'Feb': 1, 'Mar': 2, 'May': 4, 'June':5, 'Jul':6, 'Aug':7, 'Sep': 8, 'Oct':9, 'Nov': 10, 'Dec': 11}
    df.Month = [month[item] for item in df.Month]
    vistype = { 'Returning_Visitor': 1, 'New_Visitor': 0, 'Other':0 }
    df.VisitorType = [vistype[item] for item in df.VisitorType]
    df['Weekend'] = df['Weekend']*1
    df['Revenue'] = df['Revenue']*1
    
    evidence = df.drop(columns = 'Revenue')
    
    Administrative = np.array(evidence['Administrative'], dtype = int)
    Administrative_Duration = np.array(evidence['Administrative_Duration'])
    Informational = np.array(evidence['Informational'], dtype = int)
    Informational_Duration = np.array(evidence['Informational_Duration'])
    ProductRelated = np.array(evidence['ProductRelated'], dtype = int)
    ProductRelated_Duration = np.array(evidence['ProductRelated_Duration'])
    BounceRates = np.array(evidence['BounceRates'])
    ExitRates = np.array(evidence['ExitRates'])
    PageValues = np.array(evidence['PageValues'])
    SpecialDay = np.array(evidence['SpecialDay'])
    Month = np.array(evidence['Month'], dtype = int)
    OperatingSystems = np.array(evidence['OperatingSystems'], dtype = int)
    Browser = np.array(evidence['Browser'], dtype = int)
    Region = np.array(evidence['Region'], dtype = int)
    TrafficType = np.array(evidence['TrafficType'], dtype = int)
    VisitorType = np.array(evidence['VisitorType'], dtype = int)
    Weekend = np.array(evidence['Weekend'], dtype = int)
    
    evidence = np.array([Administrative, Administrative_Duration, Informational,Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay, Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend], dtype = object).T.tolist()
    
    labels = df['Revenue'].tolist()
    
    return (evidence, labels)
    raise NotImplementedError

"""## Define function to train model"""

def train_model(X_train, y_train):
   
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors = 1)
    model = model.fit(X_train, y_train)
    return model
    raise NotImplementedError

"""## Define function to evaluate model"""

def evaluate(y_test, predictions):    
   
    from sklearn.metrics import confusion_matrix
    result = confusion_matrix(y_test, predictions)
    specificity= result[0,0]/(result[0,0]+result[1,0])
    sensitivity= result[1,1]/(result[1,1]+result[0,1])
    return sensitivity,specificity
    
    raise NotImplementedError

"""## Run the main function"""

main()

"""## Record results after every run

## First run

Correct: 4070

Incorrect: 862

True Positive Rate: 46.28%

True Negative Rate: 87.98%

## Second run

Correct: 4116
    
Incorrect: 816
    
True Positive Rate: 44.33%
    
True Negative Rate: 89.70%
"""
