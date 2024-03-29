#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    ### your code goes here
    cleaned_data = []
    errors = (net_worths-predictions)**2 #Sum of squared errors
    cleaned_data = zip(ages, net_worths, errors) #making a list of tuples
    cleaned_data = sorted(cleaned_data, key=lambda x:x[2][0], reverse=True) #sorting based on error decreasing order
    limit = int(len(ages)*0.1) #taking 10 percent
    return cleaned_data[limit:] #returning the rest 90

