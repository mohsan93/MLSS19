import random

def linear_experiment(classifier, Xdata, Ydata, scoring = ["explained_variance", "r2" ,"mean_squared_error", "mean_absolute_error"], cv=10):
    #explained variance: how many percent of the variance are explained?
    #r^2: How many percent of the absolut squared error are explained
    random.seed(0)
    results = cross_validate(classifier, Xdata, Ydata, scoring = scoring, cv=cv, return_train_score=False)
    rval = {}
    for i in results:
        rval[i] =  results[i].mean()
    for i in ["test_mean_squared_error", "test_mean_absolute_error"]:
        if rval[i] < 0:
            rval[i] = rval[i]*-1
    return rval
