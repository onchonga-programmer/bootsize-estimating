import pandas
#make a dictionary of data of boot sizes 
#and harness sizes
data={
    'boot_size':[1,2,3,4,5,6,7,8,9,10,
                11,12,13,14,15,16,17,18,
                19,20],
    'harness_size':[21,22,23,24,25,26,27,
               28,29,30,31,32,33,34,35,
               36,37,38,39,40],

}
dataset=pandas.DataFrame(data)
print(dataset)
# this is the input now lets create a model
# Load a library to do the hard work for us
import statsmodels.formula.api as smf

# First, we define our formula using a special syntax
# This says that boot_size is explained by harness_size
formula = "boot_size ~ harness_size"

# Create the model, but don't train it yet
model = smf.ols(formula = formula, data = dataset)

# Note that we have created our model but it does not 
# have internal parameters set yet
if not hasattr(model, 'params'):
    print("Model selected but it does not have parameters set. We need to train it!")
    # Train (fit) the model so that it creates a line that 
# fits our data. This method does the hard work for us
fitted_model = model.fit()

# Print information about our model now it has been fit
print("The following model parameters have been found:")
print("Line slope:", fitted_model.params.iloc[1])
print("Line Intercept:", fitted_model.params.iloc[0])

import matplotlib.pyplot as plt

# Show a scatter plot of the data points and add the fitted line
# Don't worry about how this works for now
plt.scatter(dataset["harness_size"], dataset["boot_size"])
plt.plot(dataset["harness_size"], fitted_model.params.iloc[1] * dataset["harness_size"] + fitted_model.params.iloc[0], 'r', label='Fitted line')

# add labels and legend
plt.xlabel("harness_size")
plt.ylabel("boot_size")
plt.legend()
harness_size = { 'harness_size' : [52.5] }

# Use the model to predict what size of boots the dog will fit
approximate_boot_size = fitted_model.predict(harness_size)

# Print the result
print("Estimated approximate_boot_size:")
print(approximate_boot_size[0])
