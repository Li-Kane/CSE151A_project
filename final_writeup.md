# Final Writeup

## Introduction
Predicting the time a diabetes patient will stay at a hospital is an important aspect of healthcare management and treatment. Using data obtained from 130 hospitals and integrated delivery networks, this project examines the records of patients diagnosed with diabetes who underwent laboratory tests, received medications, and had hospital stays lasting up to 14 days.

It is important to have a good predictive model so that hospitals can more accurately allocate resources, such as beds, medications, and staff for better patient care. Failure to provide proper diabetes care not only increases the managing costs for the hospitals (as the patients are readmitted) but also impacts the morbidity and mortality of the patients, who may face complications associated with diabetes. This model can have various applications, including preventative care, cost management, resource optimization, and overall healthcare efficiency.

## **Methods**

In this section, we will go over the various techniques used for preprocessing, data exploration, and building our models.

**exploration**

For our exploration, we started by trimming down to the features we deemed interesting or worthy of analysis. These were 'race', 'gender', 'age', 'num_procedures', 'num_medications', 'number_emergency', 'number_diagnoses', 'diabetesMed', 'readmitted', 'time_in_hospital.' We then visualized them using a heatmap of the correlation matrix and a pairplot using the seaborn library. We did a lot of our preprocessing prior to exploration to facilitate this. For example dropping columns with null values and encoding our non-numerical values. When we made the correlation matrix and heatmap we made a few observations, which can be seen [here.](https://colab.research.google.com/drive/1O5QPe9oywuKqfh5coIJUJ67HEC2UzZCR#scrollTo=sL_QoLrghkGQ)
<br><br>

**preprocessing**

For our preprocessing we did the following. In the original dataset the 'readmitted' column was the 'y' and everything else was the 'X.' We wanted to predict 'time in hospital' so we reattached readmitted to X and separated out 'time in hospital' to y.

Our unprocessed data had a lot of non-numerical categories and we fixed all of these in different ways. We label encoded the gender column, setting 'male' to 1 and 'female' to 0 as we iterated through the dataframe. Then we onehot encoded the race section using pd.getdummies. The 'readmitted' column had one value for readmitted in < 30 days, one for > 30 days, and one for never readmitted. We label encoded these as 0, 1, and 2 respectively, matching the numerical hierarchy. The values in the column age all described a 10 year range in a form 20-30, so we replaced those with the average of that range. For example, 20-30 was replaced with 25. This was done with simple iteration through the column.

Our dataset was extremely large with around 100,000 entries. Because our dataset was so big, we felt comfortable dropping the rows with "NaN" or other null values.
<br><br>

**model 1 methods**

For our first model we did polynomial regression of various degrees, starting with 1st degree (linear) and going up to degree 5. We used the sklearn regression libraries for this and split the data into train and test sets using our [own code](https://colab.research.google.com/drive/1O5QPe9oywuKqfh5coIJUJ67HEC2UzZCR#scrollTo=jjM6gJUQMW3M&line=4&uniqifier=1) with an 80/20 split. We evaluated and compared these polynomial models using both train MSE and test MSE.

**model 2 methods**

For our second model we used a neural network and tuned it with various activation functions and neuron layouts. In addition, we did categorical classification by changing the size of the output layer and one-hot encoding our ytrain, utilizing the fact that 'time in hospital' was always an integer value between 1 and 14. These models were evaluated with MSE for regression versions and accuracy/classification reports for the categorical classification versions.

**model 3 methods**

For our third model we utilized the imbalanced learn

5. Results

**model 1 results**

Overall, the best MSE for our linear and polynomial regression model using OLS was around 6.56 on the test, for a standard deviation of around 2.56 days on our model predictions. Our classification model could reach up to 22% accuracy with an MSE of around 6.3 when we multiply the probability predictions by the days of each class. The decision tree model had a best MSE of 6.711323110010316. Even when we were to use the entire dataset without dropping columns, MSE at the lowest was around 4, for a standard deviation of 2 days on the model error and a classification accuracy of up to 27%, decision tree with entire data had MSE of ~5.4.

**model 2 results**


**model 3 results**


8. Discussion



10. Conclusion
11. Collaboration

## 5.) Your final model (model 3) and final results summary (this should be the last paragraph in D)

## 6.) Your GitHub must be made public by the morning of the next day of the submission deadline.
