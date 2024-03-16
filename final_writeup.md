# Final Writeup

## Introduction
Predicting the time a diabetes patient will stay at a hospital is an important aspect of healthcare management and treatment. Using data obtained from 130 hospitals and integrated delivery networks, this project examines the records of patients diagnosed with diabetes who underwent laboratory tests, received medications, and had hospital stays lasting up to 14 days.

It is important to have a good predictive model so that hospitals can more accurately allocate resources, such as beds, medications, and staff for better patient care. Failure to provide proper diabetes care not only increases the managing costs for the hospitals (as the patients are readmitted) but also impacts the morbidity and mortality of the patients, who may face complications associated with diabetes. This model can have various applications, including preventative care, cost management, resource optimization, and overall healthcare efficiency.

## **Methods**

In this section, we will go over the various techniques used for preprocessing, data exploration, and building our models.

**exploration**

For our exploration, we started by trimming down to the features we deemed interesting or worthy of analysis. These were 'race', 'gender', 'age', 'num_procedures', 'num_medications', 'number_emergency', 'number_diagnoses', 'diabetesMed', 'readmitted', 'time_in_hospital.' We then visualized them using a heatmap of the correlation matrix and a pairplot using the seaborn library. We did a lot of our preprocessing prior to exploration to facilitate this. For example dropping columns with null values and encoding our non-numerical values. When we made the correlation matrix and heatmap we made a few observations, which can be seen [here.](https://colab.research.google.com/drive/1O5QPe9oywuKqfh5coIJUJ67HEC2UzZCR#scrollTo=sL_QoLrghkGQ)
<br>

**preprocessing**

For our preprocessing we did the following. In the original dataset the 'readmitted' column was the 'y' and everything else was the 'X.' We wanted to predict 'time in hospital' so we reattached readmitted to X and separated out 'time in hospital' to y.

Our unprocessed data had a lot of non-numerical categories and we fixed all of these in different ways. We label encoded the gender column, setting 'male' to 1 and 'female' to 0 as we iterated through the dataframe. Then we onehot encoded the race section using pd.getdummies. The 'readmitted' column had one value for readmitted in < 30 days, one for > 30 days, and one for never readmitted. We label encoded these as 0, 1, and 2 respectively, matching the numerical hierarchy. The values in the column age all described a 10 year range in a form 20-30, so we replaced those with the average of that range. For example, 20-30 was replaced with 25. This was done with simple iteration through the column.

Our dataset was extremely large with around 100,000 entries. Because our dataset was so big, we felt comfortable dropping the rows with "NaN" or other null values.
<br>

**model 1 methods**

For our first model we did polynomial regression of various degrees, starting with 1st degree (linear) and going up to degree 5. We used the sklearn regression libraries for this and split the data into train and test sets using our [own code](https://colab.research.google.com/drive/1O5QPe9oywuKqfh5coIJUJ67HEC2UzZCR#scrollTo=jjM6gJUQMW3M&line=4&uniqifier=1) with an 80/20 split. We evaluated and compared these polynomial models using both train MSE and test MSE.

**model 2 methods**

For our second model we used a neural network and tuned it with various activation functions and neuron layouts. In addition, we did categorical classification by changing the size of the output layer and one-hot encoding our ytrain, utilizing the fact that 'time in hospital' was always an integer value between 1 and 14. These models were evaluated with MSE for regression versions and accuracy/classification reports for the categorical classification versions.

**model 3 methods**

For our third model we utilized the imbalanced learn

## Results

**model 1 results**

Model 1 is set to perform linear regression and polynomial regression. Linear regression produced a MSE of 6.76 on the test subset. Polynomial regression is then used to evaluate the model performance by examining potential underfitting and overfitting. Polynomial regression of degree 2 has the best MSE value of 6.56 on the test subset. Since we are predicting the length of stay in days for a diabetes patient, linear regression and polynomial regression evaluate the correlation between the input features and the out target. The resulting MSE can be used as a reference in helping evaluate other models. The performance of linear regression and polynomial regression models shows a low correlation between the input features and the out target.

**model 2 results**

Model 2 is a neural network sequential classifier. To address the low correlation problem elevated by model 1, we choose to build a neural network. Neural network functions to determine a suitable weight for each input feature. We build the neural network to increase the weight of more prominent features in seeking a better model performance. However, we are still only able to get a MSE of 6.59. We further examine the result. The predicted result has an average error of  ± 2 days, and, by visualizing the error, the majority of the predicted result has an error within the range of ± 2.5 days. In accomplishing our task, the result was not unacceptable if the result is only used for a suggestion.

**model 3 results**

Model 3 includes a decision tree classifier and a decision tree regressor. For the decision tree classifier, we are expecting an inferior performance as our predicting target does not have a good affinity with the classifier, yet we still build the classifier for a reference in evaluating other models. As expected, the decision tree classifier resulted in a MSE of 11.62. For the decision tree regressor, the MSE on the test subset is 6.71. 

For all 3 models, we are getting a MSE around 6.5, which indicates our models do not have great performance. This could potentially be the limitation of the dataset. Nonetheless, considering our task does not rely on an accurate prediction, our models can provide a rough suggestion on the time of stay for diabetes patients. 

## Discussion

## Collaboration

## Collaboration
