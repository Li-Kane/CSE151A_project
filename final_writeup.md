# Final Writeup

## Introduction
Predicting the time a diabetes patient will stay at a hospital is an important aspect of healthcare management and treatment. Using data obtained from 130 hospitals and integrated delivery networks, this project examines the records of patients diagnosed with diabetes who underwent laboratory tests, received medications, and had hospital stays lasting up to 14 days.

It is important to have a good predictive model so that hospitals can more accurately allocate resources, such as beds, medications, and staff for better patient care. Failure to provide proper diabetes care not only increases the managing costs for the hospitals (as the patients are readmitted) but also impacts the morbidity and mortality of the patients, who may face complications associated with diabetes. This model can have various applications, including preventative care, cost management, resource optimization, and overall healthcare efficiency. Our project focuses on predicting the length of hospital stays for diabetes patients, a task that is pivotal in enhancing patient care and resource allocation within healthcare facilities. By analyzing a comprehensive dataset comprising 101,766 patient records with 47 features, we aim to develop a predictive model that effectively forecasts hospital stay durations based on key patient characteristics and medical history.

## **Methods**

In this section, we will go over the various techniques used for preprocessing, data exploration, and building our models.

**exploration**

For our exploration, we started by trimming down to the features we deemed interesting or worthy of analysis. These were 'race', 'gender', 'age', 'num_procedures', 'num_medications', 'number_emergency', 'number_diagnoses', 'diabetesMed', 'readmitted', 'time_in_hospital.' We then visualized them using a heatmap of the correlation matrix and a pairplot using the seaborn library. We did a lot of our preprocessing prior to exploration to facilitate this. For example dropping columns with null values and encoding our non-numerical values. To make sure we didn't drop any important data, we plotted the correlation matrix as a heatmap for both the features we used (left) an the features we dropped (right). 
<p float="left">
  <img src="/images/features-heatmap.png" width="400" />
  <img src="/images/all-features-heatmap.png" width="400" /> 
</p>
As shown in the heatmaps, the features we dropped had less correlataion than those we used. </br>

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

For our third model we utilized a Decision Tree Regressor using Grid Search with cross-validation. It searches for the best combination of hyperparameters specified in the param_grid dictionary, evaluates them using negative mean squared error, and outputs the best parameters along with the mean squared training error.

## Results

**model 1 results**
Overall, the best MSE for our linear and polynomial regression model using OLS was around 6.56 on the test, for a standard deviation of around 2.56 days on our model predictions. Increasing the degree of the polynomial improved the performance a little but eventually caused overfitting at degree four. Degree 3 gave us the best results.  
<img src="/images/linear-poly-loss.png" width="400" />

**model 2 results**
Our neural network achieved similar MSE to our linear/polynomial models when used for regression. When we used it for categorical classification by one-hot encoding y, we reached 22% accuracy with an MSE of around 6.3 when we multiply the probability predictions by the days of each class.  
<img src="/images/model-loss.png" width="400" />

**model 3 results**
The decision tree model had a best MSE of 6.711323110010316. Even when we were to use the entire dataset without dropping columns, MSE at the lowest was around 4, for a standard deviation of 2 days on the model error and a classification accuracy of up to 27%, decision tree with entire data had MSE of ~5.4.  
<img src="/images/precision-recall.png" width="400" />


## Discussion

**model 1 discussion**

Model 1 is set to perform linear regression and polynomial regression. Linear regression produced a MSE of 6.76 on the test subset. Polynomial regression is then used to evaluate the model performance by examining potential underfitting and overfitting. Polynomial regression of degree 2 has the best MSE value of 6.56 on the test subset. Since we are predicting the length of stay in days for a diabetes patient, linear regression and polynomial regression evaluate the correlation between the input features and the out target. The resulting MSE can be used as a reference in helping evaluate other models. The performance of linear regression and polynomial regression models shows a low correlation between the input features and the out target.

**model 2 discussion**

Model 2 is a neural network sequential classifier. To address the low correlation problem elevated by model 1, we choose to build a neural network. Neural network functions to determine a suitable weight for each input feature. We build the neural network to increase the weight of more prominent features in seeking a better model performance. However, we are still only able to get a MSE of 6.59. We further examine the result. The predicted result has an average error of  ± 2 days, and, by visualizing the error, the majority of the predicted result has an error within the range of ± 2.5 days. In accomplishing our task, the result was not unacceptable if the result is only used for a suggestion.

**model 3 discussion**

Model 3 includes a decision tree classifier and a decision tree regressor. For the decision tree classifier, we are expecting an inferior performance as our predicting target does not have a good affinity with the classifier, yet we still build the classifier for a reference in evaluating other models. As expected, the decision tree classifier resulted in a MSE of 11.62. For the decision tree regressor, the MSE on the test subset is 6.71.

For all 3 models, we are getting a MSE around 6.5, which indicates our models do not have great performance. This could potentially be the limitation of the dataset. Nonetheless, considering our task does not rely on an accurate prediction, our models can provide a rough suggestion on the time of stay for diabetes patients. 

## Conclusion

At the end of the day, our model had a typical error of within +/-2.5 days of the truth, and while our attempts to improve it didn't make it much worse, it didn't get much better either. I think if I asked a doctor "how long will I be in here" and he could guess within 2.5 days the length of my stay, I'd be pretty happy with that answer and our data wasn't very correlated to start with, but its frustrating to fail to improve it after multiple tries.

One thing we could have done better is we made a lot of decisions regarding what we'd predict and what features we used at the start, and then we ended up with all these low correlation features. We would've liked to check the whole data set's correlation with our target rather than cutting so much stuff out first.

## Collaboration

Lorentz (dropped the class, but we're crediting him) - data verification in milestone 2.

Asher - Data preprocessing in milestone two. Leadership and organization.

Shijun - Data preprocessing and writeup.

Rocky - Data exploration, model 2, and writeup.

Javier - Model 2, Introduction, and Methods 

Su - Neural Network Models, Decision Tree Regressor, and write up.

Wesley - Data preprocessing, model 2, model 3.
