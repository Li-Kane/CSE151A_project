# CSE151A_project
Repository containing the final group project code for CSE151A  
**Members:** Lorentz Tuazon, Shijun Lun, Javier De La Cruz Zuniga, Rocky Hankin, Kane Li, Wesley Kiang, Su Aye, Asher James  
<a target="_blank" href="https://colab.research.google.com/github/Li-Kane/CSE151A_project/blob/main/CSE151A_Project.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Citations
The Diabetes Dataset:
Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.

# Milestone 4
In this milestone, we reflect on our first model and create a second one trying to improve on it in order to answer the following questions.

**Evaluate your data, labels and loss function. Were they sufficient or did you have have to change them?** <br>
    Using MSE for our data last time seemed reasonable given we were doing linear regression. Taking the square root of the MSE indicates our average prediction was two days off the actual result, which seems fine but we'd like to do better. We'll try some other loss functions and manipulating our data a bit as we proceed. 
Train your second model

Having already tried linear and polynomial regression and been dissatisfied, we're going to move up in complexity with a neural network. It's pretty clear to us that we were underfitting if anything. We tried a lot of different combinations for our neural network, with different amounts of neurons and optimizers and activation functions. For readability we've trimmed a lot of fat, a lot of the many models we've tried, also because they were spread across different group member's notebooks. We also tried a version where we converted y_test to a one-hot encoding since there were only 14 possible values for it in the dataset, and did categorical cross-entropy with that.

**Evaluate your model compare training vs test error** <br>
    Our models don't preform any better than the linear/polynomial regression we did last time. "Given my condition, how long will I be in the hospital for?" Our test MSE isn't much different from our previous models. In the last step we found a degree three polynomial was the most effective and none of our models have beaten it, only equaled it roughly. We appear stuck around the 6.5 mse (or 22% accuracy when we do categorical crossentropy). On the brightside, the gap between our training MSE and testing MSE remained about the same and pretty small, so we don't appear to be overfitting.

I think it's important to ask ourselves "what does 6.5 mse even mean?" Because that's pretty abstract. So what we did is we took one of our better models and plotted all the test set predictions on a scatterplot based on their error. The error not being the mean squared error, just the distance between the truth and the prediction. What we found is that for our massive dataset, the vast vast majority of our predictions fall within a couple days of the actual result. around 0-2.5 days. That's not so bad. But we would like to do better and try to narrow that down even more. 

**Where does your model fit in the fitting graph, how does it compare to your first model?** <br>
    Adding neurons and testing different activation functions had little effect on the output. We seemed stuck around the same 6.5ish mse, or 22% accuracy when we did categorical classification. So its tempting to say its in the same place as before. So far we've only seen overfitting with degree 4 polynomial regression in the last milestone. 
Did you perform hyper parameter tuning? K-fold Cross validation? Feature expansion? What were the results?

We did not for this milestone. We accomplished something similar to hyper parameter tuning by just trying a bunch of different stuff manually but not to the same scale or thoroughness. It didn't have any good results, really. Like we've said, we can't seem to break past the 6.5 mse we started with. A lot of things didn't make it worse, but that's all we can say about them.

**What is the plan for the next model you are thinking of and why?** <br>
    Scaling up complexity hasn't helped much, so we're thinking about maybe aggregating. We have a lot of models that all function 'ok,' so we could aggregate their predictions like we've seen in class. Hopefully their predictions are different in a way that their average converges towards the truth.

**Conclusion section: What is the conclusion of your 2nd model? What can be done to possibly improve it? How did it perform to your first and why?** <br>
    While we didn't get signs of overfitting, our problem just isn't much more complicated than a linear/polynomial regression. We're using a bunch of values to try and weight them and get a single value. Upping complexity doesn't seem to help. We're also dealing with low correlation between our features and target to begin with, so its unclear what the peak accuracy we're shooting for is. Still, we've learned a lot of things in class that we can try for our final submission.

# Milestone 3

**Model Evaluation**  
Since we are attempting to predict days in hospital, which is an int, we decided to start with linear regression. When we evaluated our model's performance, even though the Train and Test were relatively the same, the model accuracy wasn't very good. To see if this was overfitting or underfitting, we tried using polynomial regression and saw that even though a degree of 2 is marginally better, the results are much the same. Therefore, it can be concluded that the current structure and data aren't well suited to the task. For future improvements, we can try to expand our dataset (such as readd dropped columns), or perhaps use bagging, where we build models of different polynomial degrees or data and take the mean of their predictions.
```
Linear Regression Model evaluation:
Train MSE:  6.788051615805185
Train Standard Deviation:  2.60538895666701
Test MSE:  6.859473380135014
Test Standard Deviation:  2.618915481159026
```

**Future Models**  
The first future model we plan on trying is to use a classic ANN/DNN the output range only ranges between 1-14 (14 classes), which is relatively narrow. Doing so will allow us to make our model much more complex with many more different combinations of layer sizes, depth, activation functions, and more.  
Our second future model plan is to use another regression model, likely polynomial or ridge regression. Even though we used polynomial regression to test how our linear regressor was working, there is much to expand upon with an improved dataset. However, since it is very similar to linear regression and what we have done so far, we may also try another form of regression such as ridge or lasso.

# Milestone 2
The task we want to perform with the above dataset is to predict the duration of a diabetes patient in the hospital. 

The key features we chose from the dataset were: 
- race: categorical variable with values Caucasian, Asian, African American, Hispanic, & other
- gender: categorical variable with values male, female, & unknown/invalid
- age: categorical variable with values grouped in ten-year intervals [0,10), [10,20), ..., [90,100)
- weight: categorical variable with values representing weight in pounds
- num_procedures: integer value indicating number of procedures performed
- num_medications: integer value indicating number of distinct generic medications administered
- num_diagnoses: integer value indicating number of diagnoses entered to system
- number_emergency: number of patient's emergency visits in preceding year
- medical_speciality: categorical variable with integer identifiers of admitting physician's specialty
- diabetesMed: categorical variable with values yes & no indicating if diabetic medication was prescribed
- readmitted: categorical variable with values <30, >30, & NO indicating days to inpatient readmission

Our target variable was chosen to be:
- time_in hospital: integer value of days between admission & discharge

Originally, the target variable given in the dataset was 'readmitted' but we joined this variable to the features dataframe & made 'time_in_hospital' as the target variable. 
After extracting the above desired features & assigning the dependent variable, we found missing values in the following features:
- race: 2273 NA values
- medical_specialty: 49949 NA values
We dropped rows with missing 'race' values & due to the high number of NA values in the 'medical_specialty' feature, we dropped the feature entirely.

We encoded the following categorical variables in the following manner:
- race: one hot encoding
- gender: converted to binary numerical values with 0 indicating female & 1 male
- age: given entries are intervals, we encoded these intervals with their midpoint
- diabetes_med: converted to binary numerical values with 0 indicating no, 1 yes
- readmitted: converted to numerical labels with 0 indicating <30, 1 for >30, 2 for NO

In order to improve training stability & performance, we used the MinMaxScalar to normale the following features: age, num_procedures, num_medications, number_emergency, & number_diagnoses. We also used StandardScalar to standardize the entire datset to a normal distribution. 

To visualize the data, we first ensure that all entries are numerical. We then used a heatmap to visualize a Pearson correlation matrix followed by a pairplot & made the following observations:
- Most features do not have strong correlations with each other or the target variable
- The feature with the highest correlation with the target variable is num_medications with a value of 0.5
- number_emergency & num_medications have a distinctive scatter plot shape, & it clearly illustrates that patients with a higher number of emergency visits in the preceding year tend to have a lower number of medications prescribed
- num_medications & age also have a distinct shape in their scatterplot in spite of having a low correlation on the heatmap, with increased prescriptions peaking in the age interval [60,70) then steeply declining
- due to the scatterplot of time_in_hospital vs. num_procedures indicating an even distribution across their numerical categorical values, further visual investigation can be done via a density plot
- num_procedures is heavily positively skewed (right-skewed) towards the highest value of 6
- caucasians also vastly overrepresent the majority of observations in the dataset with approximately a ratio of 3:1 as compared to other racial categories

The dataset is split into training & testing sets with a ratio of 8:2.

During the data verification step, the following issues came up:
- the datatype of the gender column, i.e. 'object', indicates that there were values unaccoutned for in the cleaning; the rows with 'Unknown/Invalid' entries in the 'gender' column were droppped
- we tested for normality by visualizing QQ plots & performing the Shapiro-Wilkes test to find out each features p-values. All p-values fall below the range of <0.05, indicating that the features were not pulled from normal distributions & that we need to revisit our implementation of how we normalized & standardized our features, or in how we encoded them, i.e. possibly using one hot encoding instead of converting to numerical values in spite of the dataset's increased dimensionality in doing so. 
- We may also want to revisit our approach to imputing our data, as dropping the rows & an entire feature may have compromised the dataset's integrity
- The low correlation between the featuers & the target variable may also indicate earlier issues in how we chose our features or encoded them.
