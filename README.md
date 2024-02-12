# CSE151A_project
Repository containing the final group project code for CSE151A

# Citations
The Diabetes Dataset:
Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.

# Editing and Commiting.
To open the file and make shared changes, click on CSE151A_Project.ipynb and then click the Open in Colab button. Or just click below:\
<a target="_blank" href="https://colab.research.google.com/github/Li-Kane/CSE151A_project/blob/main/CSE151A_Project.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>\
To commit to this repository, simply on the Google Colab go to file->Save a Copy in Github

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
