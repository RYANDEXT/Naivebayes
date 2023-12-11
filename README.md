# GaussianNB
Gaussian Naive Bayes (GaussianNB) classifier 

Simple Gaussian Naive Bayes classifier implementation. It also implements 5-fold cross-validation and confusion_matrix 

## Dataset

Heart_disease_cleveland_new.csv

Attribute and Class Information:
   

Attribute Information:
  The data set was downloaded from the UCI website.

  age: age in years
  sex: sex (1 = male; 0 = female)
  cp: chest pain type
    Value 1: typical angina
    Value 2: atypical angina
    Value 3: non-anginal pain
    Value 4: asymptomatic
  trestbps: resting blood pressure (in mm Hg on admission to the
  hospital)
  chol: serum cholestoral in mg/dl
  fbs: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
  restecg: resting electrocardiographic results
    Value 0: normal
    Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
  thalach: maximum heart rate achieved
  exang: exercise induced angina (1 = yes; 0 = no)
  oldpeak: ST depression induced by exercise relative to rest
  slope: the slope of the peak exercise ST segment
    Value 1: upsloping
    Value 2: flat
    Value 3: downsloping
  ca: number of major vessels (0-3) colored by flourosopy (for calcification of vessels)
  thal: results of nuclear stress test (3 = normal; 6 = fixed defect; 7 = reversable defect)
  num: target variable representing diagnosis of heart disease (angiographic disease status) in any major vessel
    Value 0: < 50% diameter narrowing
    Value 1: > 50% diameter narrowing
