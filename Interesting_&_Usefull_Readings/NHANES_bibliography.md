# NHANES Dataset Analysis Documentation
> Focus on cardiovascular biomarkers, frailty assessments, and neurological markers

## Table of Contents
- [Overview](#overview)
- [Dataset Information](#dataset-information)
- [Research Focus](#research-focus)
- [Methodologies](#methodologies)
- [Tools and Technologies](#tools-and-technologies)
- [Bibliography](#bibliography)

## Overview
This research project leverages the National Health and Nutrition Examination Survey (NHANES) dataset to investigate the relationships between cardiovascular biomarkers, frailty indicators, and neurological markers in the US population. The study aims to comprehensively analyze these health parameters by utilizing various statistical and analytical techniques.
The primary focus areas include:
- Analysis of cardiovascular biomarkers to assess heart health and disease risk
- Evaluation of frailty measurements and their correlations with other health indicators
- Investigation of neurological markers where available in the dataset
This documentation serves as a comprehensive record of the analytical methods, tools, and bibliographic resources used throughout the research process. The project emphasizes the technical aspects of data analysis, including statistical methodologies, programming approaches, and data processing techniques employed to extract meaningful insights from the NHANES dataset.

## Association Between Oral Health and Frailty Among American Older Adults

### Dataset Information
- 2011-2014
- 2368 community-dwelling adults aged 60 years and older

### Variables of Interest
#### Frailty Outcome
- Frailty was measured with the 49-item frailty index:
    1. cognition
    2. dependence
    3. depression
    4. comorbidities
    5. hospital utilization
    6. general health
    7. physical performance
    8. anthropometry
    9. laboratory values
- A value between 0 and 1 was assigned according to the severity of the deficit
- The frailty index value is expressed as the number of the acquired de cits by the participant divided by the total number of potential deficits

#### Explanatory Variables
- Number of teeth
- Periodontal disease (in this study was categorized into 2 groups,“No/Mild periodontitis” and“Moderate/Severe periodontitis.")

#### Covariates
- Gender
- Ethnicity
- Education
- Poverty-income ratio
- smoking status

### Research Focus
The research question of this study is whether oral health indicators, namely number of teeth and periodontal disease, are associated with frailty index among older American adults?

### Methodologies
#### Data Preprocessing
- excluding participants who did not have complete oral health examination or had missing data in any other covariate
-  All participants with fewer than 2 teeth were excluded from the periodontal disease analysis 
- If participants did not fall in any of
the definitions the authors used for perioodontitis, they were de ned in“no periodontitis” group
- They created a nutritional intake
variable by summing up the 13 micronutrient variables (0 indicates
adequate intake of all the nutrients, 13 indicates inadequate intake of all the nutrients)

#### Statistical Analysis
- Descriptive statistics were carried out stratified by frailty status
- **Negative binomial regression** was used to test the association between frailty index and each of the oral health indicators, namely number of teeth (0-32) and periodontitis (No/Mild versus Moderate/
Severe periodontitis).
- For each oral health indicator, 3 models were constructed: adjusted according to the covariates
- STATA version 16s

#### Machine Learning Approaches 
- no machine learning integration was performed

### Results and Findings
#### Key Outcomes
- This study demonstrated that in a nationally representative sample of older American adults, oral health indicated by tooth loss and periodontal disease was associated with frailty index. 
- The significant association between tooth loss and frailty persisted even after adjusting of important covariates.
- Oral health is associated with the frailty index, and nutritional intake appears to have a modest effect on the association. 
- Periodontal disease has a weaker association with frailty compared with number of teeth.  

#### Visualizations
- 3 tables with Mean, Max-Min range, p-value and rate ratio (RR):
    1. Demographic, Socioeconomic and Oral Health Caharacteristics
    2. Negative Binomial Regression Models Showing the Associations Between Number of Teeth and Frailty
    3. Negative Binomial Regression Models Showing the Associations Between Periodontal Disease and Frailty Index

## Periodontitis and low cognitive performance: A population-based study

### Dataset Information
- NHANES 2011-2014 database
- 2086 participants ≥60 anni
- Representative of 77.1 million US non-institutionalized adults

### Variables of Interest
#### Cognitive Performance Outcome
- Cognitive tests:
    1. CERAD-WL (Consortium to Establish a Registry for Alzheimer's diseas)
    2. AFT (animal fluency test)
    3. DSST (digit symbol substitution test)
    4. Global cognition score (Standardized sum of the preceding)

#### Explanatory Variables
- Periodontal state (none/mild, moderate, severe according to criteria CDC/AAP)
- Average PPD (probing pocket depth)
- Average CAL (clinical attack level)

#### Covariates
1. Age
2. Gender
3. Smoking status
4. Level of family poverty
5. Education level
6. Consumption of alcohol

### Mediators
- Diabetes
- Hypertension
- Cardiovascular/cerebrovascular disease
- Systemic inflammation marker (counts white blood cells and platelets)

### Research Focus
Study the association between periodontitis and low cognitive performance in elderly adults ( 60 years) in a representative USA sample.

### Methodologies
#### Data Preprocessing
- Excluding edentulous subjects
- Excluded subjects without complete periodontal examination
- Subjects who have not completed at least one cognitive test are excluded
- Dichotomized cognitive performance using the unweighted lower quartile
- Global cognition score calculation as the sum of the standardized z-scores of CERAD-WL, AFT and DSST

#### Statistical Analysis
- Continous variables expressed with mean and std, categorical variables expressed with proportions and standard error
- Simple and Multiple Regression to assess association between periodontitis and low cognitive performance
- Analysis adjusted for confounding a priori
- Logistic Regression for mediators analysis to study the role of each potential mediator
- Software: STATA BE v17.1

#### Machine Learning Approaches 
- no machine learning integration was performed

### Results and Findings
#### Key Outcomes
- Moderate and severe periodontitis significantly associate low DSST performance
- Stronger associations for the women
- Average CAL associated with low performance in global cognition, AFT and DSST

#### Visualizations
- 4 detailed tables with OR, IC95% and p-value
- 1 figure with cognitive test distributions per each periodontal state

## A systematic comparison of machine learning algorithms to develop and validate prediction model to predict heart failure risk in middle-aged and elderly patients with periodontitis (NHANES 2009 to 2014)

### Dataset Information
- NHANES 2009-2014
- Total 2876 participants with periodontitis
- Training set: 1980 subjects (2009-2012)
- Validation set: 896 subjects (2013-2014)

### Variables of Interest
#### Outcome
- Heart failure risk in middle-aged and elderly patients with periodontitis. Binary class.

#### Independent Variables
Key predictors identified: 
- Age
- Race
- Myocardial infarction status
- Diabetes mellitus status

#### Covariates
- Demographics: gender, education, marital status
- Clinical: BMI, waist circumference, hypertension, CHD
- Lifestyle: smoking status, alcohol consumption
- Socioeconomic: poverty-to-income ratio
- Health behaviors: sleep time, physical activity, sedentary time

### Research Focus
The goal of this study was to develop and validate a prediction model based on machine learning algorithms for the risk of heart failure in middle-aged and elderly participants with periodontitis.

### Methodologies
#### Data Preprocessing
- Patients with age > 40 years
- Patients with missing data were excluded
- Categorization of variables:
    - smoking status
    - alcohol consumption
    - Diabetes

#### Statistical Analysis
-  Description and logistic regression analysis to achieve nationally representative values
    - The t test was used to compare continuous variables between the 2 groups.
    - The chi-square test or Fisher exact test was used to determine the differences between groups when comparing categorical variables between the 2 groups.
    - Odds ratio (OR) and 95% confidence intervals (CI) were utilized as effect estimates and P < .05 was considered statistically significant. 
- Univariate and multivariate logistic regression analysis to identify the independent risk factors for heart failure 
- R software (version 4.3.0)

#### Machine Learning Approaches 
Machine Learning Models Compared:
1. Logistic regression
2. K-nearest neighbor
3. Support vector machine
4. Random forest
5. Gradient boosting machine
6. Multilayer perceptron

Validation Approach:
- 10-fold cross-validation on training set
- External validation on 2013-2014 dataset
- Performance evaluated using ROC curves and AUC

### Results and Findings
#### Key Outcomes
1. The independent predictors of the risk of heart failure in participants with periodontitis were age, race, myocardial infarction, and diabetes mellitus status.
2. Training set (10-fold cross-validation):
    - K-nearest neighbor: 0.936 (best performing)
    - GBM: 0.927
    - Random forest: 0.889
    - SVM: 0.859
    - Logistic regression: 0.848
    - MLP: 0.666
3. External validation:
    - K-nearest neighbor: 0.949 (best performing)
    - Random forest: 0.933
    - Logistic regression: 0.854
    - GBM: 0.855
    - MLP: 0.74
    - SVM: 0.647
4. Variable importance ranked in descending order:
    - Myocardial infarction
    - Age
    - Diabetes
    - Race

#### Visualizations
- Receiver operating characteristic curve analysis is used in the external validation set to check the performance of each model.
- Tables for demographic information, univariate and multivariate logistic regression analysis


## Prognostic significance of subjective oral dysfunction on the all-­cause mortality

### Dataset Information
- NHANES database 1999-2002
- Total of 7827 participants who completed oral functions data
- Follow-up period: from baseline to death or December 31, 2006

### Variables of Interest
#### Primary Outcome
- All-cause mortality

#### Subjective Oral Dysfunction Components
Three key components assessed via self-reported questionnaire:
- Limited eating ability
- Dry mouth
- Difficult swallowing

#### Covariables
- Socio-demographic: age, sex, race, smoking history
- Medical comorbidities
- Recreational activity
- Biochemistry profiles:
    - Creatinine
    - Alanine aminotransferase
    - Serum fasting glucose
    - Total cholesterol
    - Total calcium
- Muscle measurements:
    - Muscle strength (quadriceps)
    - Appendicular skeletal muscle mass

### Research Focus
To examine the association between subjective oral dysfunction and all-cause mortality in the US population. 

Honorable mention for this paper for a well-done introduction: the different indices to evaluate frailty are well exposed and it is mentioned the longitudinal study in Japan that defined oral frailty as containing 3 or more of 6 characteristics (included the number of natural teeth, tongue pressure, articulatory skill, chewing ability and perceived eating and swallowing difficulties). 

"Each frailty scale focused on different domains such as biological, physical, cognitive and deficit accumulation."

"Oral hypofunction was defined by the Japanese Society of Gerodontology (JSG) including seven components: oral dryness, poor oral hygiene, decreased occlusal force, decreased tongue pressure, reduced tongue-­lips motor function, decreased chewing and swallowing function."

### Methodologies
#### Data Preprocessing
- They excluded individuals who lacked data for covariables
- It appears to be a complete case analysis where they simply excluded participants with missing data on key variables.
- They classified study population into 4 groups:
    - group 1 (without any components of subjective oral dysfunction), 
    - group 2 (with one component of subjective oral dysfunction), 
    - group 3 (with two components of subjective oral dysfunction), 
    - group 4 (with 3 components of subjective oral dysfunction).

#### Study Design
- Cross-sectional observational study

#### Assessment Methods
- Subjective oral dysfunction assessment:
    1. Limited eating ability: "always, very often and often" responses
    2. Dry mouth: "yes" responses
    3. Difficult swallowing: "yes" responses
- Muscle strength: Isokinetic strength testing of right quadriceps
- Muscle mass: Dual-energy X-ray absorptiometry (DEXA)

#### Statistical Analysis
- SPSS version 18
- One-way ANOVA and chi-square test were applied for analysing socio-­demographic characteristics, laboratory variables and medical comorbidities. 
- Kaplan-Meier survival curves
- Cox proportional hazard models to assess the relationship between subjective oral dysfunction and all-­ cause mortality 
- Three covariate-adjusted models:
    1. Model 1: unadjusted
    2. Model 2: adjusted for basic variables
    3. Model 3: fully adjusted model

#### Machine Learning Approaches 
- No Machine Learning algorithms implemented in this research.

### Results and Findings
#### Key Outcomes
- Significant relationship between subjective oral dysfunction and all-cause mortality in fully adjusted model
- Dose-dependent effect: more components of dysfunction associated with worse mortality risk
- HRs in fully adjusted model:
    - Group 2 (one component): 1.269
    - Group 3 (two components): 1.649
    - Group 4 (three components): 3.185
- Limited eating ability inversely associated with muscle strength

#### Visualizations
- Three Kaplan-Meier curves showing cumulative survival classified by each component of subjective oral dysfunction
- Tables showing:
    - Population characteristics
    - Hazard ratios of all-cause mortality
    - Associations between muscle strength/mass and oral dysfunction components


## Machine learning approaches for predicting frailty base on multimorbidities in US adults using NHANES data (1999–2018)

### Dataset Information
- NHANES data from 1999-2018 (10 survey cycles)
- Total sample of 46,187 adults (representative of 185,602,706 US adults)
- Initial sample of 116,876 participants, with 73,990 eliminated due to incomplete data
- Weighted mean age: 47.16 years
- Gender distribution: 21,613 (51.2%) women

### Variables of Interest

#### Frailty Outcome
- Modified 36-item deficit accumulation frailty index:
    1. Count the number of deficits present in each individual
    2. Divide this number by the total of possible deficits (36)
    3. The result is a score ranging from 0 to 1, where higher values indicate greater fragility.
- Frailty classification:
  - Non-Frailty (frailty index ≤ 0.10)
  - Pre-Frailty (frailty index > 0.10 to < 0.25)
  - Frailty (frailty index ≥ 0.25)

#### Variables used
- Demographics: Age, Sex, Ethnicity, Educational Level, Marital Status
- Chronic conditions (defined by laboratory indicators):
  - Chronic Kidney Disease (CKD)
  - Diabetes Mellitus (DM)
  - Hyperlipidemia
  - Nonalcoholic Fatty Liver Disease (NAFLD)
  - Hypertension
  - Anemia
- Self-reported conditions:
  - Chronic Obstructive Pulmonary Disease (COPD)
  - Stroke
  - Coronary Heart Disease (CHD)
  - Asthma
  - Congestive Heart Failure (CHF)
  - Parkinson's Disease
  - Arthritis
  - Epilepsy

### Research Focus
The study aimed to:
1. Analyse the individual impact of diseases on frailty in the presence of multimorbidities
2. Construct a predictive model for the early identification of frailty using machine learning approaches
3. Explore the non-linear relationship between age and frailty
4. Identify critical factors influencing the progression from non-frailty to pre-frailty and from pre-frailty to frailty

### Methodologies

#### Data Preprocessing
- Elimination of participants without complete data
- Frailty assessment using a modified 36-item deficit accumulation frailty index
- Disease identification through both laboratory indicators and self-reported medical history
- Data analyzed following analytical guidelines and recommended survey weights for NHANES data

#### Statistical Analysis
- Weighted means with standard error (SE) for continuous variables
- Unweighted frequencies with weighted percentages for categorical variables
- t-test for continuous variables
- Rao-Scott Chi-Square test for categorical variables
- Restricted cubic spline (RCS) to explore non-linear association between age and frailty
- Survey-weighted multivariable logistic regressions adjusted for potential confounders
- Statistical significance defined as P-value < 0.05
- All analyses performed using R (version 4.3.2)

#### Machine Learning Approaches
- Feature selection using three algorithms:
  1. Joint Mutual Information Maximisation (JMIM)
  2. Lasso regression (LR)
  3. Random forest (RF)
- Selection of features appearing twice or more in the top 10 of the three screening methods as inclusion features. Feature candidate those that appeared only once.
- Construction of 8 different feature combination models evaluated by AUC
- Nested cross-validation approach:
  - Inner loop: 10-fold cross-validation for hyperparameter optimization
  - Outer loop: 5-fold cross-validation for model validation
- Six machine learning algorithms tested:
  1. Decision tree
  2. Logistic Regression (LR)
  3. k-Nearest Neighbor (KNN)
  4. Random Forest (RF)
  5. Recursive Partitioning and Regression Trees (RPART)
  6. eXtreme Gradient Boosting (XGBoost)
- Model performance evaluated using:
  - Receiver operating characteristic curve (ROC)
  - Precision-recall curves (PRC)
  - Areas under the ROC (AUC)
  - Area under the PRC (AU-PRC)
  - Calibration curve analysis
  - Decision Curve Analysis (DCA)

### Results and Findings

#### Key Outcomes
1. **Age-Frailty Relationship**:
   - Non-linear association between age and frailty
   - Critical turning point at 49 years old
   - Before 49 years: age appears to function as a protective factor against frailty
   - After 49 years: increased susceptibility to frailty

2. **Key Impacting Variables on Frailty**:
   **Primary tier** (highest impact):
   - Anemia
   - Arthritis
   - Diabetes Mellitus
   - Coronary Heart Disease
   - Hypertension
   - Congestive Heart Failure
   - Stroke
   
   **Secondary tier** (moderate impact):
   - Parkinson's disease
   - COPD
   - Asthma
   
   **Tertiary tier** (minimal impact):
   - Hyperlipidemia
   - Chronic Kidney Disease
   - Non-Alcoholic Fatty Liver Disease

3. **Different Disease Impact by Frailty Stage**:
   - In transition from non-frailty to pre-frailty: chronic conditions had greater impact
   - In transition from pre-frailty to frailty: acute conditions or complications (CHF, stroke, CHD) had greater impact

4. **Machine Learning Model Performance**:
   - XGBoost model showed highest performance (AUC = 0.8828 and AU-PRC = 0.624)
   - Final set of 13 predictive variables: Age, DM, CKD, COPD, Stroke, Hypertension, CHF, CHD, Arthritis, Anemia, Ethnicity, Educational Level, and Sex
   - Approximately 31,900 models built and evaluated during the optimization process

#### Visualizations
- **Figure 1**: Prevalence of Frailty in sociodemography showing:
  - Mean frailty index by age and sex
  - Frailty prevalence by age and sex
  - The non-linear association between age and frailty
  - Frailty prevalence by educational level, marital status, and ethnicity

- **Figure 2**: Adjusted logistic analysis showing the association between frailty and multimorbidity:
  - Part A: Adjusted multimorbidity logistic regression of non-Frailty versus pre-Frailty
  - Part B: Adjusted multimorbidity logistic regression of pre-Frailty versus Frailty

- **Figure 3**: Feature selection process:
  - Feature ranking via Random Forest, JMIM, and lasso regression
  - AUC values of different feature combination models

- **Figure 4**: Performance evaluation of models:
  - ROC comparison of 6 machine learning algorithms
  - AU-PRC comparison of 6 algorithms
  - Calibration curve for the XGBoost model
  - Decision curve analysis of XGBoost model

### Conclusions
1. Age 49 represents a critical threshold where physiological reserves are sufficient to resist external stressors; beyond this age, the cumulative effects of aging and external challenges overpower the body's innate resilience.

2. Chronic diseases (e.g., anemia, arthritis, diabetes) are trigger factors of frailty, while acute diseases or exacerbations (e.g., CHF, stroke) are contributing factors that accelerate the body's decline.

3. The XGBoost frailty prediction model, with its high performance (AUC = 0.8828), simplicity, and clinical value, holds potential for practical implementation in healthcare settings.

4. The model's accuracy, coupled with low data collection costs and ease of use, makes it highly applicable in clinical and community settings for early intervention and improved patient outcomes.


## Association between frailty index and cognitive dysfunction in older adults: insights from the 2011–2014 NHANES data

### Dataset Information
- NHANES data from 2011-2014 (2 survey cycles)
- Total sample of 2,574 adults aged 60 years and older

### Variables of Interest

#### Frailty Outcome
- Frailty index based on 70 health deficiencies across physical, functional, psychological, and social health variables
- Calculation formula: frailty index = total number of deficits present/total number of deficits measured
- Classification thresholds:
  - Frailty (frailty index ≥ 0.25)
  - Pre-frailty (frailty index: 0.12-0.25)
  - No frailty (frailty index < 0.12)

#### Variables used
- Demographics: Age, Gender, Race/Hispanic origin, Marital status, Education, Ratio of family income to poverty
- Health behaviors: Physical activity, Smoking status (≥100 cigarettes in life), Alcohol consumption (≥12 drinks/year)
- Medical conditions: Hypertension, Diabetes, Daily low-dose aspirin use
- Cognitive measures:
  - Animal Fluency (AF): verbal fluency test
  - Digit Symbol Substitution Test (DSST): processing speed and working memory
  - Consortium to Establish a Registry for Alzheimer's Disease Word Learning (CERAD-WL): immediate learning
  - CERAD Delayed Recall (CERAD-DR): delayed memory

### Research Focus
The study aimed to:
1. Investigate the relationship between frailty index and cognitive dysfunction in older adults
2. Evaluate the influence of covariates on this relationship through subgroup analysis and interaction
3. Explore potential non-linear relationships between frailty index and cognitive function
4. Determine whether different cognitive domains are affected differently by frailty status

### Methodologies

#### Data Preprocessing
- Exclusion of participants under age 60
- Removal of cases with missing data on key covariates
- Exclusion of participants lacking frailty index information
- Exclusion of those who failed to complete all four cognitive tests
- Application of sampling weights using the formula: wt = 1/2 * WTMEC2YR to account for complex survey design

#### Statistical Analysis
- Weighted means with standard deviation for continuous variables
- Unweighted counts with weighted percentages for categorical variables
- Chi-square tests for categorical variables and T-tests for continuous variables
- Weighted logistic regression analysis with three models:
  - Crude model: no adjustments
  - Model 1: adjusted for age, gender, and race
  - Model 2: fully adjusted for all covariates
- Frailty index categorized into tertiles for sensitivity analysis
- Subgroup analysis and interaction testing across demographic and health characteristics
- Restricted cubic spline regression model to explore non-linear relationships
- Two-sided statistical testing with significance at p < 0.05
- All analyses performed using R (version 4.2.3)

#### Machine Learning Approaches
- Not applied to this study

### Results and Findings

#### Key Outcomes
1. **Frailty-Cognitive Function Association**:
   - Significant association between frailty index and all cognitive test scores
   - Odds ratios for cognitive dysfunction with increasing frailty index
   - Highest tertile of frailty index showed significantly increased risk of cognitive dysfunction compared to lowest tertile.

2. **Non-linear Relationship**:
   - Significant non-linear association between frailty index and Animal Fluency (p for non-linearity < 0.001)
   - Risk of cognitive dysfunction began to increase when frailty index exceeded 0.19

3. **Interaction Effects**:
   - For DSST, significant interactions were found with race, marital status, education level, income-to-poverty ratio, hypertension, and diabetes
   - No significant interactions were observed for Animal Fluency test, suggesting a more universal association

#### Visualizations
- **Figure 1**: Flowchart of patient exclusion showing the selection process from initial 19,931 participants to final 2,574 study population
- **Figure 2**: Forest plot of subgroup analysis illustrating the relationship between frailty index and cognitive function stratified by demographic and health characteristics
- **Figure 3**: Restricted cubic spline plot demonstrating the non-linear relationship between frailty index and risk of cognitive dysfunction, with a marked threshold effect around 0.19


## Paper Title

### Dataset Information
- List of specific NHANES cycles analyzed
- Sample size and demographic information
- Data collection periods

### Variables of Interest
#### Frailty Outcome
- Frailty indices used

#### Variables used

### Research Focus
Detailed description of research questions and hypotheses.

### Methodologies
#### Data Preprocessing
- Data cleaning procedures
- Missing data handling
- Outlier detection and treatment

#### Statistical Analysis
- Statistical methods employed
- Software packages and versions
- Analysis pipelines

#### Machine Learning Approaches 
- Algorithms used
- Feature selection methods
- Model validation techniques

### Results and Findings
#### Key Outcomes
- Major findings
- Statistical significance
- Clinical relevance

#### Visualizations
- Key figures and plots
- Data interpretation
- Meaningful patterns

## Bibliography
```
[1] Faisal F. Hakeem MSc a, b, *, Eduardo Bernabé PhD a, Wael Sabbah PhD a. (2020). Association Between Oral Health and Frailty Among American Older Adults. JAMDA. https://doi.org/10.1016/j.jamda.2020.07.023
```
```
[2] Crystal Marruganti, Giacomo Baima, Mario Aimetti, Simone Grandini, Mariano Sanz, Mario Romandini (2023). Periodontitis and low cognitive performance: A population-based study. J Clin Periodontol. https://doi.org/10.1111/jcpe.13779
```
```
[3] Yicheng Wang, Yuan Xiao, Yan Zhang (2023) A systematic comparison of machine learning algorithms to develop and validate prediction model to predict heart failure risk in middle-aged and elderly patients with periodontitis (NHANES 2009 to 2014). Medicine (Baltimore). 10.1097/MD.0000000000034878
```
```
[4] Zhe-­Yu Yang, Wei-­ Liang Chen (2021) Prognostic significance of subjective oral dysfunction on the all-­cause mortality. Wiley, J Oral Rehabilitation. 10.1111/joor.13281
```
```
[5] Teng Li, Xueke Li, Haoran XU, Yanyan Wang (2024) Machine learning approaches for predicting frailty base on multimorbidities in US adults using NHANES data (1999–2018). Computer Methods and Programs in Biomedicine. https://doi.org/10.1016/j.cmpbup.2024.100164
```
```
[6] Jianlong Zhou, Yadi Li, Lv Zhu and Rensong Yue (2024) Association between frailty index and cognitive dysfunction in older adults: insights from the 2011–2014 NHANES data. Frontiers in Aging Neuroscience. 10.3389/fnagi.2024.1458542
```

--------------------------------------

## Notes
- Last updated: [28/02/2025]
- Contact: [silvano.quarto@gmail.com]
- Project Status: [sixth paper added]