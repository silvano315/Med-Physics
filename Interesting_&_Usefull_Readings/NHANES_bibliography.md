# NHANES Dataset Analysis Documentation
> Focus on cardiovascular biomarkers, frailty assessments, and neurological markers

## Table of Contents
- [Overview](#overview)
- [Dataset Information](#dataset-information)
- [Research Focus](#research-focus)
- [Methodologies](#methodologies)
- [Tools and Technologies](#tools-and-technologies)
- [Bibliography](#bibliography)
- [Results and Findings](#results-and-findings)
- [Future Work](#future-work)

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

--------------------------------------

## Notes
- Last updated: [27/01/2025]
- Contact: [silvano.quarto@gmail.com]
- Project Status: [first paper added]