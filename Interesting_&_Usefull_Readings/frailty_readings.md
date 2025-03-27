# Brain health assessment in cardiometabolic diseases

Hi everyone, I'm a Researcher and Data Scientist at the Healthcare hospital in Bari. I'm working on the PNRR project "Brain health assessment in cardiometabolic diseases: impact of frailty and biomarkers on healthcare management". I perform statistical analysis and data science initiatives to uncover patterns in brain health and cardiometabolic disease data.

This would be a readme file about some interesting and useful readings about some topics I'm working on during my job:

1. "Blood biomarkers of Alzheimer’s disease in the community: Variation by chronic diseases and inflammatory status" [here](https://alz-journals.onlinelibrary.wiley.com/doi/full/10.1002/alz.13860)
2. "Explainable machine learning model for pre‐frailty risk assessment in community‐dwelling older adults" [here](https://onlinelibrary.wiley.com/doi/full/10.1002/hcs2.120)
3. "Defining the Role of Frailty in the Transition from Mild Cognitive Impairment to Dementia and in Dementia Progression" [here](https://karger.com/dem/article-abstract/53/2/57/896186/Defining-the-Role-of-Frailty-in-the-Transition?redirectedFrom=fulltext)
4. "Older patients affected by COVID-19: investigating the existence of biological phenotypes" [here](https://bmcgeriatr.biomedcentral.com/articles/10.1186/s12877-024-05473-5)
5. "Frailty and risk of cardiovascular disease and mortality" [here](10.1371/journal.pone.0272527)

## Explainable machine learning model for pre‐frailty risk assessment in community‐dwelling older adults

This paper presents an innovative machine learning framework called **PACIFIC** (interPretable ACcurate and effICient pre-FraIlty Classification) to assess the risk of pre-frailty in community-dwelling older adults. This study, based on data from the China Health and Retirement Longitudinal Study (CHARLS), analyzed 3,141 adults aged 60 years and older.
The authors have well represented the framework in figure 1 which I will try to summarize as follows.

(a) Dataset:
- Show the data stream from the CHARLS Dataset
- Data is organized into seven key dimensions:
    1. Laboratory Measurements 
    2. Physical Factors 
    3. Demographics 
    4. Health Behaviors 
    5. Comorbidity and Medical Histories 
    6. Healthcare Service Utilization 
    7. Environmental Factors
- Selection of features is done through literature reviews and expert consultations
- Output is labelled using the Physical Frailty Phenotype (you can find it [here](https://en.wikipedia.org/wiki/Frailty_syndrome)).

(b) Model:
- Shows two main components:
    1. Recursive Feature Elimination (a LighGBM, with fast training and high accuracy, allows thme to obtain 57 features from itial 80 ones)
    2. Staking-CatBoost Distillation Module (with a compromise between predictive power and inference time; moreover very efficienct in handling high cardinal features)
        - They first trained several different tree-based models with k-fold bagging as base model,
        - They used Random Search with 100 iterations and 5 fold cross-validation to find best hyperparameters for each model,
        - They stacked them using ensemble selection,
        - The stacking model is distilled into a CatBoost model

(d) Interpretation of the pre-frailty classifier:
- For eXplainability module, they used Tree Explainer, SHAP (SHapley Additive exPlanations) values, SAGE (Shapley Additive Global Explanation) values.
- View how the model interprets and classifies risks
- Shows the global importance of different features
- They found out that the living city, BMI, and peak expiratory flow (PEF) were the three most significant contributors to the risk of pre‐frailty.

(e) Pre-frailty Risk Score:
- They also presented individual explanation with an example of Pre-frailty Risk Score Calculator
- The output value is the risk score for the individual. The base value is the mean risk score.

The PACIFIC framework's explainability approach offers several key advantages for both clinical practice and research:
1. Provides transparent and interpretable risk assessments
2. Enables clinicians to understand the specific factors contributing to an individual's pre-frailty risk
3. Supports evidence-based decision making through clear visualization of risk factors.
4. Quantifies the relative importance of different risk dimensions
5. Allows for targeted intervention strategies based on individual risk profiles
6. Enables monitoring of intervention effectiveness through clear metrics
7. Provides insights into population-level risk factors

This comprehensive approach not only improves the accuracy of pre-frailty risk assessment but also makes the results actionable and meaningful for both healthcare providers and patients, bridging the gap between complex machine learning models and practical clinical application.

## Predicting mortality and re-hospitalization for heart failure: a machine-learning and cluster analysis on frailty and comorbidity

This paper presents a comprehensive analysis using machine learning techniques and cluster analysis to predict adverse outcomes in older adults with heart failure, with a particular focus on integrating frailty and comorbidity assessments. The study, conducted at a tertiary care center in Italy, analyzed 571 patients aged 65 years and older who were discharged after acute decompensated heart failure (ADHF).

(a) Study Design and Data Collection:

- Single-center, prospective study of patients discharged from a geriatric unit with diagnosed ADHF
- All patients underwent a comprehensive geriatric assessment (CGA) at hospital admission
- Key assessments included:
    1. Cognitive evaluation using Short Portable Mental Status Questionnaire (SPMSQ)
    2. Basic (ADL) and instrumental (IADL) activities of daily living
    3. Comorbidity burden evaluated through Charlson Comorbidity Index (CCI)
    4. Frailty degree assessed using Clinical Frailty Scale (CFS)
    5. Blood tests including creatinine and brain natriuretic peptide (BNP)
- Primary **outcome** was a composite of re-hospitalization for heart failure or all-cause death within six months following discharge.

(b) Statistical Analysis Approach:

- Cox Regression Analysis:
    - Identified clinical and biochemical factors associated with 6-month mortality or re-hospitalization
    - Univariate and multivariate analyses revealed that BNP, CFS, and CCI were independent determinants of adverse outcomes
    - CFS showed stronger predictive capacity (AUC 0.702) compared to CCI (0.581) and BNP (0.597)
- Random Forest Analysis:
    - Used for feature selection with 70% training and 30% testing split
    - Identified the most important predictors of adverse outcomes
    - BNP (importance value 23.10), age (20.65), CFS (19.82), CCI (12.53), and creatinine (9.13) emerged as the most influential predictors
    - Achieved an impressive out-of-bag error rate of only 2.26%
- K-means Clustering:
    - Applied to stratify patients based on frailty (CFS), comorbidity burden (CCI), and BNP levels
    - Used silhouette approach to determine the optimal number of clusters (four)
    - Created distinct phenogroups with different risk profiles
    - Verified results with hierarchical agglomerative clustering as a secondary analysis

(c) Cluster Phenogroups Identified:

- Cluster 1 (Very Frail)
- Cluster 2 (Pre-frail with Low BNP)
- Cluster 3 (Pre-frail with High BNP)
- Cluster 4 (Non-frail):

(d) Clinical Implications:

- Frailty assessment using CFS provides substantial prognostic value beyond traditional risk factors
- The integration of frailty, comorbidity, and BNP levels allows for more targeted risk stratification
- Machine learning approaches can identify distinct phenotypes among older heart failure patients
- The identified clusters demonstrate significantly different risk profiles that could guide personalized care:
    - Cluster 1 and 3 patients might benefit from more intensive monitoring and support
    - Cluster 2 patients (pre-frail) could potentially benefit from interventions to improve functional status
    - Cluster 4 patients have relatively good prognosis with standard care

This study demonstrates the value of incorporating frailty assessment into heart failure risk models and shows how machine learning techniques can identify clinically meaningful patient subgroups with different risk profiles. The findings suggest that tailoring treatment strategies based on these phenotypes could potentially improve outcomes in older adults with heart failure.


## Defining the Role of Frailty in the Transition from Mild Cognitive Impairment to Dementia and in Dementia Progression

This paper investigates the critical role of frailty in predicting the transition from Mild Cognitive Impairment (MCI) to dementia and in the progression through different stages of dementia. The study analyzed 1,284 participants attending a Cognitive Disturbances and Dementia unit from January 2021 to May 2023, with a focus on using the Clinical Frailty Scale (CFS) as a practical assessment tool.

(a) Study Design

- **Study Type**: Retrospective study spanning from January 2021 to May 2023
- **Participants**: 1,284 individuals from the Centre for Cognitive Disturbances and Dementia (CDCD) unit
- **Evaluation Schedule**: Participants were evaluated at baseline and every 6 months
  - 669 had at least one follow-up visit
  - 669 had two visits
  - 429 had three visits
  - 279 had four visits
  - 193 had five visits
  - 73 had six visits
- **Total Visits**: 2,927 visits analyzed
- **CDR Distribution at Baseline**:
  - 57 participants with CDR of 0 (no dementia)
  - 773 with CDR of 0.5 (questionable dementia/MCI)
  - 273 with CDR of 1 (mild dementia)
  - 136 with CDR of 2 (moderate dementia)
  - 45 with CDR of 3 (severe dementia)

(b) Assessment Methods

The study employed a comprehensive set of measurement tools:

1. **Cognitive Measures**:
   - Mini-Mental State Examination (MMSE)
   - Clock Drawing Test
   - Geriatric Depression Scale (GDS)
   - Free and Cued Selective Reminding Test
   - Rey Figure Copy and Recall
   - Phonemic and Semantic Fluencies
   - Digit Span (Forward and Backward)
   - Screening for Aphasia in Neurodegeneration (SAND)
   - Trail Making Test (Parts A and B)
   - Visuoperceptual Reasoning Subtest of the WAIS-IV

2. **Neuropsychiatric Assessment**:
   - Neuropsychiatric Inventory (NPI)

3. **Functional Independence**:
   - Basic Activities of Daily Living (BADL)
   - Instrumental Activities of Daily Living (IADL)

4. **Comorbidities and Medication Effects**:
   - Cumulative Illness Rating Scale for Geriatrics (CIRS-G)
   - Anticholinergic Burden (ACB) Scale

5. **Frailty Assessment**:
   - Clinical Frailty Scale (CFS) - a 9-point scale from 1 (very fit) to 9 (terminally ill)
   - CFS ≥ 5 indicates the presence of frailty

6. **Disease Severity**:
   - Clinical Dementia Rating Global Score (CDR-G)
   - CDR Sum of Boxes (CDR-SoB)

(c) Statistical Analysis Methods

The statistical methodology in this study is particularly robust and carefully designed to address the complex, longitudinal nature of dementia progression. Here's a detailed summary:

1. **Descriptive Statistics Approach**:
   - Continuous variables: reported as median with interquartile range (IQR)
   - Categorical variables: reported as counts and percentages
   - Test selection rationale:
     - **Kruskal-Wallis H test**: Used for continuous variables because it's a non-parametric test that doesn't require normal distribution assumptions, making it ideal for comparing multiple independent groups
     - **Fisher's test**: Selected for categorical variables instead of chi-square because it's more accurate when dealing with cells that have low expected counts

2. **Multilevel Regression Models**:
   - **Model Type**: Mixed-effects hierarchical multilevel logistic regression
   - **Rationale**: This sophisticated approach accounts for:
     - Repeated measures on the same individuals over time
     - The time- and severity-dependent nature of variables across different CDR groups
     - The nested structure of the data (multiple observations within patients)

3. **Variable Selection Process**:
   - Initial univariate analysis to identify potential predictors (p < 0.100)
   - Multicollinearity check using Variance Inflation Factor (VIF < 2.5)
   - Linearity assessment via Box-Tidwell procedure to ensure continuous variables related linearly to the logit of the dependent variable

4. **Multiple Comparisons Control**:
   - Benjamini-Hochberg False Discovery Rate (FDR) correction
   - **Rationale**: This method provides better statistical power than traditional Bonferroni correction while still controlling for Type I errors when conducting multiple tests

5. **Stepwise Multiple Regression Analysis**:
   - Used to determine contributors to frailty
   - Model validation included checking:
     - Independence of residuals (Durbin-Watson statistics)
     - Homoscedasticity (visual inspection)
     - Absence of outliers (studentized deleted residuals)
     - Normality of residuals (Q-Q plots)

6. **Effect Size Reporting**:
   - Adjusted R² values provided (0.49 for frailty contributors model, indicating a large effect size)
   - Odds ratios with 95% confidence intervals for logistic regression models

7. **Missing Data Handling**:
   - Rigorous data collection protocols eliminated the need for imputation or exclusion methods

(d) Key Findings

1. **Relationship Between Frailty and Dementia Severity**:
   - Frailty (CFS scores) significantly increased with higher CDR groups
   - Participants with CDR 0 and 0.5 had similar frailty levels
   - Significant increases in frailty were observed in CDR groups 1, 2, and 3

2. **Predictors of Disease Progression**:
   - In the overall multilevel multivariate analysis, three factors emerged as significant predictors:
     - Age (OR: 1.03, 95% CI: 1.01-1.06, p = 0.014)
     - CFS score (OR: 1.37, 95% CI: 1.14-1.65, p < 0.001)
     - MMSE score (OR: 0.94, 95% CI: 0.91-0.97, p < 0.001)

3. **MCI to Dementia Conversion**:
   - In patients with MCI (CDR 0.5), the same factors predicted conversion to dementia:
     - Age (OR: 1.04, 95% CI: 1.01-1.07, p = 0.024)
     - CFS score (OR: 1.34, 95% CI: 1.12-1.69, p < 0.001)
     - MMSE score (OR: 0.94, 95% CI: 0.91-0.97, p < 0.001)

4. **Progressors vs. Non-Progressors**:
   - Patients who progressed to higher CDR stages showed significantly higher baseline CFS scores
   - Average CFS score differences between progressors and non-progressors

5. **Contributors to Frailty**:
   - Six factors significantly contributed to frailty (CFS scores):
     - Age (β = 0.23, p < 0.001)
     - Education level (β = -0.04, p = 0.046)
     - Anticholinergic burden (β = 0.10, p < 0.001)
     - Comorbidity burden (CIRS-G) (β = 0.22, p < 0.001)
     - Cognitive impairment (MMSE) (β = -0.39, p < 0.001)
     - Neuropsychiatric symptoms (NPI) (β = 0.15, p < 0.001)
   - The model explained 49% of the variance in frailty (adjusted R² = 0.49)

(e) Visual Representation of Findings

The paper's Figure 1 effectively visualizes two key aspects of the relationship between frailty and dementia:

1. **Panel 1a**: Shows how CFS scores increase across CDR groups, with:
   - Similar scores for CDR 0 and 0.5
   - Significant step-wise increases for CDR 1, 2, and 3
   - Statistical significance indicators between groups

2. **Panel 1b**: Compares CFS scores between:
   - Stable patients (those who remained at the same CDR level)
   - Progressors (those who transitioned to a higher CDR level)
   - Shows consistently higher frailty scores in progressors across all CDR groups

This visualization supports the study's conclusion that frailty is both a marker of current disease state and a predictor of future progression.

## Older patients affected by COVID-19: investigating the existence of biological phenotypes

This paper investigates the existence of distinct biological phenotypes in older patients hospitalized for COVID-19 using a panel of aging biomarkers. The study analyzes data from the FRACOVID Project, an observational multicenter study conducted in Northern Italy.

(a) Study Design and Population:

- Data collected from consecutive patients with COVID-19 hospitalized between February and May 2020
- Study conducted in acute Geriatric and Infectious disease wards of San Gerardo Hospital (Monza) and Civili Hospital (Brescia)
- Inclusion criteria: Age over 60 years, positive PCR test for SARS-CoV-2, Clinical Frailty Scale score ≤7

(b) Data Collection:

- Comprehensive assessment using structured case report form and Research Electronic Data Capture platform
- Data collected through in-person interviews, medical examinations, and review of medical records
- Information gathered on demographics, clinical characteristics, functional status, chronic diseases, and COVID-19 severity
- Frailty status assessed using the Clinical Frailty Scale (CFS)
- Blood samples collected at admission for biomarker analysis

(c) Biomarker Panel:

- The study analyzed 7 key aging biomarkers:
  1. Cystatin C (reflects renal function and mortality risk)
  2. Growth differentiation factor 15 (GDF-15) (marker of mitochondrial dysfunction)
  3. Interleukin-1 beta (IL-1β) (inflammatory marker of aging)
  4. Interleukin-6 (IL-6) (pro-inflammatory cytokine associated with frailty)
  5. N-terminal pro-B-type natriuretic peptide (NT-proBNP) (cardiac health marker)
  6. Plasminogen activator inhibitor-1 (PAI-1) (associated with cellular senescence)
  7. Tumor necrosis factor alpha (TNF-α) (pro-inflammatory marker of inflammaging)

(d) Statistical Analysis:

- The researchers employed an unsupervised hierarchical clustering approach to identify biological phenotypes
- This methodology was chosen because it allows for the identification of natural groupings in the data without predefined classifications
- Data preparation involved several critical steps:
  1. Out-of-range biomarker values were removed to ensure data quality
  2. Log-transformation was applied to normalize the distribution of biomarker values
  3. Values were centered and scaled using mean and standard deviation to standardize different measurement scales
- Ward's method with squared Euclidean distance was selected for clustering because:
  1. Ward's method minimizes the variance within clusters, creating more compact groups
  2. This approach is particularly effective for identifying clusters with similar biomarker profiles
- The optimal number of clusters was determined through visual inspection of the dendrogram (hierarchical tree diagram)
- Between-cluster comparisons used appropriate statistical tests:
  1. One-way ANOVA for continuous variables
  2. Chi-square tests for categorical variables with adequate cell counts
  3. Fisher's exact tests for categorical variables with small expected frequencies
- The Kruskal-Wallis test was specifically chosen to compare biomarker concentrations across phenotypes because:
  1. It's a non-parametric test that doesn't assume normal distribution
  2. It's robust for comparing medians across multiple groups
  3. It's appropriate for biomarker data that may remain skewed even after transformation
- All analyses were conducted using R 4.3.0

(e) Identified Biological Phenotypes:

- The analysis revealed three distinct biological phenotypes:
  1. "Inflammatory" phenotype (40.7% of participants):
  2. "Organ dysfunction" phenotype (37.1% of participants):
  3. "Unspecific" phenotype (22.0% of participants):

(f) Clinical Implications:

- The identified biological phenotypes correlate with different clinical and functional characteristics
- The "organ dysfunction" phenotype demonstrated the highest prevalence of frailty, disability, and chronic conditions
- The "inflammatory" phenotype showed the most pronounced systemic inflammatory response


## Frailty and risk of cardiovascular disease and mortality

This study investigates the association between frailty and the risk of developing cardiovascular disease (CVD) and mortality using data from a prospective cohort study conducted in Singapore.

(a) Study Design and Population:
* Data from the Singapore Longitudinal Ageing Study (SLAS)
* Participants recruited in two phases: SLAS-1 (2003-2004) and SLAS-2 (2009-2013)
* Final sample included 5,015 community-dwelling participants aged ≥55 years
* Follow-up period of up to 10 years, ending in December 2017

(b) Data Collection:
* Frailty status assessed according to **modified Fried criteria** with five components:
  1. Shrinking/weight loss (BMI <18.5 kg/m² and/or unintentional weight loss ≥4.5 kg)
  2. Weakness (measured by knee extension strength or rising from chair test)
  3. Slowness (gait speed <0.8m/s or POMA score <9)
  4. Exhaustion (response of "not at all" to energy question from SF-12)
  5. Low activity (no participation in any physical activity)
* CVD/mortality events as outcomes.
* Covariate variables included sociodemographic data, lifestyle behaviors, cardiometabolic risk factors, medication therapies, depression (GDS), cognitive function (MMSE), and blood biomarkers (hemoglobin, albumin, creatinine, white blood cell count)

(c) Frailty Measurement:
* Frailty status categorized as:
  1. Robust (0 points)
  2. Pre-frailty (1-2 points)
  3. Frailty (3-5 points)
* Prevalence in the sample: Robust 50.1%, Pre-frailty 46.2%, Frailty 3.7%

(d) Statistical Analysis:
* **Hierarchical Cox proportional hazard models** used to estimate hazard ratios (HR) between frailty status and incident CVD/mortality
* **Competing-risks survival regression models** employed to estimate sub-distribution hazard ratios (SHR) between frailty status/components and specific CVD outcomes
* Hierarchical adjustment in five progressive models:
  1. Model 1: Adjusted for age and sex (fundamental demographic confounders)
  2. Model 2: Added race, education, housing (socioeconomic determinants of health that may influence both frailty and CVD risk)
  3. Model 3: Added smoking, alcohol, central obesity, raised triglycerides, reduced HDL-C, hypertension, diabetes, raised LDL-C, statin therapy, antiplatelet therapy, anticoagulant therapy (traditional cardiovascular risk factors and treatments)
  4. Model 4: Added depression (GDS) and cognitive impairment (MMSE) (psychological comorbidities known to be associated with both frailty and CVD)
  5. Model 5: Added blood biomarkers (albumin, hemoglobin, creatinine, white blood cell count) (biological markers of inflammation and organ function)
* This hierarchical approach was specifically chosen to:
  1. Identify potential mediating pathways between frailty and CVD outcomes
  2. Evaluate how the association changes after adjustment for specific domains of covariates
* Time-to-event defined as the length of time between baseline and first recorded CVD event
* Sensitivity analysis excluding CVD cases within 1 year after baseline to address potential reverse causality
* Statistical significance set at p<0.05 (two-sided)
* All analyses conducted using Stata 13.0

(e) Key Findings:
* Pre-frailty and frailty were significantly associated with:
  1. Overall CVD: pre-frailty HR=1.26 (95%CI: 1.02-1.56), frailty HR=1.54 (95%CI: 1.00-2.35) after adjustment for cardiometabolic and vascular risk factors (Model 3)
  2. Fatal CVD: pre-frailty HR=1.70 (95%CI: 1.05-2.77), frailty HR=2.48 (95%CI: 1.14-5.37) in the fully adjusted model (Model 5)
  3. All-cause mortality: pre-frailty HR=1.40 (95%CI: 1.17-1.67), frailty HR=2.03 (95%CI: 1.48-2.80) in the fully adjusted model (Model 5)
* The association with overall CVD became non-significant after adjustment for GDS depression and MMSE cognitive impairment (Model 4)