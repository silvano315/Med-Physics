# Brain health assessment in cardiometabolic diseases

Hi everyone, I'm a Researcher and Data Scientist at the Healthcare hospital in Bari. I'm working on the PNRR project "Brain health assessment in cardiometabolic diseases: impact of frailty and biomarkers on healthcare management". I perform statistical analysis and data science initiatives to uncover patterns in brain health and cardiometabolic disease data.

This would be a readme file about some interesting and useful readings about some topics I'm working on during my job:

1. "Blood biomarkers of Alzheimer’s disease in the community: Variation by chronic diseases and inflammatory status" [here](https://alz-journals.onlinelibrary.wiley.com/doi/full/10.1002/alz.13860)
2. "Explainable machine learning model for pre‐frailty risk assessment in community‐dwelling older adults" [here](https://onlinelibrary.wiley.com/doi/full/10.1002/hcs2.120)

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