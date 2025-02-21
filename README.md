# Med-Physics

A comprehensive repository combining Medical Physics with Data Science and AI Engineering, focused on medical data analysis, state-of-the-art healthcare applications, computer vision, and model explainability.

## 🎯 Project Overview

Med-Physics is a research-oriented repository that bridges the gap between Medical Physics and Advanced Data Science techniques. This is an evolving project that aims to:

- Analyze and process various types of medical data (EEG, neuroimaging, cardiometabolic biomarkers, etc.)
- Test and implement state-of-the-art AI models and tools for healthcare
- Experiment with transfer learning and fine-tuning of pre-trained medical AI models
- Develop computer vision solutions for medical imaging
- Explore explainability techniques for healthcare AI systems
- Create reproducible research workflows
- Document best practices in medical data science

## 🤖 AI Engineering Focus

The repository actively explores and implements cutting-edge AI solutions in healthcare:

### Pre-trained Models & Tools
- Integration and testing of SOTA healthcare models from HuggingFace
- Experimentation with leading medical imaging models
- Fine-tuning pre-trained models for specific medical tasks
- Benchmarking different model architectures

### Healthcare AI Applications
- Medical image segmentation and classification
- Disease prediction and progression modeling
- Biomarker analysis and patient stratification
- EEG signal processing and analysis

> Note: The list of models and applications will expand as new tools are tested and integrated. Each implementation will be documented in dedicated notebooks with performance analyses and use cases.

## 🗂️ Repository Structure

```
Med-Physics/
├── data/                      
│   ├── raw/                   
│   ├── processed/             
│   └── external/             
├── src/                    
│   ├── data_processing/   
│   ├── models/               
│   │   ├── traditional/      
│   │   └── deep_learning/ 
│   ├── visualization/        
│   └── explainability/       
├── notebooks/                 
│   ├── exploratory/          
│   ├── model_development/    
│   └── results_analysis/    
├── docs/                     
│   ├── data_documentation/   
│   ├── model_documentation/ 
│   └── research_papers/       
├── tests/                    
├── configs/                   
├── mlflow/                   
│   ├── mlruns/               
│   └── artifacts/             
└── results/               
    ├── figures/            
    ├── models/               
    └── reports/              
```

## 📊 Data Sources

The repository works with various types of medical data:

- **Health Examination Data**: Periodontal measurements, Clinical assessments
- **Questionnaire Data**: Demographics, Health behaviors, Medical conditions
- **Neuroimaging**: MRI

Primary data sources include:
- NHANES (National Health and Nutrition Examination Survey)
- ACDC (Automated Cardiac Diagnosis Challenge) dataset


## 🛠️ project boards

### Periodontal Status and Functional Domains Analysis
This [project](notebooks/NHANES_analysis.ipynb) investigates the relationship between periodontal disease severity and various functional domains using NHANES data. The analysis focuses on:
- Assessment of periodontal status (None/Mild, Moderate, Severe) using CDC/AAP criteria
- Evaluation of five key functional domains:
  - Locomotion (standing difficulty)
  - Cognitive function (concentration)
  - Vitality (weight changes and appetite)
  - Psychological status (depression and interest)
  - Sensory capabilities (hearing and vision)

The analysis pipeline includes:
- Data preprocessing and feature engineering
- Descriptive statistics generation using TableOne
- Univariate logistic regression analysis
- Results visualization and reporting

Key tools:
- R version 4.4.2
- Packages: tableone, dplyr, flextable
- Statistical methods: logistic regression with odds ratios and 95% CI

Results are presented in publication-ready tables showing associations between functional domains and periodontal disease severity, stratified by gender and overall population.

### SAM Zero-Shot Segmentation from Scribbles and SegFormer Fine-tuning Pipeline
In the first part of the [project](notebooks/cardiac_MRI_segmentation_with_SAM.ipynb), I used SAM's zero-shot capabilities for cardiac segmentation:
- Utilization of pre-trained SAM model without fine-tuning
- Scribble-based prompt generation from existing annotations
- Zero-shot generalization to cardiac structures

In the second part, I implemented a complete pipeline for SegFormer fine-tuning:
- Base architecture: Pre-trained SegFormer-B0
- Supervised training on cardiac slices
The pipeline includes:
- Custom dataset with augmentation
- Combined loss function (Dice + Cross Entropy)
- Detailed logging with Weights & Biases
- Checkpoint management and early stopping

Frameworks and libraries:
- PyTorch
- HuggingFace Transformers
- Albumentations for data augmentation

The code is structured with:
- Modular component testing you can see in the [src folder](src/src_ACDC_ds/)
- Incremental implementation

This second part has to be fully tested!

### Other Medical Physics projects 

During my last years, I worked on other medical physics projects such as:
1. "An eXplainability Artificial Intelligence approach to brain connectivity in Alzheimer's disease" published on [Frontiers](https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2023.1238065/full)
2. Deep Learning for Pneumonia Detection from Chest X-rays, you can see details [here](https://github.com/Silvano315/Pneumonia_Detection)


## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Med-Physics.git
cd Med-Physics
```

2. Set up the environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Initialize MLflow:
```bash
mlflow ui
```

4. Start exploring the notebooks in the `notebooks/` directory

## 🚧 Project Status

This repository is actively under development. New models, tools, and applications are being tested and integrated regularly. Check the project boards and issues for current focus areas and upcoming features.

## 📝 License

This project is licensed under the Apache License - see the LICENSE file for details.
