# CVClassification
Project for my thesis around automated Classification of Curriculum Vitaes Using Large Language Models for Expertise, Role, and Organizational Unit Identification

# Explanations about each file

## App
This folder contains the code to run the app.

## active_learning_pipeline.py
This file contains the code for the active learning pipeline. Here we use a LLM to select the most informative samples to label, create a training set, and retrain the model. The model made here is not used in the final version of the project, since the zero-shot approach is less costly in terms of time and money.

## cv_classification_pipeline.py
This file contains the code for the classification pipeline. Here we are going to use LLMs to classify the resumes according to 3 categories: Expertise, Role, and Organizational Unit. To do this, we are going to use a pipeline with 3 steps:

1. Parse the resumes
2. Clean the resumes
3. Classify the resumes

To classify the resumes, we are going to use a pipeline with 3 steps:

1. Classify the resumes according to the category Expertise
2. Classify the resumes according to the category Role
3. Classify the resumes according to the category Organizational Unit

Each of these steps will be trained with a different model, and the results will be saved in a json file. Probably we will use the output of each model as input for the next one to better classify the resumes.

## human_labeling_interface.py
This file contains the code to create a human labeling interface. Here we let a human label the samples and save them to be used in the active learning pipeline.

## process_resumes.py
This file contains the code to process the resumes to be used in the classification. Also not used in the final version of the project.

## zero-shot-approach.py
This file contains the code for the zero-shot approach to classify the resumes. Here we use prompt engineering to classify the resumes according to 3 categories: Work Experience, Education, and Extra-Skills.


