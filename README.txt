
# Environment Setup

1. Create a new conda environemnt  `conda create --name ml python=3.11`
2. Activate the new environment   `activate ml`
3. Install the packages from requirements.txt `pip install -r requirements.txt`
4. Register the env with jupyter - `python -m ipykernel install --user --name=ml`
5. Access the project folder from dropbox link and save - https://gatech.box.com/s/jbuy8kwy59chep9ggxam36bkib0v5u23

# DATA
1. The datasets required for the project will be downloaded directly from UCI Machine Learning classification_report
2. Dataset urls
    a. https://archive.ics.uci.edu/dataset/20/census+income
    b. https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

# EDA
1. Each dataset has its corresponding EDA python notebook explainng the features and their relations.
2. eda_census_income.ipynb
3. eda_dry_bean_dataset.iptnb


# MODELS
1. Each machine learning model is applied for the dataset and saved as seperate python notebook.

boosted_dt_census_income.ipynb
boosted_dt_dry_bean.ipynb

decision_tree_census_income.ipynb
decision_tree_dry_bean.ipynb

knn_census_income.ipynb
knn_dry_bean.ipynb

svm_census_income.ipynb
svm_dry_bean.ipynb



# RUN THE MODELS
1. Open command prompt and navigate to the project folder
2. Activate the conda env - `activate ml`
3. Start the jupter lab - `jupyter lab`
4. Open the respective the model and dataset python notebook and run the full notebook. It will download the required data and train the model.
5. The trained models are saved as pickle files in ./model
