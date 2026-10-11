# Titanic Survival Predictor

Decision tree model that estimates the probability of a Titanic passenger surviving, based on sex, age and ticket class. Trained on the Kaggle Titanic dataset and deployed as a Gradio app on Hugging Face Spaces.

Demo: https://huggingface.co/spaces/scimldata/titanic-survivor-predictor

## Structure

```
data/                 Kaggle Titanic dataset
ml_web_app.ipynb      data exploration and model training
titanic_model_dtc/    Gradio app deployed to Hugging Face
```

## Running locally

```
pip install -r titanic_model_dtc/requirements.txt
cd titanic_model_dtc
python app.py
```

## Data

[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data), Kaggle.
