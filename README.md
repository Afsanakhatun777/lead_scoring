The Lead Conversion Prediction App is a machine-learning-powered Streamlit application that helps businesses identify which leads are most likely to convert into customers.
It combines XGBoost for structured data and BERT embeddings for textual features (lead messages), giving both numerical and contextual insights into conversion probability.
Bulk Upload Support – Upload multiple leads via CSV or Excel file.

 ## Text + Numeric Intelligence – Uses BERT embeddings for message text and structured ML for numerical features.
## Probability-Based Results – Shows the top N leads with the highest likelihood of conversion.
## Interactive Interface – Built with Streamlit for an easy and intuitive user experience.
 ## Downloadable Reports – Export predictions as CSV for further analysis.


## Project Workflow

# Data Cleaning & Encoding

Missing numerical values → replaced with median
Missing categorical values → replaced with “Unknown”
Categorical columns → encoded using OrdinalEncoder

# Text Embeddings

“Lead_Message” column processed using bert-base-uncased model from Hugging Face Transformers.
Sentence embeddings (768 dimensions) extracted and appended to the dataset.

# Model Training

Model: XGBoostClassifier
Evaluation metrics: Accuracy, Precision, Recall, F1-score
Saved using joblib as xgb_model.pkl and encoder.pkl

# Streamlit Interface

Accepts user input or CSV/Excel upload
Preprocesses new data (encoding, embeddings)
Predicts conversion probability for each lead
Displays top N leads based on probability

Future Improvements

🔹 Fine-tune BERT embeddings on domain-specific data.
🔹 Add visualization dashboards (conversion distribution, feature importance).
🔹 Integrate SHAP/LIME explainability.
🔹 Deploy using Docker or Streamlit Cloud.
🔹 Implement automated email follow-ups for high-probability leads.

Acknowledgments

## Hugging Face Transformersfor BERT embeddings

## Streamlit for building the web interface

## XGBoost for the classification model
