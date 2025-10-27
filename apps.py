import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import BertTokenizer, BertModel
import os

# --- Load saved models and encoders ---
@st.cache_resource
def load_resources():
    encoder = joblib.load("encoder.pkl")
    model = joblib.load("xgb_model.pkl")
    tokenizer = BertTokenizer.from_pretrained("bert_tokenizer/")
    bert_model = BertModel.from_pretrained("bert_model/")
    return encoder, model, tokenizer, bert_model

encoder, model, tokenizer, bert_model = load_resources()

st.title("🏢 Lead Conversion Prediction App")
st.write("Upload your leads data (CSV/Excel) to predict whether each lead will convert or not.")

# --- File Upload Section ---
uploaded_file = st.file_uploader("📤 Upload Lead Dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        user_df = pd.read_csv(uploaded_file)
    else:
        user_df = pd.read_excel(uploaded_file)

    st.success("✅ File uploaded successfully!")
    st.write("Here’s a quick preview of your data:")
    st.dataframe(user_df.head())
    
    
    top_n = st.number_input(
        "🔢 Enter number of top leads to display:",
        min_value=1,
        max_value=len(user_df),
        value=5,
        step=1,
        help="This will show the top N leads with the highest probability of conversion."
    )

    # --- Prediction Button ---
    if st.button("🚀 Run Predictions"):
        with st.spinner("Processing data and generating predictions..."):
            # 1️⃣ Handle Missing Values
            for col in user_df.select_dtypes(include=['object']).columns:
                user_df[col] = user_df[col].fillna("Unknown")
            for col in user_df.select_dtypes(include=['int64', 'float64']).columns:
                user_df[col] = user_df[col].fillna(user_df[col].median())

            # 2️⃣ Encode categorical columns
            if 'Lead_Message' in user_df.columns:
                categorical_cols = user_df.select_dtypes(include='object').columns.drop('Lead_Message')
            else:
                categorical_cols = user_df.select_dtypes(include='object').columns

            user_df[categorical_cols] = encoder.transform(user_df[categorical_cols])

            # 3️⃣ Load Precomputed Embeddings (Optional)
            # if 'lead_embeddings.npy' in [f.lower() for f in os.listdir()]:
            #     st.info("Using precomputed BERT embeddings for Lead_Message...")
            #     embeddings_np = np.load("lead_embeddings.npy")
            # else:
            #     st.warning("No precomputed embeddings found — generating now (this may take time).")
            #     sentences = user_df['Lead_Message'].tolist()
            #     inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
            #     with torch.no_grad():
            #         outputs = bert_model(**inputs)
            #     embeddings = outputs.last_hidden_state[:, 0, :]
            #     embeddings_np = embeddings.numpy()
            #     np.save("lead_embeddings.npy", embeddings_np)
            
            if 'Lead_Message' in user_df.columns:
                st.info("Generating BERT embeddings for Lead_Message — please wait...")
                sentences = user_df['Lead_Message'].astype(str).tolist()
                inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

            with torch.no_grad():
                outputs = bert_model(**inputs)

            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embeddings
            embeddings_np = embeddings.numpy() 
            
            
            
            

            # 4️⃣ Combine Embeddings with Original Data
            embedding_df = pd.DataFrame(
                embeddings_np,
                columns=[f'emb_{i}' for i in range(embeddings_np.shape[1])]
            )

            final_input = user_df.drop('Lead_Message', axis=1).reset_index(drop=True)
            final_input = pd.concat([final_input, embedding_df], axis=1)

            # 5️⃣ Predict
            # preds = model.predict(final_input)

#             # 6️⃣ Display and Download
#             user_df['Predicted_Converted'] = preds
#             st.success("✅ Predictions completed!")
#             st.dataframe(user_df[['Lead_Message', 'Predicted_Converted']].head())

#             # Download button
#             csv = user_df.to_csv(index=False).encode('utf-8')
#             st.download_button(
#                 "📥 Download Predictions as CSV",
#                 data=csv,
#                 file_name="lead_predictions.csv",
#                 mime="text/csv"
#             )

# else:
#     st.info("Please upload a CSV or Excel file to start prediction.")
                # 5️⃣ Predict probabilities
        pred_probs = model.predict_proba(final_input)[:, 1]
        preds = (pred_probs >= 0.5).astype(int)

        user_df['Predicted_Converted'] = preds
        user_df['Conversion_Probability'] = np.round(pred_probs, 3)

        # 6️⃣ Allow user to select top N leads by probability
        
        top_leads = user_df.sort_values(by='Conversion_Probability', ascending=False).head(top_n)

        st.success("✅ Predictions completed!")
        st.write(f"### 🔝 Top {top_n} Leads Most Likely to Convert")
        st.dataframe(top_leads[['Lead_Message', 'Conversion_Probability', 'Predicted_Converted']])

        # 7️⃣ Download full results
        csv = user_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Full Predictions as CSV",
            data=csv,
            file_name="lead_predictions_with_probabilities.csv",
            mime="text/csv"
        )
