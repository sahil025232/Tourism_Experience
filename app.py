import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. PAGE SETUP & DATA LOADING
# ==========================================
st.set_page_config(page_title="Tourism Analytics", layout="wide")
st.title("🌍 Tourism Experience Analytics & Recommender")

@st.cache_data
def load_data():
    # Update filenames to include .zip and specify compression
    df = pd.read_pickle('df_model.pkl.zip', compression='zip')
    items = pd.read_pickle('items_df.pkl.zip', compression='zip')
    
    # Update filenames to include .gz
    clf_model = joblib.load('rf_classifier.pkl.gz')
    encoder = joblib.load('label_encoder.pkl.gz')
    
    ui_matrix = df.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)
    return df, items, clf_model, encoder, ui_matrix

df_model, items_df, rf_classifier, label_encoder, user_item_matrix = load_data()

# ==========================================
# 2. SIDEBAR: USER INPUTS
# ==========================================
st.sidebar.header("User Details")
# Input for Recommendations
user_id_input = st.sidebar.selectbox("Select User ID for Recommendations:", df_model['UserId'].unique()[:50])

# Inputs for Visit Mode Prediction (Classification)
st.sidebar.subheader("Predict Visit Mode")
continent = st.sidebar.selectbox("Continent", df_model['Continent_Encoded'].unique())
country = st.sidebar.selectbox("Country", df_model['Country_Encoded'].unique())
visit_year = st.sidebar.slider("Visit Year", int(df_model['VisitYear'].min()), int(df_model['VisitYear'].max()), 2024)
visit_month = st.sidebar.slider("Visit Month", 1, 12, 6)
attraction_type = st.sidebar.selectbox("Attraction Type Code", df_model['AttractionType_Encoded'].unique())
rating = st.sidebar.slider("Expected Rating", 1.0, 5.0, 4.0)

# ==========================================
# 3. CLASSIFICATION: VISIT MODE PREDICTION
# ==========================================
st.header("🎯 Predictive Analytics")
if st.sidebar.button("Predict Visit Mode"):
    # Construct feature array exactly as trained in Phase 2[cite: 1]
    input_features = pd.DataFrame([[continent, country, visit_year, visit_month, attraction_type, rating]], 
                                  columns=['Continent_Encoded', 'Country_Encoded', 'VisitYear', 
                                           'VisitMonth', 'AttractionType_Encoded', 'Rating'])
    
    prediction_encoded = rf_classifier.predict(input_features)[0]
    # Decode back to readable text (assuming you fit VisitMode with this encoder in Phase 1)
    # If the exact classes aren't in the encoder, you can map it manually based on your data.
    try:
        prediction_text = label_encoder.inverse_transform([int(prediction_encoded)])[0]
    except:
        prediction_text = f"Class {prediction_encoded}"
        
    st.success(f"**Predicted Visit Mode:** {prediction_text}")

# ==========================================
# 4. RECOMMENDATION SYSTEM
# ==========================================
st.header("🗺️ Personalized Attraction Suggestions")

def get_recommendations(target_user_id, num_recs=5):
    """Memory-efficient collaborative filtering block."""
    if target_user_id not in user_item_matrix.index:
        return ["User not found."]
    
    target_vector = user_item_matrix.loc[target_user_id].values.reshape(1, -1)
    sim_scores = cosine_similarity(target_vector, user_item_matrix)[0]
    sim_series = pd.Series(sim_scores, index=user_item_matrix.index)
    similar_users = sim_series.drop(target_user_id).sort_values(ascending=False).index
    
    user_visited = df_model[df_model['UserId'] == target_user_id]['AttractionId'].tolist()
    recommendations = {}
    
    for sim_user in similar_users:
        sim_ratings = user_item_matrix.loc[sim_user].replace(0, np.nan).dropna()
        for attr, rat in sim_ratings.items():
            if attr not in user_visited:
                recommendations[attr] = recommendations.get(attr, 0) + rat
        if len(recommendations) >= num_recs * 3:
            break
            
    top_attrs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:num_recs]
    rec_ids = [attr for attr, score in top_attrs]
    return items_df[items_df['AttractionId'].isin(rec_ids)]['Attraction'].tolist()

if st.button(f"Get Recommendations for User {user_id_input}"):
    recs = get_recommendations(user_id_input)
    st.write("Based on similar users, we recommend:")
    for i, rec in enumerate(recs, 1):
        st.write(f"{i}. **{rec}**")

st.markdown("---")

# ==========================================
# 5. DATA VISUALIZATIONS DASHBOARD
# ==========================================
st.header("📊 Tourism Insights Dashboard")
# Display visualizations of popular attractions, regions, and user segments[cite: 1]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Popular Attractions")
    # Get the top 10 Attraction IDs and their actual visit counts from the main dataset
    top_attractions_counts = df_model['AttractionId'].value_counts().head(10)
    
    # Map the IDs to their actual names
    # Create a dictionary mapping ID -> Name from items_df
    id_to_name = dict(zip(items_df['AttractionId'], items_df['Attraction']))
    
    # Replace the IDs in our counts series with the actual names
    top_attr_names = top_attractions_counts.rename(index=id_to_name)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(y=top_attr_names.index, x=top_attr_names.values, palette="mako", ax=ax)
    ax.set_xlabel("Number of Visits")
    ax.set_ylabel("Attraction")
    st.pyplot(fig)

with col2:
    st.subheader("User Segments by Continent")
    continent_counts = df_model['Continent'].value_counts()
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.pie(continent_counts.values, labels=continent_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax2.axis('equal') 
    st.pyplot(fig2)
