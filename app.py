import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page config
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide"
)

# Load the model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('models/svm_model.joblib')
    vectorizer = joblib.load('models/vectorizer.joblib')
    return model, vectorizer

try:
    model, vectorizer = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Title and description
st.title("📧 Email Spam Classifier")
st.markdown("""
This application uses a Support Vector Machine (SVM) classifier to detect spam emails.
Enter your email text below to check if it's spam or not.
""")

# Input text area
email_text = st.text_area("Enter email text:", height=200)

if st.button("Classify Email"):
    if email_text:
        # Transform the text
        text_vectorized = vectorizer.transform([email_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        # Display result
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("📨 This email is likely SPAM!")
            else:
                st.success("✉️ This email appears to be HAM (not spam).")
        
        with col2:
            st.info(f"Confidence: {probability[int(prediction)]:.2%}")
        
        # Show feature importance (if available)
        st.subheader("Important Features")
        feature_names = vectorizer.get_feature_names_out()
        transformed_text = text_vectorized.toarray()[0]
        
        # Get top 10 features present in the email
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Value': transformed_text
        })
        feature_importance = feature_importance[feature_importance['Value'] > 0]
        feature_importance = feature_importance.sort_values('Value', ascending=False).head(10)
        
        st.bar_chart(feature_importance.set_index('Feature'))
        
    else:
        st.warning("Please enter some text to classify.")

# Add information about the model
with st.expander("About the Model"):
    st.markdown("""
    This spam classifier uses:
    - Support Vector Machine (SVM) algorithm
    - TF-IDF vectorization for text preprocessing
    - Trained on a dataset of labeled spam and ham emails
    
    The model achieves good performance in identifying spam emails while minimizing false positives.
    """)

# Footer
st.markdown("---")
st.markdown("Created as part of ML Course Project | Source code available on GitHub")
