import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit  # for sigmoid function
import warnings
warnings.filterwarnings('ignore')

def get_prediction_probability(model, X):
    """Get prediction probability, falling back to decision_function if predict_proba is unavailable."""
    try:
        # First try predict_proba
        return model.predict_proba(X)
    except AttributeError:
        try:
            # Fall back to decision_function with sigmoid
            decision_scores = model.decision_function(X)
            # Convert to probabilities using sigmoid function
            proba = expit(decision_scores)
            # Return in the same format as predict_proba (2 columns)
            return np.column_stack([1 - proba, proba])
        except AttributeError:
            # If neither method is available, return confidence as [0, 1] or [1, 0]
            predictions = model.predict(X)
            return np.array([[0, 1] if pred else [1, 0] for pred in predictions])

# Set page configuration
st.set_page_config(
    page_title="SweetTooth - Diabetes Predictor",
    page_icon="�",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Apply dark theme with custom styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .main-title {
        text-align: center;
        color: #FF69B4;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description with candy emoji
st.markdown("<h1 class='main-title'>🍬 SweetTooth 🍬<br>Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.write("Take control of your health with our advanced diabetes prediction system.")

# Load the model and scaler
try:
    # Try loading with joblib first
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except:
        # If joblib fails, try with pickle
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
except Exception as e:
    st.error(f"""
    Error loading model files. Please ensure:
    1. Both 'best_model.pkl' and 'scaler.pkl' exist in the directory
    2. Files are not corrupted
    3. All required packages are installed
    
    Error details: {str(e)}
    """)
    st.info("Try reinstalling scikit-learn with: pip install scikit-learn --upgrade")
    st.stop()

# Create input fields with validation
def get_user_input():
    st.write("### Please enter patient information:")
    
    # Pregnancies
    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0,
        max_value=20,
        value=0,
        help="Enter number of times pregnant (0-20)"
    )
    
    # Glucose
    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=0,
        max_value=500,
        value=120,
        help="Enter glucose level from oral glucose tolerance test"
    )
    
    # Blood Pressure
    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=0,
        max_value=300,
        value=70,
        help="Enter diastolic blood pressure"
    )
    
    # Skin Thickness
    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0,
        max_value=100,
        value=20,
        help="Enter triceps skin fold thickness"
    )
    
    # Insulin
    insulin = st.number_input(
        "Insulin Level (mu U/ml)",
        min_value=0,
        max_value=1000,
        value=79,
        help="Enter 2-Hour serum insulin"
    )
    
    # BMI
    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.1,
        help="Enter Body Mass Index"
    )
    
    # Diabetes Pedigree Function
    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47,
        step=0.001,
        help="Enter Diabetes Pedigree Function value"
    )
    
    # Age
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=30,
        help="Enter age in years"
    )
    
    # Create a dictionary of all inputs
    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    
    return user_data

# Get user input
user_input = get_user_input()

# Add a predict button
if st.button('Predict Diabetes Status'):
    # Convert input to numpy array and scale it
    input_array = np.array(list(user_input.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = get_prediction_probability(model, input_scaled)
    
    # Add note about probability calculation method
    if not hasattr(model, 'predict_proba'):
        st.info("Note: Probability values are approximated using the model's decision function.")
    
    # Display prediction with styled results
    st.markdown("### 🎯 Results")
    
    # Create a styled box for the results
    if prediction[0] == 1:
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 10px; background-color: rgba(255, 0, 0, 0.1); border: 2px solid #ff4b4b;">
                <h3 style="color: #ff4b4b;">⚠️ High Risk of Diabetes</h3>
                <p style="font-size: 18px;">Probability: {prediction_proba[0][1]:.2%}</p>
                <p>Please consult with a healthcare professional for proper medical advice.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 10px; background-color: rgba(0, 255, 0, 0.1); border: 2px solid #00ff00;">
                <h3 style="color: #00ff00;">✅ Low Risk of Diabetes</h3>
                <p style="font-size: 18px;">Probability: {prediction_proba[0][1]:.2%}</p>
                <p>Keep maintaining a healthy lifestyle!</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Display input values
    st.write("### Input Summary")
    df = pd.DataFrame([user_input])
    st.write(df)

# Add information about the model
with st.sidebar:
    st.markdown("### 🍬 Welcome to SweetTooth!")
    st.write("""
    SweetTooth is an intelligent diabetes prediction system trained on the Pima Indians Diabetes Database.
    
    Our model analyzes your health parameters to assess diabetes risk with high accuracy.
    
    All inputs are carefully validated to ensure reliable predictions.
    """)
    
    st.markdown("### 📝 Input Guidelines")
    st.write("""
    - Pregnancies: Number of times pregnant (0-20)
    - Glucose: Plasma glucose concentration (0-500 mg/dL)
    - Blood Pressure: Diastolic blood pressure (0-300 mm Hg)
    - Skin Thickness: Triceps skin fold thickness (0-100 mm)
    - Insulin: 2-Hour serum insulin (0-1000 mu U/ml)
    - BMI: Body Mass Index (0-100)
    - Diabetes Pedigree Function: (0-3)
    - Age: Age in years (0-120)
    """)