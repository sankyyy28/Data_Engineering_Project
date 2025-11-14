import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced model loading with multiple fallback options
@st.cache_resource
def load_model():
    try:
        with open('bigmart_best_model.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        
        model = None
        
        # Case 1: It's a tuple containing multiple objects
        if isinstance(loaded_data, tuple):
            st.sidebar.info(f"Found tuple with {len(loaded_data)} elements")
            
            # Try to find a model with predict method
            for i, item in enumerate(loaded_data):
                if hasattr(item, 'predict'):
                    model = item
                    st.sidebar.success(f"✓ Found model at position {i}")
                    break
            
            # If no model found, try common patterns
            if model is None:
                # Common pattern: (model, preprocessor, feature_names)
                if len(loaded_data) >= 1:
                    model = loaded_data[0]
                    st.sidebar.warning("Using first element of tuple as model")
        
        # Case 2: It's a single model object
        elif hasattr(loaded_data, 'predict'):
            model = loaded_data
            st.sidebar.success("✓ Single model object loaded")
        
        # Case 3: It's a dictionary or other structure
        elif isinstance(loaded_data, dict):
            if 'model' in loaded_data:
                model = loaded_data['model']
                st.sidebar.success("✓ Found model in dictionary under 'model' key")
            elif 'estimator' in loaded_data:
                model = loaded_data['estimator']
                st.sidebar.success("✓ Found model in dictionary under 'estimator' key")
        
        if model is None:
            st.sidebar.error("Could not identify model in pickle file")
            return None
            
        # Test if model can make predictions
        test_input = np.array([[1.0] * 10])  # Simple test input
        try:
            _ = model.predict(test_input)
            st.sidebar.success("✓ Model prediction test passed")
            return model
        except:
            st.sidebar.warning("Model loaded but prediction test failed")
            return model
            
    except FileNotFoundError:
        st.sidebar.error("❌ Model file 'bigmart_best_model.pkl' not found")
        st.sidebar.info("Please ensure the file is in the same directory as this script")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

# Alternative: Create a dummy model for testing
def create_dummy_model():
    """Create a simple model for testing if real model fails"""
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=10, random_state=42)
    # Train on dummy data
    X_dummy = np.random.rand(100, 10)
    y_dummy = np.random.rand(100) * 5000 + 1000
    model.fit(X_dummy, y_dummy)
    return model

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 2rem 0;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_input_for_prediction(input_dict):
    """Convert input data to format expected by model"""
    # This is a simplified version - you'll need to adjust based on your actual model requirements
    try:
        # Convert categorical variables to numerical (simplified approach)
        fat_content_map = {'Low Fat': 0, 'Regular': 1}
        outlet_size_map = {'Small': 0, 'Medium': 1, 'High': 2}
        location_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
        
        # Create feature array (adjust based on your model's expected features)
        features = [
            input_dict['Item_Weight'],
            input_dict['Item_Visibility'],
            input_dict['Item_MRP'],
            fat_content_map.get(input_dict['Item_Fat_Content'], 0),
            outlet_size_map.get(input_dict['Outlet_Size'], 0),
            location_map.get(input_dict['Outlet_Location_Type'], 0),
            input_dict['Outlet_Age']
        ]
        
        return np.array([features])
        
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

def main():
    # Sidebar with enhanced diagnostics
    with st.sidebar:
        st.title("👤 Developer Profile")
        st.subheader("Sanket Sanjay Sonparate")
        st.write("Data Scientist | ML Engineer")
        
        st.subheader("📧 Contact")
        st.write("sonparatesanket@gmail.com")
        
        st.subheader("🔗 Social Links")
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?logo=github)](https://github.com/sankyyy28)")
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/sanket-sonparate-018350260)")
        
        st.markdown("---")
        st.subheader("🔧 Model Status")
        
        # Load model and show status
        model = load_model()
        
        if model is None:
            st.error("❌ Model not available")
            if st.button("🔄 Use Demo Model"):
                st.session_state.use_demo_model = True
                st.rerun()
        else:
            st.success("✅ Model loaded successfully")
        
        st.markdown("---")
        st.subheader("📊 App Info")
        st.write("BigMart Sales Prediction using Machine Learning")

    # Main content
    st.markdown('<div class="main-header">🛒 BigMart Sales Predictor</div>', unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📝 Product & Outlet Details</div>', unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            # Product Information
            st.subheader("📦 Product Information")
            col1a, col1b, col1c = st.columns(3)
            
            with col1a:
                item_weight = st.number_input("Item Weight", min_value=0.0, max_value=50.0, value=12.5, step=0.1, help="Weight of the item in kg")
                item_fat_content = st.selectbox("Fat Content", ["Low Fat", "Regular"], help="Fat content category")
                
            with col1b:
                item_visibility = st.slider("Item Visibility", min_value=0.0, max_value=0.5, value=0.07, step=0.001, help="Visibility percentage in store")
                item_type = st.selectbox("Item Type", ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables"], help="Product category")
                
            with col1c:
                item_mrp = st.number_input("Item MRP", min_value=0.0, max_value=500.0, value=140.0, step=1.0, help="Maximum Retail Price")
                item_identifier = st.text_input("Item Identifier", value="FDA15", help="Unique item ID")
            
            # Outlet Information
            st.subheader("🏪 Outlet Information")
            col2a, col2b = st.columns(2)
            
            with col2a:
                outlet_identifier = st.selectbox("Outlet Identifier", ["OUT010", "OUT013", "OUT017", "OUT018"], help="Unique outlet ID")
                outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"], help="Size of the outlet")
                
            with col2b:
                outlet_location_type = st.selectbox("Location Type", ["Tier 1", "Tier 2", "Tier 3"], help="City tier classification")
                outlet_type = st.selectbox("Outlet Type", ["Grocery Store", "Supermarket Type1"], help="Type of retail outlet")
            
            # Establishment year
            establishment_year = st.slider("Establishment Year", 1985, 2020, 2010, help="Year when outlet was established")
            outlet_age = 2024 - establishment_year
            
            st.info(f"Outlet Age: {outlet_age} years")
            
            submit_button = st.form_submit_button("🚀 Predict Sales", use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">🎯 Prediction Result</div>', unsafe_allow_html=True)
        
        if submit_button:
            # Get model (either real or demo)
            if model is None and st.session_state.get('use_demo_model', False):
                model = create_dummy_model()
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning("⚠️ Using demo model for prediction")
                st.info("Real model file 'bigmart_best_model.pkl' not found or corrupted")
                st.markdown('</div>', unsafe_allow_html=True)
            
            if model is None:
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.error("❌ Model failed to load. Please check the model file.")
                st.markdown("""
                **Troubleshooting steps:**
                1. Ensure `bigmart_best_model.pkl` is in the same folder
                2. Check if the file is not corrupted
                3. Verify Python and library versions match
                4. Click 'Use Demo Model' in sidebar for testing
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                try:
                    # Prepare input data
                    input_data = {
                        'Item_Weight': item_weight,
                        'Item_Visibility': item_visibility,
                        'Item_MRP': item_mrp,
                        'Item_Fat_Content': item_fat_content,
                        'Outlet_Size': outlet_size,
                        'Outlet_Location_Type': outlet_location_type,
                        'Outlet_Age': outlet_age
                    }
                    
                    # Make prediction
                    features = preprocess_input_for_prediction(input_data)
                    if features is not None:
                        prediction = model.predict(features)[0]
                        
                        # Display results
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.metric("Predicted Sales", f"${prediction:,.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Insights
                        st.subheader("📈 Sales Insights")
                        if prediction > 4000:
                            st.success("🔥 Excellent sales potential!")
                        elif prediction > 2000:
                            st.info("💡 Good performance expected")
                        else:
                            st.warning("📊 Consider strategy adjustments")
                        
                        # Performance gauge
                        st.progress(min(int(prediction / 50), 100))
                        st.caption(f"Sales potential: {min(prediction / 50, 100):.1f}%")
                        
                except Exception as e:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.error(f"Prediction error: {str(e)}")
                    st.info("The model loaded but encountered an error during prediction.")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            # Default state before prediction
            st.info("👆 Fill the form and click **Predict Sales** to see results")
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center;'>
                <h3 style='color: #6c757d;'>Prediction Results</h3>
                <p>Your sales prediction will appear here after submitting the form</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer section
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d;'>
        <p>BigMart Sales Predictor • Built with Streamlit • 
        <a href='https://github.com/sankyyy28' style='color: #1f77b4;'>GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    if 'use_demo_model' not in st.session_state:
        st.session_state.use_demo_model = False
    main()
