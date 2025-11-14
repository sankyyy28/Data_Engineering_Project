import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os

# Set page configuration
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Handle missing dependencies gracefully
def check_dependencies():
    missing_deps = []
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("Pillow")
    
    return missing_deps

# Check for missing dependencies
missing_dependencies = check_dependencies()

if missing_dependencies:
    st.error(f"⚠️ Missing dependencies: {', '.join(missing_dependencies)}")
    st.info("Please install required packages using:")
    for dep in missing_dependencies:
        if dep == "scikit-learn":
            st.code("pip install scikit-learn")
        elif dep == "Pillow":
            st.code("pip install Pillow")
    st.stop()

# Now safely import sklearn since we've checked it exists
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

# Enhanced model loading with multiple fallback options
@st.cache_resource
def load_model():
    try:
        if not os.path.exists('bigmart_best_model.pkl'):
            st.sidebar.error("❌ Model file 'bigmart_best_model.pkl' not found")
            return None
            
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
            if model is None and len(loaded_data) > 0:
                model = loaded_data[0]
                st.sidebar.warning("Using first element of tuple as model")
        
        # Case 2: It's a single model object
        elif hasattr(loaded_data, 'predict'):
            model = loaded_data
            st.sidebar.success("✓ Single model object loaded")
        
        # Case 3: It's a dictionary or other structure
        elif isinstance(loaded_data, dict):
            for key in ['model', 'estimator', 'regressor', 'classifier']:
                if key in loaded_data and hasattr(loaded_data[key], 'predict'):
                    model = loaded_data[key]
                    st.sidebar.success(f"✓ Found model in dictionary under '{key}' key")
                    break
        
        if model is None:
            st.sidebar.error("Could not identify model in pickle file")
            return None
            
        return model
            
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

# Create a simple demo model for testing
def create_demo_model():
    """Create a simple demo model for testing"""
    try:
        # Create simple demo data
        np.random.seed(42)
        X_demo = np.random.rand(100, 7)
        y_demo = 1500 + X_demo[:, 0] * 3000 + X_demo[:, 1] * 2000 + np.random.randn(100) * 500
        
        model = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=3)
        model.fit(X_demo, y_demo)
        return model
    except Exception as e:
        st.error(f"Error creating demo model: {e}")
        return None

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_input(input_dict):
    """Preprocess input data for prediction"""
    try:
        # Map categorical variables to numerical values
        fat_content_map = {'Low Fat': 0, 'Regular': 1}
        outlet_size_map = {'Small': 0, 'Medium': 1, 'High': 2}
        location_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
        outlet_type_map = {
            'Grocery Store': 0, 
            'Supermarket Type1': 1, 
            'Supermarket Type2': 2, 
            'Supermarket Type3': 3
        }
        
        # Create feature array (simplified - adjust based on your actual model)
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
    # Initialize session state
    if 'use_demo_model' not in st.session_state:
        st.session_state.use_demo_model = False
    
    # Sidebar
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
        
        # Try to load the real model first
        real_model = load_model()
        
        if real_model is None:
            st.error("❌ Real model not available")
            st.markdown("""
            **Options:**
            1. Ensure `bigmart_best_model.pkl` exists
            2. Use demo model below for testing
            """)
            
            if st.button("🔄 Use Demo Model", key="demo_btn"):
                st.session_state.use_demo_model = True
                st.rerun()
        else:
            st.success("✅ Real model loaded")
            st.session_state.use_demo_model = False
        
        if st.session_state.use_demo_model:
            st.warning("⚠️ Using demo model")
        
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
                item_weight = st.number_input("Item Weight", min_value=0.0, max_value=50.0, value=12.5, step=0.1)
                item_fat_content = st.selectbox("Fat Content", ["Low Fat", "Regular"])
                
            with col1b:
                item_visibility = st.slider("Item Visibility", min_value=0.0, max_value=0.3, value=0.07, step=0.001)
                item_type = st.selectbox("Item Type", [
                    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", 
                    "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
                    "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
                    "Breads", "Starchy Foods", "Others", "Seafood"
                ])
                
            with col1c:
                item_mrp = st.number_input("Item MRP", min_value=0.0, max_value=500.0, value=140.0, step=1.0)
                item_identifier = st.text_input("Item Identifier", value="FDA15")
            
            # Outlet Information
            st.subheader("🏪 Outlet Information")
            col2a, col2b = st.columns(2)
            
            with col2a:
                outlet_identifier = st.selectbox("Outlet Identifier", [
                    "OUT010", "OUT013", "OUT017", "OUT018", "OUT019",
                    "OUT027", "OUT035", "OUT045", "OUT046", "OUT049"
                ])
                outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
                
            with col2b:
                outlet_location_type = st.selectbox("Location Type", ["Tier 1", "Tier 2", "Tier 3"])
                outlet_type = st.selectbox("Outlet Type", [
                    "Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"
                ])
            
            # Establishment year
            establishment_year = st.slider("Establishment Year", 1985, 2020, 2010)
            outlet_age = 2024 - establishment_year
            
            st.info(f"Outlet Age: {outlet_age} years")
            
            submit_button = st.form_submit_button("🚀 Predict Sales", use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">🎯 Prediction Result</div>', unsafe_allow_html=True)
        
        if submit_button:
            # Determine which model to use
            if st.session_state.use_demo_model:
                current_model = create_demo_model()
                model_type = "demo"
            else:
                current_model = real_model
                model_type = "real"
            
            if current_model is None:
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.error("❌ No model available for prediction")
                st.markdown("""
                **Please:**
                1. Check if model file exists, or
                2. Use the demo model from sidebar
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
                    
                    # Preprocess and predict
                    features = preprocess_input(input_data)
                    if features is not None:
                        prediction = current_model.predict(features)[0]
                        
                        # Ensure prediction is reasonable
                        prediction = max(0, min(prediction, 10000))
                        
                        # Display results
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.metric("Predicted Sales", f"${prediction:,.2f}")
                        if model_type == "demo":
                            st.caption("⚠️ Using demo model - results for demonstration only")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Sales insights
                        st.subheader("📈 Sales Insights")
                        if prediction > 4500:
                            st.success("🔥 **Excellent!** High sales potential detected")
                        elif prediction > 2500:
                            st.info("📊 **Good** - Solid performance expected")
                        elif prediction > 1000:
                            st.warning("💡 **Moderate** - Consider optimization")
                        else:
                            st.error("📉 **Low** - Review product placement and pricing")
                        
                        # Performance gauge
                        performance_pct = min(100, int((prediction / 6000) * 100))
                        st.progress(performance_pct)
                        st.caption(f"Sales performance: {performance_pct}% of target")
                        
                except Exception as e:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("This might be due to model compatibility issues.")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            # Default state before prediction
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.info("👆 **How to use:**")
            st.markdown("""
            1. Fill in product details on the left
            2. Provide outlet information  
            3. Click **Predict Sales** button
            4. View prediction results here
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 1rem;'>
        <p>BigMart Sales Predictor • Built with Streamlit • 
        <a href='https://github.com/sankyyy28' style='color: #1f77b4; text-decoration: none;'>GitHub Repository</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
