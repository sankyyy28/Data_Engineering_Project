import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def safe_import_sklearn():
    """Safely import sklearn with error handling"""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor, None
    except ImportError as e:
        return None, f"scikit-learn not installed: {e}"
    except Exception as e:
        return None, f"Error importing sklearn: {e}"

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'use_demo_model' not in st.session_state:
    st.session_state.use_demo_model = False
if 'real_model' not in st.session_state:
    st.session_state.real_model = None

@st.cache_resource
def load_real_model():
    """Load the real model with error handling"""
    try:
        if not os.path.exists('bigmart_best_model.pkl'):
            return None, "Model file 'bigmart_best_model.pkl' not found"
        
        with open('bigmart_best_model.pkl', 'rb') as file:
            model_data = pickle.load(file)
        
        # Handle different model formats
        if hasattr(model_data, 'predict'):
            return model_data, "Real model loaded successfully"
        elif isinstance(model_data, tuple) and len(model_data) > 0:
            for item in model_data:
                if hasattr(item, 'predict'):
                    return item, "Real model loaded from tuple"
            return model_data[0], "Using first item from tuple as model"
        elif isinstance(model_data, dict):
            for key in ['model', 'estimator', 'regressor']:
                if key in model_data and hasattr(model_data[key], 'predict'):
                    return model_data[key], f"Real model loaded from dict key: {key}"
        
        return None, "Could not extract model from file"
        
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def create_demo_model():
    """Create a demo model for testing"""
    try:
        GradientBoostingRegressor, error = safe_import_sklearn()
        if error:
            return None, error
        
        np.random.seed(42)
        # Create sample training data
        n_samples = 500
        X_demo = np.random.rand(n_samples, 10)
        # Simulate sales: base + features + noise
        y_demo = 1500 + X_demo[:, 0] * 2500 + X_demo[:, 1] * 1800 + np.random.randn(n_samples) * 300
        
        model = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=4)
        model.fit(X_demo, y_demo)
        return model, "Demo model created successfully"
        
    except Exception as e:
        return None, f"Error creating demo model: {str(e)}"

def preprocess_input(input_dict):
    """Preprocess input data for prediction"""
    try:
        # Map categorical variables to numerical
        fat_map = {'Low Fat': 0, 'Regular': 1}
        size_map = {'Small': 0, 'Medium': 1, 'High': 2}
        location_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
        type_map = {
            'Grocery Store': 0,
            'Supermarket Type1': 1,
            'Supermarket Type2': 2, 
            'Supermarket Type3': 3
        }
        item_type_map = {
            'Dairy': 0, 'Soft Drinks': 1, 'Meat': 2, 'Fruits and Vegetables': 3,
            'Household': 4, 'Baking Goods': 5, 'Snack Foods': 6, 'Frozen Foods': 7,
            'Breakfast': 8, 'Health and Hygiene': 9, 'Hard Drinks': 10, 'Canned': 11,
            'Breads': 12, 'Starchy Foods': 13, 'Others': 14, 'Seafood': 15
        }
        
        # Create feature array
        features = [
            input_dict['weight'],
            input_dict['visibility'],
            input_dict['mrp'],
            fat_map.get(input_dict['fat_content'], 0),
            item_type_map.get(input_dict['item_type'], 0),
            size_map.get(input_dict['outlet_size'], 0),
            location_map.get(input_dict['location_type'], 0),
            type_map.get(input_dict['outlet_type'], 0),
            input_dict['outlet_age'],
            1.0  # Bias term
        ]
        
        return np.array([features]), None
        
    except Exception as e:
        return None, f"Preprocessing error: {str(e)}"

def main():
    # Sidebar
    with st.sidebar:
        st.title("👤 Developer Profile")
        st.subheader("Sanket Sanjay Sonparate")
        st.write("Data Scientist | ML Engineer")
        
        st.markdown("📧 **Contact**")
        st.write("sonparatesanket@gmail.com")
        
        st.markdown("🔗 **Social Links**")
        st.markdown("[GitHub](https://github.com/sankyyy28) • [LinkedIn](https://www.linkedin.com/in/sanket-sonparate-018350260)")
        
        st.markdown("---")
        st.subheader("🔧 Model Status")
        
        # Load real model if not already loaded
        if not st.session_state.model_loaded:
            with st.spinner("Checking for model..."):
                real_model, message = load_real_model()
                if real_model is not None:
                    st.session_state.real_model = real_model
                    st.session_state.model_loaded = True
                    st.success("✅ " + message)
                else:
                    st.error("❌ " + message)
                    st.info("Using demo model instead")
                    st.session_state.use_demo_model = True
        
        if st.session_state.real_model:
            st.success("✅ Real model ready!")
        else:
            st.warning("⚠️ Using demo model")
            
        st.markdown("---")
        st.subheader("📊 App Info")
        st.write("Predict BigMart sales using machine learning")

    # Main content area
    st.markdown('<div class="main-header">🛒 BigMart Sales Predictor</div>', unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📝 Product & Outlet Details</div>', unsafe_allow_html=True)
        
        # Input form
        with st.form("sales_form"):
            st.subheader("📦 Product Information")
            
            # Product inputs in columns
            p_col1, p_col2, p_col3 = st.columns(3)
            
            with p_col1:
                item_weight = st.number_input("Item Weight (kg)", 
                                            min_value=0.0, max_value=30.0, 
                                            value=12.5, step=0.1)
                item_fat = st.selectbox("Fat Content", ["Low Fat", "Regular"])
                
            with p_col2:
                item_visibility = st.slider("Visibility %", 
                                          min_value=0.0, max_value=0.2, 
                                          value=0.07, step=0.001)
                item_type = st.selectbox("Product Type", 
                                       ["Dairy", "Soft Drinks", "Meat", "Snack Foods", 
                                        "Frozen Foods", "Bakery", "Canned", "Household"])
                
            with p_col3:
                item_mrp = st.number_input("Item MRP ($)", 
                                         min_value=0.0, max_value=300.0, 
                                         value=140.0, step=1.0)
                item_id = st.text_input("Item ID", value="FDA15")
            
            st.subheader("🏪 Outlet Information")
            
            # Outlet inputs in columns
            o_col1, o_col2 = st.columns(2)
            
            with o_col1:
                outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
                outlet_location = st.selectbox("Location Tier", ["Tier 1", "Tier 2", "Tier 3"])
                
            with o_col2:
                outlet_type = st.selectbox("Store Type", 
                                         ["Grocery Store", "Supermarket Type1", 
                                          "Supermarket Type2", "Supermarket Type3"])
                establishment_year = st.slider("Established Year", 1985, 2020, 2010)
            
            outlet_age = 2024 - establishment_year
            st.info(f"**Outlet Age:** {outlet_age} years")
            
            # Submit button
            submit_btn = st.form_submit_button("🚀 Predict Sales", use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">🎯 Prediction Result</div>', unsafe_allow_html=True)
        
        if submit_btn:
            # Determine which model to use
            if st.session_state.real_model:
                current_model = st.session_state.real_model
                model_type = "real"
                model_message = "Using trained model"
            else:
                current_model, demo_message = create_demo_model()
                model_type = "demo"
                model_message = "Using demo model"
            
            if current_model is None:
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.error("❌ No model available for prediction")
                st.write("Please check if scikit-learn is installed and try again.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                try:
                    # Prepare input data
                    input_data = {
                        'weight': item_weight,
                        'visibility': item_visibility,
                        'mrp': item_mrp,
                        'fat_content': item_fat,
                        'item_type': item_type,
                        'outlet_size': outlet_size,
                        'location_type': outlet_location,
                        'outlet_type': outlet_type,
                        'outlet_age': outlet_age
                    }
                    
                    # Preprocess and predict
                    features, preprocess_error = preprocess_input(input_data)
                    
                    if preprocess_error:
                        st.error(f"Preprocessing error: {preprocess_error}")
                    else:
                        prediction = current_model.predict(features)[0]
                        prediction = max(0, min(prediction, 10000))  # Reasonable bounds
                        
                        # Display prediction
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.metric("Predicted Sales", f"${prediction:,.2f}")
                        st.caption(f"📊 {model_message}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Sales insights
                        st.subheader("📈 Sales Insights")
                        if prediction > 4000:
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.success("**Excellent!** High sales potential")
                            st.markdown('</div>', unsafe_allow_html=True)
                        elif prediction > 2500:
                            st.markdown('<div class="info-box">', unsafe_allow_html=True)
                            st.info("**Good performance** expected")
                            st.markdown('</div>', unsafe_allow_html=True)
                        elif prediction > 1500:
                            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                            st.warning("**Moderate** - Consider optimization")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-box">', unsafe_allow_html=True)
                            st.error("**Low sales** - Review strategy")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Performance gauge
                        progress_val = min(100, int((prediction / 5000) * 100))
                        st.progress(progress_val)
                        st.caption(f"Performance: {progress_val}% of target")
                        
                except Exception as e:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.write("Please check your input values and try again.")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            # Default state before prediction
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.info("**Ready for Prediction**")
            st.write("""
            1. Fill in product details
            2. Provide outlet information  
            3. Click **Predict Sales**
            4. View results here
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Quick info
            st.markdown("---")
            st.subheader("💡 Tips")
            st.write("• Higher visibility → Better sales")
            st.write("• Supermarkets > Grocery stores")
            st.write("• Tier 1 locations perform best")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 1rem;'>
        <p>Built with Streamlit • 
        <a href='https://github.com/sankyyy28' style='color: #1f77b4;'>GitHub</a> • 
        <a href='https://www.linkedin.com/in/sanket-sonparate-018350260' style='color: #1f77b4;'>LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
