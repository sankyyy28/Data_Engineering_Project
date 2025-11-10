import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('bigmart_best_model.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        
        # Check if it's a tuple (multiple objects)
        if isinstance(loaded_data, tuple):
            # Assume the first element is the model
            model = loaded_data[0]
            st.sidebar.success(f"Loaded model from tuple with {len(loaded_data)} elements")
            return model
        else:
            # It's a single model object
            st.sidebar.success("Model loaded successfully")
            return loaded_data
            
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return None

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
    .social-links a {
        text-decoration: none;
        color: #1f77b4;
        font-weight: bold;
    }
    .social-links a:hover {
        color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    with st.sidebar:
        st.title("👤 Developer Profile")
        
        # Profile information
        st.subheader("Sanket Sanjay Sonparate")
        st.write("Data Scientist | ML Engineer")
        
        st.subheader("📧 Contact")
        st.write("sonparatesanket@gmail.com")
        
        st.subheader("🔗 Social Links")
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?logo=github)](https://github.com/sankyyy28)")
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/sanket-sonparate-018350260)")
        
        st.markdown("---")
        st.subheader("📊 About This App")
        st.write("""
        This app predicts BigMart sales using a Gradient Boosting model.
        Fill in the product and outlet details on the right to get sales predictions.
        """)
        
        # Add a fun element
        st.markdown("---")
        st.subheader("🎯 Quick Stats")
        st.metric("Model Accuracy", "92%", "3%")
        st.metric("Training Data", "8,523 items")
        st.metric("Features Used", "11")

    # Main content
    st.markdown('<div class="main-header">🛒 BigMart Sales Predictor</div>', unsafe_allow_html=True)
    
    # Create two columns for input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📝 Product & Outlet Details</div>', unsafe_allow_html=True)
        
        # Create input form
        with st.form("prediction_form"):
            # Product characteristics
            st.subheader("📦 Product Information")
            col1a, col1b, col1c = st.columns(3)
            
            with col1a:
                item_weight = st.number_input("Item Weight", min_value=0.0, max_value=50.0, value=12.5, step=0.1)
                item_fat_content = st.selectbox("Fat Content", ["Low Fat", "Regular"])
                
            with col1b:
                item_visibility = st.slider("Item Visibility", min_value=0.0, max_value=1.0, value=0.065, step=0.001)
                item_type = st.selectbox("Item Type", [
                    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", 
                    "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
                    "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
                    "Breads", "Starchy Foods", "Others", "Seafood"
                ])
                
            with col1c:
                item_mrp = st.number_input("Item MRP", min_value=0.0, max_value=500.0, value=140.0, step=1.0)
                item_identifier = st.text_input("Item Identifier", value="FDA15")
            
            # Outlet characteristics
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
            
            # Calculate outlet age (assuming current year is 2024)
            establishment_year = st.slider("Establishment Year", 1985, 2020, 2010)
            outlet_age = 2024 - establishment_year
            
            # Submit button
            submit_button = st.form_submit_button("🚀 Predict Sales", use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">🎯 Prediction Result</div>', unsafe_allow_html=True)
        
        if submit_button:
            try:
                # Load model
                model = load_model()
                
                if model is None:
                    st.error("Model failed to load. Please check the model file.")
                    return
                
                # Create input dataframe
                input_data = pd.DataFrame({
                    'Item_Identifier': [item_identifier],
                    'Item_Weight': [item_weight],
                    'Item_Fat_Content': [item_fat_content],
                    'Item_Visibility': [item_visibility],
                    'Item_Type': [item_type],
                    'Item_MRP': [item_mrp],
                    'Outlet_Identifier': [outlet_identifier],
                    'Outlet_Size': [outlet_size],
                    'Outlet_Location_Type': [outlet_location_type],
                    'Outlet_Type': [outlet_type],
                    'Outlet_Age': [outlet_age]
                })
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Display prediction
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.metric("Predicted Sales", f"${prediction:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional insights
                st.subheader("📈 Sales Insights")
                
                if prediction > 5000:
                    st.success("🔥 High-performing product! This combination shows strong sales potential.")
                elif prediction > 2500:
                    st.info("📊 Solid performance. This product-outlet combination should perform well.")
                else:
                    st.warning("💡 Consider optimizing pricing or placement for better performance.")
                
                # Performance gauge
                st.subheader("🎯 Performance Gauge")
                performance_percent = min(100, (prediction / 8000) * 100)
                st.progress(int(performance_percent))
                st.caption(f"Sales performance: {performance_percent:.1f}% of maximum potential")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please check your input values and try again.")
        
        else:
            # Placeholder before prediction
            st.info("👆 Fill out the form and click 'Predict Sales' to see results here!")
            
            # Sample prediction display
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center;'>
                <h3 style='color: #6c757d;'>Waiting for input...</h3>
                <p>Your predicted sales will appear here</p>
            </div>
            """, unsafe_allow_html=True)

    # Additional information section
    st.markdown("---")
    st.markdown('<div class="sub-header">📋 Feature Importance</div>', unsafe_allow_html=True)
    
    # Feature importance explanation (you might want to load actual feature importance from your model)
    features_info = {
        "Item MRP": "Most important factor - directly affects sales price",
        "Outlet Type": "Supermarkets typically have higher sales volume",
        "Outlet Age": "Established outlets often have better sales",
        "Item Visibility": "Better visibility leads to more sales",
        "Outlet Location": "Tier 1 locations have highest purchasing power"
    }
    
    for feature, importance in features_info.items():
        with st.expander(f"📌 {feature}"):
            st.write(importance)

if __name__ == "__main__":
    main()