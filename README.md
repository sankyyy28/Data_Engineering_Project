# BigMart Sales Predictor 

## Overview
A machine learning web application that predicts sales for BigMart products using a Gradient Boosting model. The app provides an intuitive interface for users to input product and outlet details to get sales predictions.

👉 Live Demo: https://dataengineeringproject.streamlit.app/

## Features
- **Sales Prediction**: Predict product sales based on various features
- **User-Friendly Interface**: Clean Streamlit-based web interface
- **Real-time Insights**: Performance gauges and sales insights
- **Developer Profile**: Contact information and social links

## Architecture Overview
<img width="2385" height="3817" alt="deepseek_mermaid_20251110_bf4f86" src="https://github.com/user-attachments/assets/29c36eba-cbe3-44fa-b348-92d2c4fb5c7d" />

## Installation & Usage

1. **Install dependencies**:
```bash
pip install streamlit pandas numpy pillow
```

2. **Run the application**:
```bash
streamlit run App.py
```

3. **Access the app**:
   - Open browser and go to `http://localhost:8501`

## File Structure
```
BigMart-Sales-Predictor/
│
├── App.py                 # Main application file
├── bigmart_best_model.pkl # Pre-trained ML model
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Model Performance
- **Accuracy**: 92%
- **Training Data**: 8,523 products
- **Features Used**: 11
- **Algorithm**: Gradient Boosting

## Developer
**Sanket Sanjay Sonparate**
- Email: sonparatesanket@gmail.com
- GitHub: [sankyyy28](https://github.com/sankyyy28)
- LinkedIn: [Sanket Sonparate](https://www.linkedin.com/in/sanket-sonparate-018350260)

## Features Importance
The model considers these as most influential factors:
1. Item MRP (Maximum Retail Price)
2. Outlet Type
3. Outlet Age
4. Item Visibility
5. Outlet Location Type

This application demonstrates end-to-end machine learning deployment with a focus on user experience and practical business insights.
