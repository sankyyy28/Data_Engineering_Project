# BigMart Sales Predictor 

## Overview
A machine learning web application that predicts sales for BigMart products using a Gradient Boosting model. The app provides an intuitive interface for users to input product and outlet details to get sales predictions.

👉 Live Demo: http://localhost:8502/

## Features
- **Sales Prediction**: Predict product sales based on various features
- **User-Friendly Interface**: Clean Streamlit-based web interface
- **Real-time Insights**: Performance gauges and sales insights
- **Developer Profile**: Contact information and social links

## Architecture Overview

```
BigMart Sales Predictor Architecture
flowchart TD
    subgraph Ingestion [📥 Data Ingestion]
        A1[📄 df_item.xml] --> A4[(MySQL: item_info)]
        A2[📄 df_outlet.xml] --> A5[(MySQL: outlet_info)]
        A3[📄 df_sales.xml] --> A6[(MySQL: sales_info)]
    end

    subgraph Processing [⚙️ Data Processing]
        A4 --> B1[🔗 Merge Tables]
        A5 --> B1
        A6 --> B1
        B1 --> B2[🧹 Cleaning & Feature Engineering]
        B2 --> B3[🔀 Train/Test Split]
    end

    subgraph Modeling [🤖 Model Training]
        B3 --> C1[📈 GradientBoostingRegressor]
        C1 --> C2[💾 Save bigmart_best_model.pkl]
    end

    subgraph Deployment [🚀 Streamlit App]
        C2 --> D1[🌐 Streamlit Web Interface]
        D1 --> D2[📊 Predict Sales]
    end


## Key Components

### 1. **User Interface (Streamlit)**
- Sidebar with developer profile and app information
- Main input form with product and outlet details
- Prediction results display with performance metrics

### 2. **Input Features**
- **Product Details**: Weight, Fat Content, Visibility, Type, MRP, Identifier
- **Outlet Details**: Identifier, Size, Location Type, Type, Age

### 3. **Machine Learning Model**
- Gradient Boosting algorithm
- 11 input features
- 92% accuracy on training data
- Model loaded from pickle file

### 4. **Output**
- Predicted sales amount in USD
- Performance insights and recommendations
- Visual performance gauge

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
