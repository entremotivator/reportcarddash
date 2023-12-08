import streamlit as st
import pandas as pd
import numpy as np

# Function to generate some sample data
def generate_data():
    data = {
        'Feature 1': np.random.randn(100),
        'Feature 2': np.random.randint(0, 10, 100),
        'Feature 3': np.random.choice(['A', 'B', 'C'], 100),
        'Feature 4': np.random.uniform(1, 100, 100),
        'Feature 5': np.random.choice([True, False], 100),
        'Feature 6': np.random.choice(['Red', 'Blue', 'Green'], 100),
        'Feature 7': np.random.normal(50, 10, 100),
        'Feature 8': np.random.exponential(5, 100),
        'Feature 9': np.random.choice([1, 2, 3, 4, 5], 100),
        'Feature 10': np.random.randint(100, 1000, 100),
    }
    return pd.DataFrame(data)

# Generate sample data
df = generate_data()

# Streamlit sidebar
st.sidebar.title("Dashboard Options")

# Example options in the sidebar
option_show_summary = st.sidebar.checkbox("Show Summary Statistics", value=True)
selected_feature = st.sidebar.selectbox("Select Feature", df.columns)

# Display the dashboard content
st.title("Streamlit Dashboard with 10 Features")

# Feature 1: Show the DataFrame
st.header("Feature 1: Display DataFrame")
st.dataframe(df)

# Feature 2: Show summary statistics
if option_show_summary:
    st.header("Feature 2: Summary Statistics")
    st.write(df.describe())

# Feature 3: Show a bar chart
st.header("Feature 3: Bar Chart")
st.bar_chart(df[selected_feature].value_counts())

# Feature 4: Show a line chart
st.header("Feature 4: Line Chart")
st.line_chart(df[selected_feature])

# Feature 5: Show a pie chart
st.header("Feature 5: Pie Chart")
st.pie_chart(df[selected_feature].value_counts())

# Feature 6: Show a scatter plot
st.header("Feature 6: Scatter Plot")
st.scatter_chart(df[[selected_feature, 'Feature 7']])

# Feature 7: Show a histogram
st.header("Feature 7: Histogram")
st.hist_chart(df[selected_feature])

# Feature 8: Show a number input
st.header("Feature 8: Number Input")
selected_number = st.number_input("Select a number", min_value=0, max_value=100, value=50)

# Feature 9: Show a text input
st.header("Feature 9: Text Input")
selected_text = st.text_input("Enter some text", "Default text")

# Feature 10: Show a date input
st.header("Feature 10: Date Input")
selected_date = st.date_input("Select a date")

# You can continue to add more features based on your requirements

# Note: Adjust the features and their representations based on your actual data and use case.
