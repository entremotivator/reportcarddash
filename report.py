import streamlit as st
import pandas as pd
import numpy as np

# Assume `tru.run_dashboard()` returns the content of your dashboard
def run_dashboard():
    # Generate data with 10 agents, LLMS Score, Cost, Feature1 to Feature14
    num_agents = 10
    data = {
        'Agent': np.random.choice([f'Agent {i}' for i in range(1, num_agents + 1)], 100),
        'LLMS Score': np.random.randint(0, 100, 100),
        'Cost': np.random.randint(1000, 5000, 100),
        'Feature1': np.random.normal(0, 1, 100),
        'Feature2': np.random.uniform(0, 1, 100),
        'Feature3': np.random.choice(['A', 'B', 'C'], 100),
        'Feature4': np.random.randint(1, 5, 100),
        'Feature5': np.random.randn(100),
        'Feature6': np.random.exponential(1, 100),
        'Feature7': np.random.gamma(2, 2, 100),
        'Feature8': np.random.logistic(0, 1, 100),
        'Feature9': np.random.poisson(5, 100),
        'Feature10': np.random.uniform(-1, 1, 100),
        'Feature11': np.random.choice(['X', 'Y', 'Z'], 100),
        'Feature12': np.random.randint(10, 20, 100),
        'Feature13': np.random.normal(5, 2, 100),
        'Feature14': np.random.uniform(0, 10, 100),
    }
    tru_lens_df = pd.DataFrame(data)
    return tru_lens_df

tru_lens_dashboard = run_dashboard()

# Add Streamlit sidebar
st.sidebar.title("TruLens Dashboard Options")

# Options in the sidebar
option_show_summary = st.sidebar.checkbox("Show Summary Statistics", value=True)
selected_agent = st.sidebar.selectbox("Select Agent", tru_lens_dashboard['Agent'].unique())

# Display the dashboard content
st.title("TruLens Dashboard")

# Option 1: Show the DataFrame
st.header("Option 1: Display TruLens DataFrame")
st.dataframe(tru_lens_dashboard)

# Option 2: Show summary statistics
if option_show_summary:
    st.header("Option 2: Summary Statistics")
    st.write(tru_lens_dashboard.describe())

# Option 3: Show a bar chart for LLMS Scores by Agent
st.header("Option 3: LLMS Scores by Agent")
bar_chart_data = tru_lens_dashboard.groupby('Agent')['LLMS Score'].mean()
st.bar_chart(bar_chart_data)

# Option 4: Show a line chart for LLMS Scores over time (assumed time data)
st.header("Option 4: LLMS Scores over Time")
time_data = np.arange(len(tru_lens_dashboard))
st.line_chart(pd.DataFrame({'Time': time_data, 'LLMS Score': tru_lens_dashboard['LLMS Score']}))

# Option 5: Show a pie chart for the distribution of LLMS Scores
st.header("Option 5: Distribution of LLMS Scores")
llms_counts = tru_lens_dashboard['LLMS Score'].value_counts()
st.pie_chart(llms_counts)


# Option 6: Show a scatter plot for LLMS Scores vs. another metric (Feature1)
st.header("Option 6: LLMS Scores vs. Feature1")
st.scatter_chart(pd.DataFrame({'LLMS Score': tru_lens_dashboard['LLMS Score'], 'Feature1': tru_lens_dashboard['Feature1']}))

# Option 7: Show a histogram of LLMS Scores
st.header("Option 7: Histogram of LLMS Scores")
st.hist_chart(tru_lens_dashboard['LLMS Score'])

# Option 8: Show a number input for custom LLMS Score threshold
st.header("Option 8: Custom LLMS Score Threshold")
custom_threshold = st.number_input("Enter LLMS Score Threshold", min_value=0, max_value=100, value=50)

# Option 9: Show a text input for filtering by Agent
st.header("Option 9: Filter by Agent")
filter_agent = st.text_input("Enter Agent Name")

# Option 10: Show a date input for filtering by date (assumed date data)
st.header("Option 10: Filter by Date")
filter_date = st.date_input("Select a Date")

# Option 11: Show a cost chart for each agent
st.header("Option 11: Cost by Agent")
cost_chart_data = tru_lens_dashboard.groupby('Agent')['Cost'].sum()
st.bar_chart(cost_chart_data)

# Option 12: Show a bar chart for Feature3
st.header("Option 12: Feature3 Distribution")
feature3_chart_data = tru_lens_dashboard['Feature3'].value_counts()
st.bar_chart(feature3_chart_data)

# Option 13: Show a bar chart for Feature4
st.header("Option 13: Feature4 Distribution")
feature4_chart_data = tru_lens_dashboard['Feature4'].value_counts()
st.bar_chart(feature4_chart_data)

# Option 14: Show a line chart for Feature5
st.header("Option 14: Feature5 over Time")
st.line_chart(pd.DataFrame({'Time': time_data, 'Feature5': tru_lens_dashboard['Feature5']}))

# Option 15: Show a scatter plot for Feature6 vs. Feature7
st.header("Option 15: Feature6 vs. Feature7")
st.scatter_chart(pd.DataFrame({'Feature6': tru_lens_dashboard['Feature6'], 'Feature7': tru_lens_dashboard['Feature7']}))

# Option 16: Show a histogram for Feature8
st.header("Option 16: Histogram of Feature8")
st.hist_chart(tru_lens_dashboard['Feature8'])

# Option 17: Show a number input for custom Feature9 threshold
st.header("Option 17: Custom Feature9 Threshold")
custom_feature9_threshold = st.number_input("Enter Feature9 Threshold", min_value=0, max_value=10, value=5)

# Option 18: Show a text input for filtering by Feature11
st.header("Option 18: Filter by Feature11")
filter_feature11 = st.text_input("Enter Feature11 Value")

# Option 19: Show a pie chart for the distribution of Feature12
st.header("Option 19: Distribution of Feature12")
st.pie_chart(tru_lens_dashboard['Feature12'].value_counts())

# Option 20: Show a bar chart for Feature13
st.header("Option 20: Feature13 Distribution")
feature13_chart_data = tru_lens_dashboard['Feature13'].value_counts()
st.bar_chart(feature13_chart_data)

# You can customize and add more options based on your specific needs

# Note: Adjust the features and their representations based on your actual data and use case.
