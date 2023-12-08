import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt


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

# Option 4: Show an area chart for Feature2
st.header("Option 4: Feature2 Distribution")
area_chart_data = tru_lens_dashboard.groupby('Agent')['Feature2'].sum()
st.area_chart(area_chart_data)

# Option 5: Show a line chart for LLMS Scores over time (assumed time data)
st.header("Option 5: LLMS Scores over Time")
time_data = np.arange(len(tru_lens_dashboard))
st.line_chart(pd.DataFrame({'Time': time_data, 'LLMS Score': tru_lens_dashboard['LLMS Score']}))

# Option 6: Show a scatter plot for LLMS Scores vs. another metric (Feature1)
st.header("Option 6: LLMS Scores vs. Feature1")
st.scatter_chart(pd.DataFrame({'LLMS Score': tru_lens_dashboard['LLMS Score'], 'Feature1': tru_lens_dashboard['Feature1']}))

# Option 7: Show an Altair chart for Feature3
st.header("Option 7: Feature3 Distribution")
chart = alt.Chart(tru_lens_dashboard).mark_bar().encode(x='Feature3', y='count()')
st.altair_chart(chart, use_container_width=True)


# Option 10: Show a PyDeck chart for Feature6 and Feature7
st.header("Option 10: Feature6 vs. Feature7 (PyDeck)")
pydeck_chart_data = pd.DataFrame({'Feature6': tru_lens_dashboard['Feature6'], 'Feature7': tru_lens_dashboard['Feature7']})
st.pydeck_chart(pydeck_chart_data)


# Option 12: Show a Graphviz chart (Example: Pie chart for Feature9)
st.header("Option 12: Feature9 Distribution (Graphviz)")
graphviz_chart_data = tru_lens_dashboard['Feature9'].value_counts()
st.graphviz_chart(graphviz_chart_data)


# You can customize and add more options based on your specific needs

# Note: Adjust the features and their representations based on your actual data and use case.
