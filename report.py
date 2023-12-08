import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

NUM_AGENTS = 10
NUM_ROWS = 100

def generate_data(num_agents=NUM_AGENTS, num_rows=NUM_ROWS):
    data = {
        'Agent': np.random.choice([f'Agent {i}' for i in range(1, num_agents + 1)], num_rows),
        'LLMS Score': np.random.randint(0, 100, num_rows),
        'Cost': np.random.randint(1000, 5000, num_rows),
        'Relevance': np.random.randint(1, 100, num_rows),
        'Groundedness': np.random.randint(1, 100, num_rows),
        'Sentiment': np.random.randint(1, 100, num_rows),
        'Model Agreement': np.random.randint(1, 100, num_rows),
        'Language Match': np.random.randint(1, 100, num_rows),
        'Toxicity': np.random.randint(1, 100, num_rows),
        'Moderation': np.random.randint(1, 100, num_rows),
        'Stereotypes': np.random.randint(1, 100, num_rows),
        'Summarization': np.random.randint(1, 100, num_rows),
        'Embeddings Distance': np.random.randint(1, 100, num_rows),
        'Feature1': np.random.normal(0, 1, num_rows),
        'Feature2': np.random.uniform(0, 1, num_rows),
        'Feature3': np.random.choice(['A', 'B', 'C'], num_rows),
        'Feature4': np.random.randint(1, 5, num_rows),
        'Feature5': np.random.randn(num_rows),
        'Feature6': np.random.exponential(1, num_rows),
        'Feature7': np.random.gamma(2, 2, num_rows),
        'Feature8': np.random.logistic(0, 1, num_rows),
        'Feature9': np.random.poisson(5, num_rows),
        'Feature10': np.random.uniform(-1, 1, num_rows),
        'Feature11': np.random.choice(['X', 'Y', 'Z'], num_rows),
        'Feature12': np.random.randint(10, 20, num_rows),
        'Feature13': np.random.normal(5, 2, num_rows),
        'Feature14': np.random.uniform(0, 10, num_rows),
    }
    return pd.DataFrame(data)

def show_dataframe(data):
    st.header("Option 1: Display TruLens DataFrame")
    st.dataframe(data)

def show_summary_statistics(data):
    st.header("Option 2: Summary Statistics")
    st.write(data.describe())

def show_llms_scores_by_agent(data):
    st.header("Option 3: LLMS Scores by Agent")
    bar_chart_data = data.groupby('Agent')['LLMS Score'].mean()
    st.bar_chart(bar_chart_data)

def show_feature2_distribution(data):
    st.header("Option 4: Feature2 Distribution")
    area_chart_data = data.groupby('Agent')['Feature2'].sum()
    st.area_chart(area_chart_data)

def show_llms_scores_over_time(data):
    st.header("Option 5: LLMS Scores over Time")
    time_data = np.arange(len(data))
    st.line_chart(pd.DataFrame({'Time': time_data, 'LLMS Score': data['LLMS Score']}))

def show_llms_scores_vs_feature1(data):
    st.header("Option 6: LLMS Scores vs. Feature1")
    st.scatter_chart(pd.DataFrame({'LLMS Score': data['LLMS Score'], 'Feature1': data['Feature1']}))

def show_feature3_distribution(data):
    st.header("Option 7: Feature3 Distribution")
    chart = alt.Chart(data).mark_bar().encode(x='Feature3', y='count()')
    st.altair_chart(chart, use_container_width=True)

def show_cost_by_feature(data):
    st.header("Option 8: Cost by Feature")
    scatter_chart_data = pd.DataFrame({'Cost': data['Cost'], 'Feature4': data['Feature4']})
    st.scatter_chart(scatter_chart_data)


def show_feature12_vs_feature13_scatter(data):
    st.header("Option 12: Feature12 vs. Feature13 Scatter Plot")
    st.scatter_chart(pd.DataFrame({'Feature12': data['Feature12'], 'Feature13': data['Feature13']}))

def show_feature14_distribution(data):
    st.header("Option 13: Feature14 Distribution")
    hist_chart_data = data['Feature14']
    st.hist_chart(hist_chart_data)

# ... Add more chart functions similarly

def main():
    tru_lens_dashboard = generate_data()

    st.sidebar.title("TruLens Dashboard Options")
    option_show_summary = st.sidebar.checkbox("Show Summary Statistics", value=True)
    
st.title("TruLens Dashboard")


    
    if option_show_summary:
        show_summary_statistics(tru_lens_dashboard)
    
    show_llms_scores_by_agent(tru_lens_dashboard)
    show_feature2_distribution(tru_lens_dashboard)
    show_llms_scores_over_time(tru_lens_dashboard)
    show_llms_scores_vs_feature1(tru_lens_dashboard)
    show_feature3_distribution(tru_lens_dashboard)
    show_cost_by_feature(tru_lens_dashboard)
    show_feature12_vs_feature13_scatter(tru_lens_dashboard)
    show_feature14_distribution(tru_lens_dashboard)
    # ... Add more chart functions

if __name__ == "__main__":
    main()

