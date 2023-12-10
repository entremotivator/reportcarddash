import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from google.colab import auth as google_auth
google_auth.authenticate_user()

import vertexai
from vertexai.preview.language_models import TextGenerationModel

def show_chatbot_page():
    st.title("Chatbot Page")

    # Input form for user parameters
    api_key = st.text_input("Enter your Vertex AI API key:")
    model_name = st.text_input("Enter the model name:", "text-bison@001")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.2, step=0.1)
    max_decode_steps = st.number_input("Max Decode Steps", min_value=1, value=256)
    top_p = st.slider("Top P", 0.1, 1.0, 0.8, step=0.1)
    top_k = st.number_input("Top K", min_value=1, value=40)
    content = st.text_area("Input Content", '''Give me ten interview questions for the role of program manager''')

    if st.button("Generate Response"):
        if api_key:
            response = predict_large_language_model_sample(
                api_key,
                model_name,
                temperature,
                max_decode_steps,
                top_p,
                top_k,
                content,
                "us-central1"
            )
            st.write(f"Response from Model: {response}")
        else:
            st.warning("Please enter your Vertex AI API key.")

def main():
  
    st.title("Large Language Model Prediction")

    # Input form for user parameters
    api_key = st.text_input("Enter your Vertex AI API key:")
    model_name = st.text_input("Enter the model name:", "text-bison@001")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.2, step=0.1)
    max_decode_steps = st.number_input("Max Decode Steps", min_value=1, value=256)
    top_p = st.slider("Top P", 0.1, 1.0, 0.8, step=0.1)
    top_k = st.number_input("Top K", min_value=1, value=40)
    content = st.text_area("Input Content", '''Give me ten interview questions for the role of program manager''')

    if st.button("Generate Response"):
        if api_key:
            response = predict_large_language_model_sample(
                api_key,
                model_name,
                temperature,
                max_decode_steps,
                top_p,
                top_k,
                content,
                "us-central1"
            )
            st.write(f"Response from Model: {response}")
        else:
            st.warning("Please enter your Vertex AI API key.")
# ... (Existing code)

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
        'Time': np.arange(num_rows),  # Added Time variable
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

def load_data(file):
    return pd.read_csv(file)

# Define your analysis functions here

def show_analysis_page():
    st.title("Analysis Page")
    
    # Generate random data for analysis
    analysis_data = generate_data()

    # Display charts for each metric
    for metric in analysis_data.columns:
        st.subheader(metric)
        st.line_chart(analysis_data[metric])

def generate_data(num_agents=10, num_rows=100):
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
        'Time': np.arange(num_rows),  # Added Time variable
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
def show_textgpt_page():
    st.title("TextGPT Page")
    
    # Add content for TextGPT page
    st.write("This is the TextGPT Page content.")

    # Display tru lens metrics charts
    st.subheader("GPT Report Card Metrics")
    
    # Generate random data for tru lens metrics
    tru_lens_data = {
        "Clarity": np.random.uniform(0, 10, 100),
        "Novelty": np.random.uniform(0, 10, 100),
        "Intent Understanding": np.random.uniform(0, 10, 100),
        "Ambiguity Handling": np.random.uniform(0, 10, 100),
        "User Engagement": np.random.uniform(0, 10, 100),
        "Error Rate": np.random.uniform(0, 10, 100),
        "Adaptability": np.random.uniform(0, 10, 100),
        "Bias Detection": np.random.uniform(0, 10, 100),
        "Ambient Context Awareness": np.random.uniform(0, 10, 100),
        "Ethical Considerations": np.random.uniform(0, 10, 100),
    }
    
    tru_lens_df = pd.DataFrame(tru_lens_data)

    # Display charts for each metric
    for metric in tru_lens_df.columns:
        st.subheader(metric)
        st.line_chart(tru_lens_df[metric])

def show_imagegpt_page():
    st.title("ImageGPT Page")

    # Generate random data for ImageGPT metrics
    imagegpt_metrics_data = generate_imagegpt_metrics_data()

    # Display charts for each metric
    for metric in imagegpt_metrics_data.columns:
        st.subheader(metric)
        st.line_chart(imagegpt_metrics_data[metric])

def generate_imagegpt_metrics_data(num_images=100):
    data = {
        'Accuracy': np.random.uniform(0.8, 1.0, num_images),
        'Precision': np.random.uniform(0.7, 1.0, num_images),
        'Recall': np.random.uniform(0.7, 1.0, num_images),
        'F1 Score': np.random.uniform(0.7, 1.0, num_images),
        'IoU (Intersection over Union)': np.random.uniform(0.6, 1.0, num_images),
        'Top-1 Accuracy': np.random.uniform(0.8, 1.0, num_images),
        'Top-5 Accuracy': np.random.uniform(0.7, 1.0, num_images),
        'Mean Average Precision (mAP)': np.random.uniform(0.6, 1.0, num_images),
        'Perceptual Metrics (SSI)': np.random.uniform(0.6, 1.0, num_images),
        'Robustness': np.random.uniform(0.7, 1.0, num_images),
        'Latency': np.random.uniform(0.1, 0.5, num_images),  # Assuming latency is in seconds
    }
    return pd.DataFrame(data)


def show_agentgpt_page():
    st.title("AgentGPT Page")

    # Generate random data for AgentGPT metrics
    agentgpt_metrics_data = generate_agentgpt_metrics_data()

    # Display charts for each metric
    st.subheader("Task Completion Metrics")
    st.line_chart(agentgpt_metrics_data[['Tasks Completed', 'Task Accuracy']])

    st.subheader("Response Time Distribution")
    response_time_chart = alt.Chart(agentgpt_metrics_data).mark_bar().encode(
        x='Response Time',
        y='count()',
        tooltip=['count()']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(response_time_chart)

    # Add more charts for new metrics
    st.subheader("New Metric 1 Distribution")
    new_metric1_chart = alt.Chart(agentgpt_metrics_data).mark_bar().encode(
        x='New Metric 1',
        y='count()',
        tooltip=['count()']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(new_metric1_chart)

    st.subheader("New Metric 2 Over Time")
    new_metric2_over_time_chart = alt.Chart(agentgpt_metrics_data).mark_line().encode(
        x='Time',
        y='New Metric 2',
        tooltip=['Time', 'New Metric 2']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(new_metric2_over_time_chart)

    # Add more charts for additional metrics...

    # Add interactive filters
    date_range = st.slider("Select Date Range", 0, len(agentgpt_metrics_data), (0, len(agentgpt_metrics_data)))
    filtered_data = agentgpt_metrics_data.iloc[date_range[0]:date_range[1], :]

    # Use filtered_data in subsequent charts

def generate_agentgpt_metrics_data(num_tasks=100):
    data = {
        'Tasks Completed': np.random.randint(20, 100, num_tasks),
        'Task Accuracy': np.random.uniform(0.8, 1.0, num_tasks),
        'Response Time': np.random.uniform(0.5, 5.0, num_tasks),
        'Task Completion Rate': np.random.uniform(0.7, 1.0, num_tasks),
        'Task Category': np.random.choice(['Sales', 'Customer Support', 'Data Entry'], num_tasks),
        'Automations Implemented': np.random.randint(5, 20, num_tasks),
        'Task Efficiency': np.random.uniform(0.6, 1.0, num_tasks),
        'New Metric 1': np.random.uniform(0, 1, num_tasks),  # New metric
        'New Metric 2': np.random.randint(1, 10, num_tasks),  # Another new metric
        'Time': np.arange(num_tasks),
        # Add more metrics as needed
    }
    return pd.DataFrame(data)

# Your existing code for other pages and functions
# ...

def show_videogpt_page():
    st.title("VideoGPT Page")

    # Generate random data for VideoGPT metrics
    videogpt_metrics_data = generate_videogpt_metrics_data()

    # Display charts for each metric
    st.subheader("Video Quality Over Time")
    video_quality_chart = alt.Chart(videogpt_metrics_data).mark_line().encode(
        x='Time',
        y='Video Quality',
        tooltip=['Time', 'Video Quality']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(video_quality_chart)

    st.subheader("Frame Rate Over Time")
    frame_rate_chart = alt.Chart(videogpt_metrics_data).mark_line().encode(
        x='Time',
        y='Frame Rate',
        tooltip=['Time', 'Frame Rate']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(frame_rate_chart)

    st.subheader("Object Detection Accuracy")
    st.line_chart(videogpt_metrics_data['Object Detection Accuracy'])

    st.subheader("Action Recognition Accuracy")
    st.line_chart(videogpt_metrics_data['Action Recognition Accuracy'])

    st.subheader("Temporal Coherence Over Time")
    temporal_coherence_chart = alt.Chart(videogpt_metrics_data).mark_line().encode(
        x='Time',
        y='Temporal Coherence',
        tooltip=['Time', 'Temporal Coherence']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(temporal_coherence_chart)

    st.subheader("Scene Transition Quality")
    st.line_chart(videogpt_metrics_data['Scene Transition Quality'])

    st.subheader("Generative Diversity Over Time")
    generative_diversity_chart = alt.Chart(videogpt_metrics_data).mark_line().encode(
        x='Time',
        y='Generative Diversity',
        tooltip=['Time', 'Generative Diversity']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(generative_diversity_chart)

    st.subheader("Audio-Visual Synchronization")
    st.line_chart(videogpt_metrics_data['Audio-Visual Synchronization'])

    st.subheader("Robustness to Occlusions")
    st.line_chart(videogpt_metrics_data['Robustness to Occlusions'])

    st.subheader("Latency Over Time")
    latency_chart = alt.Chart(videogpt_metrics_data).mark_line().encode(
        x='Time',
        y='Latency',
        tooltip=['Time', 'Latency']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(latency_chart)

def generate_videogpt_metrics_data(num_frames=100):
    data = {
        'Time': np.arange(num_frames),
        'Video Quality': np.random.uniform(0.7, 1.0, num_frames),
        'Frame Rate': np.random.uniform(24, 60, num_frames),
        'Object Detection Accuracy': np.random.uniform(0.7, 1.0, num_frames),
        'Action Recognition Accuracy': np.random.uniform(0.7, 1.0, num_frames),
        'Temporal Coherence': np.random.uniform(0.7, 1.0, num_frames),
        'Scene Transition Quality': np.random.uniform(0.7, 1.0, num_frames),
        'Generative Diversity': np.random.uniform(0.7, 1.0, num_frames),
        'Audio-Visual Synchronization': np.random.uniform(0.7, 1.0, num_frames),
        'Robustness to Occlusions': np.random.uniform(0.7, 1.0, num_frames),
        'Latency': np.random.uniform(0.1, 0.5, num_frames),  # Assuming latency is in seconds
    }
    return pd.DataFrame(data)


def show_codegpt_page():
    st.title("CodeGPT Page")
    st.altair_chart(code_completion_chart)


def show_audiotts_page():
    st.title("AudioTTS/GPT Page")

    # Generate random data for AudioTTS/GPT metrics
    audiotts_metrics_data = generate_audiotts_metrics_data()

    # Chart for Naturalness
    st.subheader("Naturalness")
    naturalness_chart = alt.Chart(audiotts_metrics_data).mark_line().encode(
        x='Time',
        y='Naturalness',
        tooltip=['Time', 'Naturalness']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(naturalness_chart)

    # Chart for Intelligibility
    st.subheader("Intelligibility")
    intelligibility_chart = alt.Chart(audiotts_metrics_data).mark_line().encode(
        x='Time',
        y='Intelligibility',
        tooltip=['Time', 'Intelligibility']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(intelligibility_chart)

    # Chart for Prosody
    st.subheader("Prosody")
    prosody_chart = alt.Chart(audiotts_metrics_data).mark_line().encode(
        x='Time',
        y='Prosody',
        tooltip=['Time', 'Prosody']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(prosody_chart)

    # Chart for Pitch Accuracy
    st.subheader("Pitch Accuracy")
    pitch_accuracy_chart = alt.Chart(audiotts_metrics_data).mark_line().encode(
        x='Time',
        y='Pitch Accuracy',
        tooltip=['Time', 'Pitch Accuracy']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(pitch_accuracy_chart)

    # Chart for Emotional Expression
    st.subheader("Emotional Expression")
    emotional_expression_chart = alt.Chart(audiotts_metrics_data).mark_line().encode(
        x='Time',
        y='Emotional Expression',
        tooltip=['Time', 'Emotional Expression']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(emotional_expression_chart)

    # Chart for Speaker Similarity
    st.subheader("Speaker Similarity")
    speaker_similarity_chart = alt.Chart(audiotts_metrics_data).mark_line().encode(
        x='Time',
        y='Speaker Similarity',
        tooltip=['Time', 'Speaker Similarity']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(speaker_similarity_chart)

    # Chart for Articulation
    st.subheader("Articulation")
    articulation_chart = alt.Chart(audiotts_metrics_data).mark_line().encode(
        x='Time',
        y='Articulation',
        tooltip=['Time', 'Articulation']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(articulation_chart)

    # Chart for Duration Control
    st.subheader("Duration Control")
    duration_control_chart = alt.Chart(audiotts_metrics_data).mark_line().encode(
        x='Time',
        y='Duration Control',
        tooltip=['Time', 'Duration Control']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(duration_control_chart)

    # Chart for Robustness to Noise
    st.subheader("Robustness to Noise")
    robustness_to_noise_chart = alt.Chart(audiotts_metrics_data).mark_line().encode(
        x='Time',
        y='Robustness to Noise',
        tooltip=['Time', 'Robustness to Noise']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(robustness_to_noise_chart)

    # Chart for Latency
    st.subheader("Latency")
    latency_chart = alt.Chart(audiotts_metrics_data).mark_line().encode(
        x='Time',
        y='Latency',
        tooltip=['Time', 'Latency']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(latency_chart)

def generate_audiotts_metrics_data(num_samples=100):
    data = {
        'Time': np.arange(num_samples),
        'Naturalness': np.random.uniform(0.7, 1.0, num_samples),
        'Intelligibility': np.random.uniform(0.7, 1.0, num_samples),
        'Prosody': np.random.uniform(0.7, 1.0, num_samples),
        'Pitch Accuracy': np.random.uniform(0.7, 1.0, num_samples),
        'Emotional Expression': np.random.uniform(0.7, 1.0, num_samples),
        'Speaker Similarity': np.random.uniform(0.7, 1.0, num_samples),
        'Articulation': np.random.uniform(0.7, 1.0, num_samples),
        'Duration Control': np.random.uniform(0.7, 1.0, num_samples),
        'Robustness to Noise': np.random.uniform(0.7, 1.0, num_samples),
        'Latency': np.random.uniform(0.1, 0.5, num_samples),
    }
    return pd.DataFrame(data)

def show_relevance_over_time(data):
    st.header("Relevance over Time")
    chart = alt.Chart(data).mark_line().encode(
        x='Time',
        y='Relevance',
        tooltip=['Time', 'Relevance']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(chart)

# Add other analysis functions here

def main():
    st.sidebar.title("GPT Report Card Dashboard")

    # Add a file uploader to the sidebar
    uploaded_file = st.sidebar.file_uploader("Option 1: Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        tru_lens_dashboard = load_data(uploaded_file)
    else:
        tru_lens_dashboard = generate_data()

    option_show_summary = st.sidebar.checkbox("Option 2: Show Summary Statistics", value=True)

    st.sidebar.header("Option 3: API Keys")
    # ... (unchanged code)

    st.title("🚀 GPT Report Card 📊")

    navigation_menu = st.sidebar.radio("Navigation", ["Dashboard", "Analysis Page", "TextGPT", "AgentGPT", "VideoGPT", "CodeGPT", "Audio/TTSGPT", "ImageGPT", "Chatbot"])


    if navigation_menu == "Dashboard":
        st.markdown("### GPT Report Card Metrics\n\n🔍 Track Language Models (LLM) and agents with detailed metrics and self-improvement skills.\n\n"
                    "✨ **Key Features:**\n"
                    "1. Real-time performance metrics 📈\n"
                    "2. Personalized improvement suggestions 🌟\n"
                    "3. Historical analysis for continuous learning 🔄\n"
                    "4. User-friendly interface for easy navigation 🖥️")
        
        sample_metric = 85.5
        st.info(f"Current Accuracy: {sample_metric}%")

        show_relevance_over_time(tru_lens_dashboard)
        # Add other dashboard functions here
   
    elif navigation_menu == "Chatbot":
        show_chatbot_page()
    elif navigation_menu == "Analysis Page":
        show_analysis_page()
    elif navigation_menu == "TextGPT":
        show_textgpt_page()
    elif navigation_menu == "AgentGPT":
        show_agentgpt_page()
    elif navigation_menu == "VideoGPT":
        show_videogpt_page()
    elif navigation_menu == "CodeGPT":
        show_codegpt_page()
    elif navigation_menu == "Audio/TTSGPT":
        show_audiotts_page()
    elif navigation_menu == "ImageGPT":
        show_imagegpt_page()
                     
        

if __name__ == "__main__":
    main()
