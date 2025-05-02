import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import tempfile
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_csv_agent

# Load environment variables
load_dotenv()

# Set the OpenAI API key for the environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# File upload section
st.title("Finanssist App")

# Remove the default file check and enforce file upload
uploaded_file = st.file_uploader("Upload your transactions CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)

        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file_path = temp_file.name
            data.to_csv(temp_file_path, index=False)

        # Initialize LangChain ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

        # Create a CSV agent
        agent_executor = create_csv_agent(
            llm,
            temp_file_path,
            agent_type="openai-tools",
            allow_dangerous_code=True,
            verbose=True
        )

        # User query input
        query = st.text_input("Enter your question about the transactions data:", key="chat_input")

        # Keep the chat open and allow multiple interactions
        if query:
            try:
                # Get answer from the Chain
                res = agent_executor.invoke({"input": query})
                st.write("Response:", res['output'])
            except Exception as e:
                st.error(f"An error occurred while processing your query: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")