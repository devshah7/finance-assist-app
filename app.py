import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import tempfile
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from openai import OpenAI
import json

# Load environment variables
load_dotenv()

# Set the OpenAI API key for the environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# File upload section
st.title("Finanssist App")

# Remove the default file check and enforce file upload
uploaded_file = st.file_uploader("Upload your transactions CSV file", type="csv")

# Use session state for persistent chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "system", 
        "content": "You are an expert personal financial assistant who helps users analyze their financial transactions. The csv file is already uploaded you just call the csv_toolcall tool to analyze the file. You can answer questions about the transactions, provide insights, provide tabular formats and suggest improvements."}]

if uploaded_file:
    try:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file_path = temp_file.name
            data.to_csv(temp_file_path, index=False)

        # Setup LangChain CSV agent
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        csv_tool = create_csv_agent(
            llm,
            temp_file_path,
            agent_type="openai-tools",
            allow_dangerous_code=True,
            verbose=False
        )

        def csv_toolcall(csv_query):
            res = csv_tool.invoke({"input": csv_query})
            return res['output']

        tools = [{
            "type": "function",
            "function": {
                "name": "csv_toolcall",
                "description": "Get the financial transaction data insights from the CSV file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "csv_query": {
                            "type": "string",
                            "description": "Query to answer about the transactions data."
                        },
                    },
                    "required": ["csv_query"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }]        

        client = OpenAI()
        input_query = st.text_input("Enter your question about the transactions data:")

        if input_query:
            st.session_state["messages"].append({"role": "user", "content": input_query})
            completion = client.chat.completions.create(
                model="gpt-4.1",
                messages=st.session_state["messages"],
                tools=tools,
            )
            system_message = completion.choices[0].message

            # Handle tool call if present
            if system_message.tool_calls:
                tool_call = system_message.tool_calls[0]
                args = json.loads(tool_call.function.arguments)
                result = csv_toolcall(args["csv_query"])
                st.session_state["messages"].append(system_message)
                st.session_state["messages"].append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
                # Re-run the conversation with the tool result
                completion = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=st.session_state["messages"],
                    tools=tools,
                )
                system_message = completion.choices[0].message

            # Display and store the assistant's reply
            if system_message.content:
                st.write(system_message.content)
                st.session_state["messages"].append({"role": "assistant", "content": system_message.content})
    except Exception as e:
        st.error(f"An error occurred: {e}")