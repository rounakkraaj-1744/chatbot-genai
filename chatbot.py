import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"

if GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY" or not GOOGLE_API_KEY:
    print("WARNING: Please replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API Key.")
    print("The chatbot might not function correctly without a valid API key.")
    if "GOOGLE_API_KEY" in os.environ:
        GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
        print("Using API key from environment variable GOOGLE_API_KEY.")
    else:
        print("No API key found. Exiting.")
        exit()

genai.configure(api_key=GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

memory = ConversationBufferMemory(memory_key="chat_history")

template = """You are a friendly and helpful AI assistant.
You are designed to answer questions and engage in natural conversations.

Current conversation:
{chat_history}
Human: {human_input}
AI:"""

prompt = PromptTemplate(input_variables=["chat_history", "human_input"], template=template)

conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

print("Chatbot initialized. Type 'exit' or 'quit' to end the conversation.")
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        response = conversation.predict(human_input=user_input)
        print(f"Chatbot: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your API key and network connection.")
        break