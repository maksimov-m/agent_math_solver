#########################LangChain Code##########################
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import time

model = OllamaLLM(model="llama3.1")



#########################Gradio Code##########################
import gradio as gr

def echo(message, history):
    answer = model.invoke(message)

    return answer

demo = gr.ChatInterface(fn=echo, type="messages", examples=["hello", "hola", "merhaba"], title="Echo Bot")
demo.launch()

