#########################LangChain Code##########################
from langchain_core.prompts import ChatPromptTemplate
from MathAgent.agent import MathAgent
from langchain_ollama.llms import OllamaLLM
import time

model = OllamaLLM(model="llama3.1")

mathagnet = MathAgent()

#########################Gradio Code##########################
import gradio as gr

def echo(message, history):
    state = mathagnet.graph.invoke({"messages": message})

    answer = state['messages'][-1].content

    return answer

demo = gr.ChatInterface(fn=echo, type="messages", examples=["hello", "hola", "merhaba"], title="Echo Bot")
demo.launch()

