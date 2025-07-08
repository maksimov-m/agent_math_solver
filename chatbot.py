#########################LangChain Code##########################
from MathAgent.agent import MathAgent

mathagnet = MathAgent()

#########################Gradio Code##########################
import gradio as gr

# TODO:  добавить историю в диалог
def solve_problem(message, history):
    state = mathagnet.graph.invoke({"messages": message})

    answer = state['messages'][-1].content

    return answer

demo = gr.ChatInterface(fn=solve_problem, type="messages", title="MathAgent")
demo.launch()

