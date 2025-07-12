#########################LangChain Code##########################
from MathAgent.agent import MathAgent

mathagnet = MathAgent()

#########################Gradio Code##########################
import gradio as gr
import logging

logger = logging.getLogger(__name__)
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)
logger.setLevel(logging.DEBUG)


# TODO:  добавить историю в диалог
def solve_problem(message, history):
    logger.info("Agent workflow start...")

    state = mathagnet.graph.invoke({"messages": message})

    logger.info("MathAgent inference success")

    answer = state['messages'][-1].content

    return answer


demo = gr.ChatInterface(fn=solve_problem, type="messages", title="MathAgent")
demo.launch(server_port=8888)
