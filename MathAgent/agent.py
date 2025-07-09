from MathAgent.math_tools import binary_to_decimal, general_response, calculator, decimal_to_binary
from MathAgent.prompts import prompt_create_steps, prompt_reformulation, prompt_description_tools, prompt_role

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain.chains import LLMChain

from langgraph.graph import StateGraph, START, END

from typing import Annotated
from typing_extensions import TypedDict


#TODO: вынести промпты в отдельный файл
#TODO: сделать нормальное отображение решения, отправлять пользователю сообщение одновременно с решением
#TODO: добавить комментарии и типизацию данных

class State(TypedDict):
    messages: Annotated[list, add_messages]
    steps: list
    problem: str
    result: str


class MathAgent:
    def __init__(self):
        #TODO: вынести данные в конфиг
        self.llm = ChatOllama(model="llama3.1", temperature=0.1)

        graph_builder = StateGraph(State)

        graph_builder.add_node("create_steps", self.create_steps)
        graph_builder.add_node("solver", self.solver)
        graph_builder.add_node("generate_answer", self.generate_answer)

        graph_builder.add_edge(START, "create_steps")
        graph_builder.add_edge("create_steps", "solver")
        graph_builder.add_edge("solver", "generate_answer")
        graph_builder.add_edge("generate_answer", END)

        self.graph = graph_builder.compile()

    def create_steps(self, state: State):
        response_schemas = [
            ResponseSchema(name="steps", type="List[Dict]", description="Список шагов решения с объяснениями"),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        # Шаблон промпта с инструкциями
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_create_steps,
            input_variables=["problem"],
            partial_variables={"format_instructions": format_instructions}
        )

        chain = LLMChain(llm=self.llm, prompt=prompt, output_parser=output_parser)

        result = chain.invoke({"problem": state['messages'][-1]})

        print("ПОЛУЧЕННЫЕ ШАГИ:", result['text']['steps'])
        return {"steps": result['text']['steps'], "problem": state['messages'][-1]}

    def reformulation(self, solve_step: str, curr_task: str, problem: str) -> str:

        prompt = prompt_reformulation.format(**{"problem": problem,
                                                "solve_step": solve_step,
                                                "curr_task": curr_task})

        result = self.llm.invoke(prompt)

        return result.content

    def solve_step(self, prompt, solve_steps, i):

        new_prompt = prompt_description_tools + prompt
        print("Текущая задача:", new_prompt)
        llm_with_tool = self.llm.bind_tools(tools=[calculator, binary_to_decimal, decimal_to_binary, general_response],
                                            tool_choice='any')
        result = llm_with_tool.invoke(prompt)

        print()
        print("Используемые инструменты:", result.tool_calls)

        for tool_call in result.tool_calls:
            try:
                selected_tool = {"calculator": calculator, "binary_to_decimal": binary_to_decimal,
                                 "decimal_to_binary": decimal_to_binary, "general_response": general_response}[
                    tool_call["name"].lower()]

                if tool_call["name"].lower() == 'calculator':
                    tool_call['args']["expression"] = tool_call['args']["expression"].replace("π", "3.14").replace(
                        "math.pi", "3.14").replace("math.", "").replace("^", "**").replace("pi", "3.14")
                elif tool_call["name"].lower() == 'general_response':
                    solve_steps += f"Шаг {i}: " + str(prompt) + "\n\n"
                    continue

                tool_msg = selected_tool.invoke(tool_call['args'])
                solve_steps += f"Шаг {i}: " + f"Функция: {tool_call['name']}, аргументы:{tool_call['args']}" + "\nОтвет: " + str(
                    tool_msg) + "\n\n"
            except Exception as ex:
                print("Error -", ex)
                solve_steps += f"Шаг {i}: " + str(prompt) + "\n\n"

        return solve_steps

    def solver(self, state):
        steps = state['steps']
        problem = state['problem']

        solve_steps = ""
        i = 1
        for step in steps:
            print("-----------------------")
            prompt = str(step)

            if len(solve_steps) > 0:
                prompt = self.reformulation(solve_steps, str(step), problem)

            solve_steps = self.solve_step(prompt, solve_steps, i)

            print()
            print("История:", solve_steps)
            i += 1
        return {"messages": [solve_steps]}

    def generate_answer(self, state):

        prompt = prompt_role.format(**{"problem": state['problem']})

        result = self.llm.invoke([prompt] + state['messages'])

        print("RESULT:", result.content)
        return {"messages": [result.content]}
