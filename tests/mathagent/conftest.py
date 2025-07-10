import pytest
from MathAgent.agent import MathAgent, State
from typing import TypedDict

class LLMFraud:

    class Result:
        def __init__(self, content):
            self.content = content

    def __init__(self):
        pass
    def invoke(self, prompt):
        result = self.Result(content=prompt)
        return result
@pytest.fixture(scope='module')
def create_llm_connection():
    llm = LLMFraud()
    agent = MathAgent(llm=llm)
    state = State(messages=['message'],
                  steps=['step 1'],
                  problem='Problem',
                  result='Result')

    return {"agent": agent, "state": state}