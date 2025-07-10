import pytest

from MathAgent.agent import MathAgent, State
from conftest import create_llm_connection


def test_build_agent(create_llm_connection):
    agent = MathAgent(llm=create_llm_connection['agent'].llm)




