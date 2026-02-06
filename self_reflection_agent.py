import json
from typing import List, Dict, Literal
from openai import OpenAI
from os import getenv
from dotenv import load_dotenv

load_dotenv()

api_key = getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")


class Memory:
    def __init__(self):
        self._messages: List[Dict[str, str]] = []

    def add_message(self, role: Literal['user', 'system', 'assistant'], content: str):
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        return self._messages

    def last_message(self):
        return self._messages[-1] if self._messages else None


# --- Self-Critique Prompt ---
REFLECTION_PROMPT = """
Reflect on your previous response...
Identify any mistakes, areas for improvement, or ways to clarify the answer, 
making it more concise and easy to understand. 
Provide a revised response if necessary in a Json Output structure:
{
    "original_response": "",
    "revisions_needed": "",
    "updated_response": ""
}
"""


class Agent:
    """A self-reflection AI Agent"""

    def __init__(self, role="Personal Assistant", instructions="Help users with any question"):
        self.llm = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.memory = Memory()
        self.memory.add_message("system", f"You're an AI Agent. Role: {role}. {instructions}")
        self.evaluator = REFLECTION_PROMPT

    def invoke(self,
               user_message: str,
               self_reflection: bool = False,
               max_iter: int = 1,
               show_logs: bool = False) -> str:

        # Clamp iterations between 1-3
        max_iter = max(1, min(3, max_iter))

        # If no reflection, just do 1 pass (0.5 * 2 = 1 loop)
        max_iter = max_iter if self_reflection else 0.5
        loops = int(2 * max_iter)

        # Add user question to memory
        self.memory.add_message("user", user_message)
        if show_logs:
            self._log("user", user_message)

        for i in range(loops):
            # Step 1: Get AI response
            response = self._get_completion()
            self.memory.add_message("assistant", response)
            if show_logs:
                self._log("assistant", response)

            # Step 2: If not last loop, ask for self-critique
            if i < loops - 1:
                self.memory.add_message("user", self.evaluator)
                if show_logs:
                    self._log("evaluator", "Requesting self-reflection...")

        return self.memory.last_message()["content"]

    def _get_completion(self) -> str:
        response = self.llm.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=self.memory.get_messages()
        )
        return response.choices[0].message.content

    def _log(self, role, content):
        print(f"\n### {role.upper()} ###")
        print(content)
        print("_" * 50)


# --- Run it ---
agent = Agent()
result = agent.invoke(
    user_message="What is an LLM? Explain in 2 sentences.",
    self_reflection=True,
    max_iter=2,
    show_logs=True
)

# Parse the final refined response
final = json.loads(result)
print("\n✅ Final Answer:", final["updated_response"])
