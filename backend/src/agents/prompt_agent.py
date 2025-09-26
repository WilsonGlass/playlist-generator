import yaml

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from src.agents.schemas import RewriteResult


class PromptAgent:
    def __init__(self):
        with open("src/agents/prompts.yaml", "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
        self.system_prompt = prompts["system_prompt"]
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        self.model = "llama3.1:8b"

    def rewrite_prompt(self, user_prompt: str) -> RewriteResult:
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=0.2,
            messages=[
                ChatCompletionSystemMessageParam(
                    role="system", content=self.system_prompt
                ),
                ChatCompletionUserMessageParam(
                    role="user", content=user_prompt
                ),
            ],
            response_format=RewriteResult,
        )

        message = completion.choices[0].message
        if message.parsed:
            return message.parsed
        elif message.refusal:
            raise RuntimeError(f"Model refused: {message.refusal}")
        else:
            raise RuntimeError("Unexpected model response")


if __name__ == "__main__":
    agent = PromptAgent()
    for p in ["songs my mom would listen to", "1990s techno with fast bpm"]:
        result = agent.rewrite_prompt(p)
        print("\n=== Test Prompt ===")
        print("Original:", result.original_prompt)
        print("Final:   ", result.final_prompt)
        print("Rewritten?", result.was_rewritten)
