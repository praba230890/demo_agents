from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


class AgentFlowLogger(BaseCallbackHandler):
    """Console logger for agent execution flow."""

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def _short(value: Any, max_len: int = 220) -> str:
        text = str(value).replace("\n", " ").strip()
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _log(self, event: str, detail: str) -> None:
        print(f"[{self._ts()}] [{event}] {detail}")

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:
        if messages:
            last = messages[-1]
            self._log("LLM_START", f"last_message={self._short(last)}")
        else:
            self._log("LLM_START", "messages=[]")

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        if prompts:
            self._log("LLM_START", f"prompt={self._short(prompts[-1])}")
        else:
            self._log("LLM_START", "prompts=[]")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            generations = getattr(response, "generations", [])
            first = generations[0][0] if generations and generations[0] else None
            if first is not None:
                text = getattr(first, "text", None)
                if text is None:
                    msg = getattr(first, "message", None)
                    text = getattr(msg, "content", first)
                self._log("LLM_END", f"output={self._short(text)}")
                return
        except Exception:
            pass

        self._log("LLM_END", "output=<unavailable>")

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        tool = getattr(action, "tool", "<unknown>")
        tool_input = getattr(action, "tool_input", "")
        self._log("AGENT_DECISION", f"tool={tool} input={self._short(tool_input)}")

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        name = serialized.get("name", "<tool>") if isinstance(serialized, dict) else "<tool>"
        self._log("TOOL_START", f"tool={name} input={self._short(input_str)}")

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self._log("TOOL_END", f"output={self._short(output)}")

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        values = getattr(finish, "return_values", {})
        self._log("AGENT_FINISH", f"return_values={self._short(values)}")


@tool
def calculate(expression: Annotated[str, "A valid python math expression."]) -> str:
    """Evaluate a safe math expression, e.g. '(24 * 7) + sqrt(81)'."""
    allowed_names = {
        "abs": abs,
        "round": round,
        "pow": pow,
        "sqrt": math.sqrt,
        "ceil": math.ceil,
        "floor": math.floor,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        value = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
    except Exception as exc:
        return f"Calculation error: {exc}"

    return str(value)


@tool
def word_count(text: Annotated[str, "Any user text."]) -> str:
    """Return how many words are in the provided text."""
    return str(len(text.split()))


def build_agent() -> Any:
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    tools = [calculate, word_count]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a practical assistant. Use tools when they improve accuracy. "
            "If no tool is needed, answer directly."
        ),
    )


def run_chat(agent: Any, logger: AgentFlowLogger) -> None:
    chat_history: list[dict[str, str]] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        result = agent.invoke(
            {"messages": chat_history + [{"role": "user", "content": user_input}]},
            config={"callbacks": [logger]},
        )
        last_message = result["messages"][-1]
        answer = getattr(last_message, "content", "")
        if not isinstance(answer, str):
            answer = str(answer)

        print(f"Agent: {answer}\\n")
        chat_history.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer},
            ]
        )


def main() -> None:
    agent = build_agent()
    logger = AgentFlowLogger()

    print("Single-agent demo (local Ollama + llama3.1:8b). Type 'exit' to quit.\\n")
    print("Flow logs are enabled: LLM calls, tool decisions, and tool execution.\\n")

    run_chat(agent, logger)


if __name__ == "__main__":
    main()
