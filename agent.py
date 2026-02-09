import json
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # 启动时加载 .env，保证 SERPAPI_API_KEY / BING_API_KEY 等对 web_search 可用

# 日志级别 INFO：可在终端看到 web_search 是否被调用及 query
logging.basicConfig(level=logging.INFO, format="%(message)s")

from ag_ui.core import RunAgentInput
from agent_loop import agent_loop
from agui import stream_agui_events, to_openai_messages, to_sse_data
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from research_utils import (
    RESEARCH_SYSTEM_PROMPT,
    normalize_answer,
    web_search,
)

app = FastAPI()


def get_weather(location: str) -> str:
    """Get the weather information for a given location."""
    return f"The weather of {location} is sunny."


# Research Agent 使用的工具：联网搜索 + 天气示例
TOOLS = [web_search, get_weather]


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {"question": "法国首都在哪里？"}
        },
    )

    question: str
    chat_history: Optional[list] = None

    def to_messages(self) -> list:
        if self.chat_history:
            return self.chat_history + [{"role": "user", "content": self.question}]
        return [
            {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": self.question},
        ]


class QueryResponse(BaseModel):
    answer: str


@app.post("/")
async def query(req: QueryRequest) -> QueryResponse:
    """
    Basic LLM API example.

    Invoke example:

    ```
    curl -X POST "http://localhost:8000/" \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the weather in Beijing today?"}'

    ```

    Response example:

    ```json
    {
        "answer": "Beijing has sunny weather today, with temperatures between 10°C and 20°C."
    }
    ```


    """

    result = ""
    async for chunk in agent_loop(req.to_messages(), TOOLS):
        if chunk.type == "tool_call" or chunk.type == "tool_call_result":
            result = ""
        elif chunk.type == "text" and chunk.content:
            result += chunk.content

    return QueryResponse(answer=normalize_answer(result))


@app.post("/stream")
async def stream(req: QueryRequest) -> StreamingResponse:
    """
    Streaming query example.
    Invoke example:

    ```shell
    curl -N -X POST "http://localhost:8000/stream" \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the weather in Beijing today?"}'

    ```

    Response example:

    ```text

    data: {"answer": "Beijing has "}

    data: {"answer": "sunny weather"}

    data: {"answer": " today, with"}

    data: {"answer": " temperatures"}

    data: {"answer": " between 10°C and 20°C."}


    ```

    """

    async def stream_response():
        full_answer = ""
        async for chunk in agent_loop(req.to_messages(), TOOLS):
            if chunk.type == "tool_call" or chunk.type == "tool_call_result":
                full_answer = ""
            elif chunk.type == "text" and chunk.content:
                full_answer += chunk.content
                yield f"data: {json.dumps({'answer': chunk.content}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'answer': normalize_answer(full_answer)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
    )


@app.post("/ag-ui")
async def ag_ui(run_agent_input: RunAgentInput) -> StreamingResponse:
    """
    AG-UI Protocol endpoint for streaming LLM interactions.

    AG-UI Protocol: https://docs.ag-ui.com/introduction
    """

    messages = to_openai_messages(run_agent_input.messages)

    async def stream_response():
        async for event in stream_agui_events(
            chunks=agent_loop(messages, TOOLS), run_agent_input=run_agent_input
        ):
            yield to_sse_data(event)

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
    )
