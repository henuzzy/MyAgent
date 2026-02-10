import json
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # 启动时加载 .env，保证 SERPAPI_API_KEY / BING_API_KEY 等对 web_search 可用

# 日志级别 INFO：可在终端看到 web_search 是否被调用及 query
logging.basicConfig(level=logging.INFO, format="%(message)s")

from ag_ui.core import RunAgentInput
from agui import stream_agui_events, to_openai_messages, to_sse_data
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from plan_solve_agent import PlanSolveResearchAgent

app = FastAPI()


_agent = PlanSolveResearchAgent()


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {"question": "法国首都在哪里？"}
        },
    )

    question: str
    chat_history: Optional[list] = None


class QueryResponse(BaseModel):
    answer: str


@app.post("/")
async def query(req: QueryRequest, response: Response) -> QueryResponse:
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

    result = await _agent.run(req.question)
    # Traceability: expose trace_id via header (body still only contains answer for evaluation)
    response.headers["X-Trace-Id"] = result.trace_id
    return QueryResponse(answer=result.answer)


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
        # 为兼容 SSE：先完整跑完 Plan->Search->Verify->Solve，再把最终答案以 SSE 输出
        result = await _agent.run(req.question)
        yield f"data: {json.dumps({'answer': result.answer}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
    )


@app.get("/trace/{trace_id}")
async def get_trace(trace_id: str):
    """
    开发调试用：获取一次请求的 trace（语言选择、证据、约束日志等）。
    线上评测不会调用该接口。
    """
    # Trace 存储：写入 .output/traces/<trace_id>.json
    import pathlib

    trace_dir = pathlib.Path(".output") / "traces"
    path = trace_dir / f"{trace_id}.json"
    if not path.exists():
        return {"error": "trace_not_found", "trace_id": trace_id}
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/ag-ui")
async def ag_ui(run_agent_input: RunAgentInput) -> StreamingResponse:
    """
    AG-UI Protocol endpoint for streaming LLM interactions.

    AG-UI Protocol: https://docs.ag-ui.com/introduction
    """

    async def stream_response():
        # 兼容保留：AG-UI 仍按“文本流”输出最终答案（不暴露中间推理）
        msgs = to_openai_messages(run_agent_input.messages)
        question = ""
        for m in reversed(msgs):
            if m.get("role") == "user" and m.get("content"):
                question = m["content"]
                break
        result = await _agent.run(question or "")
        # 复用 AG-UI 的事件流格式：直接输出最终答案文本
        async for event in stream_agui_events(
            chunks=_single_text_chunk_stream(result.answer), run_agent_input=run_agent_input
        ):
            yield to_sse_data(event)

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
    )


async def _single_text_chunk_stream(text: str):
    """
    Provide a minimal async iterator compatible with agui.stream_agui_events.
    """
    from agent_loop import Chunk

    yield Chunk(step_index=0, type="text", content=text)
