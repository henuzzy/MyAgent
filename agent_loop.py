import inspect
import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)
from inspect import iscoroutinefunction
from typing import (
    Any,
    AsyncIterator,
    Callable,
    List,
    Literal,
    Optional,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from skills import (
    SkillIntegrationTools,
    SkillMetadata,
    build_skills_system_prompt,
    discover_skills,
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class ToolCall:
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict] = None


@dataclass
class Chunk:
    step_index: int
    type: Literal["text", "tool_call", "tool_call_result"]
    content: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[Any] = None


def python_type_to_json_type(t):
    """Map Python types to JSON types."""
    if t is str:
        return "string"
    elif t is int:
        return "integer"
    elif t is float:
        return "number"
    elif t is bool:
        return "boolean"
    elif t is list or get_origin(t) is list:
        return "array"
    elif t is dict or get_origin(t) is dict:
        return "object"
    return "string"


def function_to_schema(func: Callable) -> dict:
    """
    Convert a Python function to an OpenAI API Tool Schema.
    """
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)

    parameters = {"type": "object", "properties": {}, "required": []}

    for name, param in signature.parameters.items():
        if name in ("self", "cls"):
            continue

        annotation = type_hints.get(name, str)
        param_type = python_type_to_json_type(annotation)

        param_info = {"type": param_type}

        if get_origin(annotation) == Literal:
            param_info["enum"] = list(get_args(annotation))
            param_info["type"] = python_type_to_json_type(type(get_args(annotation)[0]))

        parameters["properties"][name] = param_info
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": parameters,
        },
    }


async def agent_loop(
    input_messages: list,
    tool_functions: List[Callable],
    skill_directories: Optional[List[str]] = ["skills"],
) -> AsyncIterator[Chunk]:
    """
    Main agent loop with skills support.

    Args:
        input_messages: List of chat messages
        tool_functions: List of tool functions available to the agent
        skill_directories: Optional list of directories to scan for skills.
                          Skills are folders containing SKILL.md files.
    """

    assert os.getenv("DASHSCOPE_API_KEY"), "DASHSCOPE_API_KEY is not set"

    # 模型：通过环境变量 DASHSCOPE_MODEL 指定，默认 qwen-plus。可选 qwen-turbo、qwen-plus、qwen-max 等。
    model_name = os.getenv("DASHSCOPE_MODEL", "qwen3-max-preview")

    client = AsyncOpenAI(
        # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        base_url = "https://apis.iflow.cn/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    # Discover and load skills metadata
    skills: List[SkillMetadata] = (
        discover_skills(skill_directories) if skill_directories else []
    )

    # Build skills system prompt if skills are available
    skills_prompt = build_skills_system_prompt(skills)

    # Prepare messages with skills context injected
    prompt_messages = input_messages.copy()
    if skills_prompt and prompt_messages:
        # Inject skills into system message if present, otherwise prepend
        if prompt_messages[0].get("role") == "system":
            original_content = prompt_messages[0].get("content", "")
            prompt_messages[0] = {
                "role": "system",
                "content": f"{original_content}\n\n{skills_prompt}",
            }
        else:
            prompt_messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"{DEFAULT_SYSTEM_PROMPT}\n\n{skills_prompt}",
                },
            )

    # Build a mapping from function name to function for quick lookup

    llm_tools = (tool_functions or []).copy()

    if skills:
        skill_tools = SkillIntegrationTools(skills)
        llm_tools.extend([skill_tools.load_skill_file, skill_tools.execute_script])

    step_index = 0
    tool_schema = [function_to_schema(tool_function) for tool_function in llm_tools]

    tool_functions_map = {func.__name__: func for func in llm_tools}

    # 仅使用基础对话+工具调用，不传 enable_search / extra_body，即未使用百炼自带联网搜索（赛题禁止）。

    # Main Agent Loop: continues as long as the model requests tool executions
    while True:
        params = {
            "model": model_name,
            "stream": True,
            "tools": tool_schema,
        }
        # 第一步强制调用 web_search，确保每次请求至少进行一次联网搜索
        if step_index == 0:
            params["tool_choice"] = {"type": "function", "function": {"name": "web_search"}}

        stream = await client.chat.completions.create(
            messages=prompt_messages, **params
        )

        tool_calls_buffer = {}

        # Process the stream
        async for chunk in stream:  # type: ChatCompletionChunk
            chunk = cast(ChatCompletionChunk, chunk)

            delta = chunk.choices[0].delta

            # Case A: Standard text content
            if delta.content:
                yield Chunk(type="text", content=delta.content, step_index=step_index)

            # Case B: Tool call fragments (accumulate them)
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    idx = tc_chunk.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": tc_chunk.id,
                            "function": {
                                "name": tc_chunk.function.name,
                                "arguments": "",
                            },
                        }
                    # Append tool arguments fragment
                    if tc_chunk.function.arguments:
                        tool_calls_buffer[idx]["function"]["arguments"] += (
                            tc_chunk.function.arguments
                        )
        if not tool_calls_buffer:
            break

        assistant_tool_calls_data = []
        sorted_indices = sorted(tool_calls_buffer.keys())

        for idx in sorted_indices:
            raw_tool = tool_calls_buffer[idx]
            assistant_tool_calls_data.append(
                {
                    "id": raw_tool["id"],
                    "type": "function",
                    "function": {
                        "name": raw_tool["function"]["name"],
                        "arguments": raw_tool["function"]["arguments"],
                    },
                }
            )

        # Append the assistant's tool call request to history
        prompt_messages.append(
            {
                "role": "assistant",
                "tool_calls": assistant_tool_calls_data,
            }
        )

        # Execute tools and yield results
        for tool_data in assistant_tool_calls_data:
            call_id = tool_data["id"]
            func_name = tool_data["function"]["name"]
            func_args_str = tool_data["function"]["arguments"]

            tool_result_content = ""
            parsed_args = {}
            tool_call = ToolCall(
                tool_call_id=call_id,
                tool_name=func_name,
                tool_arguments={},
            )

            try:
                # Parse JSON arguments
                parsed_args = json.loads(func_args_str)
                tool_call.tool_arguments = parsed_args

                # Notify caller that we are about to execute a tool
                yield Chunk(
                    step_index=step_index,
                    type="tool_call",
                    tool_call=tool_call,
                )

                # Execute the function if it exists
                if func_name in tool_functions_map:
                    if func_name == "web_search":
                        logger.info(
                            "[Research Agent] web_search 被调用: query=%s",
                            parsed_args.get("query", ""),
                        )
                    func = tool_functions_map[func_name]
                    # Note: If tools are async, use await
                    if iscoroutinefunction(func):
                        result = await func(**parsed_args)
                    else:
                        result = func(**parsed_args)
                    tool_result_content = str(result)
                else:
                    tool_result_content = f"Error: Tool '{func_name}' not found."

            except json.JSONDecodeError as e:
                tool_result_content = f"Error: Failed to parse tool arguments JSON: {func_args_str}. Error: {e}"
                yield Chunk(
                    step_index=step_index,
                    type="tool_call",
                    tool_call=tool_call,
                )
            except Exception as e:
                tool_result_content = f"Error: Execution failed - {str(e)}"

            yield Chunk(
                type="tool_call_result",
                tool_result=tool_result_content,
                step_index=step_index,
                tool_call=tool_call,
            )

            prompt_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_result_content,
                }
            )
        step_index += 1
