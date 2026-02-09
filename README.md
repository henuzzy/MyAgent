# 基于 LLM 大语言模型 Agent 的示例

这是一个基于 LLM 的 Agent 示例，演示了如何借助 LLM Function Calling 实现 Agent Loop，并通过集成 [Agent Skills](https://agentskills.io) 规范，实现一个支持工具调用的可扩展 Agent。

## 如何启动项目

1. **安装依赖**（在项目根目录执行）：
   ```bash
   pip install -r requirements.txt
   ```

2. **配置环境变量**：在项目根目录创建 `.env` 文件，至少填写百炼 API Key；若需联网搜索，可再填 SerpAPI 或 Bing 的 Key：
   ```text
   DASHSCOPE_API_KEY=sk-你的百炼Key
   SERPAPI_API_KEY=你的SerpAPI_Key   # 可选，用于 Research Agent 联网搜索
   # DASHSCOPE_MODEL=qwen-plus      # 可选，不写则默认 qwen-plus
   ```

3. **启动服务**：
   ```bash
   uvicorn agent:app --reload --host 0.0.0.0 --port 8000
   ```

4. 浏览器访问 **http://localhost:8000/docs** 可查看并调试 API；或使用：
   ```bash
   curl -X POST "http://localhost:8000/" -H "Content-Type: application/json" -d "{\"question\": \"法国首都在哪里？\"}"
   ```

更详细的本地运行说明见下方 [本地运行](#本地运行) 小节。

## 代码结构

项目结构如下：

```plain
project/
├── agent.py          # 入口：FastAPI app，Research Agent 接口与答案归一化
├── agent_loop.py     # Agent Loop：LLM 与工具执行，模型由 DASHSCOPE_MODEL 指定
├── agui.py           # AG-UI Protocol 事件流转换
├── research_utils.py  # Research Agent：答案归一化、系统提示、联网搜索 web_search
├── skills.py         # Agent Skills 集成
├── skills/           # 技能目录
│   ├── get-current-time/
│   │   ├── SKILL.md
│   │   └── scripts/
│   │       └── now.py
│   └── create_plan/
│       └── SKILL.md
└── README.md
```

### agent.py 规范

`agent.py` 是默认的入口文件，**必须**提供一个全局的 `FastAPI` 对象，命名为 `app`，例如以下：

```python
from fastapi import FastAPI
from agent_loop import agent_loop

app = FastAPI()

@app.post("/")
async def query(req: QueryRequest) -> QueryResponse:
    messages = [{"role": "user", "content": req.question}]
    result = ""
    async for chunk in agent_loop(messages, []):
        if chunk.type == "text" and chunk.content:
            result += chunk.content
    return QueryResponse(answer=result)
```

> **注意**：平台会自动识别并加载 `agent.py` 中的 `app` 对象来启动服务。

### agent_loop.py 核心逻辑

`agent_loop.py` 提供了核心的 Agent Loop 逻辑，支持：

- 流式 LLM 调用
- 工具调用（Tool Calling）和执行
- 自动将工具结果回传给 LLM 继续对话
- Agent Skills 集成

## Agent Skills 技能系统

本示例集成了 [Agent Skills](https://agentskills.io) 规范，通过技能扩展 Agent 的能力。技能是包含 `SKILL.md` 文件的文件夹，Agent 会自动发现并在合适的时机激活相应技能。

### 内置示例技能

| 技能名称 | 描述 |
|----------|------|
| `get-current-time` | 获取当前系统时间 |
| `create-plan` | 创建可执行的计划 |

### 创建新技能

在 `skills/` 目录下创建技能文件夹，包含 `SKILL.md` 文件：

```yaml
---
name: my-skill
description: 技能描述，说明何时使用该技能。
---

# 技能标题

技能指令内容...
```

更多关于 Agent Skills 规范，请参阅 [agentskills.io](https://agentskills.io)。

## 环境配置

本示例使用阿里云百炼模型服务，请通过环境变量配置相应的 API Key（`DASHSCOPE_API_KEY`）。你可以通过`.env`文件，或是在代码中配置环境变量。

### 如何确认使用的是哪个模型、是否用了百炼自带搜索

- **当前使用的模型**：在 `agent_loop.py` 中，模型名由环境变量 **`DASHSCOPE_MODEL`** 决定；未设置时默认为 **`qwen-plus`**。在项目根目录的 `.env` 里可写：
  ```bash
  DASHSCOPE_MODEL=qwen-plus
  ```
  或改为 `qwen-turbo`、`qwen-max` 等百炼支持的模型名。在代码里也可直接搜索 `model_name` 或 `DASHSCOPE_MODEL` 确认实际取值。

- **是否使用了百炼自带的联网搜索/网页抓取**：本仓库**没有**使用。百炼若开启自带搜索，需要在请求里传 `enable_search: true` 或 `extra_body={"enable_search": True}`（见[阿里云文档](https://www.alibabacloud.com/help/zh/model-studio/web-search)）。本项目中调用 `client.chat.completions.create` 时**只传了 `model`、`messages`、`stream`、`tools`**，没有传 `enable_search` 或 `extra_body`，因此是纯基础模型 + 自建工具（如 `web_search`），符合赛题“禁止使用百炼自带搜索”的要求。你可在 `agent_loop.py` 中搜索 `chat.completions.create` 和 `params` 自行核对。

- **如何确认是否进行了联网搜索**：`web_search` 作为工具提供给模型，**是否调用由模型决定**。启动服务后，每次模型调用 `web_search` 时，终端会输出 `[Research Agent] web_search 被调用: query=xxx`。若没有该输出，说明模型未调用搜索，直接凭记忆回答了。可尝试更换更强模型（如 `DASHSCOPE_MODEL=qwen-max`）或优化系统提示，提高模型对搜索的依赖。

## 本地运行

在本地运行前需要安装依赖并配置 API Key，按以下步骤操作即可。

### 1. 安装依赖

建议使用虚拟环境（可选）：

```bash
# 创建虚拟环境（可选）
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 安装项目依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

本示例使用阿里云百炼（DashScope）模型，需要先设置环境变量 `DASHSCOPE_API_KEY`：

- **方式一**：在项目根目录创建 `.env` 文件（不要提交到 Git），内容示例：
  ```text
  DASHSCOPE_API_KEY=sk-xxxxxxxx
  ```
  若使用 `python-dotenv`，可在 `agent.py` 开头加载：`load_dotenv()`。

- **方式二**：在终端中临时设置后启动：
  ```bash
  # Windows PowerShell
  $env:DASHSCOPE_API_KEY="sk-xxxxxxxx"

  # Windows CMD
  set DASHSCOPE_API_KEY=sk-xxxxxxxx

  # macOS/Linux
  export DASHSCOPE_API_KEY=sk-xxxxxxxx
  ```

API Key 可在 [阿里云百炼控制台](https://bailian.console.aliyun.com/) 获取。

### 3. 启动服务

在项目根目录执行：

```bash
uvicorn agent:app --reload --host 0.0.0.0 --port 8000
```

浏览器访问 `` 可打开 FastAPI 自带的 API 文档并调试接口。

### 4. 快速测试

```bash
curl -X POST "http://localhost:8000/" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"北京今天天气怎么样？\"}"
```

## 开发调试

### 基于 OpenAPI 的调试开发

通过平台的调试页面，可以直接使用基于 OpenAPI 规范的交互式调试界面进行开发测试：

1. 点击运行，可以进入API调试页面。
2. 可以直接在页面上填写请求参数并发送请求
3. 实时查看响应结果，以及链路追踪信息

### 对话调试

支持通过AG-UI Protocol的对话面板进行调试，支持通过对话面板查看请求的详细信息。对话面板的请求会使用AG-UI Protocol，请求服务的`/ag-ui`接口。

### 链路追踪（Trace）

平台会自动注入 instrument 来收集链路追踪数据，当通过对话面板，或是OpenAPI调试时，支持通过查看链路查看请求的详细信息。

Trace 数据包含完整的请求链路信息，包括请求耗时、LLM 调用详情、Token 使用量、错误信息（如有）等。

## 服务部署调试

完成调试后，点击服务部署，可以将应用部署到PAI-EAS，服务可以通过以下的方式调用。

> **注意**：`endpoint` 和 `token` 需要在部署服务页面查看获取。

### 普通请求接口

**Endpoint**: `POST /`

**请求体**:

```shell

curl -X POST "https://<your-endpoint>/" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{"question": "北京今天的天气如何?"}'
```

**响应体**:

```json
{
    "answer": "北京今天的天气晴朗，气温在10℃到20℃之间，空气质量良好。"
}
```

### 流式请求接口（SSE）

**Endpoint**: `POST /stream`

**请求体**:

```shell
curl -N -X POST "https://<your-endpoint>/stream" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{"question": "北京今天的天气如何?"}'
```

**响应格式** (Server-Sent Events):

```text
data: {"answer":"我"}

data: {"answer":"目前无法直接"}

data: {"answer":"获取实时天气信息。"}
```

### AG-UI Protocol 接口

**Endpoint**: `POST /ag-ui`

**请求体**:

```shell
curl -N -X POST "https://<your-endpoint>/ag-ui" \
  --header "Content-Type: application/json" \
  --data '{
    "messages": [
      {
        "content": "你好",
        "id": "msg_1",
        "role": "user"
      }
    ],
    "runId": "run_456",
    "threadId": "thread_123",
    "context": [],
    "tools": [],
    "forwardedProps": {},
    "state": null
  }'

```

**响应格式** (Server-Sent Events):

```text
data: {"type": "RUN_STARTED", "threadId": "thread_123", "runId": "run_456", "input": {"threadId": "thread_123", "runId": "run_456", "messages": [{"id": "27248f27-ae30-4742-b1fe-930a362a384c", "role": "user", "content": "你好"}], "tools": [], "context": [], "forwardedProps": {}}}

data: {"type": "TEXT_MESSAGE_START", "messageId": "ae5c8d44-e20a-43de-89f4-3da64e02b95a", "role": "assistant"}

data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "ae5c8d44-e20a-43de-89f4-3da64e02b95a", "delta": "你好"}

data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "ae5c8d44-e20a-43de-89f4-3da64e02b95a", "delta": "！有什么"}

data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "ae5c8d44-e20a-43de-89f4-3da64e02b95a", "delta": "我可以帮助你的"}

data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "ae5c8d44-e20a-43de-89f4-3da64e02b95a", "delta": "吗？"}

data: {"type": "TEXT_MESSAGE_END", "messageId": "ae5c8d44-e20a-43de-89f4-3da64e02b95a"}

data: {"type": "RUN_FINISHED", "threadId": "thread_123", "runId": "run_456"}

```
http://localhost:8000/docs