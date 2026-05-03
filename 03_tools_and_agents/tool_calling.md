# Tool Calling

> `@tool`, `ToolNode`, `bind_tools`, parallel tool calls, error handling

---

### Q1. How do you define a tool that LangGraph can call?

**Answer:**

Use the `@tool` decorator from `langchain_core.tools`. The function signature, type hints, and docstring become the schema the LLM sees.

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Return the sum of a and b."""
    return a + b
```

For richer schemas, use a Pydantic args model:

```python
from pydantic import BaseModel, Field

class SearchArgs(BaseModel):
    query: str = Field(description="The search query")
    top_k: int = 5

@tool(args_schema=SearchArgs)
def search(query: str, top_k: int) -> list[str]:
    """Search the corpus and return up to top_k results."""
    ...
```

---

### Q2. What does `bind_tools` do?

**Answer:**

`llm.bind_tools(tools)` returns a new model runnable that includes the tool schemas in every request. When invoked, the model can respond with a normal message **or** with one or more `tool_calls` attached to the AIMessage.

```python
llm_with_tools = llm.bind_tools([add, search])
response = llm_with_tools.invoke([HumanMessage("what is 2+3?")])
print(response.tool_calls)
# [{"name": "add", "args": {"a": 2, "b": 3}, "id": "call_abc"}]
```

`bind_tools` does not invoke the tools — it only declares them to the model. Execution is the graph's job.

---

### Q3. What is `ToolNode` and what does it do?

**Answer:**

`ToolNode` (from `langgraph.prebuilt`) is a ready-made graph node that:

1. Reads the latest message in `state["messages"]`.
2. Iterates over each `tool_call` on that message.
3. Looks up the tool by name from the list it was constructed with.
4. Invokes each tool with the call's arguments.
5. Wraps each result in a `ToolMessage` keyed by the tool call's `id`.
6. Returns `{"messages": [...ToolMessages]}`.

```python
from langgraph.prebuilt import ToolNode
tool_node = ToolNode([add, search])
```

It runs all tool calls in the message **in parallel** by default.

---

### Q4. Sketch the canonical agent loop using `ToolNode`.

**Answer:**

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

llm_with_tools = llm.bind_tools(tools)

def agent(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")  # back to agent after tools

graph = builder.compile()
```

The cycle `agent → tools → agent` continues until the model produces a message with no `tool_calls`, at which point routing exits to `END`.

---

### Q5. How does LangGraph handle parallel tool calls?

**Answer:**

If the model returns an AIMessage with multiple `tool_calls`, `ToolNode` invokes them concurrently (using `asyncio.gather` under `astream` / `ainvoke`, or threads under `invoke`). All resulting `ToolMessage`s are appended in a single state update, then routed back to the model.

This matches the OpenAI / Anthropic tool-use protocol where one assistant turn may emit several independent tool requests.

---

### Q6. How do you pass a per-invocation secret (e.g. user API key) to a tool without putting it in state?

**Answer:**

Tools can accept the `RunnableConfig` and injected values via `langchain_core.tools.InjectedToolArg` and `InjectedState` / `InjectedStore`:

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.runnables import RunnableConfig

@tool
def search_my_inbox(query: str, config: RunnableConfig) -> list[str]:
    """Search the user's inbox."""
    api_key = config["configurable"]["inbox_api_key"]
    ...
```

Args annotated with `InjectedToolArg` are **stripped from the schema** the LLM sees — the model never knows about them and cannot supply them.

---

### Q7. How do you let a tool read the current graph state?

**Answer:**

Use `InjectedState`:

```python
from langgraph.prebuilt import InjectedState
from typing import Annotated

@tool
def summarise_history(state: Annotated[MessagesState, InjectedState]) -> str:
    """Summarise the conversation so far."""
    return summariser.invoke(state["messages"])
```

The state is provided by the runtime, again invisible to the LLM. Use sparingly: a tool that needs broad state visibility is often a sign the logic belongs in a regular node.

---

### Q8. What happens when a tool raises an exception inside `ToolNode`?

**Answer:**

By default `ToolNode` catches the exception, formats it as a `ToolMessage` with the error text, and lets the graph continue. The model sees the error and can recover (retry with different args, ask the user, give up).

To customise:

```python
ToolNode(tools, handle_tool_errors=True)              # default-ish
ToolNode(tools, handle_tool_errors=False)             # re-raise
ToolNode(tools, handle_tool_errors="Please retry.")   # custom message
ToolNode(tools, handle_tool_errors=lambda e: f"Failed: {e}")
```

For unrecoverable errors (auth failure, missing dependency), `handle_tool_errors=False` lets the exception surface and aborts the run.

---

### Q9. Why does each tool call have an `id`, and what happens if it is missing in the response?

**Answer:**

The `id` links an assistant's tool request to its corresponding `ToolMessage` reply. Models like Anthropic's reject conversations where a tool request has no matching reply with the same id.

`ToolNode` always sets `tool_call_id` on every emitted `ToolMessage`. If you build tool replies manually, copy the id verbatim:

```python
ToolMessage(content=result, tool_call_id=call["id"])
```

Mismatched or missing ids cause mysterious "tool result not found" errors at the next model invocation.

---

### Q10. How can a tool return rich data (not just a string)?

**Answer:**

`ToolMessage.content` accepts strings, lists of content blocks (text, images), or any JSON-serialisable structure. Additionally, a tool can return both content for the model and a separate "artifact" for the application:

```python
from langchain_core.tools import tool

@tool(response_format="content_and_artifact")
def fetch_image(url: str) -> tuple[str, bytes]:
    """Fetch an image and return a description plus the raw bytes."""
    img = download(url)
    return f"Fetched {url}", img
```

The string becomes the model-visible content; the bytes are stored on `ToolMessage.artifact` for downstream Python code.

---

### Q11. How would you let one tool dynamically delegate to a subgraph?

**Answer:**

Inside the tool, invoke the subgraph as a runnable:

```python
@tool
def deep_research(topic: str) -> str:
    """Run deep research on a topic."""
    result = research_subgraph.invoke({"topic": topic})
    return result["report"]
```

Pros: simple, the model treats research as a single atomic capability. Cons: the subgraph runs synchronously inside the tool node, so its intermediate streaming is hidden from the outer graph.

For richer integration, route via an explicit graph edge to a subgraph node and return control to the agent afterwards.

---

### Q12. What are the trade-offs of using `bind_tools` with `tool_choice`?

**Answer:**

`bind_tools(tools, tool_choice="search")` forces the model to emit a call to the named tool on its next response. Useful for:

- **Structured output** — define a schema as a tool and force the model to "call" it.
- **Guard-rails** — require the model to invoke a validator before answering.

Costs: removes the model's discretion. If forced into a tool the model cannot answer, it may produce nonsense args. Reserve for narrow gates, not general agent behaviour.

---

### Q13. How do you trim or summarise the message history before each LLM call to keep token counts manageable?

**Answer:**

Insert a node before the agent that mutates `messages`:

```python
from langchain_core.messages.utils import trim_messages

def trim(state: MessagesState):
    return {"messages": trim_messages(state["messages"], max_tokens=4000, ...)}

builder.add_node("trim", trim)
builder.add_edge(START, "trim")
builder.add_edge("trim", "agent")
```

Or use a custom message reducer that automatically trims (see *Channels and Reducers* Q5). For long conversations, pair trimming with a periodic **summarisation node** that condenses old turns into a single system message.
