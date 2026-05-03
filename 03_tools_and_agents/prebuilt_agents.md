# Prebuilt Agents

> `create_react_agent`, structured responses, custom prompts

---

### Q1. What does `create_react_agent` give you out of the box?

**Answer:**

`create_react_agent` (from `langgraph.prebuilt`) returns a fully compiled graph implementing the **ReAct** pattern (Reason + Act): an agent loop that alternates between LLM reasoning and tool execution until the model produces a final answer.

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model=llm, tools=[search, calculator])
agent.invoke({"messages": [HumanMessage("What is the square root of the population of Leeds?")]})
```

You get: an agent node, a `ToolNode`, conditional routing, and `MessagesState` plumbing — all wired correctly. Suitable for the common case; drop down to `StateGraph` for anything bespoke.

---

### Q2. How do you customise the system prompt for a prebuilt React agent?

**Answer:**

Pass `prompt` (a string, a `SystemMessage`, a callable, or a `ChatPromptTemplate`):

```python
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a UK-focused research assistant. Cite sources.",
)
```

If `prompt` is a callable `(state) -> list[BaseMessage]`, it builds the message list dynamically — useful for injecting per-thread context (user profile, retrieval results) without polluting state.

---

### Q3. How do you get a structured (Pydantic) final answer from `create_react_agent`?

**Answer:**

Pass `response_format`:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    summary: str
    confidence: float
    sources: list[str]

agent = create_react_agent(model=llm, tools=tools, response_format=Answer)
result = agent.invoke({"messages": [...]})
result["structured_response"]  # Answer(...)
```

Internally LangGraph appends a final node that re-prompts the model to emit a tool call matching the `Answer` schema, parses it, and stores it in the `structured_response` channel.

---

### Q4. What state does `create_react_agent` use, and can it be extended?

**Answer:**

Default state is `AgentState` (essentially `MessagesState` plus `remaining_steps`). To add custom channels:

```python
from langgraph.prebuilt.chat_agent_executor import AgentState

class MyState(AgentState):
    user_id: str
    retrieved_docs: list[str]

agent = create_react_agent(model=llm, tools=tools, state_schema=MyState)
```

Tools and prompt callables can then read these extra fields via `InjectedState`.

---

### Q5. How does `create_react_agent` decide when to stop?

**Answer:**

It exits when the most recent AI message has **no `tool_calls`**. The conditional edge after the agent node routes to `END` in that case, otherwise to `ToolNode`. There is also a built-in `remaining_steps` counter that decrements on each loop and forces termination when exhausted (preventing runaway loops).

---

### Q6. When should you reach for `create_react_agent` and when should you build a custom `StateGraph`?

**Answer:**

Use `create_react_agent` when:

- Your agent is a single LLM with a tool belt.
- The default ReAct loop matches your control flow.
- You want tested, idiomatic plumbing without boilerplate.

Build a custom `StateGraph` when:

- Multiple specialised sub-agents must collaborate.
- Routing depends on rich state, not just "is there a tool call?".
- You need explicit human-in-the-loop interrupts at specific points.
- The graph is not strictly a single agent loop (planner → many workers → synthesiser).

A common evolution path: prototype with `create_react_agent`, then refactor to `StateGraph` once requirements stabilise.

---

### Q7. How do you attach a checkpointer to a prebuilt agent?

**Answer:**

Pass it to the constructor:

```python
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=MemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": [HumanMessage("hi")]},  config)
agent.invoke({"messages": [HumanMessage("again")]}, config)  # remembers
```

Same `thread_id` semantics as a custom graph.

---

### Q8. Can a prebuilt agent be used as a subgraph node?

**Answer:**

Yes — a compiled agent is a normal `Runnable`, so it can be added as a node in a larger graph:

```python
researcher = create_react_agent(model=llm, tools=research_tools)

builder = StateGraph(OuterState)
builder.add_node("research", researcher)
builder.add_node("write", writer)
builder.add_edge("research", "write")
```

Outer state must contain the channels the agent expects (`messages`), or you adapt input/output via small wrapper functions.

---

### Q9. What is the `remaining_steps` channel for, and how is it consumed?

**Answer:**

`remaining_steps` is a counter initialised by the prebuilt agent (default 25). Each loop the agent decrements it; when it reaches zero, the conditional edge forces termination even if the model wants more tool calls.

This is **independent** of the global `recursion_limit` (which counts supersteps). `remaining_steps` is a per-agent budget that protects multi-agent graphs from one runaway sub-agent exhausting the whole supergraph.

---

### Q10. How do you stream tokens from a prebuilt agent?

**Answer:**

Use `stream_mode="messages"` (or `astream_events`):

```python
async for chunk, metadata in agent.astream(
    {"messages": [HumanMessage("...")]} ,
    stream_mode="messages",
):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

Each yielded tuple is `(message_chunk, metadata)`. Filter on `metadata["langgraph_node"]` if you only want tokens from the final answer node, not from intermediate agent reasoning.

---

### Q11. How do you add HITL approval before tool execution in a prebuilt agent?

**Answer:**

Use the `interrupt_before` (or `interrupt_after`) compile option — except `create_react_agent` exposes this directly:

```python
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],
)
```

When the agent decides to call a tool, the graph pauses **before** `ToolNode` runs. The application inspects the proposed tool calls, optionally edits them via `update_state`, then resumes with `agent.invoke(None, config)`.

---

### Q12. What is the `version` parameter on `create_react_agent`?

**Answer:**

Controls the internal graph implementation. `version="v2"` (the modern default in recent LangGraph releases) uses the `Send` API for parallel tool calls and integrates better with structured responses; `version="v1"` is the original implementation kept for backward compatibility.

Stick with the default unless you have a specific compatibility reason; revisit when upgrading LangGraph versions because defaults shift.
