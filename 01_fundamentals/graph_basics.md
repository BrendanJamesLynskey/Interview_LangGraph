# Graph Basics

> `StateGraph`, nodes, edges, START/END, compilation

---

### Q1. What is LangGraph and how does it differ from LangChain's LCEL?

**Answer:**

LangGraph is a low-level orchestration framework for building **stateful, multi-actor** LLM applications as **directed graphs**. LangChain's LCEL (LangChain Expression Language) composes runnables into linear or branching pipelines that are fundamentally **DAGs** with limited cycle support.

Key differences:

| Aspect | LCEL | LangGraph |
|--------|------|-----------|
| Topology | DAG | Cyclic graph (loops allowed) |
| State | Implicit, per-step | Explicit shared schema |
| Persistence | None built-in | Checkpointers, threads |
| Use case | Chains, RAG | Agents, multi-agent, HITL |

LangGraph is the right tool when execution must loop (an agent that calls tools until done), pause and resume, or coordinate multiple actors.

---

### Q2. What are the three primitives of a LangGraph?

**Answer:**

1. **State** — a typed schema (TypedDict or Pydantic model) describing the data passed between nodes.
2. **Nodes** — Python functions (or runnables) that take the current state and return a partial update.
3. **Edges** — connections that determine which node runs next, either fixed or conditional.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    count: int

def increment(state: State) -> dict:
    return {"count": state["count"] + 1}

builder = StateGraph(State)
builder.add_node("inc", increment)
builder.add_edge(START, "inc")
builder.add_edge("inc", END)

graph = builder.compile()
graph.invoke({"count": 0})  # {"count": 1}
```

---

### Q3. What are `START` and `END`?

**Answer:**

`START` and `END` are **virtual nodes** representing the entry and exit points of the graph.

- `START` — the implicit source. Every graph must have at least one edge from `START` to a real node.
- `END` — the implicit sink. Reaching `END` terminates execution and returns the final state.

A graph without an edge from `START` raises a compile-time error. A graph that never reaches `END` (and has no other terminating condition) loops until the recursion limit is hit.

```python
from langgraph.graph import START, END
builder.add_edge(START, "first_node")
builder.add_edge("last_node", END)
```

---

### Q4. What does `compile()` do, and why is it a separate step?

**Answer:**

`compile()` validates the graph topology and returns a **`CompiledStateGraph`** — an immutable, executable runnable. Validation includes:

- All nodes referenced in edges exist.
- The graph is reachable from `START`.
- Channel reducers are consistent.
- Checkpointer (if any) is correctly attached.

Compilation is separate from construction so the same builder can be configured with different checkpointers, interrupts, or store backends without rebuilding the topology:

```python
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["human_review"],
)
```

A compiled graph implements the LangChain `Runnable` interface — `invoke`, `stream`, `batch`, `astream`.

---

### Q5. What is the difference between `add_edge` and `add_conditional_edges`?

**Answer:**

- **`add_edge(source, target)`** — unconditional. After `source` runs, `target` always runs next.
- **`add_conditional_edges(source, router, mapping)`** — the `router` function inspects the current state and returns a key (or list of keys) that selects the next node(s) via the optional `mapping`.

```python
def route(state: State) -> str:
    return "tools" if state["needs_tool"] else "respond"

builder.add_conditional_edges(
    "agent",
    route,
    {"tools": "tool_node", "respond": END},
)
```

If `mapping` is omitted, the router's return value is treated as the literal node name. Returning a **list** of keys triggers parallel execution of all matched nodes.

---

### Q6. What does a node function return, and how is the state updated?

**Answer:**

A node returns a **partial dict** containing only the channels it wants to update. LangGraph merges the partial into the current state using each channel's **reducer**.

```python
class State(TypedDict):
    count: int
    log: list[str]

def step(state: State) -> dict:
    return {"count": state["count"] + 1}  # log is unchanged
```

If a channel uses the default (overwrite) reducer, returning `{"count": 5}` replaces the previous value. If the channel uses `operator.add` or `add_messages`, the returned value is appended to (or merged with) the existing value.

A node may also return `None` to indicate no state change.

---

### Q7. How are parallel branches expressed?

**Answer:**

Add multiple outgoing edges from one node, or return multiple destination keys from a conditional router. LangGraph runs all matched targets **concurrently within a superstep**.

```python
builder.add_edge("planner", "research")
builder.add_edge("planner", "draft")  # research and draft run in parallel
builder.add_edge("research", "synthesise")
builder.add_edge("draft", "synthesise")
```

State updates from parallel branches are merged at the end of the superstep using the channels' reducers. This is why **non-overwrite reducers** (lists, sets) are essential for fan-out / fan-in patterns — two branches both writing to a single overwrite channel would cause a conflict.

---

### Q8. What is a "superstep" in LangGraph's execution model?

**Answer:**

LangGraph executes in **supersteps** following the **Pregel** / Bulk Synchronous Parallel model:

1. All nodes scheduled for the current superstep run **concurrently**.
2. Their state updates are collected.
3. Reducers merge updates into the shared state.
4. The next set of scheduled nodes is determined from edges and conditional routes.
5. Repeat until no nodes are scheduled or `END` is reached.

This model gives deterministic semantics for parallelism: a node always sees a consistent snapshot of state, never partial mid-superstep updates.

---

### Q9. What is the recursion limit and why does it exist?

**Answer:**

The recursion limit caps the number of **supersteps** a single invocation can execute. It guards against infinite loops in cyclic graphs (an agent that keeps calling tools without ever reaching `END`).

```python
graph.invoke({"messages": [...]}, {"recursion_limit": 50})
```

The default has changed between releases: it was **25** in LangGraph 0.x, and in 1.2.14 it is **10007** (measured; set in `langgraph/_internal/_config.py` and overridable with the `LANGGRAPH_DEFAULT_RECURSION_LIMIT` environment variable), although some docs still say 1000. Set it explicitly instead of relying on the default.

Hitting the limit raises `GraphRecursionError`. For long-running agents, raise the limit deliberately rather than disabling the check — and consider whether the agent's halting condition is correct.

---

### Q10. Can a node be a Runnable instead of a plain function?

**Answer:**

Yes. Any object implementing the LangChain `Runnable` interface can be a node, including LCEL chains, prompts piped to models, or other compiled graphs (subgraphs).

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([("user", "{question}")])
chain = prompt | llm

builder.add_node("ask", chain)
```

The runnable receives the full state dict as input and its output is treated as the partial state update — it must return a dict whose keys match state channels.

---

### Q11. How do you visualise a compiled graph?

**Answer:**

`get_graph()` returns a `Graph` object with rendering helpers:

```python
graph.get_graph().draw_mermaid()        # Mermaid source string
graph.get_graph().draw_mermaid_png()    # PNG bytes (requires extras)
graph.get_graph().draw_ascii()          # ASCII art
```

In a Jupyter notebook:

```python
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())
```

Visualisation is invaluable for debugging routing logic in agents with many conditional edges.

---

### Q12. What happens if two nodes write to the same overwrite channel in the same superstep?

**Answer:**

LangGraph raises an `InvalidUpdateError` at runtime — the default (last-write-wins) reducer cannot deterministically merge concurrent writes.

To support concurrent writes, declare the channel with a reducer that **defines a merge**:

```python
from typing import Annotated
import operator

class State(TypedDict):
    results: Annotated[list, operator.add]  # concatenate
```

Now two parallel nodes can each return `{"results": [...]}` and the lists are concatenated.

---

### Q13. What is the difference between `invoke`, `stream`, and `astream`?

**Answer:**

| Method | Sync/Async | Returns |
|--------|------------|---------|
| `invoke(input)` | Sync | Final state dict |
| `stream(input)` | Sync | Iterator of intermediate updates |
| `ainvoke(input)` | Async | Final state dict |
| `astream(input)` | Async | Async iterator of intermediate updates |

`stream` and `astream` accept a `stream_mode` argument: `"values"` (full state after each step), `"updates"` (just the delta from the most recent node), `"messages"` (token-level for LLM nodes), or `"debug"` (verbose checkpointing info).

---

### Q14. Can a graph node skip updating state?

**Answer:**

Yes — return `None` or an empty dict `{}`. The node still ran (side effects, logging, tool calls happened) but no channels are modified.

```python
def log_only(state: State) -> None:
    logger.info("state seen: %s", state)
    return None
```

This is useful for observation, metrics, or fire-and-forget side effects that shouldn't influence subsequent routing.

---

### Q15. How do you pass configuration to a node at invocation time without putting it in state?

**Answer:**

Use the `config` parameter, accessed by adding a second argument of type `RunnableConfig` to the node:

```python
from langchain_core.runnables import RunnableConfig

def node(state: State, config: RunnableConfig) -> dict:
    user_id = config["configurable"]["user_id"]
    return {"who": user_id}

graph.invoke({...}, config={"configurable": {"user_id": "alice"}})
```

`configurable` values are **per-invocation** and not part of the persisted state. Common uses: thread IDs, user IDs, model selection, feature flags.
