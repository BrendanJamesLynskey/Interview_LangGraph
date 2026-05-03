# Streaming and Human-in-the-Loop

> `astream`, stream modes, `interrupt`, approval gates, state edits

---

### Q1. What stream modes does LangGraph support?

**Answer:**

| Mode | What it yields |
|------|----------------|
| `"values"` | Full state after each superstep |
| `"updates"` | Just the partial update from the most recent node(s) |
| `"messages"` | LLM token chunks (chunk, metadata) tuples |
| `"custom"` | Arbitrary values yielded via `dispatch_custom_event` |
| `"debug"` | Verbose checkpoint and step diagnostics |

You may pass a list to combine modes: `stream_mode=["updates", "messages"]` interleaves both kinds of events.

---

### Q2. How do you stream individual LLM tokens?

**Answer:**

```python
async for chunk, meta in graph.astream(
    {"messages": [HumanMessage("...")]} ,
    stream_mode="messages",
):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

`chunk` is a `BaseMessageChunk`, `meta` includes `langgraph_node` and run metadata. This requires the LLM to support streaming (most do) and to be invoked via a streaming-capable runnable.

---

### Q3. How do you stream from inside subgraphs?

**Answer:**

Pass `subgraphs=True`:

```python
async for ns, chunk in graph.astream(input, stream_mode="updates", subgraphs=True):
    print(ns, chunk)   # ns is the subgraph namespace path
```

Without this flag, only the parent graph's events are emitted. With it, every level of nesting streams up, tagged by the namespace tuple so the consumer can attribute updates correctly.

---

### Q4. What is `astream_events` and when do you prefer it over `astream`?

**Answer:**

`astream_events(input, version="v2")` yields a richer, finer-grained event stream including LangChain-level events (`on_chat_model_start`, `on_tool_end`, `on_chain_stream`, …). Each event includes `event`, `name`, `data`, `metadata`, `tags`, `run_id`.

Use it when:

- You need fine-grained UI updates (per-tool spinner, per-model progress).
- You want to filter by tags or run IDs.
- You need to capture intermediate inputs/outputs of inner runnables.

Use `astream` when you only need state updates or token streams.

---

### Q5. What is `interrupt` and how does it work?

**Answer:**

`interrupt(value)` (from `langgraph.types`) pauses the graph mid-node, persists state, and returns control to the caller along with the value. The application can then decide what to do — show a UI, prompt for approval, edit state — and resume by calling `invoke(Command(resume=user_input), config)`.

```python
from langgraph.types import interrupt, Command

def review(state):
    decision = interrupt({"draft": state["draft"], "ask": "approve?"})
    return {"approved": decision == "yes"}

# First call pauses:
graph.invoke({"draft": "..."}, config)
# UI shows the draft, user clicks approve, then:
graph.invoke(Command(resume="yes"), config)
```

`interrupt` requires a checkpointer — without persistence there is nowhere to save the paused state.

---

### Q6. How does `interrupt` differ from `interrupt_before` / `interrupt_after`?

**Answer:**

- **`interrupt(value)`** — *dynamic*, called inside a node. Pauses at exactly that line, with a payload of your choosing.
- **`interrupt_before=["nodename"]`** — *static*, declared at compile time. Pauses **just before** the named node runs, with no payload.
- **`interrupt_after=["nodename"]`** — *static*, pauses just after.

Use the static form for predictable approval gates ("always pause before tool execution"). Use the dynamic form when the pause condition depends on state or tool arguments.

---

### Q7. How do you resume a graph after an interrupt?

**Answer:**

Call `invoke` (or `stream`) with `Command(resume=value)` and the same `config`:

```python
graph.invoke(Command(resume="approved"), config)
```

The framework loads the checkpoint, re-runs the interrupted node from its start, and the second invocation of `interrupt` returns the resume value (the first call returned the pause payload to the caller; the second returns the user's response to the node).

For static interrupts (`interrupt_before`), pass `None` instead — there is no value to inject:

```python
graph.invoke(None, config)
```

---

### Q8. How do you implement an approval gate that lets the user edit the proposed action?

**Answer:**

Pause before tool execution, expose the proposed tool calls, accept either an approval signal or an edited message:

```python
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],
)

# Run until pause:
agent.invoke(initial, config)

# Inspect proposed calls:
snap = agent.get_state(config)
proposed = snap.values["messages"][-1].tool_calls

# If user wants to edit, write a replacement message:
edited = AIMessage(id=snap.values["messages"][-1].id, content="", tool_calls=[...edited...])
agent.update_state(config, {"messages": [edited]}, as_node="agent")

# Resume:
agent.invoke(None, config)
```

Editing relies on `add_messages`'s id-aware replacement.

---

### Q9. What is `dispatch_custom_event` and how does it surface in streams?

**Answer:**

A node can emit a custom event that propagates through the stream:

```python
from langchain_core.callbacks.manager import adispatch_custom_event

async def slow_node(state):
    await adispatch_custom_event("progress", {"pct": 25})
    ...
    await adispatch_custom_event("progress", {"pct": 100})
```

Subscribers using `astream_events` see `{"event": "on_custom_event", "name": "progress", ...}`. Useful for emitting milestones (e.g. "downloading", "embedding", "indexing") without polluting the state schema.

---

### Q10. How do you forcibly cancel a running graph?

**Answer:**

If using `astream` / `ainvoke`, cancel the awaitable / async generator from the consumer (`task.cancel()`). The run's `RunnableConfig` carries cancellation tokens; LangGraph propagates the cancellation to in-flight LLM and tool calls (subject to those runnables honouring async cancellation).

For sync runs there is no portable cancellation primitive — design long jobs as async, or break them into smaller supersteps so the next checkpoint provides a natural pause point.

---

### Q11. How do you avoid replaying expensive side effects when resuming after `interrupt`?

**Answer:**

`interrupt` re-runs the **entire interrupted node** from its start when resumed. Anything done before the `interrupt(...)` call executes a second time. Mitigations:

1. **Place `interrupt` early** — perform side effects only after the resume value is in hand.
2. **Move side effects into a separate node** that runs after the approval node.
3. **Idempotent side effects** — e.g. write with a deterministic key so re-execution is a no-op.

Static `interrupt_before`/`interrupt_after` do not have this issue because they pause between nodes, not inside one.

---

### Q12. How do you implement a "wait for external webhook" pattern?

**Answer:**

Use `interrupt` to pause, expose the thread ID externally (e.g. send to a callback URL), and resume when the webhook fires:

```python
def await_webhook(state):
    request_id = create_external_request(state["payload"])
    response = interrupt({"request_id": request_id})
    return {"response": response}
```

The web handler that receives the webhook looks up the `request_id` to find the thread, then calls `graph.invoke(Command(resume=webhook_data), config)`. State persists indefinitely in the checkpointer, so the agent can sleep for hours or days.

---

### Q13. How does streaming behave with checkpointers attached?

**Answer:**

Streaming and checkpointing compose cleanly: each yielded event corresponds to either a node start, a node completion, or a token, and the checkpoint write happens at superstep boundaries. The consumer sees both real-time updates *and* gets durable resume points.

Cost: each superstep does a checkpoint write, so very chatty graphs that yield many supersteps amplify storage I/O. Tune by combining work into fewer, larger nodes if checkpoint write cost becomes a bottleneck.
