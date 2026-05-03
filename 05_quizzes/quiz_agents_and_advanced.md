# Quiz — Agents and Advanced Patterns

> Tools, multi-agent, streaming, HITL

---

### Q1. What does `bind_tools(tools)` do, and what does it not do?

**Answer:**

It returns a model runnable that includes the tool schemas in every request, so the LLM can emit `tool_calls`. It does **not** execute tools — that is the job of `ToolNode` or equivalent.

---

### Q2. Why does every tool call have an `id`?

**Answer:**

To pair the assistant's tool request with its corresponding `ToolMessage` reply. Most provider APIs reject conversations where ids do not match between request and response.

---

### Q3. How does `ToolNode` handle multiple `tool_calls` in a single AIMessage?

**Answer:**

It invokes them in parallel and returns a single state update containing all the resulting `ToolMessage`s. This matches the OpenAI / Anthropic protocol where one assistant turn can request several tool calls.

---

### Q4. What is `InjectedState` used for, and what is the user-facing benefit?

**Answer:**

It injects the current graph state as a tool argument that is **stripped from the schema the LLM sees**. The model never knows about the parameter and cannot supply it, but the tool implementation can read state freely.

---

### Q5. When does `create_react_agent` exit its loop?

**Answer:**

When the most recent AI message has no `tool_calls`. There is also a `remaining_steps` budget (default-bounded) that forces termination if exhausted.

---

### Q6. How do you make `create_react_agent` return a Pydantic structured response?

**Answer:**

Pass `response_format=YourPydanticModel` to the constructor. The agent appends a final node that re-prompts the model to emit a tool call matching the schema, then exposes the parsed object on `result["structured_response"]`.

---

### Q7. Describe the supervisor pattern in one sentence.

**Answer:**

A central supervisor LLM routes work to specialist agents, receives their output, and decides the next step until done.

---

### Q8. What is `Command` and why does it matter for multi-agent designs?

**Answer:**

`Command` lets a node return both a state update and a routing decision in one object, including handoffs to nodes in a parent graph. It collapses the "node + conditional edge" pair into a single, checkpointable primitive.

---

### Q9. What is the `Send` API for?

**Answer:**

For dynamic fan-out — schedule N parallel executions of a node, where N depends on runtime data, each with its own custom input dict. The standard implementation of map-reduce in LangGraph.

---

### Q10. A worker invoked via `Send` reads only its `Send` payload. How does it write back to shared state?

**Answer:**

The worker's return dict is merged into the parent state via the parent's reducers. Reads are scoped to the `Send` payload; writes propagate through the normal channel mechanism — which is why fan-out keys typically use `operator.add`.

---

### Q11. Name three stream modes and what each emits.

**Answer:**

- `"values"` — full state after each superstep.
- `"updates"` — partial update from the most recent node(s).
- `"messages"` — token-level streaming from LLM nodes, as `(chunk, metadata)` tuples.

---

### Q12. Why does `interrupt(...)` require a checkpointer?

**Answer:**

Because it persists the paused state and metadata so that a later `invoke(Command(resume=...), config)` can reload exactly where execution stopped. Without persistence there is nowhere to save the pause point.

---

### Q13. Static `interrupt_before=["X"]` vs dynamic `interrupt(value)` inside a node — which would you use to pause for approval before any tool call?

**Answer:**

Static `interrupt_before=["tools"]` — the pause condition is predictable and structural, no payload-shaping is needed beyond reading the proposed tool calls from state.

---

### Q14. What is a known gotcha when resuming after `interrupt(...)`?

**Answer:**

The interrupted node re-runs from its start. Anything done before the `interrupt(...)` call executes a second time. Place `interrupt` early in the function, or move side effects to a separate node.

---

### Q15. How do you stream events from a subgraph?

**Answer:**

Pass `subgraphs=True` to `astream` (or `astream_events`). Each yielded item is tagged with its namespace path so the consumer can attribute updates to the correct level of the hierarchy.

---

### Q16. When would you reach for hierarchical teams over a single supervisor?

**Answer:**

When the number of specialists per supervisor would otherwise exceed roughly 10, making routing accuracy degrade. Hierarchical teams group related specialists under mid-level supervisors so each routing decision is among a small candidate set.

---

### Q17. What is a handoff tool in the swarm pattern?

**Answer:**

A tool that, when invoked by the active agent's LLM, returns a `Command(goto="other_agent", ...)` transferring control. The active agent simply lists handoff tools alongside its own tools and lets the LLM choose when to delegate.

---

### Q18. Pattern recommendation: a workflow processes 100 uploaded PDFs, extracting structured data from each. What LangGraph constructs do you compose?

**Answer:**

A planner node decomposes the upload into items, a conditional edge fans out via `Send("worker", {"pdf": p})` per file, each worker is a small subgraph (extract → validate → record), results are concatenated via `operator.add`, and a final reduce node aggregates and emits the report. Add a checkpointer so failures resume from the last completed worker, and stream `updates` so the UI can show per-file progress.
