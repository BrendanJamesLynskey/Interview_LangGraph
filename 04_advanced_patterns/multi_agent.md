# Multi-Agent Architectures

> Supervisor, swarm, hierarchical teams, handoffs via `Command`

---

### Q1. What problem do multi-agent architectures solve?

**Answer:**

Single-agent loops degrade when the tool belt grows large or when distinct skills have conflicting prompting requirements (e.g. an SQL writer vs. a literary editor). Multi-agent designs decompose the problem into specialists, each with a focused tool set and prompt, coordinated by an explicit topology.

Benefits: smaller per-call context, clearer accountability, easier evaluation per agent. Costs: more orchestration code, more points of failure, more LLM calls.

---

### Q2. Describe the **supervisor** pattern.

**Answer:**

A **supervisor** LLM sits at the centre and routes work to specialist agents. After each specialist returns, control flows back to the supervisor, which decides the next specialist or finishes.

```
                 ┌──────────────┐
                 │  Supervisor  │
                 └──┬───────┬───┘
            ┌───────┘       └───────┐
        ┌───▼───┐               ┌───▼───┐
        │ Coder │               │Writer │
        └───────┘               └───────┘
```

Supervisor decisions are typically expressed as **routing tool calls** — the supervisor "calls" a `route_to_coder` tool whose effect is to transition the graph to that node.

---

### Q3. Describe the **swarm** pattern.

**Answer:**

Agents transfer control to each other directly via **handoff tools**. There is no central supervisor; the active agent is whichever one most recently received a handoff. The graph state tracks the active agent so that resuming a thread routes back to the right place.

Pros: less indirection, agents share full context. Cons: every agent needs the prompts and tools to perform handoff decisions correctly.

LangGraph provides `langgraph-swarm` as a packaged implementation.

---

### Q4. What is `Command` and why is it central to multi-agent designs?

**Answer:**

`Command` (from `langgraph.types`) lets a node return both a **state update** and a **routing instruction** in one object:

```python
from langgraph.types import Command
from typing import Literal

def supervisor(state) -> Command[Literal["coder", "writer", "__end__"]]:
    decision = pick_next(state)
    return Command(
        goto=decision,
        update={"trace": state["trace"] + [decision]},
    )
```

Without `Command`, you would need a node *and* a conditional edge function — `Command` collapses them into a single primitive that is also serialisable in checkpoints.

`Command(graph=Command.PARENT, goto="...")` lets a subgraph node hand off to a node in the parent graph, the foundation of hierarchical handoffs.

---

### Q5. How do you implement a handoff tool?

**Answer:**

A handoff tool, when invoked, returns a `Command` that transitions to the target agent and (optionally) seeds its messages:

```python
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

@tool
def transfer_to_writer(state: Annotated[dict, InjectedState]) -> Command:
    """Hand off to the writer agent with the current context."""
    return Command(
        goto="writer",
        graph=Command.PARENT,                     # hop up to parent graph
        update={"messages": state["messages"]},   # carry context
    )
```

The supervisor / current agent simply lists this tool alongside its normal tools; the LLM "calls" it when it decides to hand off.

---

### Q6. Describe the **hierarchical teams** pattern.

**Answer:**

A top-level supervisor delegates to **mid-level supervisors**, each of which manages a team of specialists. Each level is its own subgraph with its own state.

```
            top-supervisor
           ┌──────┴──────┐
        research        writing
        team-sup        team-sup
       ┌──┴──┐         ┌──┴──┐
     web   db        editor  fact-check
```

Useful when domains are large enough that one supervisor would be overwhelmed by 30+ agent options. Each subgraph reports a summary upward; the top supervisor never sees individual specialist messages.

---

### Q7. How is state shared (or isolated) across a multi-agent topology?

**Answer:**

Three options:

1. **Single shared state** — all agents read/write the same `MessagesState`. Simple, but the message log can become noisy and prompts grow long.

2. **Per-agent private subgraph state** — each agent is a subgraph with its own schema; the parent passes in only what's needed and reads back a summary. Cleaner isolation, more mapping code.

3. **Hybrid** — shared `messages` plus private scratchpads per agent. Common in practice.

Pick based on whether agents must see each other's reasoning. Reviewers usually need full context; tool-runners often do not.

---

### Q8. How do you prevent supervisor loops where agents bounce back to the supervisor without progress?

**Answer:**

Defensive measures, in increasing order:

1. **Step counter in state**, increment on each supervisor visit, force `END` after N.
2. **Supervisor prompt** explicitly instructed to terminate when no new information has been added.
3. **Memory** of which agents have been visited; supervisor cannot route to the same agent twice without justification.
4. **Recursion limit** as the last-resort safety net.

A persistent loop usually signals a missing tool or an under-specified termination condition rather than a routing bug.

---

### Q9. How do you stream output from a multi-agent graph in a way that the UI can attribute tokens to the correct agent?

**Answer:**

`astream_events` (or `stream_mode="messages"`) emits a `metadata` dict with `langgraph_node` (the immediate node) and `langgraph_path` (the full hierarchy down through subgraphs). Use these to label streamed tokens:

```python
async for event in graph.astream_events({"messages": [...]} , version="v2"):
    if event["event"] == "on_chat_model_stream":
        node = event["metadata"].get("langgraph_node")
        token = event["data"]["chunk"].content
        ui.append(node, token)
```

`subgraphs=True` on `stream` propagates updates from inner graphs as well — without it, subgraph internals are hidden.

---

### Q10. What are the failure modes specific to multi-agent designs and how do you mitigate them?

**Answer:**

| Failure | Mitigation |
|---------|------------|
| Agents looping over each other | Step caps, memory of visits, clear termination criteria |
| Lost context across handoffs | Carry a structured handoff payload, not raw messages |
| Conflicting writes to shared state | Use list reducers; partition state per agent |
| Cost blowup from N specialists per turn | Force the supervisor to pick **one** at a time |
| Hard to debug | Always run with a checkpointer and stream events to a tracer (LangSmith) |

---

### Q11. How does the supervisor pattern compare to a single-agent ReAct loop with many tools?

**Answer:**

| Dimension | Big-tool-belt ReAct | Supervisor + specialists |
|-----------|---------------------|--------------------------|
| Latency per turn | Lower | Higher (extra hop) |
| Token cost per turn | Lower | Higher (multiple LLM calls) |
| Prompt complexity | Bloats with tool count | Stays small per agent |
| Tool-selection accuracy | Degrades past ~10 tools | Stable as you add specialists |
| Debuggability | Single trace | One trace per agent |

Heuristic: stick with one agent until tool count or prompt length forces specialisation.

---

### Q12. How would you add a "reviewer" agent that vetoes another agent's output?

**Answer:**

Insert a reviewer node downstream of the producer, returning a `Command` that either accepts (`goto=END`) or rejects (`goto="producer"`) with feedback:

```python
def reviewer(state) -> Command[Literal["producer", "__end__"]]:
    verdict = review_chain.invoke(state["draft"])
    if verdict.ok:
        return Command(goto=END, update={"final": state["draft"]})
    return Command(
        goto="producer",
        update={"feedback": verdict.notes, "draft": None},
    )
```

The producer reads `feedback` from state on its next visit. Cap the cycle with a retry counter to avoid endless rewrites.
