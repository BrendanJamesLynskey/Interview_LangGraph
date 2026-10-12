# Subgraphs and Send API

> Composition, isolated state, dynamic fan-out with `Send`, map-reduce

---

### Q1. What is a subgraph and how is it added to a parent graph?

**Answer:**

A subgraph is a compiled `StateGraph` used as a node inside another graph. The parent treats it as a black box runnable — calls it with input, gets back a final state.

```python
inner = inner_builder.compile()

builder = StateGraph(OuterState)
builder.add_node("inner", inner)
builder.add_edge(START, "inner")
```

The framework manages the subgraph's checkpoints under a separate **namespace**, so its internal state does not collide with the parent's.

---

### Q2. How does state map between parent and subgraph?

**Answer:**

Two cases:

1. **Shared keys** — if the parent state includes channels with the same names as the subgraph state, those channels are passed through automatically. Convenient when both share `messages`.

2. **Disjoint schemas** — wrap the subgraph in a function that adapts:

```python
def adapter(state: OuterState) -> dict:
    inner_input = {"q": state["question"]}
    inner_output = inner.invoke(inner_input)
    return {"answer": inner_output["a"]}

builder.add_node("inner", adapter)
```

Use option 2 when you want strict isolation between layers.

---

### Q3. Why use a subgraph instead of just adding more nodes to one big graph?

**Answer:**

- **Reusability** — the same subgraph can be a node in many parent graphs.
- **Encapsulation** — internal state and intermediate nodes are hidden.
- **Independent evolution** — change the subgraph's internals without touching the parent.
- **Independent persistence** — subgraph checkpoints are separately addressable.
- **Recursion** — a subgraph can be invoked many times in parallel via `Send`.

Cost: extra namespace bookkeeping and slightly more obscure traces.

---

### Q4. What is the `Send` API and what problem does it solve?

**Answer:**

`Send(node, input)` schedules an additional execution of `node` with a **custom input dict**, independent of the current state. Returning a list of `Send` objects from a conditional edge produces **dynamic fan-out**: the number of parallel branches is determined at runtime.

```python
from langgraph.types import Send

def fan_out(state):
    return [Send("worker", {"item": item}) for item in state["items"]]

builder.add_conditional_edges("planner", fan_out)
```

This is the classic **map** in map-reduce — one item per `Send`, all running concurrently.

---

### Q5. How is `Send` different from a regular conditional edge that returns multiple node names?

**Answer:**

| Aspect | Multiple names | `Send` |
|--------|----------------|--------|
| Input each branch sees | Full state | Custom per-branch dict |
| Number of branches | Fixed at design time | Dynamic at runtime |
| Branch identity | One per named edge | One per `Send` instance |
| Use case | Static fan-out | Map-reduce, per-item processing |

`Send` is the only mechanism for **runtime-sized parallelism** in LangGraph.

---

### Q6. Sketch a map-reduce graph using `Send`.

**Answer:**

```python
from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class State(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]
    summary: str

def map_items(state):
    return [Send("worker", {"item": i}) for i in state["items"]]

def worker(payload: dict) -> dict:
    return {"results": [process(payload["item"])]}

def reduce(state: State) -> dict:
    return {"summary": summarise(state["results"])}

builder = StateGraph(State)
builder.add_node("worker", worker)
builder.add_node("reduce", reduce)
builder.add_conditional_edges(START, map_items, ["worker"])
builder.add_edge("worker", "reduce")
builder.add_edge("reduce", END)

graph = builder.compile()
```

All `worker` invocations run in parallel; `results` is concatenated by `operator.add`; `reduce` sees the joined list.

---

### Q7. What input does a `Send`-targeted node receive?

**Answer:**

Whatever dict you put inside the `Send`. It is **not** merged with the parent state — the worker node sees only what was sent. This is intentional: each worker should be a pure function of its own slice of work.

If a worker needs additional shared context, include it explicitly in the payload, or read from the long-term `Store`.

---

### Q8. How does the worker write back to the parent state if it doesn't receive the parent state?

**Answer:**

The worker's return dict is **still merged into the parent state** via the parent's reducers. The asymmetry is: workers read only their `Send` payload but **write to the shared state**.

This is why fan-out keys typically use `operator.add` — many workers concurrently writing to one channel must combine, not overwrite.

---

### Q9. Can `Send` target a subgraph?

**Answer:**

Yes — the target name can be any node, including a subgraph node. This is one of the most powerful patterns in LangGraph: dynamic fan-out into a richly structured worker pipeline.

```python
worker_subgraph = build_worker_pipeline().compile()

builder.add_node("worker", worker_subgraph)
builder.add_conditional_edges("planner", lambda s: [Send("worker", {"item": i}) for i in s["items"]])
```

Each `Send` triggers an independent run of the worker subgraph, with its own internal namespace and checkpoints.

---

### Q10. What happens if a worker raises during a parallel `Send` fan-out?

**Answer:**

The superstep fails and the parent graph raises the error. What happens to the siblings depends on the checkpointer:

- **Without a checkpointer**, the run is simply lost, including the siblings' results.
- **With a checkpointer**, the writes of the workers that finished are kept as *pending writes* on the last checkpoint. Resuming the thread (`graph.invoke(None, config)`) re-runs **only the failed worker**; the siblings are not called again. Checked against LangGraph 1.2.14: fan out to items 1, 2, 3 with item 2 raising, then resume, and only item 2 runs again, giving `out == [1, 2, 3]`.

To let the reduce step carry on despite failures, wrap each worker's body in try/except and return an error marker:

```python
def worker(payload):
    try:
        return {"results": [process(payload["item"])]}
    except Exception as e:
        return {"errors": [{"item": payload["item"], "error": str(e)}]}
```

The reduce step then aggregates both successes and failures and decides what to do.

---

### Q11. How can `Send` and `Command` be combined?

**Answer:**

A node can return a `Command` whose `goto` is a list of `Send` objects:

```python
def planner(state) -> Command:
    items = decompose(state["task"])
    return Command(
        update={"plan": items},
        goto=[Send("worker", {"item": i}) for i in items],
    )
```

This is the cleanest way to express: "update state and dynamically fan out in one step." Without `Command`, you would split the same logic across a node and a conditional edge.

---

### Q12. When should you choose subgraphs vs `Send` vs separate compiled graphs?

**Answer:**

- **Subgraph** — when a section of logic is *static* (always present, fixed topology) but you want encapsulation. Think: "the research pipeline is always one of the steps."

- **`Send`** — when fan-out arity is *dynamic*, determined per invocation by the data. Think: "process one chunk per uploaded document."

- **Separate compiled graph** invoked via tool — when the sub-system is conceptually independent (its own product surface, its own auth, its own checkpointer) and the agent merely *consults* it.

These compose: an agent can use `Send` to fan out into a subgraph that itself contains an inner agent.
