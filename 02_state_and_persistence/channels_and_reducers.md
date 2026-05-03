# Channels and Reducers

> Default overwrite, `add_messages`, `operator.add`, custom reducers

---

### Q1. What is a "channel" in LangGraph terminology?

**Answer:**

A channel is a **named, typed slot** in the graph's state. Each key of the state TypedDict (or each Pydantic field) is one channel. Channels are the unit of:

- **Versioning** — every checkpoint records each channel's version.
- **Update merging** — each channel has a reducer.
- **Subscription** — a node "reads" the channels it depends on and "writes" the channels it returns.

Internally LangGraph borrows the channel concept from the Pregel paper: independent message-passing slots between supersteps.

---

### Q2. What are the built-in reducer choices and when do you pick each?

**Answer:**

| Reducer | Use when |
|---------|----------|
| (default) overwrite | Single value, single writer per superstep |
| `operator.add` | Lists you want to concatenate |
| `operator.or_` | Sets you want to union |
| `add_messages` | Chat history with id-aware merging and `RemoveMessage` support |
| Custom callable | Anything else (max, min, dict merge, dedup, …) |

Pick the most restrictive option that still works — defaulting to `operator.add` everywhere creates ever-growing lists and silently masks duplicate writes.

---

### Q3. Why does fan-out / fan-in require non-overwrite reducers?

**Answer:**

When the graph fans out (one node → many parallel branches → one merge node), each branch returns a partial state update for the merge. With the default reducer, two branches both writing to channel `result` would conflict; LangGraph raises `InvalidUpdateError`.

A reducer that defines an associative merge (concatenation, union, max) lets the framework combine the branches deterministically:

```python
class State(TypedDict):
    findings: Annotated[list[dict], operator.add]

def search_web(state):    return {"findings": [{"src": "web", ...}]}
def search_db(state):     return {"findings": [{"src": "db",  ...}]}
def synthesise(state):    return {"summary": summarise(state["findings"])}
```

Both searchers can run in parallel; their `findings` lists are concatenated before `synthesise` sees them.

---

### Q4. Walk through what happens internally when a node returns `{"messages": [HumanMessage("hi")]}` with the `add_messages` reducer.

**Answer:**

1. The node finishes; LangGraph collects its return value.
2. For each key in the returned dict, the framework looks up the channel's reducer.
3. `add_messages(current_messages, [HumanMessage("hi")])` is invoked.
4. It coerces any dicts in the update to `BaseMessage` instances.
5. It scans for messages whose `id` matches an existing message — those replace, others append.
6. `RemoveMessage` instances delete matching ids.
7. The result becomes the new value of `messages`.
8. The channel's version is bumped; the checkpointer (if any) records the new value.

---

### Q5. How can a custom reducer enforce a sliding window of the last N messages?

**Answer:**

```python
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

WINDOW = 20

def windowed(current: list[BaseMessage], update: list) -> list[BaseMessage]:
    merged = add_messages(current, update)
    return merged[-WINDOW:]

class State(TypedDict):
    messages: Annotated[list[BaseMessage], windowed]
```

Caveat: LangGraph reducers are applied per superstep, so transient bursts within a single superstep are still all merged before truncation. For more sophisticated trimming, prefer running a dedicated `trim_messages` node in the graph rather than encoding the policy in the reducer.

---

### Q6. What is wrong with the following reducer?

```python
def merge(current: list, update: list) -> list:
    current.extend(update)
    return current
```

**Answer:**

It **mutates the input**. LangGraph relies on the reducer producing a **new** value so the checkpointer can record the diff and so historical snapshots remain immutable. Mutating `current` corrupts older checkpoints, breaks time travel, and can cause subtle non-determinism on retries.

The correct form returns a fresh list:

```python
def merge(current: list, update: list) -> list:
    return [*(current or []), *update]
```

---

### Q7. How do reducers interact with checkpointing?

**Answer:**

Each channel is versioned independently. When a superstep ends:

1. Reducers compute new channel values from the partial updates.
2. For each changed channel, the version is incremented.
3. The checkpointer persists the channel value (typically using a `BaseCheckpointSaver.put`) along with version metadata.
4. On replay, the checkpointer loads each channel by version, and the graph resumes execution from the next scheduled node.

Because reducers are pure, replay is deterministic provided the node code itself is deterministic (or its non-determinism is captured in state).

---

### Q8. Can a node update a channel that is not in its declared input?

**Answer:**

Yes — by default a node sees the **full state** and may write any channel, even if it never reads it. LangGraph does not enforce a strict per-node input/output schema unless you opt in with subgraph private schemas.

This is convenient but can hide implicit coupling. For complex graphs, declaring per-node input/output schemas (or splitting into subgraphs) makes data flow explicit.

---

### Q9. How would you implement a "first-write-wins" reducer?

**Answer:**

```python
_SENTINEL = object()

def first_wins(current, update):
    if current is None or current is _SENTINEL:
        return update
    return current  # ignore subsequent writes
```

Use case: caching a value computed early in the graph so later parallel branches do not overwrite it.

A subtle alternative: store the value in a channel with the default reducer and gate writes by routing — only a single node ever runs that updates it. Reducer logic should be the *last* line of defence, not the primary control mechanism.

---

### Q10. What does a reducer signature look like for typed lists of Pydantic models?

**Answer:**

```python
from pydantic import BaseModel
from typing import Annotated, TypedDict
import operator

class Finding(BaseModel):
    source: str
    text: str

class State(TypedDict):
    findings: Annotated[list[Finding], operator.add]
```

`operator.add` works on lists of any element type. If you need uniqueness by `source`, write a custom reducer that dedupes:

```python
def merge_findings(current: list[Finding], update: list[Finding]) -> list[Finding]:
    seen = {f.source: f for f in (current or [])}
    for f in update:
        seen[f.source] = f
    return list(seen.values())
```

---

### Q11. Two parallel nodes both want to set `final_answer: str`. How do you resolve this?

**Answer:**

Several options, in increasing order of structure:

1. **Refactor the graph** — only one node should write the final answer; route the other branches into a join node that picks among them.
2. **Promote to a list** — change the channel to `Annotated[list[str], operator.add]` and let downstream code pick the best.
3. **Custom reducer** with a tie-breaker (e.g. longest, highest scoring).

Resorting to a "pick one of two" reducer often hides a genuine modelling problem in the graph topology. Prefer the refactor.

---

### Q12. How does LangGraph decide which nodes run in the next superstep?

**Answer:**

A node is scheduled for the next superstep if **any of its read channels were updated** in the current superstep, and there is an edge (static or conditional) leading to it. This is the **subscription** model from Pregel: nodes are activated by data availability, not just by static topology.

Combined with conditional edges, this means a node may sit idle for many supersteps and re-activate only when relevant state changes — useful for long-running agents that occasionally invoke specialised reviewers.
