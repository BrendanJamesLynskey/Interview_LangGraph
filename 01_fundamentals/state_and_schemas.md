# State and Schemas

> TypedDict, Pydantic, input/output schemas, private state

---

### Q1. What types can be used to declare a graph's state schema?

**Answer:**

Three options:

1. **`TypedDict`** (most common) — lightweight, no runtime validation.
2. **Pydantic `BaseModel`** — runtime validation, defaults, custom serialisation.
3. **`dataclass`** — pure-Python, mutable, no validation.

```python
from typing import TypedDict
class StateA(TypedDict):
    foo: str

from pydantic import BaseModel
class StateB(BaseModel):
    foo: str = ""

from dataclasses import dataclass
@dataclass
class StateC:
    foo: str = ""
```

Pick Pydantic when you want LangGraph to **reject malformed updates** at the channel boundary; pick TypedDict for speed and simplicity.

---

### Q2. How do you declare a reducer on a channel?

**Answer:**

Use `typing.Annotated` with the reducer as the second argument:

```python
from typing import Annotated, TypedDict
import operator
from langgraph.graph.message import add_messages

class State(TypedDict):
    counter: int                                    # default: overwrite
    log: Annotated[list[str], operator.add]         # concatenate
    messages: Annotated[list, add_messages]         # message-aware merge
```

The reducer is a function `(current, update) -> merged`. It runs whenever a node returns a value for that channel.

---

### Q3. What is the difference between the input schema, the state schema, and the output schema?

**Answer:**

By default the same schema is used for all three. You can specialise them:

```python
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(TypedDict):
    question: str
    answer: str
    intermediate: list[str]   # private, not visible at boundaries

builder = StateGraph(OverallState, input=InputState, output=OutputState)
```

- **Input schema** filters what the caller may pass in.
- **Output schema** filters what `invoke` returns.
- **Overall schema** is what nodes see and write — it can include private channels not exposed externally.

---

### Q4. What is "private state" in LangGraph?

**Answer:**

Private state refers to channels in the **overall schema** that are not present in the **input** or **output** schemas. Nodes can read and write them freely, but they are invisible to external callers.

It is also possible for a single node (or subgraph) to declare a **private input schema** unique to that node, letting it accept fields that are not declared on the parent state at all. This is useful when an internal node needs scratch space without polluting the global schema.

---

### Q5. What does the default channel reducer do?

**Answer:**

The default reducer is **last-write-wins overwrite**. If a node returns `{"foo": 5}`, the new value of `foo` is `5`, regardless of what was there before. If two parallel nodes both write to `foo` in the same superstep, an `InvalidUpdateError` is raised.

To accept concurrent writes, declare an explicit reducer (e.g. `operator.add` for lists, `operator.or_` for sets, or a custom merge function).

---

### Q6. What does `add_messages` do that `operator.add` does not?

**Answer:**

`add_messages` (from `langgraph.graph.message`) is purpose-built for chat history:

1. **Coerces** dict / `BaseMessage` inputs into `BaseMessage` subclasses.
2. **Deduplicates / replaces** messages with matching `id` fields rather than blindly appending.
3. **Supports `RemoveMessage`** — yielding a `RemoveMessage(id=...)` deletes that message from history.

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, RemoveMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Update with id replaces existing message of same id
{"messages": [HumanMessage(id="m1", content="updated")]}

# Remove a message
{"messages": [RemoveMessage(id="m1")]}
```

`operator.add` would naively append, leading to duplicated or stale messages.

---

### Q7. How do you write a custom reducer?

**Answer:**

A reducer is any callable `(current, update) -> merged`. For example, a "max" reducer that keeps the larger of two integers:

```python
def keep_max(current: int | None, update: int) -> int:
    if current is None:
        return update
    return max(current, update)

class State(TypedDict):
    high_water: Annotated[int, keep_max]
```

Reducers must be **pure** and **deterministic** — they can be invoked multiple times during checkpoint replay.

---

### Q8. Can state values be mutable objects, and is in-place mutation safe?

**Answer:**

You can store mutable objects (lists, dicts, custom classes), but **in-place mutation inside a node is unsafe**. The reducer compares the returned value to the prior state to compute the diff that gets checkpointed; mutating the input means the "before" snapshot already reflects the change, so the diff is empty and the checkpointer will not record the update.

Always return a **new object** from a node:

```python
# WRONG
def bad(state):
    state["log"].append("x")
    return {"log": state["log"]}

# RIGHT
def good(state):
    return {"log": state["log"] + ["x"]}
```

This is one of the most common bugs in early LangGraph code.

---

### Q9. How does Pydantic validation interact with state updates?

**Answer:**

When the state schema is a Pydantic `BaseModel`, every channel update is validated against the model on merge. A node returning a value that fails validation raises a `ValidationError` immediately, before the partial is committed.

Trade-off: Pydantic validation has measurable overhead. For high-throughput agents, prefer TypedDict; for user-facing applications where malformed updates would corrupt downstream state, prefer Pydantic.

---

### Q10. How do you reset a list channel to empty mid-graph?

**Answer:**

With `operator.add` you cannot — the reducer only concatenates. Use a **custom reducer** that interprets a sentinel (or recognises a particular update shape) as a reset:

```python
def add_or_reset(current: list, update: list | None) -> list:
    if update is None:
        return []
    return (current or []) + update

# Node returning {"items": None} clears the list.
```

For messages, `RemoveMessage(id=msg.id)` for each message in `state["messages"]` clears the channel.

---

### Q11. What is the difference between `MessagesState` and rolling your own?

**Answer:**

`langgraph.graph.MessagesState` is a prebuilt TypedDict containing exactly one channel:

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

It saves boilerplate for the common case of a chat agent. You can subclass it to add extra channels:

```python
class State(MessagesState):
    user_id: str
    retrieved_docs: list[str]
```

For non-chat graphs (data pipelines, planners that don't track messages), define your own schema directly.

---

### Q12. How do you inspect the current state of a graph at any point?

**Answer:**

If a checkpointer is attached, `graph.get_state(config)` returns a `StateSnapshot` containing the channel values, the next node(s) scheduled to run, the configurable values, and metadata (step number, timestamp, parent checkpoint).

```python
config = {"configurable": {"thread_id": "1"}}
snapshot = graph.get_state(config)
print(snapshot.values)        # current state
print(snapshot.next)          # tuple of upcoming nodes
print(snapshot.metadata)      # CheckpointMetadata
```

`graph.get_state_history(config)` yields all historical snapshots — the foundation of time travel.
