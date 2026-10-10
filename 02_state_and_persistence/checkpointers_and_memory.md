# Checkpointers and Memory

> `MemorySaver`, SQLite/Postgres, threads, short-term vs long-term memory, time travel

---

### Q1. What is a checkpointer and what problem does it solve?

**Answer:**

A checkpointer is a pluggable persistence layer that **saves the full state of the graph after every superstep**. It enables:

- **Resumption** — interrupt a graph (HITL approval, crash) and resume exactly where it stopped.
- **Conversation memory** — chat threads survive across separate `invoke` calls.
- **Time travel** — replay or fork from any historical step.
- **Auditability** — inspect every intermediate state for debugging or compliance.

Without a checkpointer, each `invoke` is stateless: prior runs are forgotten the moment the call returns.

---

### Q2. What checkpointer implementations ship with LangGraph?

**Answer:**

| Implementation | Package | Use case |
|----------------|---------|----------|
| `MemorySaver` | `langgraph.checkpoint.memory` | Tests, demos, ephemeral notebooks |
| `SqliteSaver` | `langgraph-checkpoint-sqlite` | Local single-process apps, prototyping |
| `PostgresSaver` | `langgraph-checkpoint-postgres` | Production multi-process / multi-host |
| Async variants | `AsyncSqliteSaver`, `AsyncPostgresSaver` | Async event loops |

You can also implement `BaseCheckpointSaver` against any KV store (Redis, DynamoDB, Mongo).

---

### Q3. What is a "thread" and how is it identified?

**Answer:**

A thread is a **logical conversation or workflow instance** — all checkpoints sharing the same `thread_id` belong to one continuous history. The thread ID is passed via `config["configurable"]["thread_id"]`:

```python
config = {"configurable": {"thread_id": "user-42-session-7"}}

graph.invoke({"messages": [HumanMessage("hi")]},  config)
graph.invoke({"messages": [HumanMessage("again")]}, config)  # continues
```

The first call creates a new thread; subsequent calls with the same `thread_id` resume from the latest checkpoint of that thread.

---

### Q4. What is the difference between short-term and long-term memory in LangGraph?

**Answer:**

- **Short-term memory** = checkpointer state. Scoped to a single thread, contains the message history and channel values for that conversation. Loaded automatically when you invoke with a known `thread_id`.

- **Long-term memory** = the **`Store`** (e.g. `InMemoryStore`, `PostgresStore`). Scoped by **namespaces** that can span threads, users, organisations. Used for facts the agent must remember across separate conversations (user preferences, learned skills, vector-indexed semantic recall).

```python
from langgraph.store.memory import InMemoryStore
store = InMemoryStore(index={"embed": embeddings, "dims": 1536})
graph = builder.compile(checkpointer=saver, store=store)
```

Inside a node, access the store via the injected `BaseStore` parameter (`store.put`, `store.search`).

---

### Q5. How does time travel work?

**Answer:**

With a checkpointer, each superstep produces a checkpoint identified by a `checkpoint_id`. You can:

1. **List history**:
   ```python
   for snap in graph.get_state_history(config):
       print(snap.config, snap.metadata["step"])
   ```

2. **Fetch a specific checkpoint**:
   ```python
   past_config = {**config, "configurable": {**config["configurable"], "checkpoint_id": "..."}}
   snap = graph.get_state(past_config)
   ```

3. **Resume from that point**:
   ```python
   graph.invoke(None, past_config)  # None means: continue from checkpoint
   ```

4. **Fork** by editing state at the checkpoint and resuming:
   ```python
   graph.update_state(past_config, {"messages": [...]})
   graph.invoke(None, past_config)  # creates a new branch
   ```

---

### Q6. Why does `invoke(None, config)` resume execution?

**Answer:**

Passing `None` as input signals "no new input — load the latest (or specified) checkpoint and continue from the next scheduled node." This pattern is used to:

- Resume after a HITL interrupt.
- Re-run from a forked checkpoint.
- Continue a graph that was paused for any reason.

If you instead pass a real input dict, LangGraph treats it as a **new invocation** that builds on top of the stored state — the input dict is merged into the state via the channels' reducers, then execution continues.

---

### Q7. How do you edit state at a specific point in history?

**Answer:**

`update_state` writes a partial state update as if a node had returned it, creating a new checkpoint:

```python
graph.update_state(
    config,                       # which thread (and optional checkpoint_id)
    {"messages": [HumanMessage("forced edit")]},
    as_node="agent",              # attribute the update to this node
)
```

`as_node` controls which conditional edges fire next. This is the mechanism behind both **HITL approval workflows** and **forking** for what-if analysis.

---

### Q8. What is a `CheckpointMetadata` record?

**Answer:**

Each checkpoint is annotated with metadata captured by the saver:

```python
{
    "source": "loop" | "input" | "update" | "fork",
    "step": 0,                # superstep index, -1 for initial input
    "writes": {...},          # the partial updates applied
    "parents": {...},         # parent checkpoint references
}
```

`source` distinguishes user-initiated updates from internal step writes. This metadata is essential when filtering history (`graph.get_state_history(config, filter={"source": "update"})`).

---

### Q9. How do checkpointers handle large state?

**Answer:**

Checkpointers serialise the entire channel value on every change, so very large blobs (long documents, big tensors) become a write-amplification problem. Mitigations:

1. **Store large blobs externally** (object store, vector DB) and keep only references in state.
2. **Trim aggressively** with reducers like `trim_messages` or sliding windows.
3. **Use Postgres `JSONB`** with table partitioning and TOAST compression.
4. **Use the long-term `Store`** for retrieval-style access patterns rather than carrying the data in the live graph state.

---

### Q10. Can two processes safely write to the same Postgres-backed thread?

**Answer:**

`PostgresSaver` uses row-level locking and version checks: if two processes attempt concurrent writes to the same `(thread_id, checkpoint_ns)` they serialise via the database. However, having two graph instances actively executing the same thread is a logical race — one will overwrite the other's progress.

Pattern: have a single worker (or queue consumer) own a thread at a time. Use a `thread_id` allocation scheme (per user, per session) that naturally prevents collisions.

---

### Q11. How does `MemorySaver` differ from `SqliteSaver` operationally?

**Answer:**

| Aspect | `MemorySaver` | `SqliteSaver` |
|--------|---------------|---------------|
| Persistence | Lost on process exit | Survives restarts |
| Concurrency | Single-process only | Single-host, OK for low concurrency |
| Setup | `MemorySaver()` | `SqliteSaver.from_conn_string("checkpoints.db")` |
| Suitable for | Tests, notebooks | Local apps, prototypes |

For multi-process or multi-host production, jump straight to `PostgresSaver`.

---

### Q12. What are checkpoint namespaces?

**Answer:**

A checkpoint is keyed by `(thread_id, checkpoint_ns, checkpoint_id)`. The **namespace** is an internal sub-key used by LangGraph to keep **subgraph state separate** from parent state — each subgraph invocation gets its own namespace string so its checkpoints do not collide with the outer graph's.

You rarely set namespace explicitly; it is managed automatically when using `add_node(name, subgraph)` patterns.

---

### Q13. How does `get_state_history` differ from `get_state`?

**Answer:**

- `get_state(config)` returns one snapshot — the checkpoint specified by `checkpoint_id`, or the latest if none is given.
- `get_state_history(config, before=None, limit=None, filter=None)` yields a stream of snapshots in **reverse chronological order**, optionally filtered by metadata or capped to `limit`.

History is invaluable for debugging: walk backwards from the failure to find the exact superstep where state went wrong.

---

### Q14. How can the long-term `Store` be used for semantic memory?

**Answer:**

Configure the store with an embedding model and dimensionality, then `put` records under namespaces and `search` semantically:

```python
store = InMemoryStore(index={"embed": embeddings_model, "dims": 1536})

# In a node:
def remember(state, *, store: BaseStore):
    store.put(("user", state["user_id"], "facts"), key="hometown", value={"text": "Leeds"})

def recall(state, *, store: BaseStore):
    hits = store.search(("user", state["user_id"], "facts"), query=state["question"], limit=3)
    return {"context": [h.value["text"] for h in hits]}
```

Items in the store are namespaced tuples, which makes per-user / per-org isolation explicit at the API level.

---

### Q15. What is the `durability` option for, and when do you use it?

**Answer:**

`durability` (on `invoke` / `stream`) controls **when** checkpoints are written. In LangGraph 1.x it replaces the older `checkpoint_during` flag, which is deprecated (`checkpoint_during=True` maps to `"async"`, `False` to `"exit"`, and passing both raises an error):

| Mode | When the checkpoint is written |
|------|--------------------------------|
| `"sync"` | after every superstep, before the next one starts (safest, slowest) |
| `"async"` (default) | after every superstep, while the next one is already running |
| `"exit"` | only when the run exits: on completion, an interrupt or an error |

Use `"exit"` when:

- The graph runs as a single atomic batch where partial replay is undesirable.
- You want to reduce write amplification on a high-throughput pipeline.

The trade-off: the thread keeps only the final checkpoint of that run, so there is no time travel within it. A node that raises is still saved on exit (you can resume from the failed step), but a hard process crash mid-run loses all of that run's progress. Use `"sync"` when every superstep must be durable before the next begins (long, expensive runs on infrastructure that can die). Checked against LangGraph 1.2.14 (`Durability` in `langgraph/types.py`, the deprecation in `pregel/main.py`).
