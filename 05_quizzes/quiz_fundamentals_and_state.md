# Quiz — Fundamentals and State

> Graphs, channels, reducers, checkpointing

---

### Q1. What is the default reducer for a state channel and what happens if two parallel nodes write to it?

**Answer:**

The default is **last-write-wins overwrite**. Two concurrent writes in the same superstep raise `InvalidUpdateError` because the framework cannot deterministically merge them. To allow parallel writes, declare a reducer such as `operator.add` or a custom merge function.

---

### Q2. Why must reducers be pure and avoid mutating their inputs?

**Answer:**

Reducers run during normal execution **and** during checkpoint replay / forking. Mutating inputs corrupts historical snapshots, breaks time travel, and can cause non-deterministic behaviour on retries. Always return a new value.

---

### Q3. What does `START` mean in a `StateGraph`?

**Answer:**

`START` is a virtual node representing the entry point. Every graph must have at least one edge from `START` to a real node, otherwise compilation fails.

---

### Q4. Name three checkpointer implementations and the typical use case for each.

**Answer:**

- `MemorySaver` — tests, demos, ephemeral notebook sessions.
- `SqliteSaver` — local single-process apps and prototypes.
- `PostgresSaver` — production multi-process / multi-host deployments.

---

### Q5. What is the role of `thread_id` in `config["configurable"]`?

**Answer:**

It identifies a **conversation or workflow instance**. All checkpoints with the same `thread_id` share a continuous history; reusing the id resumes the latest state of that thread.

---

### Q6. How does `add_messages` differ from `operator.add` when used as a reducer for a list of messages?

**Answer:**

`add_messages` coerces inputs to `BaseMessage`, replaces existing messages with matching `id`, and supports `RemoveMessage` to delete entries. `operator.add` blindly concatenates, leading to duplicates and stale entries.

---

### Q7. What is a "superstep"?

**Answer:**

One iteration of the Pregel-style execution loop: all currently scheduled nodes run concurrently, their updates are merged via reducers, and the next set of nodes is scheduled. Each superstep produces (at most) one checkpoint.

---

### Q8. What is the purpose of the `recursion_limit`?

**Answer:**

It caps the number of **supersteps** in a single invocation (default 25 in LangGraph 0.x, 10007 in 1.2.14; set it explicitly), guarding against infinite cyclic loops in agentic graphs. Hitting the limit raises `GraphRecursionError`.

---

### Q9. How does the input schema differ from the overall state schema?

**Answer:**

The input schema declares which fields a caller may pass in via `invoke`. The overall schema is what nodes see and write — it can include private channels not exposed at the input boundary. Likewise, the output schema filters what is returned.

---

### Q10. What does `graph.update_state(config, partial, as_node="X")` do?

**Answer:**

It writes a new checkpoint as if node `X` had returned `partial`. Routing then proceeds as it would after node `X`. This is the foundation of HITL state edits and of forking from historical checkpoints.

---

### Q11. Why does `invoke(None, config)` resume execution?

**Answer:**

`None` signals "no new input — load the latest checkpoint and continue from the next scheduled node." Used to resume after an interrupt or to continue from a forked checkpoint.

---

### Q12. What is the difference between short-term memory (checkpointer) and long-term memory (`Store`)?

**Answer:**

The checkpointer holds per-thread state — message history and channel values for one conversation. The `Store` holds cross-thread, namespace-scoped data used for facts that persist across separate conversations (user preferences, semantic recall). Both can coexist on the same compiled graph.

---

### Q13. A node returns `{}`. What happens?

**Answer:**

The node still ran (any side effects executed) but no channels are updated. The graph proceeds to the next scheduled node based on edges and conditional routes.

---

### Q14. Inside a Pydantic-state graph, a node returns a value that fails validation. What happens?

**Answer:**

A `ValidationError` is raised immediately during channel merge, before the partial is committed. Pydantic state is the recommended choice when malformed updates would corrupt downstream behaviour.

---

### Q15. Name two ways to inspect the current state of a thread.

**Answer:**

`graph.get_state(config)` returns a `StateSnapshot` for the latest (or specified) checkpoint. `graph.get_state_history(config)` yields all historical snapshots in reverse chronological order.

---

### Q16. Why might you set `durability="exit"` on `invoke`?

**Answer:**

To skip the per-superstep checkpoint writes and persist only when the run exits (on completion, an interrupt or an error). Reduces storage I/O for high-throughput pipelines but loses time-travel resolution within that run, and a hard process crash loses the run's progress. (`durability` replaces the deprecated `checkpoint_during=False` in LangGraph 1.x; the default is `"async"`.)

---

### Q17. Two parallel branches both want to write a single string to one channel. How do you redesign?

**Answer:**

Either route both branches into a join node that picks one (cleanest), or promote the channel to `Annotated[list[str], operator.add]` and let downstream code combine them. Avoid hacky "first-write-wins" reducers as they hide modelling problems.

---

### Q18. What does `compile()` validate?

**Answer:**

That every referenced node exists, the graph is reachable from `START`, channel reducers are consistent, and any attached checkpointer/interrupts are correctly configured. Returns an immutable, runnable `CompiledStateGraph`.
