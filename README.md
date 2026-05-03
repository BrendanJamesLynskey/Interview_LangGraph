# LangGraph — Interview Preparation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Frameworks-blue.svg)](https://langchain-ai.github.io/langgraph/)

A focused collection of notes, questions, and worked examples covering LangGraph — the graph-based orchestration framework for stateful, multi-actor LLM applications. Each file is self-contained and designed for active recall: read the question, attempt an answer, then check the detailed explanation below it.

---

## Table of Contents

| # | Section | Topics |
|---|---------|--------|
| 01 | [Fundamentals](#01-fundamentals) | Graphs, nodes, edges, state, compilation |
| 02 | [State and Persistence](#02-state-and-persistence) | Channels, reducers, checkpointers, memory, time travel |
| 03 | [Tools and Agents](#03-tools-and-agents) | Tool calling, ReAct, prebuilt agents, structured output |
| 04 | [Advanced Patterns](#04-advanced-patterns) | Multi-agent, subgraphs, streaming, human-in-the-loop, Send API |
| 05 | [Quizzes](#05-quizzes) | Mixed-topic self-assessment |

---

## 01 Fundamentals

- [Graph Basics](01_fundamentals/graph_basics.md) — `StateGraph`, nodes, edges, START/END, compilation
- [State and Schemas](01_fundamentals/state_and_schemas.md) — TypedDict, Pydantic, input/output schemas, private state

## 02 State and Persistence

- [Channels and Reducers](02_state_and_persistence/channels_and_reducers.md) — Default overwrite, `add_messages`, `operator.add`, custom reducers
- [Checkpointers and Memory](02_state_and_persistence/checkpointers_and_memory.md) — `MemorySaver`, SQLite/Postgres, threads, short-term vs long-term memory, time travel

## 03 Tools and Agents

- [Tool Calling](03_tools_and_agents/tool_calling.md) — `@tool`, `ToolNode`, `bind_tools`, parallel tool calls, error handling
- [Prebuilt Agents](03_tools_and_agents/prebuilt_agents.md) — `create_react_agent`, structured responses, custom prompts

## 04 Advanced Patterns

- [Multi-Agent Architectures](04_advanced_patterns/multi_agent.md) — Supervisor, swarm, hierarchical teams, handoffs via `Command`
- [Subgraphs and Send API](04_advanced_patterns/subgraphs_and_send.md) — Composition, isolated state, dynamic fan-out with `Send`, map-reduce
- [Streaming and Human-in-the-Loop](04_advanced_patterns/streaming_and_hitl.md) — `astream`, stream modes, `interrupt`, approval gates, state edits

## 05 Quizzes

- [Quiz — Fundamentals and State](05_quizzes/quiz_fundamentals_and_state.md) — Graphs, channels, reducers, checkpointing
- [Quiz — Agents and Advanced Patterns](05_quizzes/quiz_agents_and_advanced.md) — Tools, multi-agent, streaming, HITL

---

## How to Use

1. **Pick a section** that matches the topic you need to revise.
2. **Read each question**, pause, and formulate your own answer before scrolling down.
3. **Compare** your answer with the detailed explanation provided.
4. **Run the code examples** in a Python interpreter with `pip install langgraph langchain` to build muscle memory.
5. **Attempt the quizzes** in `05_quizzes/` to test yourself under light time pressure.

## Related Repositories

| Repository | Description |
|------------|-------------|
| [Interview_Agentic_Frameworks](https://github.com/BrendanJamesLynskey/Interview_Agentic_Frameworks) | Broader survey of LangGraph, LangChain, CrewAI |
| [Interview_Agentic_AI](https://github.com/BrendanJamesLynskey/Interview_Agentic_AI) | Agent design patterns, planning, evaluation |
| [Interview_Agentic_Engineering](https://github.com/BrendanJamesLynskey/Interview_Agentic_Engineering) | Production engineering for agentic systems |
| [Interview_MCP](https://github.com/BrendanJamesLynskey/Interview_MCP) | Model Context Protocol — tool/resource servers for agents |

## Contributing

Contributions are welcome. Please open an issue or submit a pull request if you spot an error or want to add new content.

## License

This project is licensed under the [MIT License](LICENSE).
