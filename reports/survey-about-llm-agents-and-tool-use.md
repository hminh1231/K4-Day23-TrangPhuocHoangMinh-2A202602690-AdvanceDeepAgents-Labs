# LLM Agents and Tool Use: A Survey

## TL;DR
- LLM agents are systems that observe environments, plan, and act; tool use is the explicit invocation of external capabilities (e.g., APIs, code execution), and common architectures split perception, planning, memory, and execution [1][2][3][4].
- Architectures range from ReAct-style thought–act–observe loops to planner–executor and bilevel graph-driven systems; recent work focuses on efficient action-space search, toolkit-level planning, and decoupled tool-selection layers [5][6][7][8].
- Benchmarks and evaluations have shifted toward trajectory-aware and execution-grounded metrics (EM, Traj-satisfy, invocation accuracy, tool-selection accuracy, parameter name F1, MRR/NDCG) to capture failures in the tool-invocation trajectory [9][10][11][12].
- Safety risks include prompt injection, malicious tools, and privilege misuse; defenses include step-level guardrails (TS-Guard/TS-Flow), hybrid static/dynamic vetting (ToolGuardian, MTGuard), sandboxing, and red-teaming frameworks [13][14][15][16].

## Background
Definition and why it matters

LLM-based agents are defined consistently across recent surveys as systems that "observe their environment, make decisions, and take actions." They are framed around modular components such as profile/brain, memory, planning, and action execution; some formalizations present agent state as a structured quintuple with elements like LLM, Objective, Memory, Action and Rethink [1][2][3][4].

Foundational work and common paradigms

Three prominent paradigms in building agents are tool use (including RAG), planning, and feedback learning. ReAct and Reflexion appear as frequently cited single-agent frameworks, and tool invocation workflows are often categorized into In-Generation Triggers, Reasoning-Acting Strategy, and Confidence-Based Invocation [2][3].

## Architectures and methods

Agent architectures fall into two broad design choices: agent-driven control flow, where the LLM decides each step's control flow (e.g., ReAct loops), and predefined-code workflows, where control flow is handled by orchestrating code and the LLM is used for specific decisions (e.g., planner–executor) [17][5].

Decoupling capability identification and tool implementation

Several sources advocate a two-stage selection: first identify an abstract capability, then map it to a concrete tool instance from a registry, using static tags or dynamic prompts; this reduces tool-selection errors and supports graceful fallback when APIs fail [18][8].

Search and planning techniques

Best-first and A* search applied to tool-call trees (ToolChain∗) and graph-driven bilevel planning (NaviAgent) are proposed to reduce search cost in large action spaces and improve success on complex tasks; toolkit-level clustering (Tool-Planner) similarly reduces instability by searching toolkits rather than raw APIs [6][7][8].

Dual-process and switch strategies

Hybrid systems pairing a fast policy with a slow deliberative model can gate deliberation based on uncertainty or failure signals; System Switch studies when to hand control from the fast model to the slow model in real-time and turn-based settings [19].

## Evaluation and benchmarks

Trajectory-aware evaluation

Recent benchmarks (TRAJECT-Bench, MCP-Bench) and unified protocols like CUBE emphasize trajectory logging, executable tool integration, and metrics that measure both per-step correctness (Invocation Accuracy, Tool Selection Accuracy, Parameter F1) and final task success (Acc, EM). These benchmarks argue that execution-grounded metrics reveal failures that output-only scoring misses [10][11][20][9].

Protocol validity and anti-gaming

Studies like "Do Agent Benchmarks Measure Capability?" and methods such as HackDetect highlight reward-hacking and exposure-based inflation, suggesting that benchmark design must consider protocol validity, adversarial exposure, and synthetic overfitting[21][21].

Practical evaluation guidance

LangChain recommends combining reference-based, reference-free, LLM-as-judge, code-based evaluators, and human annotation, and stresses that application-level evals should target the system's own data and failure modes rather than relying solely on public benchmarks [12].

## Safety, reliability, and risks

Classes of safety failures

Tool-using agents introduce new vectors: prompt injection that alters plans, malicious/Trojan tools uploaded to repositories that execute harmful actions when selected, and privilege misuse where tools act with overbroad rights [22][13].

Defensive layers and runtime controls

Proposed defenses include proactive step-level guardrails (TS-Guard and TS-Flow) that reduce unsafe tool invocations and improve benign task completion under prompt injection [13]; ToolGuardian's layered vetting and runtime authorisation encodes declarative policies for admission and runtime constraints [14]; MTGuard combines contextual inspection with dynamic analysis and post-execution verification [15]. System-level sandboxing guidance recommends strict OS-level egress and filesystem controls and virtualization isolation to limit damage from tool execution [16].

Red-teaming and measurement

Executable red-teaming frameworks like REDAgentBench and MalTool demonstrate that adversarial tool uploads and interactions are feasible at scale and must be part of evaluation and defense pipelines [23][22].

## Trends and open problems

Recent changes (last two years)

The field has moved toward trajectory-aware, execution-grounded evaluation (TRAJECT-Bench, MCP-Bench) and toward declarative and hybrid vetting strategies for tool safety (ToolGuardian, MTGuard) as responses to demonstrated malicious-tool attacks and prompt injection [10][11][14][15][22].

Open problems

- Standardising tool registries, schemas, and runtime contracts to reduce selection errors and enable provable safety checks [8][18].
- Robustness to malicious or Trojan tools remains unsolved; current vetting methods balance false positives and negatives, and dynamic runtime behaviours can evade static analysis [22][14][15].
- Benchmark validity and anti-gaming: exposure and synthetic strategies can inflate scores, so we need better red-teaming, holdout protocols, and executable testbeds [21][23][9].

## References
[1] From language to action: a review of large language models as autonomous agents and tool users | Artificial Intelligence Review | Springer Nature Link. web. https://link.springer.com/article/10.1007/s10462-025-11471-9 (2026-01-06)
[2] A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning. web. https://arxiv.org/html/2406.05804v5 (N/A)
[3] Large Language Model Agent: A Survey on Methodology, Applications and Challenges. web. https://arxiv.org/html/2503.21460 (N/A)
[4] From Question Answering to Task Completion: A Survey on Agent System and Harness Design. hf-search. https://huggingface.co/papers/2606.20683 (2026-06-14)
[5] Plan-and-Execute Agents - LangChain. web. https://www.langchain.com/blog/planning-agents (2024-02-13)
[6] ToolChain∗: Efficient Action Space Navigation in Large Language Models with A∗ Search. web. https://ar5iv.labs.arxiv.org/html/2310.13227 (unknown)
[7] NaviAgent: Graph‑Driven Bilevel Planning for Scalable Tool Orchestration. web. https://arxiv.org/html/2506.19500 (unknown)
[8] Tool-Planner: Task Planning with Clusters across Multiple Tools. web. https://arxiv.org/pdf/2406.03807v4.pdf (unknown)
[9] Evaluation and Benchmarking of LLM Agents: A Survey - arXiv. web. https://arxiv.org/html/2507.21504v1 (unknown)
[10] TRAJECT-Bench: A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use. web. https://exa.ai/library/publication/1rwb8gwkl59 (2025-10-06)
[11] MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers. web. https://exa.ai/library/publication/n1tngp5py72 (2025-08-28)
[12] Evaluating LLMs and Agents: Benchmarks, Evals & Guardrails (LangChain). web. https://www.langchain.com/resources/how-to-evaluate-llms (2026-06-23)
[13] ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback. web. https://arxiv.org/html/2601.10156 (N/A)
[14] ToolGuardian: Declarative Security for AI Agent-Tool Interactions. web. https://arxiv.org/pdf/2607.21835.pdf (unknown)
[15] Hybrid Analysis for Secure MCP Tool Use in LLM Agents (MTGuard). web. https://arxiv.org/html/2607.25297v1 (2026-07-28)
[16] Practical Security Guidance for Sandboxing Agentic Workflows and Managing Execution Risk (NVIDIA AI Red Team). web. https://developer.nvidia.com/blog/practical-security-guidance-for-sandboxing-agentic-workflows-and-managing-execution-risk/ (2026-01-30)
[17] Anthropic's Effective Agents Framework: A Pattern Map. web. https://agentpatterns.ai/patterns/agent-design/anthropic-effective-agents-framework (unknown)
[18] Tool Selection by Large Language Model (LLM) Agents (TDCommons). web. https://www.tdcommons.org/cgi/viewcontent.cgi?article=9446&context=dpubs_series (unknown)
[19] System Switch: When Should a Fast Decision Model Stop and Think? (hf-daily). hf-daily. https://huggingface.co/papers/2610.09683 (2026-10-07)
[20] CUBE: A Standard for Unifying Agent Benchmarks. hf-search. https://huggingface.co/papers/2603.15798 (2026-03-16)
[21] Do Agent Benchmarks Measure Capability? Protocol Validity in the Age of Agentic AI. hf-search. https://huggingface.co/papers/2607.22368 (2026-07-24)
[22] MalTool: Malicious Tool Attacks on LLM Agents - Berkeley RDI. web. https://rdi.berkeley.edu/blog/maltool/ (2026-03-03)
[23] REDAgentBench: Executable Red Teaming and Faithful Measurement of LLM Agent Systems. hf-search. https://huggingface.co/papers/2608.10669 (2026-08-11)
