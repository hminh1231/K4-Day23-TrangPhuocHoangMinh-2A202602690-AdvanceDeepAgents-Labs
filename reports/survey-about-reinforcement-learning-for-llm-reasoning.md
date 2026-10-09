# Reinforcement learning for LLM reasoning: a survey

## TL;DR

- Reinforcement learning (RL) has been shown to elicit chain-of-thought and stepwise reasoning under outcome-based and process-based reward schemes, but success depends critically on data distribution and reward design [1][2][3].
- Process Reward Models (PRMs) that score intermediate steps improve reranking and GSM8K accuracy, while Outcome Reward Models (ORMs) typically boost final-answer metrics on harder benchmarks like MATH; hybrid or trajectory-aware rewards often outperform either alone [4][5][6].
- Algorithm choice and credit assignment matter: specialized methods (VinePPO, Sequence-Level PPO, Reinforce-Rej, Expert Iteration) and improved credit assignment significantly improve training stability and long-horizon reasoning performance [5][7][8].
- Empirical gains are typically modest but task-dependent: reported improvements range from a few percentage points to ~12.8 pp in select experiments; sample-efficiency and compute remain major constraints [9][10][11].
- Open problems include reward hacking/deceptive alignment for generative RMs, scaling PRMs efficiently, and rigorous benchmarks for long-horizon agentic reasoning [12][11][13].

## Background

Reinforcement learning for LLM reasoning applies RL machinery — policies, rewards, environments — to improve models' multi-step problem solving and chain-of-thought generation. The RLHF pipeline (instruction-tuning, preference collection, reward modeling, RL optimization) is the standard training-time approach to align LLMs to human preferences and reasoning behavior [2]. RL can be applied at inference time as a navigator over reasoning actions (RLoT) or at training time with policy-gradient methods adapted to sequence generation [14][8].

Foundational theoretical work shows that outcome-based policy gradient can cause Transformers to develop interpretable iterative algorithms that implement reasoning strategies, but this emergence depends on having sufficient mass of "simple examples" in the training distribution to bootstrap longer chains [1].

## Architectures and methods

There are three families of methods for applying RL to LLM reasoning. First, standard RLHF-style training optimizes a policy against a reward model with policy gradients, often implemented with PPO or its variants; RLHF remains the dominant production approach to align LLM behavior [2][15]. Second, inference-time navigators (RL-of-Thought, RLoT) learn lightweight decision policies that select logical actions during chain-of-thought generation and use PRMs as single-step rewards [14]. Third, specialized credit-assignment and sequence-level methods (VinePPO, SPPO, Reinforce-Rej, Expert Iteration) adapt the RL algorithm to the long-horizon, structured nature of CoT traces by decoupling value functions or improving Monte Carlo credit assignment [5][7][8][9].

Comparative studies highlight that PPO is strong in many RLHF settings, but new methods sometimes outperform it on reasoning tasks: VinePPO reports refined credit assignment that beats PPO, Sequence-Level PPO reformulates long-chain reasoning as a contextual bandit to address instability, and Reinforce-Rej trades off simplicity for improved stability and sample efficiency [5][7][8]. Agentic RL frameworks (AGILE) combine memory, tools and expert consultation with PPO training to handle complex conversational or multi-step tasks [16].

## Rewards and evaluation

Reward design splits into outcome-based rewards (ORMs) that score final answer correctness and process-based rewards (PRMs) that provide step-level feedback on reasoning traces. Early experiments show PRMs can dramatically increase GSM8K accuracy (PRM-Max on GSM8K) while ORMs are more effective on MATH in some studies; aggregation method (max, mean, trajectory-aware) strongly affects results [3][4][6].

Generative reward models that produce CoT rationales followed by verdicts (Think-RM, GenPRM) provide richer supervision for long-horizon tasks and can reduce data requirements by scoring stepwise reasoning and enabling verification steps [11][4]. Trajectory-aware PRMs (ReasonFlux) and hybrid rewards that mix hard/verifiable signals with continuous shaping improve convergence and robustness against reward hacking [17][18]. RewardBench 2 evaluates hundreds of RMs across domains and finds high correlation (Pearson 0.87) between RM scores and downstream performance under BoN sampling, yet absolute RM performance is lower than prior benchmarks, indicating room for improvement [6].

Evaluation best practices include reporting final-answer accuracy (GSM8K, MATH, BigBench Hard), step-level correctness or faithfulness, and sensitivity to reward aggregation and adversarial inputs [9][6][19].

## Empirical results and benchmarks

Across multiple studies, RL-based interventions yield task-dependent improvements. DialCoT with PPO reports a 6.2% average improvement across four datasets when using FlanT5-XXL and a ~2% PPO-specific ablation gain [10]. MATH-SHEPHERD's step-by-step PPO with PRM raises Mistral-7B GSM8K accuracy from 77.9% to 84.1% and MATH from 28.6% to 33.0% in reported experiments [19]. Controlled studies find Expert Iteration can outperform PPO in sample efficiency and final performance on some reasoning tasks, while PPO with ORM guidance gives around 5% improvement over SFT baselines in some settings [9].

Computational costs and stability remain concerns: OPPO accelerates PPO-based RLHF training by up to 2.8× while preserving convergence, addressing practical bottlenecks for applying PPO at scale [20]. Results vary by reward structure: a 'hard' reward strategy achieved 40% final accuracy in one evaluation vs 28% for continuous rewards on GSM8K in another study [18].

## Trends and open problems

Recent two-year trends (2024–2026) show a move from simple outcome rewards to richer process-aware and generative reward models (Think-RM, GenPRM, ReasonFlux) and a focus on credit assignment and sequence-level RL (VinePPO, SPPO) to handle long CoT traces [11][5][7][4][17]. There is also growing interest in inference-time RL navigators (RLoT) and agentic world models to provide realistic training environments and evidence-grounded oversight for long-horizon agents [14][13][21].

Open problems include: preventing deceptive alignment and reward hacking in generative reward models, scalable training of PRMs for long contexts, rigorous metrics for step-level faithfulness, better benchmarks for learning-from-interaction (e.g., Learn2Play), and understanding when outcome-only RL can provably elicit reasoning versus when process supervision is needed [12][6][13][1].

## References
[1] Outcome-Based RL Provably Leads Transformers to Reason, but Only With the Right Data. web. https://ar5iv.labs.arxiv.org/html/2601.15158 (unknown)
[2] Reinforcement Learning from Human Feedback - arXiv. web. https://arxiv.org/html/2504.12501v2 (unknown)
[3] Let's Reinforce Step by Step. web. https://ar5iv.labs.arxiv.org/html/2311.05821 (unknown)
[4] GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning. hf-search. https://huggingface.co/papers/2504.00891 (2025-04-01)
[5] VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment. hf-search. https://huggingface.co/papers/2410.01679 (2024-10-02)
[6] RewardBench 2: Advancing Reward Model Evaluation. web. https://arxiv.org/html/2506.01937v2 (unknown)
[7] SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks. hf-search. https://huggingface.co/papers/2604.08865 (2026-04-10)
[8] A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce. hf-search. https://huggingface.co/papers/2504.11343 (2025-04-15)
[9] Teaching Large Language Models to Reason with Reinforcement Learning. arxiv. https://arxiv.org/html/2403.04642v1 (2024-03-07)
[10] DialCoT Meets PPO: Decomposing and Exploring Reasoning Paths in Smaller Language Models. web. https://exa.ai/library/publication/ygy742972br (unknown)
[11] Think-RM: Enabling Long-Horizon Reasoning in Generative Reward Models. hf-search. https://huggingface.co/papers/2505.16265 (2025-05-22)
[12] Outcome Accuracy is Not Enough: Aligning the Reasoning Process of Reward Models. hf-search. https://huggingface.co/papers/2602.04649 (2026-02-04)
[13] From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation. hf-daily. https://huggingface.co/papers/2610.06100 (2026-10-05)
[14] RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning. web. https://arxiv.org/abs/2505.14140v3 (unknown)
[15] Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study. hf-search. https://huggingface.co/papers/2404.10719 (2024-04-16)
[16] AGILE: A Novel Reinforcement Learning Framework of LLM Agents. hf-search. https://huggingface.co/papers/2405.14751 (2024-05-23)
[17] ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs. hf-search. https://huggingface.co/papers/2506.18896 (2025-06-23)
[18] A Reward Structure Showdown in Reasoning Models Training. web. https://arxiv.org/html/2511.13016v1 (unknown)
[19] MATH-SHEPHERD (mirror). web. https://d6108366.hf-mirror.com/papers/2312.08935 (unknown)
[20] OPPO: RLHF acceleration (ICLR 2026). web. https://proceedings.iclr.cc/paper_files/paper/2026/file/6b4fd4a4607f57fe65b5e276bdb17ed1-Paper-Conference.pdf (unknown)
[21] What Did the Agent Actually Do? Evidence-Grounded Oversight for Long-Horizon Agents. hf-daily. https://huggingface.co/papers/2610.06406 (2026-10-05)
