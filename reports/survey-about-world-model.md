# Survey: World Models in Machine Learning
## TL;DR
- World models are neural generative or predictive models that learn environment dynamics and enable planning or policy learning inside imagined trajectories [1].
- Latent dynamics architectures (deterministic+stochastic state, VAE-based) enable efficient planning from pixels and improved sample efficiency versus model-free baselines on DM Control and Atari [2][3][4].
- Recent trends shift toward discrete tokenization + autoregressive Transformers (IRIS, \u0394-IRIS, iVideoGPT) and repurposing LLMs as world models/planners for web agents (WebDreamer, RAP) [5][6][7].

## Background
Ha & Schmidhuber trained generative neural networks to learn compressed spatial-temporal representations of RL environments and demonstrated training agents inside hallucinated "dream" rollouts [1][8]. Later work operationalized this approach as latent-dynamics model-based RL: PlaNet proposed a latent dynamics model with deterministic and stochastic transition components and introduced "latent overshooting" to plan in latent space from pixels [2], and Dreamer applied latent imagination to learn behaviors and reported improved data efficiency and computation time on control tasks [3].

## Architectures and methods
Latent-dynamics world models typically separate representation learning and transition modeling. PlaNet and Dreamer use RSSM-like hybrids with deterministic and stochastic components, variational objectives and techniques such as latent overshooting and KL-balancing to stabilise learning [2][3][9].

Discrete-token + transformer backbones (IRIS, \u0394-IRIS, iVideoGPT) convert visual observations into discrete tokens via an autoencoder and apply autoregressive Transformers to model multimodal sequences of observations, actions and rewards; these architectures claim improved simulation efficiency and sample efficiency in MBRL when paired with appropriate tokenization [10][6][5].

Backbone comparisons (RNN vs Transformer vs S4) find that S4-based world models (S4WM) can provide superior long-range memory and efficiency compared to RNN and Transformer backbones in MBRL settings [11].

There is also a line of work re-purposing LLMs as world models/planners: RAP uses an LLM for next-state prediction with MCTS for planning; WebDreamer trains dedicated world models synthesized from LLM-generated interactions, showing large gains for web agents [12][7][13].

## Benchmarks and evaluation
Common benchmarks for world models include the DeepMind Control Suite (continuous control), Atari (discrete action, pixel input), Procgen (generalization), and Crafter (diverse open-ended achievements) [2][3][14][15]. Papers report metrics such as cumulative episode return, normalized scores, sample-efficiency measured by environment steps, and aggregated achievement rates [4][2].

Key empirical findings include Dreamer reporting strong sample-efficiency and improved average performance on the control suite relative to strong model-free baselines; the paper reports average scores across tasks and training-time comparisons between Dreamer and D4PG in its experiments [4][16].

## Trends and open problems
In the last two years the field has moved toward large discrete-token transformers for world modeling (IRIS family, iVideoGPT, \u0394-IRIS) to improve simulation and planning efficiency, and toward integrating LLM reasoning with explicit simulation (RAP, WebDreamer) for web agents and complex planning tasks [10][5][6][7][12]. Open problems include robust long-horizon prediction, generalization to novel environments, combining object-centric physics with neural dynamics, and grounding LLM-based world models in real-interaction data rather than purely synthetic reasoning [17][11][13].

## References
[1] World Models. web. https://arxiv.org/abs/1803.10122 (2018-05-09)
[2] Learning Latent Dynamics for Planning from Pixels. hf-search. https://huggingface.co/papers/1811.04551 (2019-06-04)
[3] Dream to Control: Learning Behaviors by Latent Imagination. hf-search. https://huggingface.co/papers/1912.01603 (2019-12-03)
[4] Dream to Control (arXiv / PDF). web. http://arxiv.org/abs/1912.01603v3 (unknown)
[5] iVideoGPT: Interactive VideoGPTs are Scalable World Models. hf-search. https://huggingface.co/papers/2405.15223 (2024-05-24)
[6] Efficient World Models with Context-Aware Tokenization (Δ-IRIS). hf-search. https://huggingface.co/papers/2406.19320 (2024-06-27)
[7] Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents. hf-daily. https://huggingface.co/papers/2411.06559 (2024-11-10)
[8] World Models (interactive). web. https://worldmodels.github.io/ (2018-03-27)
[9] Mastering Atari with Discrete World Models (DreamerV2). web. https://arxiv.org/abs/2010.02193 (unknown)
[10] Transformers are Sample-Efficient World Models (IRIS). hf-search. https://huggingface.co/papers/2209.00588 (2023-03-01)
[11] Facing Off World Model Backbones: RNNs, Transformers, and S4. hf-search. https://huggingface.co/papers/2307.02064 (2023-07-05)
[12] Reasoning with Language Model is Planning with World Model (RAP). web. https://aclanthology.org/anthology-files/pdf/emnlp/2023.emnlp-main.507.pdf (unknown)
[13] Understanding the planning of LLM agents: A survey. web. https://arxiv.org/abs/2402.02716 (2024-02-05)
[14] Procgen Benchmark. web. https://arxiv.org/abs/1912.01588 (unknown)
[15] Benchmarking the Spectrum of Agent Capabilities (Crafter). web. https://arxiv.org/abs/2109.06780 (unknown)
[16] Model-Based Reinforcement Learning for Atari (SimPLe). hf-search. https://huggingface.co/papers/1903.00374 (2024-04-03)
[17] Understanding World or Predicting Future? A Comprehensive Survey of World Models. hf-search. https://huggingface.co/papers/2411.14499 (2024-11-21)
