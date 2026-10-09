# Efficient Inference and Small Language Models: a Survey

## TL;DR
- Practical inference speedups for LLMs combine low-bit quantization, memory-aware kernels like FlashAttention, and structured pruning or distillation to trade minimal quality loss for large latency/memory gains [1][2][3].
- Small, latency-optimized models (e.g., Mistral Small 3, Llama/Llama-2 at 7B–13B) use fewer layers, grouped-query attention (GQA) and dense parameterization to reach high throughput on commodity GPUs [4][5][6].
- Compiler/runtime work (FlashAttention, Triton kernels, AutoTriton) and system techniques (offloading, paged weights, pipelining) are essential to realize quantization and pruning gains in wall-clock latency [1][7][8][9].

## Background
Efficient inference for large language models (LLMs) aims to reduce latency, memory footprint and energy without substantially degrading downstream task quality. Core approaches include lower numerical precision (post-training quantization, quantization-aware training), structured or unstructured pruning, knowledge distillation into smaller models, and software/hardware co-design such as IO-aware attention and kernel fusion. Foundational algorithmic contributions include FlashAttention for IO-aware attention [1], GPTQ and AWQ for weight quantization strategies [2][6], and SmoothQuant for activation/weight transformation enabling W8A8 quantization [10].

## Foundations and Techniques
Techniques cluster into quantization, pruning/distillation, and token/sequence-level methods.

Quantization: Post-training and activation-aware methods let models run in low-bit formats with small accuracy drops. GPTQ (accurate post-training quantization) enables 3–4 bit weight quantization and reports up to 4.5x inference speedups on some GPUs [2]. AWQ (activation-aware weight quantization) protects a small fraction of salient weights to reduce quantization error and is designed for on-device scenarios [6]. SmoothQuant migrates activation outliers into weight space to enable W8A8 with little loss and reports up to 1.56x speedup and 2x memory reduction [10]. Larger surveys synthesize these methods and emphasize the two-step structure of pre-quantization transforms and quantization error mitigation [11].

Pruning & Distillation: Structured pruning approaches like Compresso and CoFi use task-aware and collaborative methods to remove parameters while preserving performance [12][3]. Wanda offers a magnitude-plus-activation pruning rule to avoid retraining and achieve efficient sparsity [12]. Distillation and sequence-level reductions (e.g., layer removal at inference) can yield large latency reductions with modest performance losses in tasks evaluated on LLaMA variants [13].

Sequence/token-level methods: training-free architecture search identifies efficient subnets that preserve accuracy while reducing GPU memory and inference time [11]. These can be combined with quantization to further lower memory requirements [11].

## Architectures for Small Models
Recent small-model families prioritize inference latency by trading parameter count and depth. Mistral Small 3 is a 24B latency-optimized design with fewer layers and engineering to reach high throughput and MMLU accuracy near larger models [5]. Llama and Llama 2 span 7B–70B but the smaller 7B/13B variants are commonly used for latency-sensitive deployment; Llama 2 also used grouped-query attention (GQA) in the 70B variant to reduce inference cost for large models [4][6]. Readme and model cards emphasize that smaller models pretrained on more tokens can be fine-tuned and quantized to run on consumer GPUs [6].

## Runtime and Compiler Optimizations
Runtime optimizations are needed to convert algorithmic gains into wall-clock improvements. FlashAttention reduces memory transfers between HBM and on-chip SRAM via tiling and reports 15% speedup on BERT-large and up to 3x on GPT-2 for long sequences [1][7]. Community implementations (FlashAttention GitHub) and FlashAttention-2 provide production-ready kernels [7].

AutoTriton and Triton-kernel generation automate creation of optimized GPU kernels, with AutoTriton using RL to generate high-performance Triton kernels [8]. FastAttention extends FlashAttention-like techniques to NPUs and low-resource GPUs, adapting tiling strategies to different hardware [14].

System-level techniques such as paged weights, CPU-GPU-I/O pipelining, and offloading (e.g., MoE-Lightning's CGOPipe) enable large models and MoEs to run on memory-constrained GPUs by overlapping IO and computation and by paging infrequently used weights [9][15].

## Benchmarks and Empirical Evaluations
Benchmarking efforts focus on latency, throughput, and the trade-off between model edits (quantization/pruning) and downstream performance. LLM-Inference-Bench provides cross-platform hardware performance evaluations for model inference and recommended configurations [16]. tinyBenchmarks offers small, representative subsets to evaluate LLMs efficiently [17]. Empirical studies show that removing some attention/MLP layers at inference can significantly reduce latency with moderate performance degradation on LLaMA-v2 tasks [13].

## Trends and Open Problems
Recent trends (2024–2026) include making MoE architectures practical for inference via expert quantization, paged expert weights, and pipelined execution; MoE-Lightning reports up to 10.3× higher throughput for Mixtral 8x7B using offloading and pipelining [9][15].

Quantization-aware training and data-free QAT (LLM-QAT) offer routes to aggressive low-bit deployment by fine-tuning models without access to original training data, improving over PTQ in some settings [3].

Open problems: combining MoE sparse architectures with low-bit quantization and efficient paging remains hard due to irregular memory access and communication costs [9][15]. Compiler-level automation for kernel generation (AutoTriton) shows promise but fully closing the gap between theoretical speedups and wall-clock gains across diverse hardware is ongoing [8][14]. Evaluation standards and benchmarks that jointly measure latency, energy, and quality (ALEM-style metrics) are still being consolidated across the community [18][19].

## References
[1] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arxiv. https://arxiv.org/abs/2205.14135 (2022-06-23)
[2] GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. arxiv. https://arxiv.org/abs/2210.17323 (2022-10-31)
[3] LLM-QAT: Data-Free Quantization Aware Training for Large Language Models. hf-search. https://huggingface.co/papers/2305.17888 (2023-05-29)
[4] Llama 2: Open Foundation and Fine-Tuned Chat Models. web. https://arxiv.science/abs/2307.09288 (2023-07-19)
[5] Mistral Small 3 | Mistral AI. web. https://mistral.ai/news/mistral-small-3/ (2025-01-30)
[6] AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration. arxiv. https://arxiv.org/abs/2306.00978 (2023-06-01)
[7] Dao-AILab/flash-attention (GitHub). web. https://github.com/Dao-AILab/flash-attention/ (2022-05-19)
[8] AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs. hf-search. https://huggingface.co/papers/2507.05687 (2025-07-08)
[9] MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs. web. https://dlnext.acm.org/doi/pdf/10.1145/3669940.3707267 (unknown)
[10] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. arxiv. https://arxiv.org/abs/2211.10438 (2022-11-18)
[11] Search for Efficient Large Language Models. hf-search. https://huggingface.co/papers/2409.17372 (2024-09-25)
[12] Compresso: Structured Pruning with Collaborative Prompting Learns Compact Large Language Models. hf-search. https://huggingface.co/papers/2310.05015 (2023-10-08)
[13] Attention Is All You Need But You Don't Need All Of It For Inference of Large Language Models. hf-search. https://huggingface.co/papers/2407.15516 (2024-07-22)
[14] FastAttention: Extend FlashAttention2 to NPUs and Low-resource GPUs. hf-search. https://huggingface.co/papers/2410.16663 (2024-10-22)
[15] Towards Efficient Mixture of Experts: A Holistic Study of Compression Techniques. web. https://case-lab-umd.github.io/Unified-MoE-Compression/ (unknown)
[16] LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators. hf-search. https://huggingface.co/papers/2411.00136 (2024-10-31)
[17] tinyBenchmarks: evaluating LLMs with fewer examples. hf-search. https://huggingface.co/papers/2402.14992 (2024-02-22)
[18] A Survey of Quantization in LLM: Unlocking Potential Hardware Efficiency. web. https://dl.acm.org/doi/10.1007/s11390-026-5979-1 (2026-03-28)
[19] On-device large language models: a survey of model compression and system optimization. web. https://www.springerprofessional.de/en/on-device-large-language-models-a-survey-of-model-compression-an/52458164 (unknown)
