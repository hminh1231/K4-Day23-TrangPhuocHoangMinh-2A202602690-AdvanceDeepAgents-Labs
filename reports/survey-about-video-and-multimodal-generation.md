# Survey: Video and Multimodal Generation

## TL;DR

- Diffusion-based approaches (latent or pixel) have become the dominant paradigm for high-fidelity, text- and multimodal-conditioned video generation, building on image diffusion backbones adapted to space-time [1][2].
- Conditioning on images, audio, and motion is implemented with multi-stage pipelines and attention-based conditioning (MVDiT, I4VGen, VMC), enabling image-as-stepping-stone and LLM-director setups [3][4][5][6].
- Evaluation relies on distribution-level metrics such as FVD and newer metrics that separate spatial vs temporal fidelity (STREAM, FVMD); captioning metrics remain imperfect and motivate learned/reference-free scores [7][8][9][10].
- Scaling to minute-level and arbitrary-length videos is an active area: recurring strategies compress to latents/tokens and use autoregressive or memory-augmented transformers (MALT, StreamingT2V, FAR) to extend context while preserving appearance [11][12][13].
- Applications include commercial systems (Runway Gen-2), synthetic-data and film tools, and substantial risks (deepfakes, misinformation) noted by government reports [14][15].

## Background

Video and multimodal generation refers to models that synthesize video conditioned on text, images, audio or other modalities. It matters now because recent diffusion-based and transformer-augmented architectures have produced high-quality short videos and opened pathways to longer, multimodal, and controllable generation [1][2][16]. Foundational work includes MoCoGAN's motion/content factorization (GAN era) and the introduction of Fréchet Video Distance (FVD) for evaluation [17][7].

## Foundations and model families

The field evolved from GAN-based models that decomposed motion and content (MoCoGAN) to diffusion-based video models that extend image U-Nets to space-time and jointly train on images and videos to reduce variance and accelerate optimization [17][1]. Video diffusion methods introduce sampling strategies for spatial and temporal extension and can be conditioned on text, images or prior frames — techniques that underpin systems like Video Diffusion Models and Video Diffusion Transformers (VDT) which add spatio-temporal attention [1][10]. Efficient decompositions separate content and motion in latent space to leverage pretrained image diffusion models, improving quality and compute efficiency [18]. FVD remains a foundational metric for distributional evaluation and correlates with human judgments in large studies [7].

## Conditioning and multimodal inputs

Text-to-video and image-conditioned pipelines commonly use multi-stage approaches: synthesize a high-quality image (or key-frame) with text-to-image models and animate it via video diffusion or transformer-based modules (I4VGen, VideoElevator, MVDiT). Benchmarks and datasets such as OpenVid-1M and T2V-CompBench support research on compositionality and multimodal alignment [3][19][4]. LLM directors plus LDM animators (Free-Bloom) show a practical modular route where language models produce structured instructions that guide diffusion-based animators [6]. Temporal attention tuning (VMC) is used to reproduce and diversify motion while preserving appearance [5].

## Evaluation: metrics and benchmarks

Metrics move beyond a single aggregate to measure spatial and temporal fidelity separately. FVD, based on I3D features, is established and correlates with human judgment on benchmarks like SCV [7]. STREAM explicitly decomposes spatial and temporal scores to avoid aggregated confounds [8], while FVMD focuses on motion consistency, aligning closer with perceived motion quality [9]. Video-captioning metrics (BLEU, METEOR, CIDEr, SPICE) are commonly used for alignment evaluation but have documented weaknesses, motivating learned or reference-free measures and human studies [10][20].

## Recent trends: long-horizon, scaling and architectures

A major recent trend is extending generation from clip-length (seconds) to minute-level videos. Strategies include compressing frames into latents or tokens (VAE/VQ-style or latent diffusion) and using autoregressive or memory-augmented generation over segments (MALT, StreamingT2V, FAR). Modules like conditional attention module (CAM) and appearance preservation module (APM) explicitly handle short- and long-term consistency, and recurrent attention or memory latents maintain context over long rollouts [11][12]. Several papers report large gains in FVD for long-horizon settings (e.g., MALT's reported 220.4 vs prior 648.4 on UCF-101 128-frame generation) [11].

## Applications and risks

Commercial tools like Runway Gen-2 demonstrate that multimodal video generation is practical and preferred in user studies over earlier baselines, and large datasets (OpenVid-1M) enable training at scale for improved realism and compositionality [14][3]. At the same time, government reports highlight risks from synthetic video technologies for misinformation and privacy harms, making detection and provenance an active concern [15].

## Trends and open problems

- Long-horizon generation: while methods scale to minutes using autoregressive or memory-augmented techniques, maintaining motion diversity and visual quality over thousands of frames is still an open challenge; approaches like LongTake and StreamingT2V are recent steps [12][21].
- Multimodal alignment and compositionality: T2V-CompBench and OpenVid-1M expose remaining limitations in object interactions, numeracy, and fine-grained control [19][3].
- Evaluation: current metrics improve on FVD by isolating temporal fidelity (STREAM, FVMD), but reference-free and human-alignment metrics remain needed [8][9][7].
- Compute and efficiency: efficient motion-content decompositions and sparse attention techniques (MC-Sparse) are being developed to make long video generation tractable [18][22].

## References
[1] Video Diffusion Models. web. https://arxiv.org/abs/2204.03458 (unknown)
[2] Video Diffusion Models. hf-search. https://huggingface.co/papers/2204.03458 (2022-04-07)
[3] OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation. hf-search. https://huggingface.co/papers/2407.02371 (2024-07-02)
[4] I4VGen: Image as Stepping Stone for Text-to-Video Generation. hf-search. https://huggingface.co/papers/2406.02230 (2024-06-04)
[5] VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models. hf-search. https://huggingface.co/papers/2312.00845 (2023-12-01)
[6] Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator. hf-search. https://huggingface.co/papers/2309.14494 (2023-09-25)
[7] Towards Accurate Generative Models of Video: A New Metric & Challenges. web. https://arxiv.org/abs/1812.01717 (2019-03-27)
[8] STREAM: Spatio-TempoRal Evaluation and Analysis Metric for Video Generative Models. hf-search. https://huggingface.co/papers/2403.09669 (2024-01-30)
[9] Fréchet Video Motion Distance: A Metric for Evaluating Motion Consistency in Videos. hf-search. https://huggingface.co/papers/2407.16124 (2024-07-23)
[10] VDT: General-purpose Video Diffusion Transformers via Mask Modeling. hf-search. https://huggingface.co/papers/2305.13311 (2023-05-22)
[11] MALT Diffusion: Memory-Augmented Latent Transformers for Any-Length Video Generation. web. https://arxiv.org/html/2502.12632 (unknown)
[12] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text. web. https://openaccess.thecvf.com/content/CVPR2025/papers/Henschel_StreamingT2V_Consistent_Dynamic_and_Extendable_Long_Video_Generation_from_Text_CVPR_2025_paper.pdf (unknown)
[13] VideoElevator: Elevating Video Generation Quality with Versatile Text-to-Image Diffusion Models. hf-search. https://huggingface.co/papers/2403.05438 (2024-03-08)
[14] Gen-2: Generate novel videos with text, images or video clips. web. https://runway.com/research/gen-2 (2023-03-11)
[15] The Science & Technology Digital Forgeries Report. web. https://www.dhs.gov/sites/default/files/2023-06/23_0630_st_digital_forgeries_report_signed.pdf (unknown)
[16] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. hf-search. https://huggingface.co/papers/2408.06072 (2024-08-12)
[17] MoCoGAN: Decomposing Motion and Content for Video Generation. hf-search. https://huggingface.co/papers/1707.04993 (2017-07-17)
[18] Efficient Video Diffusion Models via Content-Frame Motion-Latent Decomposition. hf-search. https://huggingface.co/papers/2403.14148 (2024-03-21)
[19] T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation. hf-search. https://huggingface.co/papers/2407.14505 (2024-07-19)
[20] Re-evaluating Automatic Metrics for Image Captioning. web. https://ar5iv.labs.arxiv.org/html/1612.07600 (unknown)
[21] LongTake: Learning to Sustain Dynamics in Long-Horizon Video Generation. hf-daily. https://huggingface.co/papers/2609.38562 (2026-09-29)
[22] MC-Sparse: Deconstructing and Closing the Dense-Sparse Attention Gap in Diffusion Transformers. hf-daily. https://huggingface.co/papers/2610.06801 (2026-10-05)
