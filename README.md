# Q-Align
Q-Align is a zero-shot, training-free framework for knowledge-based visual question answering (KB-VQA) that improves LLM-based VQA by aligning question semantics with fine-grained visual evidence.

Instead of relying on coarse global captions, Q-Align extracts question-relevant entities and keywords, fusing visual grounding with scene graphs to locate key local regions. It then generates multi-granularity captions (both global and local) to provide structured, targeted visual evidence for LLMs.

The framework consists of three main modules:

QKIE: Question-guided Key Information Extraction

SG-GF: Spatially Aligned Scene Graph-Grounding Fusion

QMG-CG: Question-guided Multi-Granularity Caption Generation

Experiments on OK-VQA and A-OKVQA demonstrate that Q-Align achieves competitive zero-shot performance without any training or parameter tuning.
