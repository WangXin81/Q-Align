# Q-Align
Q-Align is a zero-shot, training-free framework for knowledge-based visual question answering (KB-VQA) that improves LLM-based VQA by aligning question semantics with fine-grained visual evidence.

Instead of relying on coarse global captions, Q-Align extracts question-relevant entities and keywords, fusing visual grounding with scene graphs to locate key local regions. It then generates multi-granularity captions (both global and local) to provide structured, targeted visual evidence for LLMs.

The framework consists of three main modules:

QKIE: Question-guided Key Information Extraction

SG-GF: Spatially Aligned Scene Graph-Grounding Fusion

QMG-CG: Question-guided Multi-Granularity Caption Generation

Experiments on OK-VQA and A-OKVQA demonstrate that Q-Align achieves competitive zero-shot performance without any training or parameter tuning.

# Figs
<img width="4371" height="2008" alt="Fig3" src="https://github.com/user-attachments/assets/011f01ec-1106-4dcf-b549-b8e209be493f" />

# Data Preparation
OK-VQA and Images

https://okvqa.allenai.org/download.html

A-OKVQA

https://github.com/allenai/aokvqa#downloading-the-dataset

Please download the OK-VQA and A-OKVQA datasets and organize them as follows:

```text
data/
├── coco/
│   ├── val2014/
├── okvqa/
│   ├── OpenEnded_mscoco_val2014_questions.json
│   └── mscoco_val2014_annotations.json
└── aokvqa/
    └── aokvqa_v1.0_val.json
```

# Model Checkpoints & Third-Party Code Setup
To run Q-Align, you need to set up the corresponding visual grounding (GroundingDINO) and scene graph (SGDet) repositories, along with their pre-trained weights. Please follow the instructions below to configure the directory structure.

## 1. GroundingDINO Configuration
We utilize GroundingDINO for question-guided visual grounding. 

1. Download the model configuration file `GroundingDINO_SwinT_OGC.py` from the official(https://github.com/IDEA-Research/GroundingDINO) repository and place it into your local `config/` directory.
2. Download the pre-trained weights (`groundingdino_swint_ogc.pth`) into the `weights/`.

## 2. Scene Graph Benchmark Setup (SGDet)
Our framework integrates scene graph parsing based on KaihuaTang/Scene-Graph-Benchmark.pytorch.

1. Pre-trained Weights: Download the SGDet checkpoint from the official repository and place it into the checkpoint/scene_graph/ directory.

2. Compiling C++/CUDA Extensions: Since this scene graph toolkit contains custom C++/CUDA operations, you must compile them locally before running the framework.

```text
Q-Align/
├── checkpoint/
│   └── scene_graph/
│       ├── config.yml
│       ├── labels.json
│       ├── last_checkpoint
│       ├── log.txt
│       ├── model_0028000.pth
│       └── VG_stanford_filtered_with_attribute_train_statistics.cache
├── config/
│   └── GroundingDINO_SwinT_OGC.py
├── data/
│   ├── a_okvqa/
│   ├── coco/
│   └── ok_vqa/
├── model/
├── Scene-Graph/
├── weight/
│   └── groundingdino_swint_ogc.pth
├── main_eval_gpt.py
├── main_eval_gpt_map_global.py
└── main_eval_gpt_map_local.py
```

# Environments

python 3.8

torch 2.4.1

openai==0.28.0

transformers==4.46.3

sentence_transformers==3.2.1
