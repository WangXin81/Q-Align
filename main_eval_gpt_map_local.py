# main_eval.py
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "D:/hf_cache"
import sys
import spacy
import torch
import json
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import openai
from model.scene_graph.scene_graph_module import SceneGraphModule
from model.caption.caption_module import CaptionModule
from model.LLM.GPT_module_local import LLMModule
from model.pipeline_local import VQA_Pipeline
from model.dataset.dataloader import VQADataset
from model.eval_utils_map import evaluate_vqa
from model.dataset.dataloader_aokvqa import AOKVQADataset

def custom_collate(batch):
    image_paths = [item['image_path'] for item in batch]
    questions = [item['question'] for item in batch]
    question_ids = [item['question_id'] for item in batch]
    gt_answers = [item['gt_answers'] for item in batch]
    return {
        "image_path": image_paths,
        "question": questions,
        "question_id": question_ids,
        "gt_answers": gt_answers
    }

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    sys.path.append(project_root)
    with open("data/final/A-OKVQA/local_filter_sg_all.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    model_name = "gpt-3.5-turbo"
    llm = LLMModule(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQA_Pipeline(llm,data,device)
    #数据加载
    question_json_path = os.path.join(project_root, "data", "ok_vqa1", "OpenEnded_mscoco_val2014_questions.json")
    annotation_json_path = os.path.join(project_root, "data", "ok_vqa1", "mscoco_val2014_annotations.json")
    image_root = os.path.join(project_root, "data", "coco", "val2014")
    aokvqa_json_root = os.path.join(project_root, "data", "a_okvqa", "aokvqa_v1p0_val.json")
    dataset =AOKVQADataset(aokvqa_json_root,image_root)
    #dataset = VQADataset(question_json_path, annotation_json_path, image_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,collate_fn=custom_collate)

    # 评估
    acc = evaluate_vqa(model, dataloader, device)
    print(f"VQA Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
