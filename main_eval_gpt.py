# main_eval.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_HOME"] = "hf_cache"
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"
import sys
import torch
import json
from torch.utils.data import DataLoader

from model.scene_graph.scene_graph_module import SceneGraphModule
from model.caption.caption_module import CaptionModule
from model.LLM.GPT_module import LLMModule
from model.pipeline import VQA_Pipeline
from model.dataset.dataloader import VQADataset
from model.dataset.dataloader_aokvqa import AOKVQADataset
from model.eval_utils import evaluate_vqa

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
    #模型初始化
    config_file = os.path.join(project_root, 'checkpoint','scene_graph','config.yml')
    config_file2 = os.path.join(project_root, 'config', 'GroundingDINO_SwinT_OGC.py')
    weight_file2 = os.path.join(project_root, 'weight', 'groundingdino_swint_ogc.pth')
    data_file = os.path.join(project_root,'Scene-Graph','datasets','vg','VG-SGG-dicts-with-attri.json')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(data_file, 'r') as file:
        data = json.load(file)

    prep_words = [
        # 多词
        'ahead of', 'close to', 'far from', 'near to', 'next to', "attach to", "connect to",
        'away from', "opposite of", "out of", "outside of", "up against",
        # --- front ---
        'in front of', 'in the front of', 'at the front of', 'on the front of',
        # --- back ---
        'in back of', 'in the back of', 'at the back of', 'on the back of', "on back of",
        # --- top ---
        'on top of', 'on the top of', 'at the top of', 'in the top of',
        # --- bottom ---
        'on the bottom of', 'at the bottom of', 'in the bottom of',
        # --- left/right ---
        'to the left of', 'on the left of',
        'to the right of', 'on the right of',
        # --- side ---
        'on the side of', 'at the side of', 'by the side of',
        # --- center/middle ---
        'in the center of', 'at the center of',
        'in the middle of', 'at the middle of',
        # --- complex ---
        'on the left side of', 'on the right side of',
        "on the upper left side of", "on the upper right side of",
        "on the lower left side of", "on the lower right side of",
        # single word
        'above', 'across', 'against', 'along', 'alongside', 'among', 'around', 'behind', 'below', 'beneath',
        'beside', 'besides', 'between', 'in', 'inside', 'near', 'on', 'onto', 'opposite', 'over', 'under',
        'underneath', 'upon',
        # 防止误匹配
        'based on', 'depend on', 'focus on', 'rely on', 'believe in',
        'engaged in', 'interested in', 'in use', 'in need of',
        'in addition to', 'in front', 'on average', 'in general',
        'on purpose', 'in fact', 'in contrast', 'on time', 'in turn',
        "in the picture", "in this picture", "in the photo", "in this photo", "in the image", "in this image",
        "in the scene", "in this scene",
        "in the background", "in the foreground",
        # -----front-----
        'in the front', 'at the front', 'on the front',
        # --- back ---
        'in the back', 'at the back', 'on the back',
        # --- top ---
        'on top', 'at the top', 'on the top',
        # --- bottom ---
        'on the bottom', 'at the bottom',
        # --- left/right ---
        'in the left', 'in the right', 'on the left', 'on the right',
        # --- side ---
        'on the side', 'at the side',
        # --- center/middle ---
        'in the center', 'at the center',
        'in the middle', 'at the middle',
        # --- complex ---
        'on the far left', 'on the far right',
        'in the top left', 'in the top right',
        'in the bottom left', 'in the bottom right',
        "in the front center", 'in the back center of',
    ]

    scene_graph = SceneGraphModule(config_file,config_file2,weight_file2,prep_words,device)

    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    #model_id = "Salesforce/blip2-opt-2.7b"
    #model_id_local = "Salesforce/blip2-flan-t5-xl"
    #model_id = "noamrot/FuseCap"
    #model_id = "Salesforce/blip-image-captioning-base"

    caption_model = CaptionModule(model_id,device)

    #model_name = "microsoft/phi-2"
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    #model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    #model_name = "EleutherAI/gpt-neo-2.7B"
    # model_name = "EleutherAI/gpt-j-6B"
    model_name = "gpt-3.5-turbo"

    llm = LLMModule(model_name)
    model = VQA_Pipeline(scene_graph,caption_model,llm,data,device)
    #数据加载
    question_json_path = os.path.join(project_root, "data", "ok_vqa1", "OpenEnded_mscoco_val2014_questions.json")
    annotation_json_path = os.path.join(project_root, "data", "ok_vqa1", "mscoco_val2014_annotations.json")
    image_root = os.path.join(project_root, "data", "coco", "val2014")
    aokvqa_json_root = os.path.join(project_root, "data", "a_okvqa", "aokvqa_v1p0_val.json")
    dataset = VQADataset(question_json_path, annotation_json_path, image_root)
    #dataset =AOKVQADataset(aokvqa_json_root,image_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,collate_fn=custom_collate)

    # 评估
    acc = evaluate_vqa(model, dataloader, device)
    print(f"VQA Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
