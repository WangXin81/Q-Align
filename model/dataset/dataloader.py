import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = "D:/hf_cache"
from torch.utils.data import Dataset

class VQADataset(Dataset):
    def __init__(self, question_json_path, annotation_json_path, image_root):
        """
        参数:
        - question_json_path: str, 问题 JSON 路径
        - annotation_json_path: str, 答案 JSON 路径
        - image_root: str, 图像目录路径
        """
        with open(question_json_path, 'r') as f:
            self.questions = json.load(f)['questions']

        with open(annotation_json_path, 'r',encoding='utf-8') as f:
            annotations = json.load(f)['annotations']
            # 构建 question_id -> list of answers 映射
            self.id2answers = {
                ann['question_id']: [ans['answer'] for ans in ann['answers']]
                for ann in annotations
            }

        self.image_root = image_root

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        question_id = q['question_id']
        image_id = q['image_id']
        question = q['question']

        # 构造图像路径
        image_filename = f"COCO_val2014_{image_id:012d}.jpg"
        image_path = os.path.join(self.image_root, image_filename)

        # 获取该问题的所有 GT 答案（长度为 10）
        gt_answers = self.id2answers.get(question_id, [])
        return {
            'image_path': image_path,
            'question': question,
            'question_id': question_id,
            'gt_answers': gt_answers
        }