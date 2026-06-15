import os
import json
from torch.utils.data import Dataset

class AOKVQADataset(Dataset):
    def __init__(self, json_path, image_root):
        """
        参数:
        - json_path: str, A-OKVQA 的 JSON 文件路径
        - image_root: str, 图像目录路径（MSCOCO val2014 或 train2014）
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.image_root = image_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 基本字段
        question_id = item.get('question_id', idx)
        image_id = item['image_id']
        question = item['question']

        # 构造图像路径（A-OKVQA 图像来源与 MSCOCO 一致）
        image_filename = f"COCO_val2014_{image_id:012d}.jpg"
        image_path = os.path.join(self.image_root, image_filename)
        # if not os.path.exists(image_path):
        #     # 若文件不存在，可尝试train2014路径
        #     image_filename = f"COCO_train2014_{image_id:012d}.jpg"
        #     image_path = os.path.join(self.image_root, image_filename)

        # ground truth 答案
        gt_answers = item.get('direct_answers', [])

        # 额外信息（用于多任务/分析）
        # choices = item.get('choices', None)
        # correct_choice_idx = item.get('correct_choice_idx', None)
        # rationales = item.get('rationales', None)

        return {
            'image_path': image_path,
            'question': question,
            'question_id': question_id,
            'gt_answers': gt_answers
            # 'choices': choices,
            # 'correct_choice_idx': correct_choice_idx,
            # 'rationales': rationales
        }
