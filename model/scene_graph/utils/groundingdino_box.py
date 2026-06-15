import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = "D:/hf_cache"
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util import box_ops
import numpy as np
from typing import List
import torch

def load_grounding_dino_model(config_path:str, weights_path:str,device):
    model = load_model(config_path, weights_path).to(device)
    return model

def groundingdino_inference(
    model,
    image_path: str,
    entity_texts: List[str],
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
):
    """
    使用 GroundingDINO 对多个 entity_texts 进行推理，返回 boxes, scores, labels
    """
    image_source, image_tensor = load_image(image_path)
    width, height = image_source.shape[1], image_source.shape[0]

    all_boxes = []
    all_scores = []
    all_labels = []

    for entity_text in entity_texts:
        if entity_text is None:
            boxes_xyxy = [[0, 0, width, height]]
            scores = [1.0]
            labels = ["entire image"]
        else:
            # print(entity_text)
            boxes, scores, labels = predict(
                model=model,
                image=image_tensor,
                caption=entity_text,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
            boxes_xyxy[:, [0, 2]] *= width
            boxes_xyxy[:, [1, 3]] *= height
            # boxes_xyxy = boxes_xyxy.cpu().numpy()
            # scores = scores.cpu().numpy()
            boxes_xyxy = boxes_xyxy.numpy()
            scores = scores.numpy()

            sorted_indices = np.argsort(-scores)
            boxes_xyxy = boxes_xyxy[sorted_indices]
            scores = scores[sorted_indices]
            labels = [labels[i] for i in sorted_indices]
        for box, score, label in zip(boxes_xyxy, scores, labels):
            # all_boxes.append(box.tolist())
            # all_scores.append(float(score))
            # all_labels.append(label)
            x1, y1, x2, y2 = box
            box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            image_area = width * height
            ratio = box_area / image_area

            if ratio <= 0.55:  # 保留小于等于55%的框
                all_boxes.append(box.tolist())
                all_scores.append(float(score))
                all_labels.append(label)
        # all_boxes.extend(boxes_xyxy.tolist())
        # all_scores.extend(scores.tolist())
        # all_labels.extend(labels)

    #print(f"推理完成，检测到 {len(all_boxes)} 个框")

    return all_boxes,all_labels,all_scores
