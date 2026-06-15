import cv2
from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo
def load_scene_graph_model(config_file, device='cpu'):
    """
    加载 scene graph 模型
    输入: config_file(str)
    输出: 已初始化的模型实例 coco_demo
    """
    # 配置
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device

    # 初始化 predictor
    coco_demo = COCODemo(cfg)
    return coco_demo
def extract_scene_graph(image_path, coco_demo):
    """
    使用已加载的模型处理图像，生成scene graph
    输入: image_path, coco_demo(已加载模型)
    输出: prediction
    """
    # 读取图像
    image = cv2.imread(image_path)
    prediction = coco_demo.compute_prediction(image)
    return prediction
def extract_scene_graph_subgraph(prediction, score_thresh=0.5):
    """
    根据置信度阈值提取过滤后的 boxes, labels, scores, relations (dict 格式)

    Returns:
        boxes: List of [x1, y1, x2, y2]
        labels: List of int
        scores: List of float
        relations: List of dict
    """

    # 1. 取出 bbox, labels, scores
    boxes = prediction.bbox.tolist() if hasattr(prediction, 'bbox') else []
    labels = prediction.get_field('pred_labels').tolist()
    scores = prediction.get_field('pred_scores').tolist()

    # 2. 过滤低分框，记录有效索引
    filtered = [(box, label, score, i) for i, (box, label, score) in enumerate(zip(boxes, labels, scores)) if score >= score_thresh]
    boxes = [x[0] for x in filtered]
    labels = [x[1] for x in filtered]
    scores = [x[2] for x in filtered]
    indices = [x[3] for x in filtered]

    # 3. 过滤关系
    rel_pair_idxs = prediction.get_field('rel_pair_idxs').tolist()
    rel_labels = prediction.get_field('pred_rel_labels').tolist()
    filtered_relations = []

    # old -> new idx
    index_map = {old_i: new_i for new_i, old_i in enumerate(indices)}
    for (subj_idx, obj_idx), rel_label in zip(rel_pair_idxs, rel_labels):
        if subj_idx in index_map and obj_idx in index_map:
            new_sub_idx = index_map[subj_idx]
            new_obj_idx = index_map[obj_idx]
            relation = {
                'subject_idx': new_sub_idx,
                'subject_label': labels[new_sub_idx],
                'predicate': rel_label,
                'object_idx': new_obj_idx,
                'object_label': labels[new_obj_idx]
            }
            filtered_relations.append(relation)
    return boxes, labels, scores, filtered_relations
