import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_iou(box1, box2):
    """
    计算两个 box 的 IoU
    :param box1: [x1,y1,x2,y2]
    :param box2: [x1,y1,x2,y2]
    :return: float, IoU
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2-x1) * max(0, y2-y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0
    else:
        return inter_area / union_area

def hungarian_match_visual_scenegraph(visual_boxes, sg_boxes, iou_threshold=0.5):
    """
    基于 IoU 的匈牙利匹配：匹配 scene graph 与 visual grounding 的框
    - 若 IoU < 阈值，则视为无法匹配
    """
    visual_boxes = np.array(visual_boxes)
    sg_boxes = np.array(sg_boxes)

    N, M = len(visual_boxes), len(sg_boxes)
    cost_matrix = np.ones((N, M))  # 初始化为高代价（默认不匹配）

    for i in range(N):
        for j in range(M):
            iou = compute_iou(visual_boxes[i], sg_boxes[j])
            if iou >= iou_threshold:
                cost_matrix[i, j] = 1 - iou  # IoU 越高，代价越小
            else:
                cost_matrix[i, j] = 1e6  # 无效匹配，给大代价\
    # 匈牙利算法求最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # 过滤掉 IOU < 阈值的匹配结果
    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        if compute_iou(visual_boxes[r], sg_boxes[c]) >= iou_threshold:
            matched_pairs.append((r, c))
    return matched_pairs

