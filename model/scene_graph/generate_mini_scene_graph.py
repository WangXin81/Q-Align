import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_HOME"] = "../../hf_cache"
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"
from sentence_transformers import SentenceTransformer
from model.scene_graph.whole_scene_graph import extract_scene_graph,extract_scene_graph_subgraph
from model.scene_graph.utils.get_keyword import extract_constrained_noun_phrases_and_prepositions
from model.scene_graph.utils.groundingdino_box import groundingdino_inference
from model.scene_graph.utils.hungarian import hungarian_match_visual_scenegraph
encoder = SentenceTransformer("all-MiniLM-L6-v2")
scene_graph_preposition_map = {
    "above": ["above","over"],
    "across": ["across"],
    "against": ["against","opposite","up against","opposite of"],
    "along": ["along","alongside"],
    "and": [],
    "at": [],
    "attached to": ["attach to","connect to"],
    "behind": ["behind", "in back of", "in the back of", "at the back of", "on the back of"],
    "belonging to": [],
    "between": ["between", "among"],
    "carrying": [],
    "covered in": [],
    "covering": [],
    "eating": [],
    "flying in": [],
    "for": [],
    "from": ["far from","away from","out of", "outside of" ],
    "growing on": [],
    "hanging from": [],
    "has": [],
    "holding": [],
    "in": ["in","inside"],
    "in front of": ["in front of", "in the front of", "at the front of", "on the front of", "ahead of"],
    "laying on": [],
    "looking at": [],
    "lying on": [],
    "made of":[],
    "mounted on": [],
    "near": ["near", "near to", "next to", "close to", "beside", "besides","around","on the left of", "to the left of", "on the right of", "to the right of", "on the side of", "at the side of","by the side of", "on the right side of", "on the left side of", "on the upper left side of", "on the upper right side of", "on the lower left side of", "on the lower right side of"],
    "of": [],
    "on": [ "on", "onto","upon","on top of", "on the top of", "at the top of", "in the top of"],
    "on back of": ["behind","in back of", "in the back of", "at the back of","on the back of" ],
    "over": [ "over", "above" ],
    "painted on": [],
    "parked on": [],
    "part of": [],
    "playing": [],
    "riding": [],
    "says": [],
    "sitting on": [],
    "standing on": [],
    "to": [],
    "under": [ "under", "underneath", "beneath", "below", "on the bottom of", "at the bottom of", "in the bottom of" ],
    "using": [],
    "walking in": [],
    "walking on": [],
    "watching": [],
    "wearing": [],
    "wears": [],
    "with": []
}
def build_vg_sg_keyword_map(matched_pairs, vg_labels):
    """
    matched_pairs: list of (vg_idx, sg_idx)
    vg_labels: list of str, vg 框对应的 keyword
    return: dict, sg_idx ➜ keyword
    """
    vg_keywords = {}
    for vg_idx, sg_idx in matched_pairs:
        vg_keywords[sg_idx] = vg_labels[vg_idx]
    return vg_keywords
def get_key(dictionary, value):
  """在字典中查找值对应的键"""
  for key, val in dictionary.items():
    if val == value:
      return key
  return None

def filter_scenegraph_and_build_boxlabel_mapping(vg_boxes, vg_labels, sg_boxes, sg_labels, sg_relations, position_words,
                                                 data):
  """
  fusion_filter
  根据匈牙利匹配结果 + 位置关系筛选 relation，并返回 boxes_with_labels + 更新后的 relations

  :param sg_relations: list of dict, 每个 dict 包含 'subject_idx','predicate','object_idx' 等
  :param matched_pairs: list of (vg_idx, sg_idx)
  :param position_words: list of str
  :param data: dict, 包含 category_id_to_name, predicate_to_idx 等
  :param sg_labels: list of int, scene graph box labels
  :param vg_keywords: dict, sg_idx ➔ keyword
  :param sg_boxes: list of [x1,y1,x2,y2], scene graph box 坐标
  :param threshold: float
  :return: updated_relations, boxes_with_labels
  """
  # Step 1️⃣: 匈牙利匹配（VG→SG）
  hungarian_matched_pairs = hungarian_match_visual_scenegraph(vg_boxes, sg_boxes)
  vg_keywords = build_vg_sg_keyword_map(hungarian_matched_pairs, vg_labels)
  # 1️⃣ 提取被匈牙利匹配到的 sg_idx
  matched_sg_idx = set([sg_idx for _, sg_idx in hungarian_matched_pairs])

  filtered_relations = []
  used_box_indices = set()
  vg_index_map = {sg_idx: vg_idx for vg_idx, sg_idx in hungarian_matched_pairs}  # SG idx -> VG idx

  # 2️⃣ 关系筛选
  for rel in sg_relations:
    subj_idx = rel['subject_idx']
    obj_idx = rel['object_idx']
    pred_idx = rel['predicate']

    pred = get_key(data['predicate_to_idx'], pred_idx)

    subject_or_object_matched = (subj_idx in matched_sg_idx) or (obj_idx in matched_sg_idx)
    if not subject_or_object_matched:
      continue
    predicate_matched = any(
      pred in scene_graph_preposition_map and pos_word in scene_graph_preposition_map[pred]
      for pos_word in position_words
    )
    if not predicate_matched:
      continue
    if subject_or_object_matched and predicate_matched:
      filtered_relations.append(rel)
      used_box_indices.update([subj_idx, obj_idx])

  # 3️⃣ 构建 boxes_with_labels
  boxes_with_labels = []
  category_id_to_name = data['idx_to_label']
  index_mapping = {}  # old idx ➔ new idx

  for new_idx, old_idx in enumerate(used_box_indices):
    # 判断是否匹配到 vg，若匹配则 label = keyword，否则 = sg label
    if old_idx in vg_keywords:
      vg_idx = vg_index_map[old_idx]
      box = vg_boxes[vg_idx]
      label_str = vg_keywords[old_idx]
    else:
      box = sg_boxes[old_idx]
      label_str = category_id_to_name[str(sg_labels[old_idx])]

    boxes_with_labels.append({
      "new_idx": new_idx,
      "original_idx": old_idx,
      "box": box,
      "label": label_str
    })

    index_mapping[old_idx] = new_idx

  # 4️⃣ 更新 relations 中的下标
  updated_relations = []
  for rel in filtered_relations:
    subj_old = rel['subject_idx']
    obj_old = rel['object_idx']

    # only keep if both indices are in used boxes
    if subj_old in index_mapping and obj_old in index_mapping:
      updated_rel = rel.copy()
      updated_rel['subject_idx'] = index_mapping[subj_old]
      updated_rel['object_idx'] = index_mapping[obj_old]
      updated_relations.append(updated_rel)

  boxes = []
  labels = []

  for item in boxes_with_labels:
    boxes.append(item["box"])
    labels.append(item["label"])

  return updated_relations, boxes, labels

def convert_relations_to_phrases(relations, labels, data):
    """
    根据 relations, labels, data 将 relations 转换为 phrase 形式
    :param relations: list of dict
    :param labels: list of str
    :param data: dict, 包含 predicate_to_idx
    :return: list of str, relation phrases
    """
    idx_to_predicate = {v: k for k, v in data['predicate_to_idx'].items()}

    relation_phrases = []

    for rel in relations:
        subj_idx = rel["subject_idx"]
        obj_idx = rel["object_idx"]
        pred_class = rel["predicate"]

        subj_label = labels[subj_idx]
        obj_label = labels[obj_idx]
        pred_label = idx_to_predicate.get(pred_class, str(pred_class))  # 如果不存在则返回数字字符串

        phrase = f"{subj_label}(local caption [{subj_idx}]) {pred_label} {obj_label}(local caption [{obj_idx}])"
        relation_phrases.append(phrase)

    return relation_phrases

def build_vg_sg_relation_and_box_mapping(
    vg_boxes,
    vg_labels,
    sg_boxes,
    sg_labels,
    sg_relations,
    data,
    position_words,
):
    """
    综合 VG + SG,fusion_all：
    1. 在函数内执行匈牙利匹配
    2. 保留所有 VG 框
    3. 根据介词匹配筛选 SG 关系，扩展框（A-介词-B）
    """

    # Step 1️⃣: 匈牙利匹配（VG→SG）
    hungarian_matched_pairs = hungarian_match_visual_scenegraph(vg_boxes, sg_boxes)
    vg_keywords = build_vg_sg_keyword_map(hungarian_matched_pairs, vg_labels)

    vg_to_sg = {vg_idx: sg_idx for vg_idx, sg_idx in hungarian_matched_pairs}
    sg_to_vg = {sg_idx: vg_idx for vg_idx, sg_idx in hungarian_matched_pairs}

    used_box_indices = set()
    filtered_relations = []

    # Step 2️⃣: 筛选满足介词条件的 SG 关系
    for rel in sg_relations:
        subj_idx = rel['subject_idx']
        obj_idx = rel['object_idx']
        pred_idx = rel['predicate']

        pred = get_key(data['predicate_to_idx'], pred_idx)

        # 判断介词是否匹配
        predicate_matched = any(
            pred in scene_graph_preposition_map and pos_word in scene_graph_preposition_map[pred]
            for pos_word in position_words
        )

        # 判断是否与 VG 匹配
        subj_matched_vg = subj_idx in sg_to_vg
        obj_matched_vg = obj_idx in sg_to_vg

        # 如果一方匹配 VG 且介词匹配，则保留
        if predicate_matched and (subj_matched_vg or obj_matched_vg):
            filtered_relations.append(rel)
            used_box_indices.update([subj_idx, obj_idx])

    # Step 3️⃣: 先构建 VG 框部分
    boxes_with_labels = []
    index_mapping = {}
    new_idx = 0

    for vg_idx, vg_box in enumerate(vg_boxes):
        boxes_with_labels.append({
            "new_idx": new_idx,
            "original_idx": f"VG_{vg_idx}",
            "box": vg_box,
            "label": vg_labels[vg_idx]
        })
        index_mapping[f"VG_{vg_idx}"] = new_idx
        new_idx += 1

    # Step 4️⃣: 再加入 SG 扩展框（B）
    category_id_to_name = data['idx_to_label']
    for old_idx in sorted(used_box_indices):
        if old_idx in sg_to_vg:  # 已有对应 VG
            continue
        label_str = category_id_to_name[str(sg_labels[old_idx])]
        box = sg_boxes[old_idx]
        boxes_with_labels.append({
            "new_idx": new_idx,
            "original_idx": old_idx,
            "box": box,
            "label": label_str
        })
        index_mapping[old_idx] = new_idx
        new_idx += 1

    # Step 5️⃣: 更新关系下标
    updated_relations = []
    for rel in filtered_relations:
        subj_old, obj_old = rel['subject_idx'], rel['object_idx']

        subj_new = (
            index_mapping.get(f"VG_{sg_to_vg[subj_old]}")
            if subj_old in sg_to_vg else index_mapping.get(subj_old)
        )
        obj_new = (
            index_mapping.get(f"VG_{sg_to_vg[obj_old]}")
            if obj_old in sg_to_vg else index_mapping.get(obj_old)
        )

        if subj_new is not None and obj_new is not None:
            new_rel = rel.copy()
            new_rel['subject_idx'] = subj_new
            new_rel['object_idx'] = obj_new
            updated_relations.append(new_rel)

    # Step 6️⃣: 输出
    boxes = [item["box"] for item in boxes_with_labels]
    labels = [item["label"] for item in boxes_with_labels]

    return updated_relations, boxes, labels

def mini_scene_graph(coco_demo,image_path,vg_model,question,prep_words,data):
    '''
    :param coco_demo:sc_model
    :param nlp:nlp
    :param encoder:encoder
    :param image_path:图像路径
    :param vg_model:vg model
    :param question:question
    :param standard_prepositions:standard_prepositions
    :param variant_to_standard_map:variant_to_standard_map
    :param word_prepositions:word_prepositions
    :param data:Scene-Graph/datasets/vg/VG-SGG-dicts-with-attri.json
    :return: sg_boxes:List of [x1, y1, x2, y2]
            filtered_relations:List of dict
    '''
    entity_text,prepositions = extract_constrained_noun_phrases_and_prepositions(question,prep_words)
    vg_boxes,vg_labels,vg_scores = groundingdino_inference(vg_model,image_path,entity_text)
    if vg_boxes == []:
        return [],[],[]
    if prepositions == []:
        return vg_boxes,vg_labels,[]
    prediction = extract_scene_graph(image_path,coco_demo)
    sg_boxes,sg_labels,sg_scores,sg_relations = extract_scene_graph_subgraph(prediction)
    #f_relations,f_boxes,f_labels = filter_scenegraph_and_build_boxlabel_mapping(vg_boxes,vg_labels,sg_boxes,sg_labels,sg_relations,prepositions,data)
    f_relations,f_boxes,f_labels = build_vg_sg_relation_and_box_mapping(vg_boxes,vg_labels,sg_boxes,sg_labels,sg_relations,data,prepositions)
    if f_relations == []:
        #return all_boxes,all_labels,[]
        return vg_boxes,vg_labels,[]
    relation_phrase = convert_relations_to_phrases(f_relations,f_labels,data)
    return f_boxes,f_labels,relation_phrase