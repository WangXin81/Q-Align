import torch

class VQA_Pipeline:
    def __init__(self,llm_model,caption_map,device="cuda" if torch.cuda.is_available() else "cpu"):
        self.llm = llm_model
        self.caption_map = caption_map
        self.device = device

    def get_captions_by_qid(self, question_id):
        """根据 question_id 获取 image_caption, boxes_caption, relation_phrase"""
        question_id = str(question_id)
        entry = self.caption_map.get(question_id, {})
        captions = entry.get("captions", {})
        image_caption = captions.get("image_caption", "")
        return image_caption

    def process_batch(self, question_ids, questions):
        assert len(question_ids) == len(questions), "image_paths and questions must align"
        print(f"Running batch of size: {len(question_ids)}")
        whole_caption_list = []
        for qid in question_ids:
            img_cap = self.get_captions_by_qid(qid)
            whole_caption_list.append([img_cap])  # 包成 list 供 LLM 使用

        # === 打印调试信息（可选）===
        for i, whole in enumerate(whole_caption_list):
            print(f"Image {i}:")
            print("----- Whole Captions -----")
            for w in whole: print(w)
            print()

        # === LLM 批处理 ===
        answers = self.llm.batch_forward(
            questions,
            whole_caption_list,
        )
        return answers