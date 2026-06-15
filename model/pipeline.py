import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = "../hf_cache"
import torch

class VQA_Pipeline:
    def __init__(self,scene_graph_model,caption_model,llm_model,data,device="cuda" if torch.cuda.is_available() else "cpu"):
        self.scene_graph = scene_graph_model
        self.caption = caption_model
        self.llm = llm_model
        self.data = data
        self.device = device
    def process_image(self,image_path,question):
        boxes,labels,relation_phrases = self.scene_graph.process(image_path,question,self.data)
        whole_captions,local_captions = self.caption.forward(image_path,question,boxes,labels)
        print("----------------------------whole_captions---------------------------")
        for whole_caption in whole_captions:
            print(whole_caption)
        print("----------------------------local_captions---------------------------")
        # local_captions = [f"[{i}] {cap}" for i, cap in enumerate(local_captions)]
        for local_caption in local_captions:
            print(local_caption)
        print("---------------------------relation_phrases--------------------------")
        for relation_phrase in relation_phrases:
            print(relation_phrase)
        answer = self.llm.forward(question, whole_captions, local_captions, relation_phrases)
        return answer

    def process_batch(self, image_paths, questions):
        assert len(image_paths) == len(questions), "image_paths and questions must align"
        print(f"Running batch of size: {len(image_paths)}")
        sg_results = [
            self.scene_graph.process(img_path, q, self.data)
            for img_path, q in zip(image_paths, questions)
        ]
        boxes_list, labels_list,relation_phrases_list = zip(*sg_results)

        # === Caption 批处理 ===
        caption_results = [
            self.caption.forward(img_path, q, boxes, labels)
            for img_path, q, boxes, labels in zip(image_paths, questions, boxes_list, labels_list)
        ]
        whole_caption_list, local_caption_list = zip(*caption_results)

        for i, (whole, local, rels) in enumerate(zip(whole_caption_list, local_caption_list, relation_phrases_list)):
            print(f"Image {i}:")
            print("----- Whole Captions -----")
            for w in whole: print(w)
            print("----- Local Captions -----")
            for l in local: print(l)
            print("----- Relations -----")
            for r in rels: print(r)
            print()

        # === LLM 批处理 ===
        answers_base = self.llm.batch_forward(
            questions,
            whole_caption_list,
            local_caption_list,
        )
        return answers_base




