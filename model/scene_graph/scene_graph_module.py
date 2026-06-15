from model.scene_graph.whole_scene_graph import load_scene_graph_model
from model.scene_graph.utils.groundingdino_box import load_grounding_dino_model
from model.scene_graph.generate_mini_scene_graph import mini_scene_graph

class SceneGraphModule:
    def __init__(self,sg_config_path,vg_config_path,vg_weight_path,prep_words,device):
        self.device = device
        self.scene_graph_model = load_scene_graph_model(sg_config_path,device)
        self.visual_grounding_model = load_grounding_dino_model(vg_config_path,vg_weight_path,device)
        self.prep_words = prep_words
    def process(self,image_path,question,data):
        boxes,labels,relations_phrase = mini_scene_graph(
            coco_demo = self.scene_graph_model,
            image_path = image_path,
            vg_model = self.visual_grounding_model,
            question = question,
            prep_words = self.prep_words,
            data = data
        )
        return boxes,labels,relations_phrase

