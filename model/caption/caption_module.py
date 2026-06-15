from PIL import Image
from model.caption.boxes_image_caption_llava import load_caption_model_llava,generate_whole_caption,generate_local_caption
from model.caption.utils.get_text_prompt import extract_keywords_from_question

class CaptionModule:
    def __init__(self,model_id,device):
        self.model,self.process = load_caption_model_llava(model_id,device)
    def extract_keywords_from_question(self,question):
        return extract_keywords_from_question(question)
    def generate_whole_captions(self,image,keywords):
        return generate_whole_caption(self.model,self.process,image,keywords,num_captions = 1)
    def generate_local_captions(self,image,boxes,labels,keywords):
        return generate_local_caption(self.model,self.process,image,boxes,labels,keywords)
    def forward(self,image_path,question,boxes,labels):
        image = Image.open(image_path).convert("RGB")
        keywords = self.extract_keywords_from_question(question)
        whole_caption = self.generate_whole_captions(image,keywords)
        local_captions = self.generate_local_captions(image,boxes,labels,keywords)
        return whole_caption,local_captions
