import torch
from model.LLM.answer_gpt import answer_question_local
from model.LLM.util.api_key import get_api_key
import openai
class LLMModule:
    def __init__(self, model_name = "gpt-3.5-turbo",api_key: str = get_api_key("OPENAI_API_KEY"),base_url: str = "https://api.chatanywhere.tech/v1",device="cuda" if torch.cuda.is_available() else "cpu"):
        # 加载模型和tokenizer
        self.model = model_name
        self.device = device
        openai.api_key = api_key
        openai.api_base = base_url

    def forward(self, question, image_caption, local_captions):
        """
        综合处理：生成答案
        :param question: 提出的问题
        :param image_caption: 图像描述
        :param local_captions: 框描述
        :return: 答案
        """
        # 直接调用answer_question生成答案
        answer = answer_question_local(image_caption,local_captions,question,self.model)
        return answer

    def batch_forward(self, questions, whole_captions_list, local_captions_list):
        answers = []
        for question, whole_caps, local_caps in zip(questions, whole_captions_list, local_captions_list):
            answer = self.forward(question, whole_caps, local_caps)
            answers.append(answer)
        return answers
