from collections import Counter
import torch
import spacy
import re
import string
import nltk
nltk.data.path.append('D:/nltk_data')
from nltk import WordNetLemmatizer
import os
import json
from sentence_transformers import SentenceTransformer,util
# 映射英文数字单词到数字
nlp = spacy.load("en_core_web_sm")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
NUMBER_WORD_TO_DIGIT = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10"
}
lemmatizer = WordNetLemmatizer()
def singular(word):
    doc = nlp(word)
    token = doc[0]
    sing =  token.lemma_  # 复数 → 单数
    return sing
def normalize_answer(ans: str, do_lemmatize: bool = True) -> str:
    """
    标准化 VQA 答案用于公平评估。
    :param ans: 输入答案字符串
    :param do_lemmatize: 是否启用词形还原（默认为True）
    :return: 规范化后的答案字符串
    """
    if not ans:
        return ""

    # 小写处理
    ans = ans.lower()

    ans = ans.replace("-", " ")

    # 去掉不需要的标点符号
    ans = re.sub(r"[\[\]{}()\"“”‘’']", "", ans)  # 去掉括号、引号等
    ans = ans.translate(str.maketrans("", "", string.punctuation))

    # 去掉冠词
    ans = re.sub(r'\b(a|an|the)\b', ' ', ans)

    # 去掉多余空格
    ans = re.sub(r'\s+', ' ', ans).strip()

    # 词形还原（如 working -> work, dogs -> dog）
    tokens = ans.split()

    # 替换数字词为数字
    tokens = [NUMBER_WORD_TO_DIGIT.get(t, t) for t in tokens]
    #tokens = [singular(t) for t in tokens]

    if do_lemmatize and len(tokens) <= 2:
        lemmatized = [lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='v'), pos='n') for word in tokens]
        # lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
        ans = " ".join(lemmatized)
    return ans

def compute_vqa_accuracy(pred_answer: str, gt_answers: list,thresholds = [0.7,0.8,0.9]):
    """
    计算单个问题的VQA准确率
    :param pred_answer: str，模型预测答案
    :param gt_answers: list of str，10个annotators的答案
    :return: Dict
    """
    # pred_answer = pred_answer.strip().lower().rstrip(".!,")
    # gt_answers = [ans.strip().lower() for ans in gt_answers]
    pred_answer = normalize_answer(pred_answer)
    gt_answers = [normalize_answer(ans) for ans in gt_answers]
    print(gt_answers)
    print(pred_answer)
    emb_pred = encoder.encode(pred_answer, convert_to_tensor=True)
    emb_gts = encoder.encode(gt_answers, convert_to_tensor=True)
    sims = util.cos_sim(emb_pred, emb_gts).cpu().numpy().flatten()
    print(sims)
    results = {}
    results["exact_acc"] = round(min(Counter(gt_answers)[pred_answer] / 3.0, 1.0), 2)
    for t in thresholds:
        matched = sum(sim >= t for sim in sims)
        acc = min(matched / 3.0, 1.0)
        results[f"threshold_{t}"] = round(acc, 2)
    return results

def save_predictions(pred_dict, save_path):
    """
    保存预测结果字典为JSON文件
    :param pred_dict: dict, {question_id: predicted_answer}
    :param save_path: str, 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(pred_dict, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved to {save_path}")

def evaluate_vqa(model, dataloader, device):
    total_score_exact_global_local = 0
    total_score_acc07_global_local = 0
    total_score_acc08_global_local = 0
    total_score_acc09_global_local = 0
    count_global_local = 0

    total_score_exact_global = 0.0
    total_score_acc07_global = 0.0
    total_score_acc08_global = 0.0
    total_score_acc09_global = 0.0
    count_global = 0

    with torch.no_grad():
        pred_answers_global = {}
        pred_answers_global_local = {}
        for batch in dataloader:
            image_paths = batch['image_path']
            questions = batch['question']
            question_ids = batch['question_id']
            gt_answers_batch = batch['gt_answers']
            print("Current batch questions:", questions)
            pred_answers_batch_global_local = model.process_batch(image_paths,questions)
            for qid,pred, gt in zip(question_ids,pred_answers_batch_global_local, gt_answers_batch):
                pred_answers_global_local[qid] = pred
                results = compute_vqa_accuracy(pred,gt)
                exact_match = results["exact_acc"]
                acc_07 = results["threshold_0.7"]
                acc_08 = results["threshold_0.8"]
                acc_09 = results["threshold_0.9"]
                total_score_exact_global_local += exact_match
                total_score_acc07_global_local += acc_07
                total_score_acc08_global_local += acc_08
                total_score_acc09_global_local += acc_09
                count_global_local += 1
                print(f"acc_now_global_local: {total_score_exact_global_local}/{count_global_local} ({round(total_score_exact_global_local / count_global_local, 4)})")
                print(f"acc_now_global_local: {total_score_acc07_global_local}/{count_global_local} ({round(total_score_acc07_global_local / count_global_local, 4)})")
                print(f"acc_now_global_local: {total_score_acc08_global_local}/{count_global_local} ({round(total_score_acc08_global_local / count_global_local, 4)})")
                print(f"acc_now_global_local: {total_score_acc09_global_local}/{count_global_local} ({round(total_score_acc09_global_local / count_global_local, 4)})")
            # for qid,pred, gt in zip(question_ids,pred_answers_batch_global, gt_answers_batch):
            #     pred_answers_global[qid] = pred
            #     # score = compute_vqa_accuracy(pred, gt)  # pred: str, gt: list[str]
            #     # total_score += score
            #     # count += 1
            #     # print(f"acc_now: {total_score}/{count} ({round(total_score / count, 4)})")
            #     results = compute_vqa_accuracy1(pred,gt)
            #     exact_match = results["exact_acc"]
            #     acc_07 = results["threshold_0.7"]
            #     acc_08 = results["threshold_0.8"]
            #     acc_09 = results["threshold_0.9"]
            #     total_score_exact_global += exact_match
            #     total_score_acc07_global += acc_07
            #     total_score_acc08_global += acc_08
            #     total_score_acc09_global += acc_09
            #     count_global += 1
            #     print(f"acc_now_global: {total_score_exact_global}/{count_global} ({round(total_score_exact_global / count_global, 4)})")
            #     print(f"acc_now_global: {total_score_acc07_global}/{count_global} ({round(total_score_acc07_global / count_global, 4)})")
            #     print(f"acc_now_global: {total_score_acc08_global}/{count_global} ({round(total_score_acc08_global / count_global, 4)})")
            #     print(f"acc_now_global: {total_score_acc09_global}/{count_global} ({round(total_score_acc09_global / count_global, 4)})")

            # print(pred_answer)
            # 准确率
            # score = compute_vqa_accuracy(pred_answer, gt_answers)
            # total_score += score
            # count += 1
            # print(f"acc_now:{total_score}/{count}")

    avg_acc_2 = total_score_exact_global_local / count_global_local if count_global_local > 0 else 0.0
    print(f"Average VQA acc_global_local: {avg_acc_2}")
    # avg_acc = total_score_exact_global / count_global if count_global > 0 else 0.0
    # print(f"Average VQA acc_global: {avg_acc}")
    #save_predictions(pred_answers_global,save_path = "D:/save")
    save_predictions(pred_answers_global_local,save_path = "D:/save")
    return avg_acc_2
