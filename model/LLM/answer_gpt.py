import openai
import time
def answer_question(image_caption,boxes_caption,relation_phrase, question,model):
    if not boxes_caption:
        prompt = f"""You are a helpful assistant for a Visual Question Answering (VQA) task.
Base your answer on the provided visual information.
If the question cannot be answered directly from the global caption or relationships, use general world knowledge or commonsense to infer the answer.
Below is the information extracted from the current image:

[Global Caption]
{image_caption}

Please answer the following question based on the above information.
If the answer requires external knowledge, combine the detected objects with your world knowledge.
****Answer with a single word if possible, otherwise the shortest phrase.Do not add adjectives unless they are specific names.****

Question: {question}
Answer:"""
    elif not relation_phrase:
        prompt = f"""You are a helpful assistant for a Visual Question Answering (VQA) task.
Base your answer on the provided visual information.
If the question cannot be answered directly from the global caption or local captions or relationships, use general world knowledge or commonsense to infer the answer.
Below is the information extracted from the current image:

[Global Caption]
{image_caption}

[Local Captions]
Here are the descriptions of detected objects in the image:
# object: description of that object
{boxes_caption}

Please answer the following question based on the above information.
If the answer requires external knowledge, combine the detected objects with your world knowledge.
****Answer with a single word if possible, otherwise the shortest phrase.Do not add adjectives unless they are specific names.****

Question: {question}
Answer:"""
    else:
        prompt = f"""You are a helpful assistant for a Visual Question Answering (VQA) task.
Base your answer on the provided visual information.
If the question cannot be answered directly from the global caption or local captions or relationships, use general world knowledge or commonsense to infer the answer.
Below is the information extracted from the current image:

[Global Caption]
{image_caption}

[Local Captions]
Here are the descriptions of detected objects in the image:
# object: description of that object
{boxes_caption}

[Relationships]
Here are the relationships between detected objects in the image:
{relation_phrase}

Please answer the following question based on the above information.
If the answer requires external knowledge, combine the detected objects with your world knowledge.
****Answer with a single word if possible, otherwise the shortest phrase.Do not add adjectives unless they are specific names.****

Question: {question}
Answer:"""
    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=5,  # 控制输出不超过 20 token
                temperature=0,
                top_p=1,
                request_timeout=20
            )
            answer = response["choices"][0]["message"]["content"]
            return answer
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # 等待后重试
    return "Error: API request failed after retries"


def answer_question_global(image_caption, question, model):
    prompt = f"""You are a helpful assistant for a Visual Question Answering (VQA) task.
Base your answer on the provided visual information.
If the question cannot be answered directly from the global caption or relationships, use general world knowledge or commonsense to infer the answer.
Below is the information extracted from the current image:

[Global Caption]
{image_caption}

Please answer the following question based on the above information.
If the answer requires external knowledge, combine the detected objects with your world knowledge.
****Answer with a single word if possible, otherwise the shortest phrase.Do not add adjectives unless they are specific names.****

Question: {question}
Answer:"""
    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=5,  # 控制输出不超过 20 token
                temperature=0,
                top_p=1,
                request_timeout=20
            )
            answer = response["choices"][0]["message"]["content"]
            return answer
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # 等待后重试
    return "Error: API request failed after retries"


def answer_question_local(image_caption, local_captions, question, model):
    if not local_captions:
        prompt = f"""You are a helpful assistant for a Visual Question Answering (VQA) task.
Base your answer on the provided visual information.
If the question cannot be answered directly from the global caption or relationships, use general world knowledge or commonsense to infer the answer.
Below is the information extracted from the current image:

[Global Caption]
{image_caption}

Please answer the following question based on the above information.
If the answer requires external knowledge, combine the detected objects with your world knowledge.
****Answer with a single word if possible, otherwise the shortest phrase.Do not add adjectives unless they are specific names.****

Question: {question}
Answer:"""
    else:
        boxes_captions = [
            cap.split("]", 1)[-1].strip() if cap.startswith("[") else cap
            for cap in local_captions
        ]
        prompt = f"""You are a helpful assistant for a Visual Question Answering (VQA) task.
Base your answer on the provided visual information.
If the question cannot be answered directly from the global caption or local captions or relationships, use general world knowledge or commonsense to infer the answer.
Below is the information extracted from the current image:

[Global Caption]
{image_caption}

[Local Captions]
Here are the descriptions of detected objects in the image:
# object: description of that object
{boxes_captions}

Please answer the following question based on the above information.
If the answer requires external knowledge, combine the detected objects with your world knowledge.
****Answer with a single word if possible, otherwise the shortest phrase.Do not add adjectives unless they are specific names.****

Question: {question}
Answer:"""
    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=5,  # 控制输出不超过 20 token
                temperature=0,
                top_p=1,
                request_timeout=20,
            )
            answer = response["choices"][0]["message"]["content"]
            return answer
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # 等待后重试
    return "Error: API request failed after retries"
