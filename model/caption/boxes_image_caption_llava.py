import os
import matplotlib.pyplot as plt
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "../../hf_cache"
import torch
from sentence_transformers import SentenceTransformer,util
from transformers import AutoProcessor,LlavaOnevisionForConditionalGeneration
from PIL import Image, ImageEnhance ,ImageDraw
import re
from collections import defaultdict
import numpy as np
from typing import List

encoder = SentenceTransformer("all-MiniLM-L6-v2")
# ====================================
# 1. 加载模型
# ====================================
def load_caption_model_llava(model_id=None,device="cuda" if torch.cuda.is_available() else "cpu"):
    if model_id is None:
        model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model_LLava = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    # 加载模型和处理器
    processor = AutoProcessor.from_pretrained(model_id)
    return model_LLava, processor
def highlight_bbox_on_image(image,bbox,darken_factor = 0.2):
    # 创建整张图变暗的副本(darken_factor<1变暗，>1变亮，=1不变)
    darkened_image = ImageEnhance.Brightness(image).enhance(darken_factor)
    # 创建 mask 以保留 bbox 区域
    ##Image.new(mode,size,color);
    ##          mode = 'L':8 bit灰度图像，像素值[0,255],0表示全黑，不保留任何区域
    mask = Image.new("L", image.size, 0)
    ##创建绘画对象，可以在mask上绘制图像，fill=255，将bbox部分保留
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=255)

    # 将亮区域（bbox）从原图抠出，贴到变暗背景上
    output = Image.composite(image, darkened_image, mask)

    return output

def extract_generated_caption(full_output: str, prompt: str = None, length: int = 2, debug: bool = False) -> List[str]:
    """
    从模型输出中去除 prompt，仅保留生成内容，并提取前最多 length 句 caption。
    行为：
      - 若引号内的末尾标点实际上是句末（引号外跟大写/数字/引号），
        则把该末尾标点移出引号（例如: "310." This -> "310". This）。
      - 否则保护引号内的 . ? !，避免把引号内的标点造成误分句。
      - 在保护后按规则分句：只在 [.?!] 后 + 空格 + 大写/数字/引号处分句。
    返回：分好的句子列表，长度最多为 length。
    """
    # ===0. 去除输出格式
    full_output = re.sub(r"<\|im_start\|>", "", full_output).strip()
    prompt = re.sub(r"<\|im_start\|>|<\|im_end\|>", "", prompt).strip()
    prompt = prompt.replace("user <image>","").strip()
    # ===1. 去除 prompt
    if prompt in full_output:
        generated_part = full_output.split(prompt, 1)[-1].strip()
    else:
        generated_part = full_output.strip()

    # === 2. 处理所有引号块（保护内部标点并在必要时把末尾标点移出引号） ===
    DOT_PH = "<DOT_IN_Q>"
    QMARK_PH = "<QMARK_IN_Q>"
    EXCL_PH = "<EXCL_IN_Q>"

    # 匹配双引号、智能引号、单引号（不处理嵌套）
    quote_pattern = re.compile(
        r'(?:"[^"]*"|“[^”]*”|(?<!\w)\'[^\'\n]*\'(?!\w))'
    )

    out_chunks = []
    last_idx = 0
    text = generated_part

    for m in quote_pattern.finditer(text):
        start, end = m.span()
        # 先把引号前的部分直接加入输出
        out_chunks.append(text[last_idx:start])

        quoted = m.group(0)          # 包含引号的整个片段，例如: "OZ YAVUZ 310."
        inner = quoted[1:-1]         # 引号内部，不包含引号本身

        # 找到引号结束后第一个非空字符，用于判断是否应该把内部末尾标点视为句末
        next_char = ''
        i = end
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text):
            next_char = text[i]

        # 判断内部最后一个字符是否为句末标点
        inner_ends_with_term = bool(inner) and inner[-1] in '.?!'

        # 如果内部有末尾标点，且引号外的第一个非空字符是大写字母/数字/引号，
        # 则认为这个末尾标点实际上是句末标志（例如: "...310." This ...）
        if inner_ends_with_term and next_char and (next_char.isupper() or next_char.isdigit() or next_char in ['"', '“', "'"]):
            # 把引号内部的最后一个标点移出：保留引号内其他标点的占位保护
            body = inner[:-1]
            final_punct = inner[-1]  # '.' 或 '?' 或 '!'
            # 保护 body 内的其它终结符
            body_pro = re.sub(r'\.', DOT_PH, body)
            body_pro = re.sub(r'\?', QMARK_PH, body_pro)
            body_pro = re.sub(r'!', EXCL_PH, body_pro)
            # 生成新的 chunk： " + protected_body + " + final_punct (放在引号外)
            new_quoted = quoted[0] + body_pro + quoted[-1] + final_punct
            out_chunks.append(new_quoted)
        else:
            # 否则：保护引号内部的所有句末标点（不移出末尾）
            inner_pro = re.sub(r'\.', DOT_PH, inner)
            inner_pro = re.sub(r'\?', QMARK_PH, inner_pro)
            inner_pro = re.sub(r'!', EXCL_PH, inner_pro)
            new_quoted = quoted[0] + inner_pro + quoted[-1]
            out_chunks.append(new_quoted)

        last_idx = end

    # 追加最后一段
    out_chunks.append(text[last_idx:])
    protected_text = "".join(out_chunks)

    if debug:
        print("DEBUG: protected_text (placeholders inserted / moved punctuation):")
        print(repr(protected_text))

    # === 3. 智能分句：仅在 [.?!] 后跟空格且下一个字符为大写/数字/引号时分句 ===
    # 用 lookbehind + lookahead：只在真正可能的句末处断
    splitter = re.compile(r'(?<=[.?!])\s+(?=[A-Z0-9"“\'])')
    raw_sentences = splitter.split(protected_text)

    if debug:
        print("DEBUG: raw_sentences (after split, before restore):")
        for i, s in enumerate(raw_sentences):
            print(i, repr(s))

    # === 4. 恢复引号内占位符 ===
    restored = []
    for s in raw_sentences:
        s2 = s.replace(DOT_PH, '.').replace(QMARK_PH, '?').replace(EXCL_PH, '!')
        restored.append(s2.strip())

    # === 5. 过滤空句并返回前 length 个 ===
    sentences = [s for s in restored if s]
    if debug:
        print("DEBUG: final sentences:")
        for i, s in enumerate(sentences[:length]):
            print(i, repr(s))
    return sentences[:length]

def format_local_captions_with_label_index(local_captions, labels):
    label_counter = defaultdict(int)
    formatted = []

    for idx, (caption, label) in enumerate(zip(local_captions, labels)):
        label_counter[label] += 1
        formatted_caption = f"[{idx}] {label}: {caption}"
        formatted.append(formatted_caption)

    return formatted

def expand_crop(image: Image.Image, box, expand_ratio=0.2):
    """
    以 box 为中心，四周各扩 expand_ratio*原图宽/高，然后 crop。
    expand_ratio=0.5 表示左右各加 0.5w，上下各加 0.5h，相当于边长放大一倍。
    """
    w, h = image.size
    x0, y0, x1, y1 = box

    # box 中心
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    w_o = x1-x0
    h_o = y1-y0

    expand_w = int(w_o * expand_ratio)
    expand_h = int(h_o * expand_ratio)

    # 新边界
    new_x0 = max(0, int(cx - (x1 - x0) / 2 - expand_w))
    new_y0 = max(0, int(cy - (y1 - y0) / 2 - expand_h))
    new_x1 = min(w, int(cx + (x1 - x0) / 2 + expand_w))
    new_y1 = min(h, int(cy + (y1 - y0) / 2 + expand_h))

    return image.crop((new_x0, new_y0, new_x1, new_y1))
def generate_local_caption(
        model_LLava,
        processor,
        image,
        boxes,
        labels,
        keywords,
):
    local_captions = []
    for box, label in zip(boxes, labels):
        try:
            highlight_image = highlight_bbox_on_image(image,box)
            # x0, y0, x1, y1 = map(int, box)
            # crop = image.crop((x0, y0, x1, y1))
            # crop = expand_crop(image,box)

            prompt_text = (
                    f"Describe the {label} in detail,"
                    f"based on these keywords: {keywords}. "
                    "Ignore any incorrect assumptions from the keywords."
            )
            # 构造 conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image"}
                    ]
                }
            ]

            # apply_chat_template 得到最终 prompt
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=highlight_image, text=prompt, return_tensors='pt').to(model_LLava.device, torch.float32)
            with torch.no_grad():
                output = model_LLava.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                )
            caption_with_prompt = processor.decode(output[0][2:], skip_special_tokens=True)
            captions = extract_generated_caption(caption_with_prompt, prompt)
            print("local_generated_captions:",captions)
            local_captions.append(captions[0])
        except Exception as e:
            print(f"[Warning] Failed to generate caption: {e}")
            continue  # 跳过当前
    if len(local_captions) == 0:
        return []
    local_captions = format_local_captions_with_label_index(local_captions, labels)
    return local_captions
def generate_whole_caption(model_LLava,processor,image,keywords,num_captions = 5):
    prompt_text = (
        f"Describe the overall image related about {keywords}."
    )
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"}
            ]
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(model_LLava.device, torch.float32)
    captions = []
    for i in range(num_captions):
        try:
            with torch.no_grad():
                output = model_LLava.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                )
            caption_with_prompt = processor.decode(output[0][2:], skip_special_tokens=True)
            generated_captions = extract_generated_caption(caption_with_prompt, prompt)
            print("whole_generated_captinos:",generated_captions)
            captions.append(generated_captions[0])
        except Exception as e:
            print(f"Error generating caption: {e}")
            captions.append("Error generating caption")
    return captions


if __name__ == "__main__":
    boxes = [[70.20237731933594, 157.18438720703125, 221.37890625, 427.2403259277344],
     [184.4833984375, 253.79953002929688, 514.282958984375, 426.9772644042969],
     [509.21881103515625, 7.38298225402832, 639.2864990234375, 427.2866516113281],
     [118.79777526855469, 17.223979949951172, 459.5704040527344, 257.8733825683594],
     [205.89682006835938, 250.04986572265625, 294.79827880859375, 374.8625793457031]]
    # #labels = ['person', 'seat', 'door', 'window', 'bag']
    image_path = "../../data/coco/val2014/COCO_val2014_000000308394.jpg"
    image = Image.open(image_path).convert("RGB")
    for box in boxes:
        # crop = highlight_bbox_on_image(image, box)
        crop = expand_crop(image,box)
        # x0, y0, x1, y1 = map(int, box)
        # crop = image.crop((x0, y0, x1, y1))
##################图像输出处理部分####################
    #关键修正：设置画布尺寸与裁剪区域完全一致
        w, h = crop.size
        plt.figure(figsize=(w / 100, h / 100), dpi=100)  # 将像素转换为英寸（1英寸=100点）

        #显示图像并关闭坐标轴
        plt.imshow(np.array(crop), aspect='equal')  # aspect='equal'保持宽高比
        plt.axis('off')

        #强制去除所有白边
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.imshow(crop)
        plt.axis('off')  # 去掉坐标轴
        plt.show()