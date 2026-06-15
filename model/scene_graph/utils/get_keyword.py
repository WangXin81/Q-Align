import spacy
import inflect
from nltk import RegexpParser
from nltk.corpus import wordnet as wn
nlp = spacy.load("en_core_web_sm")
COCO_NONE = ['airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']
LOCALIZATION_BLACKLIST = {
    # ----- 顶级统称 -----
    "thing", "object", "item", "device", "structure", "area", "place", "location",
    "surface", "product", "equipment", "tool", "apparatus", "facility", "container",
    "piece", "stuff","substance",
    # ----- 属性维度 -----
    "position", "direction", "material", "type", "kind", "part", "portion",
    # ----- 相对方位 -----
    "eating", "landing", "front", "back", "top", "bottom", "corner", "side", "end",
    # ----- 场景级抽象 -----
    "traffic", "weather", "scenery"
}
QUANT_EVAL_ADJ = [
    # 数量 & 泛指
    "all", "any", "each", "every", "some", "few", "fewer", "many", "much", "more", "most", "several",
    "numerous", "countless", "plenty", "various", "multiple", "enough", "sufficient", "insufficient",
    "extra", "additional", "further", "other", "another", "same", "different", "certain",
    # 顺序 & 界限
    "first", "second", "third", "last", "final", "next", "previous", "initial", "ultimate",
    "main", "major", "minor", "primary", "secondary", "chief", "principal", "central",
    # 程度 & 强调
    "very", "quite", "rather", "pretty", "fairly", "too", "so", "extremely", "highly", "deeply",
    "really", "truly", "absolutely", "totally", "completely", "entirely", "partly", "partially",
    "almost", "nearly", "hardly", "barely", "scarcely", "just", "only", "even", "still",
    # 评价 好/坏
    "good", "better", "best", "great", "excellent", "perfect", "wonderful", "fantastic", "amazing",
    "awesome", "brilliant", "outstanding", "superb", "fine", "nice", "lovely", "beautiful", "pretty",
    "bad", "worse", "worst", "awful", "terrible", "horrible", "disgusting", "nasty", "ugly", "gross",
    "poor", "mediocre", "average", "ordinary", "normal", "regular", "standard", "typical",
    "special", "particular", "specific", "unique", "rare", "common","cool", "usual", "unusual",
    "popular", "famous", "well-known", "unknown", "obscure", "infamous", "notorious",
    "important", "significant", "main", "key", "critical", "crucial", "vital", "essential",
    "useful", "helpful", "valuable", "worthless", "useless", "pointless", "hopeless",
    "easy", "difficult", "hard", "simple", "complex", "complicated", "straightforward",
    "cheap", "expensive", "free", "costly", "affordable", "pricey", "reasonable",
    "clean", "dirty", "tidy", "messy", "neat", "sloppy", "fresh", "stale", "new", "old",
    "healthy", "unhealthy", "safe", "dangerous", "risky", "secure", "unsafe", "harmful",
    "happy", "sad", "funny", "serious", "boring", "interesting", "exciting", "dull",
    "lucky", "unlucky", "good-luck", "bad-luck"
]
def singularize_with_spacy(word,tag):
    doc = nlp(word)
    token = doc[0]
    if token.morph.get("Number") == ["Plur"] or tag in ["NNS","NNPS"]:
        sing =  token.lemma_  # 复数 → 单数
        if sing.lower() == word.lower():
            inflect_engine = inflect.engine()
            singular = inflect_engine.singular_noun(sing)
            if singular == False:
                return word
            return singular
        return sing
    return word  # 单数保持不变
def spacy_pos_tag(sentence):
    """
    用 spaCy 做分词和词性标注，并返回 NLTK 格式的 (token, Penn-Treebank tag)
    """
    doc = nlp(sentence)
    return [(token.text, token.tag_) for token in doc]

import re

def extract_matched_prepositions(question: str, preposition_list: list) -> list:
    """
    从question中提取与给定介词列表匹配的介词（从长到短匹配，避免重叠）

    参数:
    - question: str，输入的问题句子
    - preposition_list: list[str]，介词或介词短语列表（例如 ['in front of', 'on', 'under', ...]）

    返回:
    - list[str]，匹配到的介词（无重复）
    """
    # Step 1: 词形还原（lemma 化）
    doc = nlp(question.lower())
    lemmatized_question = " ".join([token.lemma_ for token in doc])

    # Step 2: 初始化匹配标记
    occupied = [0] * len(lemmatized_question)
    matched = []
    # 按长度从长到短排序，避免短词（如 "in"）覆盖长词（如 "in front of"）
    sorted_preps = sorted(preposition_list, key=lambda x: -len(x))

    def mark_span(start, end):
        for i in range(start, end):
            occupied[i] = 1

    for prep in sorted_preps:
        pattern = r'\b' + re.escape(prep) + r'\b'  # 保证匹配完整词组
        for match in re.finditer(pattern, lemmatized_question):
            start, end = match.start(), match.end()
            # 检查是否与已有匹配重叠
            if sum(occupied[start:end]) == 0:
                mark_span(start, end)
                matched.append(prep)
                # break  # 可选，是否只取第一个出现的

    # 去重但保留原出现顺序
    final_matched = []
    seen = set()
    for m in matched:
        if m not in seen:
            seen.add(m)
            final_matched.append(m)

    return final_matched


def is_concrete_most_common_exact(word):
    """
    判断名词是否是具体物体，只看最常用的、名字完全匹配输入单词的义项
    """
    word = word.lower().strip()
    if word in LOCALIZATION_BLACKLIST:
        return False,f"'{word}' can't be localize."
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return False, f"'{word}' not found in WordNet."

    # 找第一个名字完全匹配输入单词的义项
    for syn in synsets:
        syn_name = syn.name().split('.')[0].lower()
        if syn_name == word:
            # 找到了最常用且名字匹配的义项
            hypernym_paths = syn.hypernym_paths()
            for path in hypernym_paths:
                lemmas = [h.name().split('.')[0] for h in path]
                if "physical_entity" in lemmas:
                    return True, {
                        "word": word,
                        "synset": syn.name(),
                        "definition": syn.definition(),
                        "lemmas_in_path": lemmas,
                        "physical_entity_found": True
                    }
            # 遍历完链子也没找到 physical_entity
            return False, {
                "word": word,
                "synset": syn.name(),
                "definition": syn.definition(),
                "lemmas_in_path": lemmas,
                "physical_entity_found": False
            }

    # 没有任何义项名字完全匹配
    return False, f"No noun sense with exact word '{word}' found."
INTERNAL_REMOVE = {
    "the","either", "a", "an", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "who's", "what's", "whats"
}
def clean_np_tokens(words_with_tags):
    """
    输入：subtree.leaves() 形式的 [(word, tag), ...]
    输出：清理后的 token 列表（保留 poss 合并、去除 DET/WH 等）
    """
    tokens = [w for w, t in words_with_tags]
    tags = [t for w, t in words_with_tags]

    cleaned = []
    i = 0
    while i < len(tokens):
        w = tokens[i]
        t = tags[i]

        # 如果是属格标记 's （POS），把它合并到前一个词（如果存在）
        if t == "POS" and len(cleaned) > 0:
            # 合并成 previous+'s'
            cleaned[-1] = cleaned[-1] + "'s"
            i += 1
            continue

        # 如果当前 token 是要在 NP 内部删除（det/疑问代词等），跳过它
        if w.lower() in INTERNAL_REMOVE and t in {"DT", "WDT", "WP", "WP$"}:
            i += 1
            continue

        # 其他正常保留（包含 JJ/NN/VBN 等，后续会挑选 head）
        cleaned.append(w)
        i += 1

    # 移除首尾的标点或空 token
    cleaned = [tok for tok in cleaned if re.search(r'\w', tok)]

    return cleaned

def extract_locatable_noun_phrases(question):
    """
    提取问题中可定位的名词短语（形容词 + 名词），并用 WordNet 判断是否为具体名词。
    保证复合名词（如 "salt water beach"）不会被拆开。
    """
    tagged = spacy_pos_tag(question)
    # 定义 NP 规则：可有形容词，后面跟一个或多个名词
    grammar = r"NP: {<DT>?<JJ.*|VBN|NN.*|POS>*<NN.*>+}"
    cp = RegexpParser(grammar)
    tree = cp.parse(tagged)
    noun_phrases = []

    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            words_with_tags = subtree.leaves()
            head_index = None
            for i in reversed(range(len(words_with_tags))):
                word, tag = words_with_tags[i]
                if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                    head_singular = singularize_with_spacy(word,tag)
                    if head_singular in COCO_NONE:
                        head_index = i
                        break
                    else:
                        is_concrete, info = is_concrete_most_common_exact(head_singular)
                        if is_concrete:
                            head_index = i
                            break

            if head_index is None:
                # NP中没有可定位名词
                continue
            head_word, head_tag = words_with_tags[head_index]
            head_singular = singularize_with_spacy(head_word, head_tag)

            # 将 subtree 的 token 做清理
            cleaned_tokens = clean_np_tokens(words_with_tags)

            # 把清理后的 token 映射回原来的 tag 信息（便于找 head）
            # 例如，如果 head_word 在 cleaned_tokens 中，确定它的新索引
            if head_word in cleaned_tokens:
                cleaned_head_idx = cleaned_tokens.index(head_word)
            else:
                continue

            # 修饰词：清理 token 中 head 之前的所有词
            modifiers = [
                w for i, w in enumerate(cleaned_tokens[:cleaned_head_idx])
                if w.lower() not in QUANT_EVAL_ADJ
            ]

            # 组合最终短语
            phrase = " ".join(modifiers + [head_singular]).lower().strip()
            if phrase:
                noun_phrases.append(phrase)

    return noun_phrases

def extract_constrained_noun_phrases_and_prepositions(question,prep_words):
    """
    提取带约束性的名词短语和介词：
    - 有修饰词：保留完整短语（修饰词+名词）
    - 无修饰词：保留单个名词
    :param question: str, 输入问题句子
    :return: (noun_phrases, prepositions)
        noun_phrases: list of str
        prepositions: list of str
    """
    prepositions = extract_matched_prepositions(question,prep_words)
    noun_phrases = extract_locatable_noun_phrases(question)
    return noun_phrases, prepositions