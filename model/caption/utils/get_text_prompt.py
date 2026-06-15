from keybert import KeyBERT

kw_model = KeyBERT(model='all-MiniLM-L6-v2')

def extract_keywords_from_question(question, top_k=1, diversity=0.5):
    """
    使用 KeyBERT 提取 VQA 问题中的关键词。
    KeyBERT 利用 BERT 向量计算每个词/短语与整句的语义相似度。

    参数:
        question (str): 输入问题文本，如 "What is the man holding in his hand?"
        top_k (int): 返回的关键词数量
        min_df (int): 最低词频阈值（默认1，对短句通常无需修改）
        diversity (float): 控制关键词之间的多样性，范围 [0,1]，越高代表去重越强

    返回:
        keywords (list[str]): 提取出的关键词列表
    """

    if not question or not isinstance(question, str):
        return []

    # 提取关键词，use_mmr=True 表示使用 Maximal Marginal Relevance，防止重复
    keywords = kw_model.extract_keywords(
        question,
        keyphrase_ngram_range=(1,1),   # 提取1-2词短语（如 "red car"）
        stop_words='english',
        use_mmr=True,
        diversity=diversity,
        top_n=top_k
    )
    # for kw,score in keywords:
    #     print(kw, score)
    # keywords 是 [(word, score)] 格式，只取 word
    return [kw for kw, score in keywords]