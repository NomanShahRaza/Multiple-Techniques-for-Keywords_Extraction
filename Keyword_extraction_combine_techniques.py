"""
@author: Noman Raza Shah
"""
#%% 
# ============================== #
# KEYWORDS EXTRACTION TECHNIQUES #
# ============================== #

def gensim_keyword(original_text):
    import gensim
    from gensim.summarization import keywords
    keywords = keywords(original_text,words=5)
    return keywords


def rake_keyword(original_text):
    from rake_nltk import Rake
    r = Rake()
    r = Rake(include_repeated_phrases=False)
    r.extract_keywords_from_text(original_text)
    keywords=r.get_ranked_phrases()[0:6]
    return keywords


def yake_keyword(original_text):
    import yake
    kww=[]
    language = "en"
    max_ngram_size = 3
    deduplication_thresold = 0.2
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 20
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(original_text)
    for kw in keywords[0:5]:
        print(kw)
        kww.append(kw)
    return kww

def bert_keyword(original_text):
    from keybert import KeyBERT
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(original_text, keyphrase_ngram_range=(1, 3), stop_words='english', 
                              use_maxsum=True, nr_candidates=20, top_n=5)
    return keywords

def senTransformer_keyword(original_text):
    from sklearn.feature_extraction.text import CountVectorizer
    n_gram_range = (1, 3)
    stop_words = "english"   
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([original_text])   #Convert a collection of text documents to a matrix of token counts.
    candidates = count.get_feature_names()
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_embedding = model.encode([original_text])
    candidate_embeddings = model.encode(candidates)
    top_n = 5
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    return keywords


def senTransformer_roberta_keyword(original_text):
    from sklearn.feature_extraction.text import CountVectorizer
    n_gram_range = (1, 3)
    stop_words = "english"
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([original_text])
    all_candidates = count.get_feature_names()
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(original_text)
    noun_phrases = set(chunk.text.strip().lower() for chunk in doc.noun_chunks)
    nouns = set()
    for token in doc:
        if token.pos_ == "NOUN":
            nouns.add(token.text)        
    all_nouns = nouns.union(noun_phrases)
    candidates = list(filter(lambda candidate: candidate in all_nouns, all_candidates))  
    from transformers import AutoModel, AutoTokenizer
    model_name = "distilroberta-base"  # used a knowledge-distilled version of RoBERTa.
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    candidate_tokens = tokenizer(candidates, padding=True, return_tensors="pt")
    candidate_embeddings = model(**candidate_tokens)["pooler_output"]  
    text_tokens = tokenizer([original_text], padding=True, return_tensors="pt")
    text_embedding = model(**text_tokens)["pooler_output"]
    candidate_embeddings = candidate_embeddings.detach().numpy()
    text_embedding = text_embedding.detach().numpy()
    from sklearn.metrics.pairwise import cosine_similarity
    top_k = 5
    distances = cosine_similarity(text_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_k:]]
    return keywords

#%%

original_text= """we are working in Artificial intelligence lab. we have few groups who are focus on the area of machine learning and artificial intelligence.
Google tackles the most challenging problems in computer science. Our teams aspire to make discoveries that impact everyone, and core to our approach is sharing 
our research and tools to fuel progress in the field. Our researchers publish regularly in academic journals, release projects as open source, and apply research 
to Google products."""


#%%
print("="*70)
print('The Keywords by using Gensim Keywords Extraction Technique is "', gensim_keyword(original_text), '"')
print("="*70)

#%%
print("="*70)
print('The Keywords by using RAKE Keywords Extraction Technique is "', rake_keyword(original_text), '"')
print("="*70)

#%%
print("="*70)
print('The Keywords by using YAKE Keywords Extraction Technique is "', yake_keyword(original_text), '"')
print("="*70)

#%%
print("="*70)
print('The Keywords by using Senetnce Transformer model is "',senTransformer_keyword(original_text), '"')
print("="*70)

#%%
print("="*70)
print('The Keywords by using Senetnce Transformer RoBERTa model is "',senTransformer_roberta_keyword(original_text), '"')
print("="*70)

#%%
print("="*70)
print('The Keywords by using BERT model is "',bert_keyword(original_text), '"')
print("="*70)









