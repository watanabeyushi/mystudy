from gensim.models import KeyedVectors
import numpy as np
import MeCab
import os

# MeCabの設定
#os.environ["MECABRC"] = "/etc/mecabrc"
tagger = MeCab.Tagger("-Owakati")

# Word2Vecモデルのロード
# 注意: モデルファイルのパスを適切に設定してください
model = KeyedVectors.load_word2vec_format('jawiki.word_vectors.200d.txt', binary=False)

# 単語ベクトルの取得
word = '札幌'
weight = model.get_vector(word)
print(f"'{word}'のベクトル:")
print(weight)

# 類似度が高い単語を取得
sim = model.most_similar([word])
print(f"\n'{word}'と類似度が高い単語:")
print(sim)

# 文章のベクトル（分散表現）を求める関数
num_features = 200

def get_sentence_vector(sentence):
    # MeCabで分かち書き
    words = tagger.parse(sentence).replace(' \n', '').split()
    # 文章ベクトルの配列を用意
    sentence_vec = np.zeros((num_features,), dtype="float32")
    
    # ボキャブラリに登録されている単語のみを使用
    wordList = []
    for w in words:
        if w in model:
            wordList.append(w)
    
    # 単語ベクトルを足していく
    for word in wordList:
        word_vec = model.get_vector(word)
        sentence_vec = np.add(sentence_vec, word_vec)
    
    # 平均を求める
    if len(wordList) > 0:
        sentence_vec = np.divide(sentence_vec, len(wordList))
    
    return sentence_vec

# 文章のベクトル化の例
sentence = "文章を分散表現に変換する方法を学ぶ"
sentence_vec = get_sentence_vector(sentence)
print(f"\n文章のベクトル: {sentence}")
print(sentence_vec)

