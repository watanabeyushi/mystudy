from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# サンプル文書
documents = [
    "私は機械学習が好きです",
    "自然言語処理を勉強しています",
    "機械学習と自然言語処理は面白い"
]

# TF-IDFベクトライザーの作成
vectorizer = TfidfVectorizer()

# 文書をTF-IDFベクトルに変換
tfidf_matrix = vectorizer.fit_transform(documents)

# 結果を表示
print("TF-IDF行列の形状:", tfidf_matrix.shape)
print("\n特徴語:", vectorizer.get_feature_names_out())
print("\nTF-IDF行列:")
print(tfidf_matrix.toarray())

# 特定の文書のTF-IDFベクトルを取得
doc_vector = tfidf_matrix[0].toarray()[0]
print("\n最初の文書のTF-IDFベクトル:")
print(doc_vector)

