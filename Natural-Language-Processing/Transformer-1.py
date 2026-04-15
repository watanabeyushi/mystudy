#%%
from transformers import pipeline

text_classification_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-marc_ja"
)

positive_text = "世界には言葉がわからなくても感動する音楽がある。"
print(text_classification_pipeline(positive_text)[0])

negative_text = "世界には言葉がでないほどひどい音楽がある。"
print(text_classification_pipeline(negative_text)[0])


#%%

nli_pipeline = pipeline(model="llm-book/bert-base-japanese-v3-jnli")

text = "二人の男性がジェット機を見ています"
entailment_text = "ジェット機を見ている人が二人います"
print(nli_pipeline({"text": text, "text_pair": entailment_text}))

contradiction_text = "二人の男性が飛んでいます"
print(nli_pipeline({"text": text, "text_pair": contradiction_text}))

neutral_text = "2人の男性が、白い飛行機を眺めています"
print(nli_pipeline({"text": text, "text_pair": neutral_text}))





