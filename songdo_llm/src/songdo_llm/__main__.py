import torch
from ollama import chat
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DebertaTokenizer,
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
)
import time
from metr.components import TrafficData
from .dataset import BaseTrafficDataModule

data = TrafficData.import_from_hdf("../datasets/metr-imc/metr-imc.h5").data
module = BaseTrafficDataModule()



LLM_MODEL_NAME = "deepseek-r1:8b"
CLASSIFICATION_MODEL_NAME = "microsoft/deberta-v3-base"

tokenizer: DebertaV2Tokenizer = DebertaV2Tokenizer.from_pretrained(
    CLASSIFICATION_MODEL_NAME
)
cls_model: DebertaV2ForSequenceClassification = (
    DebertaV2ForSequenceClassification.from_pretrained(
        CLASSIFICATION_MODEL_NAME, num_labels=1
    )
)



# stream = chat(
#     model="llama3.1",
#     messages=[
#         {
#             "role": "user",
#             "content": """
# In South Korea, June 24 is not a nationally recognized holiday or commemorative day with any widespread official significance. It falls just one day before June 25, which is the date the Korean War (often referred to in Korea as “6·25 전쟁”) began in 1950. Because June 25 is widely commemorated for its historical importance, June 24 may be noted in relation to those events, but there is no special observance or public holiday specifically designated on June 24.

# If you are looking for significant June dates in Korea:
# 	•	June 6 is 현충일 (Memorial Day), honoring the sacrifices of fallen soldiers.
# 	•	June 10 is the anniversary of the 6·10 민주항쟁 (June 10 Democratic Uprising) in 1987.
# 	•	June 25 marks the outbreak of the Korean War in 1950.

# Aside from these dates, there is no official or widely celebrated event on June 24.

# ------

# Considering this information, traffic flow in June 24 will be normal as usual or not?
# """,
#         }
#     ],
#     stream=True,
# )

# stream = chat(
#     model=LLM_MODEL_NAME,
#     messages=[
#         {
#             "role": "system",
#             "content": "From now on, the user will only ask about Korean commemorative dates. Your answers should describe how important the date is to Koreans. If the date the user asks about is not considered special in Korea, please respond with, \"That date is not related to Korea.\"",
#         },
#         {
#             "role": "user",
#             "content": "Is there any special meaning of August 15 in Korea?",
#         }
#     ],
#     stream=True,
# )

# result: str = ""
# for chunk in stream:
#     word = chunk["message"]["content"]
#     print(word, end="", flush=True)
#     result += word
# print()

# inputs = tokenizer(result, return_tensors="pt", padding=True, truncation=True)

# cls_model.eval()
# with torch.no_grad():
#     outputs = cls_model(**inputs)
#     # outputs.logits shape: (batch_size, 1)
#     logits = outputs.logits

# print(logits)
