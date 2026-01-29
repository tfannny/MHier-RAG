import logging
from abc import ABC, abstractmethod

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv
import logging

import os

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="/root/autodl-tmp/model/sbert"): # sentence-transformers/multi-qa-mpnet-base-cos-v1
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)


class Qwen3LocalEmbeddingModel(BaseEmbeddingModel):
    # 如果模型下载到了本地,model_name填模型保存目录
    def __init__(self, model_name="/root/autodl-tmp/model/Qwen3-Embedding-0.6B"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        logging.debug(f"create embedding by Qwen3-Embedding-0.6B")
        return self.model.encode(text)


class Qwen3EmbeddingModel(BaseEmbeddingModel):
    # 如果模型下载到了本地,model_name填模型保存目录
    def __init__(self, model_name="text-embedding-v4"):
        self.model = model_name
        load_dotenv()
        self.client = OpenAI(api_key=os.environ["QWEN_API_KEY"],
                             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    def create_embedding(self, text):
        client = self.client
        completion = client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=1024,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
            encoding_format="float"
        )
        return completion.model_dump_json()
