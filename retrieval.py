import json
import logging
from typing import List, Tuple, Dict, Union
import pickle
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from reranking import LLMReranker
import base64
import re
from raptor import RetrievalAugmentation

_log = logging.getLogger(__name__)


def clean_json_string(json_str):
    json_str = re.sub(r'^(```|json\s*)\s*', '', json_str, flags=re.MULTILINE | re.IGNORECASE)
    json_str = re.sub(r'\s*(```|json\s*)$', '', json_str, flags=re.MULTILINE | re.IGNORECASE)
    return json_str.strip()


def parse_answer_string(answer_string):
    if answer_string.startswith("{"):
        pass
    else:
        answer_string = "{" + answer_string + "}"
    try:
        cleaned_string = clean_json_string(answer_string)
        parsed_dict = json.loads(cleaned_string)

        expected_keys = ['step_back_answer', 'reasoning_summary']
        if "schema" in parsed_dict.keys():
            parsed_dict = parsed_dict["schema"]

        for key in expected_keys:
            parsed_dict[key] = parsed_dict.get(key, None)

        return parsed_dict
    except json.JSONDecodeError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_step_back_query(query: str) -> str:
    system_prompt = '''you must return with follow format: {"schema": {"reasoning_summary": "string (concise 1-2 sentence summary)","step_back_answer": "string (direct answer to the step-back question)"}'''

    load_dotenv()
    llm = OpenAI(
        api_key=os.environ["QWEN_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    completion = llm.chat.completions.create(model="qwen-turbo",
                                             temperature=0,
                                             messages=[
                                                 {"role": "system", "content": system_prompt},
                                                 {"role": "user",
                                                  "content": "What is a step-back question of this question: " + query},
                                             ],
                                             extra_body={"enable_thinking": True},
                                             stream=True,
                                             )

    reasoning_content = ""
    answer_content = ""
    is_answering = False

    for chunk in completion:
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
            continue

        delta = chunk.choices[0].delta

        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            reasoning_content += delta.reasoning_content

        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                is_answering = True
            answer_content += delta.content
    print(answer_content)
    parsed_dict = parse_answer_string(answer_content)

    if parsed_dict is None:
        print("PARSED = None")
    else:
        answer_content_tmp = parsed_dict.get("step_back_answer")
        if answer_content_tmp is None:
            print("answer_content_tmp is None")
        elif len(answer_content_tmp) <= 10:
            print("answer is too short")
        else:
            answer_content = answer_content_tmp
            print(answer_content)

    return answer_content


class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key="",
            base_url="",
        )
        return llm

    def _load_dbs(self):
        all_dbs = []
        # Get list of JSON document paths
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        vector_db_files = {db_path.stem: db_path for db_path in self.vector_db_dir.glob('*.faiss')}

        for document_path in all_documents_paths:
            stem = document_path.stem
            if stem not in vector_db_files:
                _log.warning(f"No matching vector DB found for document {document_path.name}")
                continue
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue

            # Validate that the document meets the expected schema
            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue

            try:
                vector_db = faiss.read_index(str(vector_db_files[stem]))
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue

            report = {
                "name": stem,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-v4")  # "text-embedding-3-large"
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_document_name(self, document_name: str, query: str, llm_reranking_sample_size: int = None,
                                  top_n: int = 3, return_parent_pages: bool = False) -> List[Tuple[str, float]]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                _log.error(f"Report '{report.get('name')}' is missing 'metainfo'!")
                raise ValueError(f"Report '{report.get('name')}' is missing 'metainfo'!")

            if report['name'] == document_name:
                target_report = report
                break

        if target_report is None:
            _log.error(f"No report found with '{document_name}' document name.")
            raise ValueError(f"No report found with '{document_name}' document name.")

        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]

        actual_top_n = min(top_n, len(chunks))

        step_back_query = get_step_back_query(query)
        embedding = self.llm.embeddings.create(
            input=query + step_back_query,
            model="text-embedding-v4"
        )

        embedding = embedding.data[0].embedding
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)

        retrieval_results = []
        seen_pages = set()

        for distance, index in zip(distances[0], indices[0]):
            distance = round(float(distance), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": distance,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": distance,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)

        is_raptor = True
        if is_raptor:
            SAVE_PATH = ""
            SAVE_PATH = os.path.join(SAVE_PATH, document_name)
            RA = RetrievalAugmentation(tree=SAVE_PATH)
            context, layer_information = RA.answer_question(question=query + step_back_query)
            return retrieval_results, context, layer_information
        else:
            return retrieval_results

    def retrieve_all(self, document_name: str) -> List[Dict]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("document_name") == document_name:
                target_report = report
                break

        if target_report is None:
            _log.error(f"No report found with '{document_name}' document name.")
            raise ValueError(f"No report found with '{document_name}' document name.")

        document = target_report["document"]
        pages = document["content"]["pages"]

        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "distance": 0.5,
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)

        return all_pages


class HybridRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.reranker = LLMReranker()

    def retrieve_by_document_name(
            self,
            document_name: str,
            query: str,
            llm_reranking_sample_size: int = 28,
            documents_batch_size: int = 1,
            top_n: int = 6,
            llm_weight: float = 0.7,
            return_parent_pages: bool = False,
            is_picture: bool = False
    ) -> List[Dict]:

        is_page_parent = True

        if is_page_parent:
            doc_name = document_name
            output_dir = ""
            all_matched_image_path = []

        # Get initial results from vector retriever 
        is_raptor = True
        if is_raptor:
            vector_results, context, layer_information = self.vector_retriever.retrieve_by_document_name(
                document_name=document_name,
                query=query,
                top_n=llm_reranking_sample_size,
                return_parent_pages=return_parent_pages
            )

            if not is_page_parent:
                return vector_results, context

            reranked_results = self.reranker.rerank_documents(
                query=query,
                documents=vector_results,
                documents_batch_size=documents_batch_size,
                llm_weight=llm_weight
            )

            if is_picture == False:
                return reranked_results[:top_n], context
            else:
                for item in reranked_results[:top_n]:
                    page_num = item['page']
                    search_pattern = f"{doc_name}_{page_num}_*.png"
                    full_pattern = os.path.join(output_dir, search_pattern)
                    import glob
                    image_path = glob.glob(full_pattern)
                    all_matched_image_path.extend(image_path)
                picture_answer = self.ask_qwen_vl(all_matched_image_path, query)
                return reranked_results[:top_n], context, picture_answer

        else:
            vector_results = self.vector_retriever.retrieve_by_document_name(
                document_name=document_name,
                query=query,
                top_n=llm_reranking_sample_size,
                return_parent_pages=return_parent_pages
            )

            # Rerank results using LLM
            reranked_results = self.reranker.rerank_documents(
                query=query,
                documents=vector_results,
                documents_batch_size=documents_batch_size,
                llm_weight=llm_weight
            )

            if is_picture == False:
                return reranked_results[:top_n]
            else:
                for item in reranked_results[:top_n]:
                    page_num = item['page']
                    search_pattern = f"{doc_name}_{page_num}_*.png"
                    full_pattern = os.path.join(output_dir, search_pattern)
                    import glob
                    image_path = glob.glob(full_pattern)
                    all_matched_image_path.extend(image_path)
                picture_answer = self.ask_qwen_vl(all_matched_image_path, query)
                return reranked_results[:top_n], picture_answer

    def ask_qwen_vl(self, image_paths, question):
        client = OpenAI(
            api_key="",
            base_url="",
        )

        """
        Call Qwen-VL to answer questions about multiple images (with English prompts)
        :param image_paths: List of image paths (local paths or URLs)
        :param question: Question text in English
        :return: Model's answer
        """
        # Construct multimodal input (images + question + English prompts)
        messages = [{
            "role": "user",
            "content": [
                # Step 1: Clear instruction to analyze images
                {
                    "type": "text",
                    "text": "You are a professional visual assistant. Please carefully analyze the following images, paying attention to objects, scenes, and details."
                },
                # Step 2: Insert all images
                *[{
                    "type": "image_url",
                    "image_url": {"url": self.image_to_base64(path)}
                } for path in image_paths],
                # Step 3: Answer the question with requirements
                {
                    "type": "text",
                    "text": f"Based on the provided images, answer the following question: {question}\n"
                            "Requirements:\n"
                            "1. If the question involves multiple images, compare their similarities and differences;\n"
                            "2. Keep the response concise and accurate, avoiding subjective speculation."
                }
            ]
        }]

        completion = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=messages,
        )
        a = completion.model_dump_json()
        picture_answer = json.loads(a)["choices"][0]["message"]["content"]
        return picture_answer

    def image_to_base64(self, image_path):
        """Convert local image to Base64 data URL"""
        with open(image_path, "rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode('utf-8')

        if image_path.lower().endswith('.png'):
            return f"data:image/png;base64,{base64_str}"
        else:
            return f"data:image/jpeg;base64,{base64_str}"
