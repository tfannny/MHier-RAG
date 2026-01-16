import json
from typing import Union, Dict, List, Optional
import re
from pathlib import Path
from retrieval import VectorRetriever, HybridRetriever
from api_requests import APIProcessor
from tqdm import tqdm
import pandas as pd
import threading
import concurrent.futures
import base64


class QuestionsProcessor:
    def __init__(
            self,
            vector_db_dir: Union[str, Path] = './vector_dbs',
            documents_dir: Union[str, Path] = './documents',
            questions_file_path: Optional[Union[str, Path]] = None,
            new_challenge_pipeline: bool = False,
            subset_path: Optional[Union[str, Path]] = None,
            parent_document_retrieval: bool = False,  # 是否在找到片段后，返回包含该片段的完整父文档页面（上下文更多）
            llm_reranking: bool = False,  # 是否让 LLM 对初步检索结果进行二次筛选（精度更高，但更慢）
            llm_reranking_sample_size: int = 20,
            top_n_retrieval: int = 10,
            parallel_requests: int = 10,  # 并发处理问题的数量
            api_provider: str = "qwen",
            answering_model: str = "qwen-turbo",
            full_context: bool = False
    ):
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.subset_path = Path(subset_path) if subset_path else None

        self.new_challenge_pipeline = new_challenge_pipeline
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context

        self.answer_details = []
        self.detail_counter = 0
        self._lock = threading.Lock()

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results, picture_answer) -> str:
        """
        将向量检索结果格式化为RAG上下文字符串
        结构：
            Text retrieved from page 1: [内容...]
            Text summarized from the relevant document: [摘要...]
            Text retrieved from images:[图片描述...]
        """
        if not retrieval_results:
            return ""

        context_parts = []
        for result in retrieval_results:
            page_number = result['page']
            text = result['text']
            context_parts.append(f'Text retrieved from page {page_number}: \n"""\n{text}\n"""')

        if len(picture_answer) != 0:
            context_parts.append(f'Text retrieved from images within pages: \n"""\n{picture_answer}\n"""')
        return "\n\n---\n\n".join(context_parts)

    def _format_retrieval_results_raptor(self, retrieval_results, context, picture_answer) -> str:
        """
        将向量检索结果格式化为RAG上下文字符串
        结构：
            Text retrieved from page 1: [内容...]
            Text summarized from the relevant document: [摘要...]
            Text retrieved from images:[图片描述...]
        """
        if not retrieval_results:
            return ""

        context_parts = []
        for result in retrieval_results:
            page_number = result['page']
            text = result['text']
            context_parts.append(f'Text retrieved from page {page_number}: \n"""\n{text}\n"""')
        context_parts.append(f'Text summarized from the relevant document: \n"""\n{context}\n"""')
        if len(picture_answer) != 0:
            context_parts.append(f'Text retrieved from images within pages: \n"""\n{picture_answer}\n"""')

        return "\n\n---\n\n".join(context_parts)

    def _validate_page_references(self, claimed_pages: list, retrieval_results: list, min_pages: int = 2,
                                  max_pages: int = 8) -> list:
        """
        防幻觉机制
        验证LLM答案中提到的所有页码是否确实来自检索结果
        如果剩余的有效参考文献少于min_pages，则添加检索结果中的前几页
        """
        if claimed_pages is None:
            claimed_pages = []

        retrieved_pages = [result['page'] for result in retrieval_results]

        validated_pages = [page for page in claimed_pages if page in retrieved_pages]

        if len(validated_pages) < len(claimed_pages):
            removed_pages = set(claimed_pages) - set(validated_pages)
            print(f"Warning: Removed {len(removed_pages)} hallucinated page references: {removed_pages}")

        if len(validated_pages) < min_pages and retrieval_results:
            existing_pages = set(validated_pages)

            for result in retrieval_results:
                page = result['page']
                if page not in existing_pages:
                    validated_pages.append(page)
                    existing_pages.add(page)

                    if len(validated_pages) >= min_pages:
                        break

        if len(validated_pages) > max_pages:
            print(f"Trimming references from {len(validated_pages)} to {max_pages} pages")
            validated_pages = validated_pages[:max_pages]

        return validated_pages

    def find_predict_value(self, json_file, query):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                if item.get('question') == query:
                    return item.get('predicted_label')

            return None

        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_answer_for_question(self, document_name: str, question: str, schema: str) -> dict:
        """
        核心逻辑：检索与上下文构建
        """
        # 选择检索器：
        # 如果开启llm_reranking，使用HybridRetriever（混合检索，通常结合向量+关键词）。
        # 否则使用普通的VectorRetriever
        if self.llm_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )

        if self.full_context:
            retrieval_results = retriever.retrieve_all(document_name)
        else:
            # 如果问题包含"color"或"figure"，会专门触发图片检索逻辑
            is_raptor = True
            if 'color' in question or 'figure' in question:
                is_picture = True
            else:
                is_picture = False

            # RAPTOR策略
            if is_raptor:
                # 调用 retriever.retrieve_by_document_name 获取相关文档片段 (retrieval_results)、上下文摘要 (context) 和图片描述 (picture_answer)
                if is_picture:
                    retrieval_results, context, picture_answer = retriever.retrieve_by_document_name(
                        document_name=document_name,
                        query=question,
                        llm_reranking_sample_size=self.llm_reranking_sample_size,
                        top_n=self.top_n_retrieval,
                        return_parent_pages=self.return_parent_pages,
                        is_picture=True
                    )
                else:
                    retrieval_results, context = retriever.retrieve_by_document_name(
                        document_name=document_name,
                        query=question,
                        llm_reranking_sample_size=self.llm_reranking_sample_size,
                        top_n=self.top_n_retrieval,
                        return_parent_pages=self.return_parent_pages,
                        is_picture=False
                    )
            else:
                if is_picture:
                    retrieval_results, picture_answer = retriever.retrieve_by_document_name(
                        document_name=document_name,
                        query=question,
                        llm_reranking_sample_size=self.llm_reranking_sample_size,
                        top_n=self.top_n_retrieval,
                        return_parent_pages=self.return_parent_pages,
                        is_picture=True
                    )
                else:
                    retrieval_results = retriever.retrieve_by_document_name(
                        document_name=document_name,
                        query=question,
                        llm_reranking_sample_size=self.llm_reranking_sample_size,
                        top_n=self.top_n_retrieval,
                        return_parent_pages=self.return_parent_pages,
                        is_picture=False
                    )

        if not retrieval_results:
            raise ValueError("No relevant context found")

        if is_picture == False:
            picture_answer = ''

        if is_raptor:
            rag_context = self._format_retrieval_results_raptor(retrieval_results, context, picture_answer)
        else:
            rag_context = self._format_retrieval_results(retrieval_results, picture_answer)

        # 将拼接好的rag_context、用户question和期望的schema（输出格式）发给LLM
        # LLM返回一个结构化的字典，包含答案、推理过程等
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data
        if self.new_challenge_pipeline:
            pages = answer_dict.get("relevant_pages", [])
            validated_pages = self._validate_page_references(pages, retrieval_results)
            answer_dict["relevant_pages"] = validated_pages
        return answer_dict

    def process_question(self, question: str, schema: str, doc_id: str):
        answer_dict = self.get_answer_for_question(doc_id=doc_id, question=question, schema=schema)
        return answer_dict

    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """Create a reference ID for answer details and store the details"""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict['step_by_step_analysis'],
                "reasoning_summary": answer_dict['reasoning_summary'],
                "relevant_pages": answer_dict['relevant_pages'],
                "response_data": self.response_data,
                "self": ref_id
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:
        """计算已处理问题的统计数据"""
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(1 for q in processed_questions if (q.get("value") if "value" in q else q.get("answer")) == "N/A")
        success_count = total_questions - error_count - na_count
        if print_stats:
            print(f"\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count / total_questions) * 100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count / total_questions) * 100:.1f}%)")
            print(f"Successfully answered: {success_count} ({(success_count / total_questions) * 100:.1f}%)\n")

        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count
        }

    def process_questions_list(self, questions_list: List[dict], output_path: str = None, submission_file: bool = False,
                               team_email: str = "", submission_name: str = "", pipeline_details: str = "") -> dict:
        """
        并发处理多个问题

        如果parallel_requests > 1，使用ThreadPoolExecutor创建线程池
        批量提交任务_process_single_question
        使用tqdm显示进度条

        每处理完一批，调用_save_progress 将结果写入磁盘
        """
        total_questions = len(questions_list)
        # 为每个问题添加索引，以便知道在哪里编写答案详情
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions  # Preallocate list for answer details
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:  # True
            for question_data in tqdm(questions_with_index, desc="Processing questions"):
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(processed_questions, output_path, submission_file=submission_file,
                                        team_email=team_email, submission_name=submission_name,
                                        pipeline_details=pipeline_details)
        else:
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i: i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        # executor.map will return results in the same order as the input list.
                        batch_results = list(executor.map(self._process_single_question, batch))
                    processed_questions.extend(batch_results)

                    if output_path:
                        self._save_progress(processed_questions, output_path, submission_file=submission_file,
                                            team_email=team_email, submission_name=submission_name,
                                            pipeline_details=pipeline_details)
                    pbar.update(len(batch_results))

        statistics = self._calculate_statistics(processed_questions, print_stats=True)

        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics
        }

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0)

        if self.new_challenge_pipeline:
            question_text = question_data.get("question")
            schema = question_data.get("answer_format")
            doc_name = question_data.get("doc_id").rsplit(".pdf", 1)[0]
        else:
            question_text = question_data.get("question")
            schema = question_data.get("schema")
        try:
            answer_dict = self.process_question(question_text, schema, doc_name)

            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref({
                    "step_by_step_analysis": None,
                    "reasoning_summary": None,
                    "relevant_pages": None
                }, question_index)
                if self.new_challenge_pipeline:
                    return {
                        "question_text": question_text,
                        "kind": schema,
                        "value": None,
                        "references": [],
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref}
                    }
                else:
                    return {
                        "question": question_text,
                        "schema": schema,
                        "answer": None,
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref},
                    }
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            if self.new_challenge_pipeline:
                return {
                    "question_text": question_text,
                    "kind": schema,
                    "value": answer_dict.get("final_answer"),
                    "references": answer_dict.get("references", []),
                    "answer_details": {"$ref": detail_ref}
                }
            else:
                return {
                    "question": question_text,
                    "schema": schema,
                    "answer": answer_dict.get("final_answer"),
                    "answer_details": {"$ref": detail_ref},
                }
        except Exception as err:
            return self._handle_processing_error(question_text, schema, err, question_index)

    def _handle_processing_error(self, question_text: str, schema: str, err: Exception, question_index: int) -> dict:
        """
        Handle errors during question processing.
        Log error details and return a dictionary containing error information.
        """
        import traceback
        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {
            "error_traceback": tb,
            "self": error_ref
        }

        with self._lock:
            self.answer_details[question_index] = error_detail

        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")

        if self.new_challenge_pipeline:
            return {
                "question_text": question_text,
                "kind": schema,
                "value": None,
                "references": [],
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref}
            }
        else:
            return {
                "question": question_text,
                "schema": schema,
                "answer": None,
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref},
            }

    def _post_process_submission_answers(self, processed_questions: List[dict]) -> List[dict]:
        """
        提交格式的答案后处理：
        1. 将页码索引从1转换为0
        2. 清除N/A（不知道/没找到）答案的引用
        3. 根据提交格式设置答案格式
        4. 在答案详情中添加分步分析
        """
        submission_answers = []

        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = "N/A" if "error" in q else (q.get("value") if "value" in q else q.get("answer"))
            references = q.get("references", [])

            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            step_by_step_analysis = None
            if answer_details_ref and answer_details_ref.startswith("#/answer_details/"):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                        step_by_step_analysis = self.answer_details[index].get("step_by_step_analysis")
                except (ValueError, IndexError):
                    pass

            # Clear references if value is N/A
            if value == "N/A":
                references = []
            else:
                # Convert page indices from one-based to zero-based (competition requires 0-based page indices, but for debugging it is easier to use 1-based)
                references = [
                    {
                        "pdf_sha1": ref["pdf_sha1"],
                        "page_index": ref["page_index"] - 1
                    }
                    for ref in references
                ]

            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
            }

            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis

            submission_answers.append(submission_answer)

        return submission_answers

    def _save_progress(self, processed_questions: List[dict], output_path: Optional[str], submission_file: bool = False,
                       team_email: str = "", submission_name: str = "", pipeline_details: str = ""):
        if output_path:
            statistics = self._calculate_statistics(processed_questions)

            # Prepare debug content
            result = {
                "questions": processed_questions,
                "answer_details": self.answer_details,
                "statistics": statistics
            }
            output_file = Path(output_path)
            debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)
            with open(debug_file, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=2)

            if submission_file:
                # Post-process answers for submission
                submission_answers = self._post_process_submission_answers(processed_questions)
                submission = {
                    "answers": submission_answers,
                    "team_email": team_email,
                    "submission_name": submission_name,
                    "details": pipeline_details
                }
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(self, output_path: str = 'questions_with_answers.json', team_email: str = "",
                              submission_name: str = "", submission_file: bool = False, pipeline_details: str = ""):
        result = self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            team_email=team_email,
            submission_name=submission_name,
            pipeline_details=pipeline_details
        )
        return result
