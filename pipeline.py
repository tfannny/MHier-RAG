from dataclasses import dataclass  # 用于定义配置类 (PipelineConfig, RunConfig)，简化类的编写
from pathlib import Path  # 用于优雅地处理文件路径
from pyprojroot import here  # 用于自动定位项目的根目录 (here())
import logging
import os
import json
import pandas as pd

from pdf_parsing import PDFParser
from parsed_reports_merging import PageTextPreparation
from text_splitter import TextSplitter
from ingestion import VectorDBIngestor,Qwen3LocalVectorDBIngestor
from ingestion import BM25Ingestor
from questions_processing import QuestionsProcessor

import requests


@dataclass
class PipelineConfig:
    """管理所有输入、输出和中间文件的路径。避免在代码中硬编码路径字符串"""

    def __init__(self, root_path: Path, subset_name: str = "", questions_file_name: str = "questions.json",
                 pdf_reports_dir_name: str = "pdf_reports", serialized: bool = False, config_suffix: str = ""):
        self.root_path = root_path
        suffix = "_ser_tab" if serialized else ""

        self.subset_path = root_path / subset_name
        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = root_path / pdf_reports_dir_name

        self.answers_file_path = root_path / f"answers{config_suffix}.json"
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / f"databases{suffix}"

        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_dbs"

        self.parsed_reports_dirname = "01_parsed_reports"
        self.parsed_reports_debug_dirname = "01_parsed_reports_debug"
        self.merged_reports_dirname = f"02_merged_reports{suffix}"
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"

        self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        self.parsed_reports_debug_path = self.debug_data_path / self.parsed_reports_debug_dirname
        self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname


@dataclass
class RunConfig:
    """控制 RAG 流程的具体行为参数"""
    use_serialized_tables: bool = False
    parent_document_retrieval: bool = False
    use_vector_dbs: bool = True
    use_bm25_db: bool = False
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30
    top_n_retrieval: int = 10
    parallel_requests: int = 10
    team_email: str = ""
    submission_name: str = ""
    pipeline_details: str = ""
    submission_file: bool = True
    full_context: bool = False
    api_provider: str = "qwen"
    answering_model: str = "qwen-turbo"  # or "gpt-4o"
    config_suffix: str = ""


class Pipeline:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json",
                 pdf_reports_dir_name: str = "pdf_reports", run_config: RunConfig = RunConfig()):
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str,
                          pdf_reports_dir_name: str) -> PipelineConfig:
        """根据运行配置设置初始化路径配置"""
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )

    def _convert_json_to_csv_if_needed(self):
        """
        Checks if subset.json exists in root dir and subset.csv is absent.
        If so, converts the JSON to CSV format.
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"

        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                df = pd.DataFrame(data)

                df.to_csv(csv_path, index=False)

            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

    # Docling automatically downloads some models from huggingface when first used
    # I wanted to download them prior to running the pipeline and created this crutch
    @staticmethod
    def download_docling_models():
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_reports_sequential(self):
        """
        串行解析PDF
        将非结构化的PDF转换为结构化的JSON数据
        """
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        pdf_parser.parse_and_export(doc_dir=self.paths.pdf_reports_dir)
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """使用多个进程并行解析 PDF 报告
        
        Args:
            chunk_size: 每个工作进程要处理的PDF文件数量
            num_workers: 要使用的并行工作进程数

        将非结构化的PDF转换为结构化的JSON数据
        """
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))

        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def merge_reports(self):
        """将复杂的JSON报告合并成一个更简单的结构，其中包含页面列表，所有文本块都合并成一个字符串"""
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)  # False
        _ = ptp.process_reports(
            reports_dir=self.paths.parsed_reports_path,  # /01_parsed_reports
            output_dir=self.paths.merged_reports_path  # /02_merged_reports
        )
        print(f"Reports saved to {self.paths.merged_reports_path}")

    def export_reports_to_markdown(self):
        """将处理后的报表导出为Markdown格式以供审阅"""
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        ptp.export_to_markdown(
            reports_dir=self.paths.parsed_reports_path,
            output_dir=self.paths.reports_markdown_path
        )
        print(f"Reports saved to {self.paths.reports_markdown_path}")

    def chunk_reports(self, include_serialized_tables: bool = False):
        """将处理后的报告拆分成更小的部分，以便更好地进行处理"""

        text_splitter = TextSplitter()

        serialized_tables_dir = None
        if include_serialized_tables:
            serialized_tables_dir = self.paths.parsed_reports_path

        text_splitter.split_all_reports(
            self.paths.merged_reports_path,  # /02_merged_reports
            self.paths.documents_dir,
            serialized_tables_dir
        )
        print(f"Chunked reports saved to {self.paths.documents_dir}")

    def create_vector_dbs(self):
        """建立索引：从分块报告中创建矢量数据库，将文本块转化为向量，实现语义检索"""
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir

        # TODO 换成本地向量模型
        # vdb_ingestor = VectorDBIngestor()
        vdb_ingestor = Qwen3LocalVectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")

    def create_bm25_db(self):
        """建立索引：从分块报表创建BM25数据库，建立BM25倒排索引，实现关键词检索"""
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path

        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")

    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        if parallel:
            self.parse_pdf_reports_parallel(chunk_size=chunk_size, max_workers=max_workers)
        else:
            self.parse_pdf_reports_sequential()

    def process_parsed_reports(self):
        """
        将已解析的 PDF 报告按以下步骤处理：
        1. 合并为更简单的 JSON 结构
        2. 导出为 Markdown 格式
        3. 将报告分块
        4. 创建矢量数据库
        """
        print("Starting reports processing pipeline...")

        print("Step 1: Merging reports...")
        self.merge_reports()

        print("Step 2: Exporting reports to markdown...")
        self.export_reports_to_markdown()

        print("Step 3: Chunking reports...")
        self.chunk_reports()

        print("Step 4: Creating vector databases...")
        self.create_vector_dbs()

        print("Reports processing pipeline completed successfully!")

    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        如果文件存在，则返回下一个可用的文件名，并在文件名后添加一个数字后缀
        例如：如果answers.json存在，则返回answers_01.json，依此类推
        """
        if not base_path.exists():
            return base_path

        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent

        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename

            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        """
        这是 RAG 的最后一步——生成答案

        流程：
        1. 初始化QuestionsProcessor，传入向量库路径、问题文件路径和LLM配置。
        2. process_all_questions: 遍历每一个问题。
        3. 检索：根据配置，从VectorDB或BM25中检索相关片段。
        4. 重排序：如果开启llm_reranking，让LLM判断哪些片段最相关。
        5. 生成：将筛选后的片段作为上下文喂给LLM(answering_model)，生成最终答案。
        6. 保存：结果保存到JSON文件，文件名会自动递增避免覆盖（通过 _get_next_available_filename）
        """
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context
        )

        output_path = self._get_next_available_filename(self.paths.answers_file_path)

        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            team_email=self.run_config.team_email,
            submission_name=self.run_config.submission_name,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")


preprocess_configs = {"ser_tab": RunConfig(use_serialized_tables=True),
                      "no_ser_tab": RunConfig(use_serialized_tables=False)}

max_st_qwenturbo8k_reasoning_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=1,
    submission_name="",
    pipeline_details="",
    api_provider="qwen",
    answering_model="qwen-turbo",
    config_suffix="_max_qwen-turbo1-llmre-reasoning"
)

# You can run any method right from this file with
# python .\src\pipeline.py
# Just uncomment the method you want to run
# You can also change the run_config to try out different configurations
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    root_path = here() / "data" / "test_set"
    print("root_path:", root_path)
    pipeline = Pipeline(root_path,
                        run_config=max_st_qwenturbo8k_reasoning_config
                        )

    # 此方法将 PDF 报告解析为 JSON 文件。它会在 debug/data_01_parsed_reports 目录中创建 JSON 文件。这些 JSON 文件将在后续步骤中使用。
    # 它还会将文档生成的原始输出存储在 debug/data_01_parsed_reports_debug 目录中。这些 JSON 文件包含大量元数据，但不会被使用。
    pipeline.parse_pdf_reports_sequential()

    # 此方法将 debug/data_01_parsed_reports 中的 JSON 转换为更简单的 JSON，即 Markdown 格式的页面列表。
    # 新的 JSON 文件位于 debug/data_02_merged_reports 中。
    pipeline.merge_reports()

    # 此方法将报告导出为纯 Markdown 格式。这些报告仅用于审阅和全文搜索配置：gemini_thinking_config
    # 新文件位于 debug/data_03_reports_markdown 目录下
    pipeline.export_reports_to_markdown()

    # 此方法将报告分割成多个数据块，用于向量化处理
    # 新的 JSON 文件位于 databases/chunked_reports 目录中。
    pipeline.chunk_reports()

    # 此方法从分块报告中创建向量数据库
    # 新文件位于 databases/vector_dbs 目录中。
    pipeline.create_vector_dbs()

    # 此方法处理问题和答案
    # 问题处理逻辑取决于 run_config
    pipeline.process_questions()
