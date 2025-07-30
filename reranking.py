import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import prompts as prompts
from concurrent.futures import ThreadPoolExecutor


class LLMReranker:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks
      
    def set_up_llm(self):
        load_dotenv()

        llm = OpenAI(
            api_key="", 
            base_url="",
        )

        return llm
    
    def get_rank_for_single_block(self, query, retrieved_document):
        user_prompt = f'/nHere is the query:/n"{query}"/n/nHere is the retrieved text block:/n"""/n{retrieved_document}/n"""/n'
        reasoning = True
        
        if reasoning: 
            completion = self.llm.chat.completions.create(model="qwen-turbo",
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt_rerank_single_block},
                    {"role": "user", "content": user_prompt},
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
            else:
                completion = self.llm.chat.completions.create(model="qwen-turbo",
                        temperature=0,
                        messages=[
                            {"role": "system", "content": self.system_prompt_rerank_single_block},
                            {"role": "user", "content": user_prompt},
                        ],
                        
                        )
                answer_content = completion.choices[0].message.content
        return answer_content 
    
    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 1, llm_weight: float = 0.7):
        """
        Rerank multiple documents using parallel processing with threading.
        Combines vector similarity and LLM relevance scores using weighted average.
        """
        # Create batches of documents documents
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight
        
        if documents_batch_size == 1:
            def process_single_doc(doc):
                # Get ranking for single document
                output = self.get_rank_for_single_block(query, doc['text'])
                import re
                match = re.search(r"Relevance Score:\s*(?:[\W\d\s]*?)?(\d*\.?\d+)", output)

                if match:
                    ranking = float(match.group(1))
                    print(f"Score: {ranking}")
                else:
                    pattern2 = r"relevance score is\s*(?:[*\s]*)?(\d*\.?\d+)"
                    match2 = re.search(pattern2, output, re.IGNORECASE)
                    if match2:
                        ranking = float(match2.group(1))
                        print(f"Score: {ranking}")
                    else:
                        ranking = 0.0
                        print("None")
                        print(output)
                
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking 
                # Calculate combined score - note that distance is inverted since lower is better
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking + 
                    vector_weight * doc['distance'],
                    4
                )
                return doc_with_score

            #Process all documents in parallel using single-block method
            with ThreadPoolExecutor() as executor:
                all_results = list(executor.map(process_single_doc, documents))
                
        else:
            def process_batch(batch):
                texts = [doc['text'] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get('block_rankings', [])
                
                if len(block_rankings) < len(batch):
                    print(f"\nWarning: Expected {len(batch)} rankings but got {len(block_rankings)}")
                    for i in range(len(block_rankings), len(batch)):
                        doc = batch[i]
                        print(f"Missing ranking for document on page {doc.get('page', 'unknown')}:")
                        print(f"Text preview: {doc['text'][:100]}...\n")
                    
                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append({
                            "relevance_score": 0.0, 
                            "reasoning": "Default ranking due to missing LLM response"
                        })
                
                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"] + 
                        vector_weight * doc['distance'],
                        4
                    )
                    results.append(doc_with_score)
                return results

            # Process batches in parallel using threads
            with ThreadPoolExecutor() as executor:
                batch_results = list(executor.map(process_batch, doc_batches))
            
            # Flatten results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        
        # Sort results by combined score in descending order
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
