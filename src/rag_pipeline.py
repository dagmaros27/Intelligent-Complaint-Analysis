import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    A complete RAG pipeline for complaint analysis that combines semantic search
    with language model generation to answer questions about customer complaints.
    """
    
    def __init__(self, 
                 vector_store_path: str = "vector_store/",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model_name: str = "microsoft/DialoGPT-medium",
                 top_k: int = 5):
        """
        Initialize the RAG pipeline with necessary components.
        
        Args:
            vector_store_path: Path to the saved vector store
            embedding_model_name: Name of the embedding model
            llm_model_name: Name of the language model for generation
            top_k: Number of top chunks to retrieve
        """
        self.vector_store_path = vector_store_path
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.top_k = top_k
        
        # Initialize components
        self.embedding_model = None
        self.vector_store = None
        self.metadata = None
        self.llm_pipeline = None
        
        self._load_components()
    
    def _load_components(self):
        """Load all necessary components for the RAG pipeline."""
        try:
            # Load embedding model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Load vector store and metadata
            logger.info("Loading vector store and metadata...")
            self._load_vector_store()
            
            # Initialize LLM pipeline
            logger.info(f"Loading language model: {self.llm_model_name}")
            self._initialize_llm()
            
            logger.info("RAG pipeline initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error loading components: {str(e)}")
            raise
    
    def _load_vector_store(self):
        """Load the FAISS vector store and associated metadata."""
        try:
            # Load FAISS index
            index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
            self.vector_store = faiss.read_index(index_path)
            
            # Load metadata
            metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
                
        except FileNotFoundError as e:
            logger.error(f"Vector store files not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def _initialize_llm(self):
        """Initialize the language model pipeline."""
        try:
            # Use a more suitable model for text generation
            device = 0 if torch.cuda.is_available() else -1
            
            # For better results, we'll use a text generation pipeline
            self.llm_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",  # Smaller, faster model
                tokenizer="microsoft/DialoGPT-small",
                device=device,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256
            )
            
        except Exception as e:
            logger.warning(f"Error loading primary model, falling back to simpler approach: {str(e)}")
            # Fallback to a simpler approach
            self.llm_pipeline = None
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a user query using the same embedding model used for indexing.
        
        Args:
            query: User's question as a string
            
        Returns:
            Query embedding as numpy array
        """
        return self.embedding_model.encode([query])[0]
    
    def retrieve_relevant_chunks(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant text chunks for a given query.
        
        Args:
            query: User's question
            
        Returns:
            List of dictionaries containing retrieved chunks and metadata
        """
        # Embed the query
        query_embedding = self.embed_query(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Perform similarity search
        scores, indices = self.vector_store.search(query_embedding, self.top_k)
        
        # Retrieve corresponding metadata
        retrieved_chunks = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                chunk_data = {
                    'chunk_text': self.metadata[idx]['chunk_text'],
                    'product': self.metadata[idx]['product'],
                    'complaint_id': self.metadata[idx]['complaint_id'],
                    'similarity_score': float(score),
                    'rank': i + 1
                }
                retrieved_chunks.append(chunk_data)
        
        return retrieved_chunks
    
    def create_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for the language model using the query and retrieved context.
        
        Args:
            query: User's question
            retrieved_chunks: List of relevant chunks from retrieval
            
        Returns:
            Formatted prompt string
        """
        # Combine retrieved chunks into context
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(
                f"[Source {i+1} - {chunk['product']}]: {chunk['chunk_text']}"
            )
        
        context = "\n\n".join(context_parts)
        
        prompt_template = """You are a financial analyst assistant for CrediTrust Financial. Your task is to analyze customer complaints and provide insights to help improve financial services.

Based on the following customer complaint excerpts, please answer the question. Focus on identifying patterns, common issues, and actionable insights. If the context doesn't contain enough information to answer the question, state that clearly.

Context:
{context}

Question: {question}

Analysis and Answer:"""

        return prompt_template.format(context=context, question=query)
    
    def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer using the language model.
        
        Args:
            prompt: Formatted prompt with context and question
            
        Returns:
            Generated answer string
        """
        if self.llm_pipeline is None:
            # Fallback to rule-based response
            return self._generate_fallback_answer(prompt)
        
        try:
            # Generate response
            response = self.llm_pipeline(
                prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            generated_text = response[0]['generated_text']
            
            # Extract only the new generated part (after the prompt)
            answer = generated_text[len(prompt):].strip()
            
            return answer if answer else self._generate_fallback_answer(prompt)
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return self._generate_fallback_answer(prompt)
    
    def _generate_fallback_answer(self, prompt: str) -> str:
        """
        Generate a simple rule-based answer when LLM fails.
        
        Args:
            prompt: The full prompt
            
        Returns:
            Fallback answer
        """
        # Extract context and question from prompt
        if "Context:" in prompt and "Question:" in prompt:
            context_start = prompt.find("Context:") + len("Context:")
            question_start = prompt.find("Question:") + len("Question:")
            
            context = prompt[context_start:prompt.find("Question:")].strip()
            question = prompt[question_start:prompt.find("Analysis and Answer:")].strip()
            
            # Simple pattern matching for common questions
            question_lower = question.lower()
            
            if any(word in question_lower for word in ['why', 'problem', 'issue', 'complaint']):
                return f"Based on the provided complaints, customers appear to be experiencing issues related to the services mentioned in the context. The specific concerns include various operational and service-related problems that require attention from the product teams."
            elif any(word in question_lower for word in ['how many', 'count', 'number']):
                return f"Based on the retrieved complaints, there are multiple instances of customer feedback available for analysis."
            else:
                return f"Based on the customer complaint data provided, there are several concerns that need to be addressed by the relevant teams."
        
        return "I'm sorry, I couldn't generate a proper response based on the available information."
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve relevant chunks and generate an answer.
        
        Args:
            query: User's question
            
        Returns:
            Dictionary containing the answer and supporting information
        """
        try:
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.retrieve_relevant_chunks(query)
            
            if not retrieved_chunks:
                return {
                    'answer': "I couldn't find relevant information to answer your question.",
                    'sources': [],
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Step 2: Create prompt
            prompt = self.create_prompt(query, retrieved_chunks)
            
            # Step 3: Generate answer
            answer = self.generate_answer(prompt)
            
            # Step 4: Format response
            response = {
                'answer': answer,
                'sources': retrieved_chunks,
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'num_sources': len(retrieved_chunks)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'query': query,
                'timestamp': datetime.now().isoformat()
            }


class RAGEvaluator:
    """
    Evaluation framework for the RAG pipeline to assess performance quality.
    """
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.evaluation_results = []
    
    def create_evaluation_questions(self) -> List[Dict[str, Any]]:
        """
        Create a comprehensive set of evaluation questions covering different aspects.
        
        Returns:
            List of evaluation questions with expected criteria
        """
        evaluation_questions = [
            {
                'question': "What are the main issues customers face with credit cards?",
                'category': 'Product-specific analysis',
                'expected_elements': ['credit card', 'billing', 'payment', 'fees']
            },
            {
                'question': "Why are BNPL customers complaining?",
                'category': 'Product-specific analysis',
                'expected_elements': ['buy now pay later', 'payment', 'installment']
            },
            {
                'question': "What problems do customers have with personal loans?",
                'category': 'Product-specific analysis',
                'expected_elements': ['personal loan', 'interest', 'approval', 'repayment']
            },
            {
                'question': "What are the most common complaints across all products?",
                'category': 'Cross-product analysis',
                'expected_elements': ['multiple products', 'common patterns']
            },
            {
                'question': "How do customers feel about money transfer services?",
                'category': 'Sentiment analysis',
                'expected_elements': ['money transfer', 'transaction', 'fees']
            },
            {
                'question': "What savings account issues should we prioritize?",
                'category': 'Actionable insights',
                'expected_elements': ['savings account', 'priority', 'action']
            },
            {
                'question': "Are there any fraud-related complaints?",
                'category': 'Risk analysis',
                'expected_elements': ['fraud', 'security', 'unauthorized']
            },
            {
                'question': "What customer service issues are mentioned most frequently?",
                'category': 'Service quality',
                'expected_elements': ['customer service', 'support', 'response']
            }
        ]
        
        return evaluation_questions
    
    def evaluate_response_quality(self, question: str, response: Dict[str, Any], 
                                expected_elements: List[str]) -> Dict[str, Any]:
        """
        Evaluate the quality of a RAG response.
        
        Args:
            question: The evaluation question
            response: RAG pipeline response
            expected_elements: Expected elements in a good answer
            
        Returns:
            Evaluation metrics and scores
        """
        answer = response.get('answer', '').lower()
        sources = response.get('sources', [])
        
        # Relevance score (1-5)
        relevance_score = self._calculate_relevance_score(answer, expected_elements)
        
        # Source quality score (1-5)
        source_score = self._calculate_source_quality(sources, expected_elements)
        
        # Completeness score (1-5)
        completeness_score = self._calculate_completeness_score(answer, question)
        
        # Overall score
        overall_score = round((relevance_score + source_score + completeness_score) / 3, 1)
        
        evaluation = {
            'question': question,
            'answer': response.get('answer'),
            'relevance_score': relevance_score,
            'source_quality_score': source_score,
            'completeness_score': completeness_score,
            'overall_score': overall_score,
            'num_sources': len(sources),
            'top_sources': sources[:2] if sources else [],
            'analysis': self._generate_analysis(relevance_score, source_score, completeness_score)
        }
        
        return evaluation
    
    def _calculate_relevance_score(self, answer: str, expected_elements: List[str]) -> int:
        """Calculate how relevant the answer is to the question."""
        if not answer or len(answer) < 20:
            return 1
        
        matches = sum(1 for element in expected_elements if element.lower() in answer)
        if matches >= 3:
            return 5
        elif matches >= 2:
            return 4
        elif matches >= 1:
            return 3
        else:
            return 2
    
    def _calculate_source_quality(self, sources: List[Dict], expected_elements: List[str]) -> int:
        """Calculate the quality of retrieved sources."""
        if not sources:
            return 1
        
        if len(sources) >= 3:
            source_score = 5
        elif len(sources) >= 2:
            source_score = 4
        else:
            source_score = 3
        
        # Check if sources contain relevant content
        relevant_sources = 0
        for source in sources[:3]:
            source_text = source.get('chunk_text', '').lower()
            if any(element.lower() in source_text for element in expected_elements):
                relevant_sources += 1
        
        if relevant_sources >= 2:
            return source_score
        elif relevant_sources >= 1:
            return max(source_score - 1, 1)
        else:
            return max(source_score - 2, 1)
    
    def _calculate_completeness_score(self, answer: str, question: str) -> int:
        """Calculate how complete the answer is."""
        if not answer:
            return 1
        
        word_count = len(answer.split())
        if word_count >= 50:
            return 5
        elif word_count >= 30:
            return 4
        elif word_count >= 20:
            return 3
        elif word_count >= 10:
            return 2
        else:
            return 1
    
    def _generate_analysis(self, relevance: int, source: int, completeness: int) -> str:
        """Generate analysis comments based on scores."""
        comments = []
        
        if relevance >= 4:
            comments.append("High relevance to question")
        elif relevance <= 2:
            comments.append("Low relevance - may need better retrieval")
        
        if source >= 4:
            comments.append("Good source quality")
        elif source <= 2:
            comments.append("Poor source retrieval")
        
        if completeness >= 4:
            comments.append("Comprehensive answer")
        elif completeness <= 2:
            comments.append("Answer too brief")
        
        return "; ".join(comments) if comments else "Average performance"
    
    def run_evaluation(self) -> pd.DataFrame:
        """
        Run complete evaluation on the RAG pipeline.
        
        Returns:
            DataFrame with evaluation results
        """
        questions = self.create_evaluation_questions()
        
        print("Running RAG Pipeline Evaluation...")
        print("=" * 50)
        
        for i, q_data in enumerate(questions, 1):
            print(f"\nEvaluating Question {i}/{len(questions)}")
            print(f"Question: {q_data['question']}")
            
            # Get RAG response
            response = self.rag_pipeline.answer_question(q_data['question'])
            
            # Evaluate response
            evaluation = self.evaluate_response_quality(
                q_data['question'], 
                response, 
                q_data['expected_elements']
            )
            evaluation['category'] = q_data['category']
            
            self.evaluation_results.append(evaluation)
            
            print(f"Overall Score: {evaluation['overall_score']}/5")
            print(f"Analysis: {evaluation['analysis']}")
        
        # Convert to DataFrame
        eval_df = pd.DataFrame(self.evaluation_results)
        
        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Average Overall Score: {eval_df['overall_score'].mean():.2f}/5")
        print(f"Average Relevance Score: {eval_df['relevance_score'].mean():.2f}/5")
        print(f"Average Source Quality: {eval_df['source_quality_score'].mean():.2f}/5")
        print(f"Average Completeness: {eval_df['completeness_score'].mean():.2f}/5")
        
        return eval_df
    
    def save_evaluation_results(self, filepath: str = "evaluation_results.csv"):
        """Save evaluation results to CSV file."""
        if self.evaluation_results:
            eval_df = pd.DataFrame(self.evaluation_results)
            eval_df.to_csv(filepath, index=False)
            print(f"Evaluation results saved to {filepath}")

