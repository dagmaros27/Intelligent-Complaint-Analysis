import gradio as gr
import pandas as pd
from datetime import datetime
import os
import sys

sys.path.append('src')

from rag_pipeline import RAGPipeline

class ComplaintChatbot:
    """
    Interactive chatbot interface for the complaint analysis RAG system.
    """
    
    def __init__(self):
        """Initialize the chatbot with RAG pipeline."""
        try:
            self.rag_pipeline = RAGPipeline()
            self.chat_history = []
            print("Chatbot initialized successfully!")
        except Exception as e:
            print(f"Error initializing chatbot: {str(e)}")
            self.rag_pipeline = None
    
    def format_sources(self, sources):
        """Format retrieved sources for display."""
        if not sources:
            return "No sources retrieved."
        
        formatted_sources = []
        for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
            product = source.get('product', 'Unknown')
            chunk_text = source.get('chunk_text', '')[:200] + "..." if len(source.get('chunk_text', '')) > 200 else source.get('chunk_text', '')
            score = source.get('similarity_score', 0)
            
            formatted_sources.append(
                f"**Source {i} - {product}** (Similarity: {score:.3f})\n"
                f"{chunk_text}\n"
            )
        
        return "\n".join(formatted_sources)
    
    def chat_response(self, message, history):
        """
        Generate response for the chat interface.
        
        Args:
            message: User's current message
            history: Chat history
            
        Returns:
            Updated history and sources
        """
        if not self.rag_pipeline:
            error_msg = "Sorry, the system is not properly initialized. Please check the setup."
            history.append([message, error_msg])
            return history, "System not available."
        
        if not message.strip():
            history.append([message, "Please enter a valid question about customer complaints."])
            return history, ""
        
        try:
            # Get response from RAG pipeline
            response = self.rag_pipeline.answer_question(message)
            
            # Format the answer
            answer = response.get('answer', 'No answer generated.')
            sources = response.get('sources', [])
            
            # Add to chat history
            history.append([message, answer])
            
            # Format sources for display
            formatted_sources = self.format_sources(sources)
            
            # Store in internal history for analytics
            self.chat_history.append({
                'timestamp': datetime.now().isoformat(),
                'question': message,
                'answer': answer,
                'num_sources': len(sources)
            })
            
            return history, formatted_sources
            
        except Exception as e:
            error_msg = f"I encountered an error while processing your question. Please try again."
            history.append([message, error_msg])
            return history, f"Error: {str(e)}"
    
    def clear_chat(self):
        """Clear the chat history."""
        return [], ""
    
    def get_example_questions(self):
        """Return example questions users can ask."""
        return [
            "What are the main issues customers face with credit cards?",
            "Why are BNPL customers complaining?",
            "What problems do customers have with personal loans?",
            "What are the most common complaints across all products?",
            "How do customers feel about money transfer services?",
            "What savings account issues should we prioritize?",
            "Are there any fraud-related complaints?",
            "What customer service issues are mentioned most frequently?"
        ]

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # Initialize chatbot
    chatbot = ComplaintChatbot()
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .chat-message {
        padding: 10px !important;
        margin: 5px 0 !important;
        border-radius: 10px !important;
    }
    .sources-box {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        padding: 15px !important;
        margin-top: 10px !important;
    }
    """
    
    # Create the interface
    with gr.Blocks(css=custom_css, title="CrediTrust Complaint Analyzer") as demo:
        gr.Markdown(
            """
            # 🏦 CrediTrust Financial - Intelligent Complaint Analyzer
            
            **Ask questions about customer complaints and get AI-powered insights!**
            
            This tool analyzes thousands of customer complaints across our financial products:
            • Credit Cards • Personal Loans • Buy Now, Pay Later (BNPL) • Savings Accounts • Money Transfers
            
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot_interface = gr.Chatbot(
                    height=500,
                    label="💬 Chat with the Complaint Analyzer",
                    show_label=True
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask about customer complaints (e.g., 'What are the main credit card issues?')",
                        label="Your Question",
                        lines=2,
                        scale=4
                    )
                    
                with gr.Row():
                    submit_btn = gr.Button("🔍 Ask", variant="primary", scale=1)
                    clear_btn = gr.Button("🗑️ Clear Chat", scale=1)
            
            with gr.Column(scale=1):
                # Sources display
                sources_display = gr.Markdown(
                    "**📋 Sources will appear here after asking a question**",
                    label="Retrieved Sources"
                )
                
                # Example questions
                gr.Markdown("### 💡 Example Questions:")
                example_questions = chatbot.get_example_questions()
                
                for i, question in enumerate(example_questions[:4], 1):
                    example_btn = gr.Button(
                        f"{question}",
                        size="sm",
                        variant="secondary"
                    )
                    example_btn.click(
                        lambda q=question: chatbot.chat_response(q, []),
                        outputs=[chatbot_interface, sources_display]
                    )
        
        # Additional information
        with gr.Accordion("ℹ️ How to Use This Tool", open=False):
            gr.Markdown(
                """
                **Getting Started:**
                1. Type your question about customer complaints in the text box
                2. Click "Ask" or press Enter
                3. Review the AI-generated analysis and supporting sources
                
                **Best Practices:**
                - Be specific about the product or issue you're interested in
                - Ask about trends, patterns, or specific problems
                - Use the example questions as inspiration
                
                **Understanding Results:**
                - The AI provides insights based on real customer complaint data
                - Sources show the actual complaint excerpts used to generate the answer
                - Similarity scores indicate how relevant each source is to your question
                """
            )
        
        # Event handlers
        def handle_submit(message, history):
            return chatbot.chat_response(message, history)
        
        def handle_clear():
            return chatbot.clear_chat()
        
        # Set up event listeners
        submit_btn.click(
            handle_submit,
            inputs=[msg_input, chatbot_interface],
            outputs=[chatbot_interface, sources_display]
        ).then(
            lambda: "",  # Clear input after submission
            outputs=[msg_input]
        )
        
        msg_input.submit(
            handle_submit,
            inputs=[msg_input, chatbot_interface],
            outputs=[chatbot_interface, sources_display]
        ).then(
            lambda: "",  # Clear input after submission
            outputs=[msg_input]
        )
        
        clear_btn.click(
            handle_clear,
            outputs=[chatbot_interface, sources_display]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            **CrediTrust Financial - Internal AI Tool** | Built with ❤️ for better customer insights
            
            *This tool is designed for internal use by Product Managers, Support Teams, and Compliance Officers.*
            """
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_gradio_interface()
    
    # Launch with specific configuration
    demo.launch(
        server_name="0.0.0.0",  
        server_port=7860,       
        share=False,            
        debug=True,             
        show_error=True         
    )