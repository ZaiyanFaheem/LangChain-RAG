# LangChain-RAG
import os
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from tqdm import tqdm
!pip install -U langchain-google-genai google-generativeai pinecone-client langchain-community


class RAGSystem:
    def __init__(self):
        """Initialize the RAG system by checking for environment variables first."""
        self.google_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
        self.index_name = "gemini-rag-index"
        
        # Validate API keys
        self._validate_credentials()
        # Configure services
        self._setup_services()

    def _validate_credentials(self):
        """Validate that all required API keys are present."""
        if not self.google_api_key:
            raise ValueError("GOOGLE_GEMINI_API_KEY environment variable is not set")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")

    def _setup_services(self):
        """Set up Google Gemini and Pinecone services."""
        try:
            # Configure Google Gemini
            genai.configure(api_key=self.google_api_key)
            
            # Initialize Pinecone with explicit API key
            self.pc = Pinecone(
                api_key=self.pinecone_api_key,
                environment=self.pinecone_environment #Corrected: Added environment parameter
            )
            
            # Create or get index
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.pinecone_environment
                    )
                )
            
            self.index = self.pc.Index(self.index_name)
            
            # Set up embedding model
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.google_api_key
            )
            
        except Exception as e:
            raise Exception(f"Error setting up services: {str(e)}")

    def process_documents(self, file_path: str):
        """Load and process documents with error handling."""
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len
            )
            return text_splitter.split_documents(documents)
        except FileNotFoundError:
            raise Exception(f"Document file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error processing documents: {str(e)}")

    def embed_and_store(self, docs):
        """Embed documents and store in Pinecone with progress tracking."""
        try:
            # Clear existing vectors
            self.index.delete(delete_all=True)
            
            batch_size = 100
            for i in tqdm(range(0, len(docs), batch_size)):
                batch = docs[i:i + batch_size]
                embeddings_batch = []
                
                for j, doc in enumerate(batch):
                    try:
                        vector = self.embeddings.embed_documents([doc.page_content])[0]
                        metadata = {
                            "text": doc.page_content,
                            "source": doc.metadata.get("source", "unknown")
                        }
                        unique_id = f"doc_{i}_{j}"
                        embeddings_batch.append((unique_id, vector, metadata))
                    except Exception as e:
                        print(f"Warning: Error embedding document {i}_{j}: {str(e)}")
                        continue
                
                if embeddings_batch:
                    self.index.upsert(vectors=embeddings_batch)
                    
        except Exception as e:
            raise Exception(f"Error storing embeddings: {str(e)}")

    def setup_qa_chain(self):
        """Set up the question-answering chain with improved prompt."""
        try:
            prompt_template = """
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer or cannot find it in the context, just say "Based on the provided context, I cannot answer this question."
            Always cite specific parts of the context to support your answer.

            Context: {context}

            Question: {question}

            Answer: Let me answer based on the provided context:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            vectorstore = LangchainPinecone.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=""
            )

            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.3,
                google_api_key=self.google_api_key,
                convert_system_message_to_human=True
            )

            # Updated retriever configuration
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Fetch top 4 most relevant chunks
            )

            return RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": PROMPT
                },
                return_source_documents=True  # Optional: returns source documents with response
            )
        except Exception as e:
            raise Exception(f"Error setting up QA chain: {str(e)}")

def main():
    # Set environment variables
    os.environ['GOOGLE_GEMINI_API_KEY'] = "YOUR_GOOGLE_API_KEY"
    os.environ['PINECONE_API_KEY'] = "YOUR_PINECONE_API_KEY"
    os.environ['PINECONE_ENVIRONMENT'] = "YOUR_PINECONE_ENVIRONMENT"
    
    try:
        # Initialize RAG system
        rag = RAGSystem()
        
        print("Processing documents...")
        docs = rag.process_documents("documents.txt")
        
        print("Embedding and storing documents...")
        rag.embed_and_store(docs)
        
        print("Setting up QA chain...")
        qa_chain = rag.setup_qa_chain()
        
        while True:
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            
            try:
                result = qa_chain.invoke({"query": query})
                print("\nResponse:", result["result"].strip())
                
                # Optionally print source documents
                # if "source_documents" in result:
                #     print("\nSources:")
                #     for i, doc in enumerate(result["source_documents"], 1):
                #         print(f"\n{i}. {doc.page_content[:200]}...")
                
            except Exception as e:
                print(f"Error processing question: {str(e)}")
                
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")

if __name__ == "__main__":
    main()
