import fitz 
from PIL import Image
import os
from dotenv import load_dotenv
import sys
from phi.agent import Agent, RunResponse
from phi.model.groq import Groq
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.agent import Agent
from phi.vectordb.chroma import ChromaDb
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder


def pdf_to_images(pdf_path, output_folder):
    pdf_document = fitz.open(pdf_path)
    images = []
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        pix = page.get_pixmap()
        image_path = f"{output_folder}/page_{page_number + 1}.png"
        pix.save(image_path)
        images.append(image_path)
    return images

# Load the .env file
load_dotenv()

# Access the GROQ API Key
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    print("GROQ API Key loaded successfully.")
else:
    sys.exit("Exiting program due to missing GROQ API Key.")

knowledge_base = PDFKnowledgeBase(
    path = "path/to/pdf",
    reader=PDFReader(chunk=True),
    vector_db=ChromaDb(collection="DenseNet",embedder=SentenceTransformerEmbedder(),persistent_client=True,path='path/to/collection/Chromadb'),
)

# agent = Agent(
#     knowledge=knowledge_base,
#     search_knowledge=True,
# )
# agent.knowledge.load(recreate=False)

# agent.print_response("Ask me about something from the knowledge base")


agent = Agent(
    model=Groq(id="llama-3.1-8b-instant"),
    description="You are an intelligent document assistant specialized in extracting and analyzing information from PDFs and images. For a given PDF or image, you will identify and extract relevant text, tables, or visual details. Then, you will summarize, analyze, or generate responses based on the extracted content to assist users in understanding or utilizing the information effectively",
    instructions=[
        "Extract all text and visual content from the provided PDF or image using OCR and semantic understanding",
        "If the document contains structured data (e.g., tables, charts), ensure it is extracted in an organized format",
        "Analyze the extracted data, providing insights or summaries based on user queries",
        "Offer contextually appropriate responses, ensuring clarity and relevance for user needs",
    ],
    markdown=True,
    show_tool_calls=True,
    knowledge=knowledge_base,
    search_knowledge=True,
)
# agent.knowledge.load(recreate=False)

agent.print_response("What Deep Networks. Answer based on the document only")



