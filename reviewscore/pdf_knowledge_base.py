"""
PDF Knowledge Base for ReviewScore.
Supports loading and processing PDF documents as knowledge sources.
"""

import os
import tempfile
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

# PDF processing libraries
try:
    import PyPDF2
    import pdfplumber
    import fitz  # PyMuPDF

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning(
        "PDF processing libraries not installed. Install with: pip install PyPDF2 pdfplumber PyMuPDF"
    )

# Text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .core import ModelConfig


class BaseKnowledgeBase:
    """Base class for knowledge bases."""

    def __init__(self, name: str, kb_type: str):
        self.name = name
        self.kb_type = kb_type


class PDFKnowledgeBase(BaseKnowledgeBase):
    """
    Knowledge base that loads content from PDF files.
    Supports multiple PDF processing backends for robust text extraction.
    """

    def __init__(self, pdf_path: Union[str, Path], name: str = "pdf_kb"):
        """
        Initialize PDF knowledge base.

        Args:
            pdf_path: Path to the PDF file
            name: Name for the knowledge base
        """
        super().__init__(name=name, kb_type="pdf")

        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self.content = ""
        self.metadata = {}
        self.chunks = []
        self._load_pdf()

    def _load_pdf(self):
        """Load and process PDF content."""
        if not PDF_SUPPORT:
            raise ImportError(
                "PDF processing libraries not available. Install with: pip install PyPDF2 pdfplumber PyMuPDF"
            )

        # Try multiple PDF processing methods for robustness
        content_methods = [
            self._extract_with_pdfplumber,
            self._extract_with_pymupdf,
            self._extract_with_pypdf2,
        ]

        for method in content_methods:
            try:
                content, metadata = method()
                if content.strip():
                    self.content = content
                    self.metadata = metadata
                    logging.info(f"Successfully loaded PDF using {method.__name__}")
                    break
            except Exception as e:
                logging.warning(f"Failed to load PDF with {method.__name__}: {e}")
                continue

        if not self.content.strip():
            raise ValueError("Failed to extract content from PDF using any method")

        # Split content into chunks for better processing
        self._chunk_content()

    def _extract_with_pdfplumber(self) -> tuple[str, Dict[str, Any]]:
        """Extract text using pdfplumber (best for tables and formatting)."""
        content = ""
        metadata = {"method": "pdfplumber", "pages": 0}

        with pdfplumber.open(self.pdf_path) as pdf:
            metadata["pages"] = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    content += f"\n--- Page {page_num + 1} ---\n{page_text}"

        return content, metadata

    def _extract_with_pymupdf(self) -> tuple[str, Dict[str, Any]]:
        """Extract text using PyMuPDF (fast and reliable)."""
        content = ""
        metadata = {"method": "pymupdf", "pages": 0}

        doc = fitz.open(self.pdf_path)
        metadata["pages"] = doc.page_count

        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text:
                content += f"\n--- Page {page_num + 1} ---\n{page_text}"

        doc.close()
        return content, metadata

    def _extract_with_pypdf2(self) -> tuple[str, Dict[str, Any]]:
        """Extract text using PyPDF2 (fallback method)."""
        content = ""
        metadata = {"method": "pypdf2", "pages": 0}

        with open(self.pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata["pages"] = len(pdf_reader.pages)

            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    content += f"\n--- Page {page_num + 1} ---\n{page_text}"

        return content, metadata

    def _chunk_content(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Split content into manageable chunks."""
        if not self.content:
            return

        # Use LangChain's text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        # Create documents for splitting
        doc = Document(page_content=self.content, metadata=self.metadata)
        self.chunks = text_splitter.split_documents([doc])

        logging.info(f"Split PDF content into {len(self.chunks)} chunks")

    def get_content(self) -> str:
        """Get the full PDF content."""
        return self.content

    def get_chunks(self) -> List[Document]:
        """Get the PDF content as chunks."""
        return self.chunks

    def search_content(self, query: str, max_chunks: int = 5) -> List[Document]:
        """
        Search for relevant content in the PDF.

        Args:
            query: Search query
            max_chunks: Maximum number of chunks to return

        Returns:
            List of relevant document chunks
        """
        if not self.chunks:
            return []

        # Simple keyword-based search (can be enhanced with embeddings)
        query_lower = query.lower()
        relevant_chunks = []

        for chunk in self.chunks:
            content_lower = chunk.page_content.lower()
            if query_lower in content_lower:
                relevant_chunks.append(chunk)

        # Return top chunks
        return relevant_chunks[:max_chunks]

    def get_metadata(self) -> Dict[str, Any]:
        """Get PDF metadata."""
        return {
            "name": self.name,
            "type": self.kb_type,
            "pdf_path": str(self.pdf_path),
            "content_length": len(self.content),
            "num_chunks": len(self.chunks),
            **self.metadata,
        }


class MultiPDFKnowledgeBase:
    """
    Knowledge base that can handle multiple PDF files.
    Useful for loading multiple papers or documents.
    """

    def __init__(self, name: str = "multi_pdf_kb"):
        self.name = name
        self.pdf_kbs: List[PDFKnowledgeBase] = []
        self.combined_content = ""

    def add_pdf(
        self, pdf_path: Union[str, Path], kb_name: Optional[str] = None
    ) -> PDFKnowledgeBase:
        """
        Add a PDF to the knowledge base.

        Args:
            pdf_path: Path to the PDF file
            kb_name: Optional name for this PDF KB

        Returns:
            PDFKnowledgeBase instance
        """
        if kb_name is None:
            kb_name = f"pdf_{len(self.pdf_kbs) + 1}"

        pdf_kb = PDFKnowledgeBase(pdf_path, name=kb_name)
        self.pdf_kbs.append(pdf_kb)

        # Update combined content
        self.combined_content += f"\n\n--- {pdf_kb.name} ---\n{pdf_kb.get_content()}"

        return pdf_kb

    def get_all_content(self) -> str:
        """Get content from all PDFs."""
        return self.combined_content

    def search_all(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search across all PDFs.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results with source information
        """
        results = []

        for pdf_kb in self.pdf_kbs:
            chunks = pdf_kb.search_content(query, max_chunks=3)
            for chunk in chunks:
                results.append(
                    {
                        "content": chunk.page_content,
                        "source": pdf_kb.name,
                        "metadata": chunk.metadata,
                    }
                )

        return results[:max_results]

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for all PDFs."""
        return {
            "name": self.name,
            "num_pdfs": len(self.pdf_kbs),
            "total_content_length": len(self.combined_content),
            "pdfs": [pdf_kb.get_metadata() for pdf_kb in self.pdf_kbs],
        }


def create_pdf_knowledge_base(
    pdf_path: Union[str, Path], name: str = "pdf_kb"
) -> PDFKnowledgeBase:
    """
    Factory function to create a PDF knowledge base.

    Args:
        pdf_path: Path to the PDF file
        name: Name for the knowledge base

    Returns:
        PDFKnowledgeBase instance
    """
    return PDFKnowledgeBase(pdf_path, name)


def create_multi_pdf_knowledge_base(
    name: str = "multi_pdf_kb",
) -> MultiPDFKnowledgeBase:
    """
    Factory function to create a multi-PDF knowledge base.

    Args:
        name: Name for the knowledge base

    Returns:
        MultiPDFKnowledgeBase instance
    """
    return MultiPDFKnowledgeBase(name)


# Example usage and testing
if __name__ == "__main__":
    # Test PDF knowledge base
    try:
        # This would be used with actual PDF files
        print("PDF Knowledge Base Test")
        print("=" * 40)

        # Example usage (replace with actual PDF path)
        # pdf_kb = create_pdf_knowledge_base("path/to/paper.pdf", "research_paper")
        # print(f"Loaded PDF: {pdf_kb.get_metadata()}")

        print("PDF Knowledge Base functionality ready!")
        print("To use: pip install PyPDF2 pdfplumber PyMuPDF")

    except Exception as e:
        print(f"Error: {e}")
