#!/usr/bin/env python3
"""
Example script showing how to use PDF knowledge bases with ReviewScore.
Demonstrates loading PDF papers and using them as knowledge sources.
"""

from pathlib import Path
from reviewscore.pdf_knowledge_base import (
    create_pdf_knowledge_base,
    create_multi_pdf_knowledge_base
)
from reviewscore.paper_faithful import PaperFaithfulReviewScore, PaperFaithfulConfig
from reviewscore.core import create_review_point, ReviewPointType


def example_single_pdf():
    """Example: Using a single PDF as knowledge base."""
    print("Example: Single PDF Knowledge Base")
    print("=" * 50)
    
    # Replace with your actual PDF path
    pdf_path = "path/to/your/paper.pdf"
    
    if not Path(pdf_path).exists():
        print(f"PDF file not found: {pdf_path}")
        print("Please provide a valid PDF path.")
        return
    
    try:
        # Create PDF knowledge base
        pdf_kb = create_pdf_knowledge_base(pdf_path, "research_paper")
        print(f"✓ Loaded PDF: {pdf_kb.name}")
        
        # Get PDF content
        content = pdf_kb.get_content()
        print(f"✓ Content length: {len(content)} characters")
        
        # Create ReviewScore evaluator
        config = PaperFaithfulConfig()
        evaluator = PaperFaithfulReviewScore(config)
        
        # Set up knowledge bases with PDF content
        evaluator.set_knowledge_bases(
            submitted_paper=content,
            annotator_knowledge="Expert knowledge about the research domain",
            referred_papers="Related work and citations"
        )
        
        # Create review points to evaluate
        review_points = [
            create_review_point(
                text="What methodology does this paper use?",
                point_type=ReviewPointType.QUESTION,
                paper_context=content,
                review_context="The reviewer asks about methodology.",
                point_id="q1"
            ),
            create_review_point(
                text="The experimental results are not convincing.",
                point_type=ReviewPointType.CLAIM,
                paper_context=content,
                review_context="The reviewer criticizes the results.",
                point_id="c1"
            ),
            create_review_point(
                text="The paper lacks proper evaluation because the dataset is too small.",
                point_type=ReviewPointType.ARGUMENT,
                paper_context=content,
                review_context="The reviewer argues about evaluation.",
                point_id="a1"
            )
        ]
        
        # Evaluate each review point
        print("\nEvaluating review points:")
        for i, point in enumerate(review_points, 1):
            result = evaluator.evaluate_with_paper_methodology(point)
            print(f"  {i}. {point.type.value}: Score={result.base_score}, Misinformed={result.is_misinformed}")
        
        print("✓ Single PDF example completed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_multi_pdf():
    """Example: Using multiple PDFs as knowledge base."""
    print("\nExample: Multi-PDF Knowledge Base")
    print("=" * 50)
    
    # Replace with your actual PDF paths
    pdf_paths = [
        "path/to/paper1.pdf",
        "path/to/paper2.pdf",
        "path/to/paper3.pdf"
    ]
    
    # Check if PDFs exist
    existing_pdfs = [path for path in pdf_paths if Path(path).exists()]
    if not existing_pdfs:
        print("No PDF files found. Please provide valid PDF paths.")
        return
    
    try:
        # Create multi-PDF knowledge base
        multi_kb = create_multi_pdf_knowledge_base("research_papers")
        print(f"✓ Created multi-PDF knowledge base: {multi_kb.name}")
        
        # Add PDFs
        for i, pdf_path in enumerate(existing_pdfs):
            pdf_kb = multi_kb.add_pdf(pdf_path, f"paper_{i+1}")
            print(f"✓ Added PDF: {pdf_kb.name}")
        
        # Get combined content
        combined_content = multi_kb.get_all_content()
        print(f"✓ Combined content length: {len(combined_content)} characters")
        
        # Search across all PDFs
        search_results = multi_kb.search_all("transformer", max_results=5)
        print(f"✓ Search results for 'transformer': {len(search_results)} results")
        
        # Create ReviewScore evaluator with combined content
        config = PaperFaithfulConfig()
        evaluator = PaperFaithfulReviewScore(config)
        
        evaluator.set_knowledge_bases(
            submitted_paper=combined_content,
            annotator_knowledge="Expert knowledge across multiple papers",
            referred_papers="Cross-paper references and citations"
        )
        
        # Test evaluation
        question = create_review_point(
            text="What are the main contributions across these papers?",
            point_type=ReviewPointType.QUESTION,
            paper_context=combined_content,
            review_context="The reviewer asks about contributions.",
            point_id="multi_pdf_q1"
        )
        
        result = evaluator.evaluate_with_paper_methodology(question)
        print(f"✓ Multi-PDF evaluation: Score={result.base_score}, Misinformed={result.is_misinformed}")
        
        print("✓ Multi-PDF example completed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_pdf_search():
    """Example: Searching within PDF content."""
    print("\nExample: PDF Content Search")
    print("=" * 50)
    
    pdf_path = "path/to/your/paper.pdf"
    
    if not Path(pdf_path).exists():
        print(f"PDF file not found: {pdf_path}")
        return
    
    try:
        # Create PDF knowledge base
        pdf_kb = create_pdf_knowledge_base(pdf_path, "searchable_paper")
        
        # Search for specific content
        search_queries = [
            "methodology",
            "experimental results",
            "conclusion",
            "transformer",
            "attention mechanism"
        ]
        
        print("Searching within PDF content:")
        for query in search_queries:
            results = pdf_kb.search_content(query, max_chunks=2)
            print(f"  '{query}': {len(results)} results")
            
            if results:
                print(f"    First result: {results[0].page_content[:100]}...")
        
        print("✓ PDF search example completed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    """Run PDF knowledge base examples."""
    print("PDF Knowledge Base Examples for ReviewScore")
    print("=" * 60)
    print("This script demonstrates how to use PDF documents as knowledge sources.")
    print("=" * 60)
    
    print("\nTo use this script:")
    print("1. Install PDF processing libraries: pip install PyPDF2 pdfplumber PyMuPDF")
    print("2. Update the PDF paths in the examples below")
    print("3. Run the script")
    
    # Run examples
    example_single_pdf()
    example_multi_pdf()
    example_pdf_search()
    
    print("\n" + "=" * 60)
    print("PDF KNOWLEDGE BASE USAGE SUMMARY")
    print("=" * 60)
    print("✓ Single PDF: Load one PDF as knowledge source")
    print("✓ Multi-PDF: Load multiple PDFs for comprehensive knowledge")
    print("✓ Content Search: Search within PDF content")
    print("✓ ReviewScore Integration: Use PDFs with ReviewScore evaluators")
    print("\nKey Features:")
    print("- Multiple PDF processing backends (PyPDF2, pdfplumber, PyMuPDF)")
    print("- Automatic content chunking for better processing")
    print("- Search functionality within PDF content")
    print("- Integration with ReviewScore evaluation system")
    print("- Support for multiple PDF documents")


if __name__ == "__main__":
    main()
