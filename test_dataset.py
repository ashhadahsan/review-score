"""
Test dataset for ReviewScore evaluation.
Creates a comprehensive dataset with human annotations for testing the system.
"""

from typing import List, Dict, Any
from reviewscore import create_review_point, ReviewPointType, Question, Claim, Argument


def create_test_dataset() -> List[Dict[str, Any]]:
    """
    Create a comprehensive test dataset with human annotations.
    Based on the paper's evaluation methodology.
    """

    # Sample paper content for testing
    paper_content = """
    Title: Attention Is All You Need: A Novel Approach to Neural Machine Translation
    
    Abstract: We propose a new simple network architecture, the Transformer, 
    based solely on attention mechanisms, dispensing with recurrence and convolutions 
    entirely. Experiments on two machine translation tasks show these models to be 
    superior in quality while being more parallelizable and requiring significantly 
    less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German 
    translation task, improving over the existing best results, including ensembles, 
    by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model 
    establishes a new single-model state-of-the-art BLEU score of 41.8 after training 
    for 3.5 days on eight P100 GPUs. We show that the Transformer generalizes well to 
    other tasks by applying it successfully to English constituency parsing with large 
    and limited training data.
    
    Introduction: Recurrent neural networks, long short-term memory (LSTM) and 
    gated recurrent neural networks in particular, have been firmly established as 
    state of the art approaches in sequence modeling and transduction problems such 
    as language modeling and machine translation. Numerous efforts have since 
    continued to push the boundaries of recurrent language models and encoder-decoder 
    architectures. Recurrent models typically factor computation along the symbol 
    positions of the input and output sequences. Aligning the positions to steps in 
    computation time, they generate a sequence of hidden states h_t, as a function 
    of the previous hidden state h_{t-1} and the input for position t. This inherently 
    sequential nature precludes parallelization within training examples, which becomes 
    critical at longer sequence lengths, as memory constraints limit batching across 
    examples. Recent work has achieved significant improvements in computational 
    efficiency through factorization tricks and conditional computation, while also 
    improving model performance in case of the latter. The fundamental constraint of 
    sequential computation, however, remains.
    
    Method: We propose the Transformer, a model architecture eschewing recurrence 
    and instead relying entirely on an attention mechanism to draw global dependencies 
    between input and output. The Transformer follows this overall architecture using 
    stacked self-attention and point-wise, fully connected layers for both the encoder 
    and decoder. The encoder maps an input sequence of symbol representations 
    (x_1, ..., x_n) to a sequence of continuous representations z = (z_1, ..., z_n). 
    Given z, the decoder then generates an output sequence (y_1, ..., y_m) of symbols 
    one element at a time. At each step the model is auto-regressive, consuming the 
    previously generated symbols as additional input when generating the next.
    
    Experiments: We evaluate the Transformer on the WMT 2014 English-to-German and 
    English-to-French translation tasks. We use the standard WMT 2014 data, which 
    consists of about 4.5M sentence pairs. We use the standard development set, 
    newstest2013, and report results on the test set, newstest2014. We use a vocabulary 
    of 37,000 tokens for English-German and 32,000 tokens for English-French. We use 
    the Adam optimizer with β1 = 0.9, β2 = 0.98 and ε = 10−9. We vary the learning 
    rate over the course of training, increasing it linearly for the first warmup_steps 
    training steps and decreasing it thereafter proportionally to the inverse square 
    root of the step number. We use warmup_steps = 4000.
    
    Results: On the WMT 2014 English-to-German translation task, the big Transformer 
    model (Transformer (big) in Table 2) outperforms the best previously reported 
    models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art 
    BLEU score of 28.4. On the WMT 2014 English-to-French translation task, the big 
    Transformer model achieves a BLEU score of 41.8, outperforming all of the 
    previously published single models, at less than 1/4 the training cost of the 
    previous state-of-the-art model.
    
    Conclusion: We presented the Transformer, the first sequence transduction model 
    based entirely on attention, replacing the recurrent layers most commonly used in 
    encoder-decoder architectures with multi-headed self-attention. For translation 
    tasks, the Transformer can be trained significantly faster than architectures 
    based on recurrent or convolutional layers. On both WMT 2014 English-to-German 
    and English-to-French translation tasks, we achieve a new state of the art. 
    In the former task our best model outperforms even all previously reported 
    ensembles. We are excited about the future of attention-based models and plan 
    to apply them to other tasks.
    """

    # Test cases with human annotations
    test_cases = [
        # Question cases
        {
            "review_point": create_review_point(
                text="What is the BLEU score achieved on the English-to-German translation task?",
                point_type=ReviewPointType.QUESTION,
                paper_context=paper_content,
                review_context="This is a question about the paper's results.",
                point_id="question_1",
            ),
            "human_annotation": {
                "is_misinformed": True,  # Question is answerable by paper
                "score": 2.0,
                "reasoning": "The paper clearly states the BLEU score of 28.4 on English-to-German translation task in the Results section.",
            },
        },
        {
            "review_point": create_review_point(
                text="How does the Transformer handle long sequences compared to RNNs?",
                point_type=ReviewPointType.QUESTION,
                paper_context=paper_content,
                review_context="This is a question about the paper's methodology.",
                point_id="question_2",
            ),
            "human_annotation": {
                "is_misinformed": False,  # Question is not directly answerable
                "score": 4.0,
                "reasoning": "While the paper mentions parallelization benefits, it doesn't provide detailed analysis of long sequence handling compared to RNNs.",
            },
        },
        # Claim cases
        {
            "review_point": create_review_point(
                text="The paper achieves a BLEU score of 28.4 on English-to-German translation.",
                point_type=ReviewPointType.CLAIM,
                paper_context=paper_content,
                review_context="This is a claim about the paper's results.",
                point_id="claim_1",
            ),
            "human_annotation": {
                "is_misinformed": False,  # Claim is factually correct
                "score": 5.0,
                "reasoning": "The paper explicitly states this BLEU score in the Results section.",
            },
        },
        {
            "review_point": create_review_point(
                text="The Transformer uses convolutional layers for sequence processing.",
                point_type=ReviewPointType.CLAIM,
                paper_context=paper_content,
                review_context="This is a claim about the paper's methodology.",
                point_id="claim_2",
            ),
            "human_annotation": {
                "is_misinformed": True,  # Claim is factually incorrect
                "score": 1.0,
                "reasoning": "The paper explicitly states that the Transformer eschews recurrence and convolutions, relying entirely on attention mechanisms.",
            },
        },
        # Argument cases
        {
            "review_point": create_review_point(
                text="The Transformer is superior to RNNs because it can be parallelized during training, which is mentioned in the introduction, and it achieves better BLEU scores as shown in the results section.",
                point_type=ReviewPointType.ARGUMENT,
                paper_context=paper_content,
                review_context="This is an argument about the paper's contributions.",
                point_id="argument_1",
            ),
            "human_annotation": {
                "is_misinformed": False,  # Argument is factually correct
                "score": 4.0,
                "reasoning": "Both premises are factually correct: the paper mentions parallelization benefits and reports superior BLEU scores.",
            },
        },
        {
            "review_point": create_review_point(
                text="The paper's results are not reliable because the authors used a different dataset than previous work, and the training time comparison is unfair since they used more powerful hardware.",
                point_type=ReviewPointType.ARGUMENT,
                paper_context=paper_content,
                review_context="This is an argument criticizing the paper's methodology.",
                point_id="argument_2",
            ),
            "human_annotation": {
                "is_misinformed": True,  # Argument contains incorrect premises
                "score": 2.0,
                "reasoning": "The paper uses standard WMT 2014 data (same as previous work) and provides fair comparisons. The hardware criticism is not substantiated in the paper.",
            },
        },
        # Edge cases
        {
            "review_point": create_review_point(
                text="Does the paper discuss the computational complexity of the attention mechanism?",
                point_type=ReviewPointType.QUESTION,
                paper_context=paper_content,
                review_context="This is a question about computational analysis.",
                point_id="question_3",
            ),
            "human_annotation": {
                "is_misinformed": False,  # Question is not answerable
                "score": 4.0,
                "reasoning": "The paper does not provide detailed computational complexity analysis of the attention mechanism.",
            },
        },
        {
            "review_point": create_review_point(
                text="The paper fails to compare with recent attention-based models.",
                point_type=ReviewPointType.CLAIM,
                paper_context=paper_content,
                review_context="This is a claim about the paper's limitations.",
                point_id="claim_3",
            ),
            "human_annotation": {
                "is_misinformed": True,  # Claim is incorrect
                "score": 2.0,
                "reasoning": "The paper is the first to propose the Transformer architecture, so there are no previous attention-based models to compare with.",
            },
        },
    ]

    return test_cases


def create_extended_test_dataset() -> List[Dict[str, Any]]:
    """
    Create an extended test dataset with more complex cases.
    """

    # Extended paper content
    paper_content = """
    Title: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    
    Abstract: We introduce a new language representation model called BERT, which stands 
    for Bidirectional Encoder Representations from Transformers. Unlike recent language 
    representation models, BERT is designed to pre-train deep bidirectional representations 
    from unlabeled text by jointly conditioning on both left and right context in all layers. 
    As a result, the pre-trained BERT model can be fine-tuned with just one additional 
    output layer to create state-of-the-art models for a wide range of tasks, such as 
    question answering and language inference, without substantial task-specific architecture 
    modifications. BERT is conceptually simple and empirically powerful. It obtains new 
    state-of-the-art results on eleven natural language processing tasks, including 
    pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy 
    to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2% 
    (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1% (5.1 point absolute improvement).
    
    Introduction: Learning representations of words is a fundamental problem in natural 
    language processing. There are two main approaches: feature-based and fine-tuning. 
    Feature-based approaches, such as ELMo, use task-specific architectures that include 
    the pre-trained representations as additional features. Fine-tuning approaches, such as 
    the Generative Pre-trained Transformer (OpenAI GPT), introduce minimal task-specific 
    parameters and are trained on the downstream tasks by simply fine-tuning all 
    pre-trained parameters. The two approaches share the same objective function during 
    pre-training, where they use unidirectional language models to learn general language 
    representations. We argue that current techniques limit the power of the pre-trained 
    representations, especially for the fine-tuning approaches. The major limitation is 
    that standard language models are unidirectional, and this limits the choice of 
    architectures that can be used during pre-training. For example, in OpenAI GPT, 
    the authors use a left-to-right architecture, where every token can only attend to 
    previous tokens in the self-attention layers of the Transformer. Such restrictions 
    are sub-optimal for sentence-level tasks, and could be very harmful when applying 
    fine-tuning based approaches to token-level tasks such as question answering, where 
    a bidirectional context is crucial.
    
    Method: BERT addresses the previously mentioned unidirectional constraint by using 
    a "masked language model" (MLM) pre-training objective, inspired by the Cloze task. 
    The masked language model randomly masks some of the tokens from the input, and the 
    objective is to predict the original vocabulary id of the masked word based only on 
    its context. Unlike left-to-right language model pre-training, the MLM objective 
    enables the representation to fuse the left and the right context, which allows us 
    to pre-train a deep bidirectional Transformer. In addition to the masked language 
    model, we also use a "next sentence prediction" task that jointly pre-trains 
    text-pair representations. The pre-training procedure follows the existing literature 
    on language model pre-training. For the pre-training corpus, we use the BooksCorpus 
    (800M words) and English Wikipedia (2,500M words). For the input representation, 
    BERT is able to represent a single sentence or a pair of sentences (e.g., [Question, 
    Answer]) in one token sequence. Throughout this work, a "sentence" can be an arbitrary 
    span of contiguous text, rather than an actual linguistic sentence. A "sequence" 
    refers to the input token sequence to BERT, which may be a single sentence or two 
    sentences packed together. We use WordPiece embeddings with a 30,000 token vocabulary. 
    The final hidden vector of the special [CLS] token is used as the aggregate sequence 
    representation for classification tasks. For non-classification tasks, a token 
    representation is used instead.
    
    Experiments: We evaluate BERT using two model sizes: BERT_BASE (L=12, H=768, A=12, 
    Total Parameters=110M) and BERT_LARGE (L=24, H=1024, A=16, Total Parameters=340M). 
    BERT_BASE was chosen to have the same model size as OpenAI GPT for comparison purposes. 
    We present results on 11 NLP tasks. On the right side of the table, we show the 
    results for BERT_LARGE. In all cases, BERT_LARGE significantly outperforms BERT_BASE. 
    The best results across all tasks are obtained by fine-tuning BERT_LARGE, which 
    achieves state-of-the-art results on 11 individual NLP tasks, including pushing the 
    GLUE benchmark to 80.5%, MultiNLI accuracy to 86.7%, and SQuAD Test F1 to 93.2%.
    
    Results: BERT_LARGE significantly outperforms BERT_BASE across all tasks. The 
    magnitude of the improvements can be attributed to the model capacity, as BERT_LARGE 
    has 3.4x more parameters than BERT_BASE. The improvements are particularly pronounced 
    for tasks that require fine-grained understanding, such as SQuAD, where BERT_LARGE 
    outperforms BERT_BASE by 4.7% absolute on Test F1. On the GLUE benchmark, BERT_LARGE 
    achieves 80.5%, which is a 7.7% absolute improvement over the previous state-of-the-art. 
    The results demonstrate that the bidirectional pre-training is crucial for achieving 
    these results, as the unidirectional models (OpenAI GPT) perform significantly worse 
    on most tasks.
    """

    # Extended test cases
    test_cases = [
        # Complex question cases
        {
            "review_point": create_review_point(
                text="What is the difference between BERT_BASE and BERT_LARGE in terms of parameters?",
                point_type=ReviewPointType.QUESTION,
                paper_context=paper_content,
                review_context="Question about model architecture.",
                point_id="question_4",
            ),
            "human_annotation": {
                "is_misinformed": True,  # Answerable by paper
                "score": 2.0,
                "reasoning": "The paper explicitly states that BERT_BASE has 110M parameters and BERT_LARGE has 340M parameters, and that BERT_LARGE has 3.4x more parameters than BERT_BASE.",
            },
        },
        # Complex claim cases
        {
            "review_point": create_review_point(
                text="BERT uses a unidirectional language model for pre-training.",
                point_type=ReviewPointType.CLAIM,
                paper_context=paper_content,
                review_context="Claim about BERT's architecture.",
                point_id="claim_4",
            ),
            "human_annotation": {
                "is_misinformed": True,  # Factually incorrect
                "score": 1.0,
                "reasoning": "The paper explicitly states that BERT is designed to pre-train deep bidirectional representations, not unidirectional ones.",
            },
        },
        # Complex argument cases
        {
            "review_point": create_review_point(
                text="BERT achieves state-of-the-art results because it uses bidirectional representations, which allows it to capture context from both directions, and it uses a masked language model objective, which enables better understanding of word relationships.",
                point_type=ReviewPointType.ARGUMENT,
                paper_context=paper_content,
                review_context="Argument about BERT's success factors.",
                point_id="argument_3",
            ),
            "human_annotation": {
                "is_misinformed": False,  # Factually correct
                "score": 5.0,
                "reasoning": "Both premises are factually correct: the paper explains that bidirectional representations and the masked language model objective are key to BERT's success.",
            },
        },
    ]

    return test_cases


def get_complete_test_dataset() -> List[Dict[str, Any]]:
    """
    Get the complete test dataset combining basic and extended cases.
    """
    basic_dataset = create_test_dataset()
    extended_dataset = create_extended_test_dataset()
    return basic_dataset + extended_dataset


def print_dataset_summary(dataset: List[Dict[str, Any]]):
    """Print a summary of the test dataset."""
    print(f"Test Dataset Summary:")
    print(f"Total test cases: {len(dataset)}")

    # Count by type
    type_counts = {}
    misinformed_counts = {}

    for case in dataset:
        point_type = case["review_point"].type
        is_misinformed = case["human_annotation"]["is_misinformed"]

        type_counts[point_type] = type_counts.get(point_type, 0) + 1
        misinformed_counts[point_type] = misinformed_counts.get(point_type, 0) + (
            1 if is_misinformed else 0
        )

    print(f"\nBy type:")
    for point_type, count in type_counts.items():
        misinformed = misinformed_counts[point_type]
        print(f"  {point_type}: {count} cases ({misinformed} misinformed)")

    print(
        f"\nOverall misinformed rate: {sum(misinformed_counts.values()) / len(dataset):.1%}"
    )


if __name__ == "__main__":
    # Create and display test dataset
    dataset = get_complete_test_dataset()
    print_dataset_summary(dataset)

    # Print sample cases
    print(f"\nSample test cases:")
    for i, case in enumerate(dataset[:3]):
        print(f"\n--- Case {i+1} ---")
        print(f"Type: {case['review_point'].type}")
        print(f"Text: {case['review_point'].text[:100]}...")
        print(f"Human annotation: {case['human_annotation']}")
