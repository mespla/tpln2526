# Introduction to computational linguistics and natural language processing (November 21, 2025)

Computational linguistics (CL) is a branch of linguistics that focuses on the theoretical understanding of language through computational models. In contrast, natural language processing (NLP) is an interdisciplinary field within artificial intelligence (AI) that aims to use computational models to process and generate language efficiently. NLP intersects with machine learning, statistics, and data science. While NLP is not primarily focused on linguistics, many of its approaches and tasks draw on linguistic theories to address the complexities of natural language. NLP cover a wide range of tasks, including part-of-speech tagging, named entity recognition, machine translation, speech recognition, and text summarization.

In this session, we will introduce some fundamental concepts and techniques in NLP, focusing specifically on text as the medium of natural language, rather than speech. While historically NLP has concentrated on English and a few major languages, recent years have seen a shift towards a more multilingual focus, with growing interest in low-resource languages.

With this in mind, the session will be organized into four main blocks. We will begin by discussing the steps involved in preparing text or a corpus for NLP applications. Next, we will explore how text can be processed at three different levels: the word level (morphology), the sentence structure level (syntax), and finally, the level of sentence meaning (semantics).

The [slides for the first session](https://raw.githubusercontent.com/mespla/tpln2526/refs/heads/main/docs/slides/session1tpln.pdf) are now available.

## Text preprocessing

### Contents to prepare before the session on 11/21/2025

As mentioned earlier, this session focuses on processing textual data. Texts come from diverse sources, languages, formats, scripts, and character encoding standards. A common preliminary step in preparing text for any NLP-related task is preprocessing it to make it suitable for the specific application. Typical preprocessing tasks include removing formatting, converting character encodings, and tokenizing. Additional steps often involve normalizing text, standardizing punctuation, and similar operations.

A helpful introduction to these strategies and their implications for various NLP tasks can be found in the article [*Comparison of text preprocessing methods*](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/43A20821D65F1C0C4366B126FC794AE3/S1351324922000213a.pdf/comparison-of-text-preprocessing-methods.pdf), which you are required to read before class. You only need to read until Section 3.5, as the strategies described in Sections 3.6 and 3.7 are not that frequent for many NLP tasks, and Section 4 describes datasets that are not relevant for this session. 1.5 hours 🕒️ in duration.

The article focuses on tokenization at the word level. However, most neural-network-based approaches rely on subword-level tokenization, which involves splitting words into fragments ranging from single characters to character groups. Some popular subword-level tokenization techniques include byte-pair encoding (BPE), unigram, and SentencePiece.

A concise and intuitive explanation of these methods can be found in the [*Tokenizers*](https://huggingface.co/docs/transformers/main/tokenizer_summary) section of the HuggingFace Transformers tutorial. Scroll to the end of the page for an overview of how these subword tokenization strategies work. 0.5 hours 🕒️ in duration.

Additionally, if you are curious about how popular large language models handle subword tokenization, explore the [Tiktokenizer](https://tiktokenizer.vercel.app). This tool simulates the tokenization process of several well-known generative neural models. Select a model from the dropdown menu in the upper-right corner and input a short text to see an example of subword tokenization. Try experimenting with texts in different languages to observe how these models manage multilingual input.

## Morphological parsing

### Contents to prepare before the session on 11/21/2025

In this section, we will explore computational approaches to modeling morphology, the study of word structure. Morphological parsing involves analyzing the components of a word to understand their role in a sentence and their contribution to the meaning of a text fragment. Morphological parsing is essential for various NLP tasks, such as word segmentation and lemmatization.

It is important to note that languages vary significantly in morphological complexity. For languages with complex morphology, morphology-aware NLP models have shown to be particularly effective, especially when only limited data is available for these languages. The paper [*Morphological Processing of Low-Resource Languages: Where We Are and What’s Next*](https://aclanthology.org/2022.findings-acl.80.pdf) offers a comprehensive overview of the state-of-the-art techniques in morphology analysis, with a focus on low-resource languages. These languages are especially challenging due to the scarcity of data for training state-of-the-art models. 1 hour 🕒️ in duration.

### Additional **optional** material: 

* An ambitious effort in recent years to create a comprehensive knowledge base for morphology across many languages is the UniMorph project. The paper [*UniMorph 2.0: Universal Morphology*](https://aclanthology.org/L18-1293v2.pdf) provides a detailed overview of this initiative.
* For a broader perspective on how computational approaches have addressed morphology over the decades, I recommend the chapter [*Computational morphology*](https://ling.auf.net/lingbuzz/007366/current.pdf?_s=R2w2FR3x92UHOOwt) in the book *What is Morphology?*, edited by Mark Aronoff and Kirsten Fudeman (Wiley Blackwell, 2022). This chapter covers the evolution of computational morphology, from rule-based methods using finite-state automata to statistical approaches based on hidden Markov models.

## Syntactic parsing

### Contents to prepare before the session on 11/21/2025

Syntactic parsing involves automatically inferring the structural relationships between words in a sentence. This task is crucial for understanding the meaning of a text fragment. In this block, we will explore the technologies that enable the analysis of word relationships within sentences and how these relationships impact meaning.

For a solid introduction to computational syntax and syntactic parsing, I recommend watching [*Depenency Parsing*](https://www.youtube.com/watch?v=f-3N0stPtbw), a lecture by Graham Neubig from the *Multilingual Natural Language Processing* course at Carnegie Mellon University's Language Technology Institute (2022). The video is divided into two parts, but you only need to watch the first part, which ends at approximately minute 38\. This lecture compares the two main syntactic parsing approaches—constituent parsing and dependency parsing—highlighting their advantages in multilingual contexts. It also covers key resources and tools, including the Universal Dependencies (UD) treebank, a foundational resource for multilingual dependency parsing, and discusses the primary applications of syntax in NLP. 0.75 hours  🕒️ in duration.

As mentioned earlier, Universal Dependencies is one of the most widely used resources for training models in dependency parsing. Several tools and libraries leverage this resource, including the Stanza library, developed by the Stanford NLP research group. To prepare for this session, familiarize yourself with Stanza using the [*Multilingual Text Processing*](https://applied-language-technology.mooc.fi/html/notebooks/part_iii/01_multilingual_nlp.html) tutorial on the Applied Language Technology platform of the University of Helsinki. 1.25 hours 🕒️ in duration.

### Additional **optional** material:

For those interested in learning more about the Universal Dependencies project, I recommend a recent [lecture](https://www.youtube.com/watch?v=rIo44KZ9nTc) by Joakim Nivre, one of the project’s founders, delivered at the Institute of Formal and Applied Linguistics, Charles University (Czech Republic) in April 2024\. The lecture, available as a one-hour video, provides deeper insights into the project.

## Semantic representation of text

### Contents to prepare before the session on 11/21/2025

This final block of the session focuses on the semantic representation of text. In NLP, there are two main approaches to representing meaning: identifying the semantic roles of text components and producing vector-based semantic representations.

The first approach, inspired by linguistic theories of semantic analysis, involves automatically labeling roles such as *experiencer*, *agent*, *theme*, or *goal* within a text fragment. For instance, in the sentence *John broke the window with a rock*, we could label *John* as the *agent*, *broke the window* as the *theme*, and *with a rock* as the *instrument*. This task, known as **semantic role labeling**, has been crucial in NLP for many years. For example, it has enhanced machine translation systems, particularly for languages with strong divergences in the way meaning is expressed from a morphological and syntactic point of view.

The second approach, **vector semantics**, focuses on building numerical representations (vectors) that capture the meaning of a text fragment. These semantic vectors are essential for various NLP tasks, including information retrieval and question answering, and they form a foundational component of neural-network-based NLP models.

In this block, we will focus on vector semantics. For a solid introduction to the basics of this topic, refer to [Chapter 6 of *Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/6.pdf) by Daniel Jurafsky and James H. Martin (2024). You only need to read up to Section 6.5; some of the remaining sections will be covered in later sessions. 1 hour 🕒️ in duration.
