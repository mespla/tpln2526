# Assignment on text similarity using vectorial representations of text
    
In Natural Language Processing (NLP), one of the primary challenges is how to represent human language in a form that computers can process. This is where ***vectorized representations of text*** come into play. Text data, such as sentences, paragraphs, or entire documents, need to be converted into numerical formats to be understood by machine learning algorithms. Vectorization transforms raw text into fixed-length, dense, or sparse vectors that capture the semantic properties of the words or sentences, enabling machines to perform tasks like classification, similarity search, and recommendation.

The methods you'll explore in this assignment — Bag of Words (BoW), TF-IDF, and Sentence Embeddings (SBERT) — are foundational techniques in NLP, each with its own strengths and weaknesses. Understanding these representations is critical for a wide range of applications in the field of NLP.

## Introduction to the work

In this task, you will work with a dataset containing a collection of article titles and their corresponding abstracts. This information has been extracted from the proceedings of the EMNLP international conference for the years 2016, 2017, and 2018. Your goal is to create vectorized representations of these articles, which can then be used to identify similarities between titles or abstracts of different papers.

As part of this assignment, you will explore various strategies for constructing these vector representations and use them to compare the similarity between text fragments. Specifically, you will:

 1. Evaluate two word-based methods for generating sparse vector representations of text: bag-of-words and TF-IDF-weighted word vectors.
 2. Use pre-trained embedding models to generate dense representations of the same text fragments.

This assignment will be completed in pairs. To make your work more manageable, a [CoLab notebook](https://colab.research.google.com/drive/18JEqt15fu-am84cz7hpkCCGXC-KUOoEc?usp=sharing) has been provided. It includes the basic structure of the tasks, along with code snippets to help you complete the assignment. You will use this CoLab notebook to:

 * Complete the required tasks.
 * Describe the results you obtain.

Remember that you will need to be logged with your GCloud account to access the CoLab notebook.

###Submission

In this assignment, your task is to implement the different strategies to produce vector representations of text and evaluate them with the dataset described in this document. You will also be required to try the different configurations and observe the results obtained. A base CoLab notebook is provided that contains the basic structure of the work to be done in this assignment. Follow the instructions in this document to both implement the solutions and to describe the results obtained and the conclusions drawn from this work. Once this is done, you should download the resulting CoLab notebook and submit it via the UACloud *Evaluation tools* before 23:59 on Sunday, December 14, 2025. The assignment must be done in pairs. Remember to include both authors' names in the notebook.

## Part 1: Bag of Words (BoW) and TF-IDF Vectorization with Scikit-Learn

In this section of the assignment, you will explore two widely used methods for vectorizing text: Bags of Words (BoW) and TF-IDF (Term Frequency-Inverse Document Frequency). Both techniques transform text into numerical vectors, but they capture different characteristics of the data.

   * *BoW*: Text fragments are converted into sparse vectors that list the words appearing in the text along with their respective counts. This method focuses exclusively on word frequency within each text fragment.
   * *TF-IDF*: This method builds on BoW by also considering the importance of a word across the entire corpus. It adjusts word counts based on how frequently the word appears in other documents, giving more weight to distinctive terms.

You will use both approaches to vectorize the titles and abstracts of research papers and then perform a similarity search. Your task will be to identify the top 3 most similar documents for a given query.

### Step 1: Preprocessing the Data

Before applying either BoW or TF-IDF, it is important to preprocess the text data to ensure that it is in a usable form. The dataset you will be working with is in the JSON format. In the CoLab mentioned above you have the code snippet to download and store it locally. For each research paper in the dataset, the JSON file contains:

 1. The title of the paper.
 2. The abstract of the paper.
 3. The URL of the paper.
 4. The venue in which it was presented.
 5. The year of publication of the work.

From these fields, you will only build vectorized representations on the title and the abstract. To do so, you can concatenate them in a single string:

  ```python
     df['text'] = df['title'] + ' ' + df['abstract']
  ```

### Step 2: Building the BoW and TF-IDF Representations

You will now create vectorized representations of the documents using Bag of Words (BoW) and TF-IDF. Both techniques are implemented in the `scikit-learn` library. To build BoW vectors, you will be using the [CountVectorizer](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) class, that automatically transforms a collection of fragment of text into a matrix of vectors consisting of a set of vectors containing the occurrences of each word. For the TF-IDF vectors, you will be using the [TfidfVectorizer](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer) class. You can use the method `fit_transform`, that takes a list of strings and creates the set of vectors representing a collection of documents. Have a look to the vectors obtained and comment in the notebook their format and their size. Have a look to the documentation of these vectorizers to better understand what do they mean.

A relevant aspect when producing vectorized representations building on words is *text preprocessing*. These vectorization methods allow some level of customization of the pre-processing steps carried out. For example, text is lowercased by default. In the same way, punctuation is ignored. However, some of these parameters can be modified; try, at least, enabling and disabling:

 * Lowercasing the text
 * Removing stop-words (you can use `stop_words='english'` in `CountVectorizer` or `TfidfVectorizer` to automatically handle this step)

Analyse the differences obtained when changing these parameters and provide a short discussion of the results.

### Step 3: Similarity Search

After creating the vector representations for the entire dataset, you will perform vector comparisons to identify similarities between these representations. In this step, you will compare the representations of three new queries against the entire dataset. Each query consists of the title and abstract of a paper published in different journals and international conferences. These queries are:

* Query 1:
    * *Title*: QUALES: Machine translation quality estimation via supervised and unsupervised machine learning.
    * *Abstract*: The automatic quality estimation (QE) of machine translation consists in measuring the quality of translations without access to human references, usually via machine learning approaches. A good QE system can help in three aspects of translation processes involving machine translation and post-editing: increasing productivity (by ruling out poor quality machine translation), estimating costs (by helping to forecast the cost of post-editing) and selecting a provider (if several machine translation systems are available). Interest in this research area has grown significantly in recent years, leading to regular shared tasks in the main machine translation conferences and intense scientific activity. In this article we review the state of the art in this research area and present project QUALES, which is under development. 
* Query 2:
    * *Title*: Learning to Ask Unanswerable Questions for Machine Reading Comprehension
    * *Abstract*: Machine reading comprehension with unanswerable questions is a challenging task. In this work, we propose a data augmentation technique by automatically generating relevant unanswerable questions according to an answerable question paired with its corresponding paragraph that contains the answer. We introduce a pair-to-sequence model for unanswerable question generation, which effectively captures the interactions between the question and the paragraph. We also present a way to construct training data for our question generation models by leveraging the existing reading comprehension dataset. Experimental results show that the pair-to-sequence model performs consistently better compared with the sequence-to-sequence baseline. We further use the automatically generated unanswerable questions as a means of data augmentation on the SQuAD 2.0 dataset, yielding 1.9 absolute F1 improvement with BERT-base model and 1.7 absolute F1 improvement with BERT-large model.
* Query 3:
    * *Title*: Unsupervised Neural Text Simplification
    * *Abstract*: The paper presents a first attempt towards unsupervised neural text simplification that relies only on unlabelled text corpora. The core framework is composed of a shared encoder and a pair of attentional-decoders, crucially assisted by discrimination-based losses and denoising. The framework is trained using unlabelled text collected from en-Wikipedia dump. Our analysis (both quantitative and qualitative involving human evaluators) on public test data shows that the proposed model can perform text-simplification at both lexical and syntactic levels, competitive to existing supervised methods. It also outperforms viable unsupervised baselines. Adding a few labelled pairs helps improve the performance further.

To do so, you will have to use the method `transform` of the vectorization object used to produce the vectors of the original dataset. The difference between the `fit_transform` and `transform` methods is that the first learns the vocabulary from the dataset and then produces the vectorized representations of the text fragments, while the second one uses the vocabulary already learned to produce the representation for new text fragments.

Finally, to compare the vector representations of the queries against the entire dataset, you can use the cosine similarity metric. This metric computes a similarity score ranging from 1 to -1 for each pair of vectors, although negative values are rare when comparing text fragments. The interpretation of the cosine similarity score is as follows:

 * A value close to 1 indicates that the two vector representations are highly similar.
 * A value close to 0 indicates that they are barely related.

You can use the class `sklearn.metrics.pairwise.cosine_similarity` to compute the pairwise cosine similarities across two lists of vectors. A code snippet is provided in the CoLab notebook mentioned above to sort the results obtained and to print the top 3 highest-scored matches.
 
### Step 4: Analysis of the results obtained

Spend some time analysing the results obtained with the two techniques and with the different parameters you modified. Describe your impression about the relation between the papers in the query and those found with the different approaches. You can also comment on how informative or easy to interpret are the scores Add some discussion on your conclusions in the notebook.

## Part 2: Sentence Embeddings with SentenceTransformers

In this section, you will try using  **sentence embeddings** for representing text with dense vectors. You will be using embedding models as provided through the [SentenceTransformers](https://sbert.net/docs/sentence_transformer/usage/usage.html) library. Have a look to the general description to use this library in the link provided. In addition to the general description provided through the link, you can find a description of some interesting use cases, such as:

 * Semantic Textual Similarity,
 * Semantic Search,
 * Retrieve & Re-Rank,
 * Clustering,
 * Paraphrase Mining,
 * Translated Sentence Mining,
 * Image Search.

Have a look to them for a better understanding of the capabilities of this type of embedding models.

Note that, unlike BoW and TF-IDF, sentence embeddings capture the semantic meaning of sentences and documents, making them more powerful for tasks like similarity search. You will use the [**SentenceTransformers**](https://sbert.net) library to generate embeddings for the dataset used in *Part 1* and will again run queries to find similar documents.

### Step 1: Trying a general purpose small monolingual model

We will first try a small model that allows to build semantic embeddings for English: the model `all-MiniLM-L6-v2`. You have a nice example on how to obtain the *n*-top matches for a query search in a collection of embeddings using the cosine similarity at: [https://www.sbert.net/examples/applications/semantic-search/README.html](https://www.sbert.net/examples/applications/semantic-search/README.html).

Note that in this case you will not need to specify any data preprocessing details. The pre-trained model already includes a sub-word tokenizer and takes care of this. Given that these components are trained, changing, for example, capitals, or removing words, could affect negatively to the performance of the model and the quality of the resulting embeddings.

Run the semantic search, and compare the results obtained to those obtained previously, and discuss the differences observed. Also, have a look to the size and the format of the embeddings. Can you draw any conclusions from this inspection?

### Step 2: Comparing other models

The `SentenceTransformers` library provides easy access to a number of [pretrained text embedding models](https://sbert.net/docs/sentence_transformer/pretrained_models.html). Try reproducing the experiment carried out in the previous step using other models. As already mentioned, the `all-MiniLM-L6-v2` model is rather small and general purpose. By *small* it should also be understood that the size of the embeddings is small, but also that the amount of text that can parse is rather small (256 sub-word tokens, according to the documentation in the library website). You can try with larger models, such as `all-distilroberta-v1`, and also with specific purpose: the `SPECTER` model is specifically aimed at detecting the similarity between two scientific papers.

Again, try to draw conclusions from the use of these different models: what is your use experience, what is the impact in the results obtained, etc. Try to describe these conclusions in the notebook.

### Step 3: Moving to a multilingual environment

Finally, you will explore the use of cross-lingual embedding models. The `SentenceTransformers` library provides a few pre-trained models of this type that enable the comparison of embeddings obtained from text fragments in different languages. You can try, for example, the model `distiluse-base-multilingual-cased-v1`. In order to be able to test the performance of these cross-lingual model, a new dataset has been created with the exact same format as the first one, but with papers extracted from the SEPLN journal. This journal allows papers in Spanish and in English. Try following the same steps you followed before with and search for the same query papers, but now on this bilingual dataset. Compare the results obtained when using a monolingual model and a cross-lingual model to obtain the embeddings. Once more, comment on your observations and the conclusions you are able to draw from them.

## Concluding remarks

After this evaluation, you have explored different strategies to obtain vector representations of text. End your notebook with a general overview of the work done and your experience in comparing these models. Try to identify the strengths and the weaknesses of the different methods compared and discuss them.
