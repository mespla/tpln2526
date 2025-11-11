# Architectures for written-text processing

!!! danger
    These materials are temporary and incomplete. If you choose to read them, you do so at your own risk.

In this module, we will study some neural models used to process texts. The professor of this module is Juan Antonio Pérez Ortiz. The module begins with a review of the functioning of logistic regression, which will help us establish the necessary knowledge to understand subsequent models. Next, we study in some detail *skip-grams*, one of the algorithms for obtaining non-contextual word *embeddings*. Then, we review the functioning of *feedforward* neural architectures and study their application to language models. The ultimate goal is to address the study of the most important architecture in current text processing systems: the transformer. Once we have studied these architectures, we will conclude with an analysis of the functioning of pretrained models (foundational models) in general, and language models in particular.

Class materials complement the reading of some chapters from a textbook ("Speech and Language Processing" by Dan Jurafsky and James H. Martin, third edition draft, available online) with annotations made by the professor.

## First session of this module (December 5, 2025)

### Contents to prepare before the session on Dec 5 {#before-text1}

The activities to complete before this class are:

- Reading and studying the contents of [this page](https://dlsi.ua.es/~japerez/materials/transformers/en/regresor/) on logistic regression. As you will see, the page indicates which contents you should read from the book. After a first reading, read the professor's annotations, whose purpose is to help you understand the key concepts of the chapter. Then, perform a second reading of the book's chapter. In total, this part should take you about 4 hours 🕒️ of work.
- Watching and studying the video tutorials in this [official PyTorch playlist](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN). Study at least the first 4 videos (“Introduction to PyTorch”, “Introduction to PyTorch Tensors”, “The Fundamentals of Autograd”, and “Building Models with PyTorch”). In total, this part should take you about 2 hours 🕒️ of work.
- Reading and studying the contents of [this page](https://dlsi.ua.es/~japerez/materials/transformers/en/embeddings/) on embeddings. As you will see, the page indicates which contents you should read from the book. After a first reading, read the professor's annotations to help you understand the key concepts of the chapter. Then, perform a second reading of the chapter from the book. In total, this part should take you about 3 hours 🕒️ of work.
- After completing the previous parts, take this [assessment test](https://forms.gle/V3U9MTHo7c9DNhkc6) on these contents. There are few questions, and it will take you a few minutes.

### Contents for the in-person session on Dec 11

In the in-person class (5 hours 🕒️ long), we will see how to implement a logistic regressor in PyTorch by following the implementations of a binary logistic regressor <a href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/logistic.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> and a multinomial one <a href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/softmax.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> discussed in [this section](https://dlsi.ua.es/~japerez/materials/transformers/en/implementacion/#code-regressor). We will also explore an implementation of the skip-gram algorithm <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/skipgram.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> discussed [here](https://dlsi.ua.es/~japerez/materials/transformers/en/implementacion/#code-skipgrams).

The idea is for you to study and slightly modify the notebooks we are working on. In a later class, a more advanced assignment involving modifying the transformer's code will be presented.

## Second session (December 12, 2025)

### Contents to prepare before the session on Dec 12 {#before-text2}

The activities to complete before this class are:

- Reading and studying the contents of [this page](https://dlsi.ua.es/~japerez/materials/transformers/en/ffw/) on feedforward neural networks and their use as very basic language models. Perform at least two readings complemented with the professor's notes as in the previous point. In total, this part should take you about 2 hours 🕒️ of work.
- Reading and studying the contents of [this page](https://dlsi.ua.es/~japerez/materials/transformers/en/attention/) as an introduction to transformers. As always, perform at least two readings complemented with the professor's notes. In total, this part should take you about 4 hours 🕒️ of work.
- After completing the previous parts, take this [assessment test](https://forms.gle/7KDwRtXcrpxsKjHp7) on these contents. There are few questions, and it will take you a few minutes.
- If you have time left, take the opportunity to review the contents of the first session.

### Contents for the in-person session on Dec 18

In the in-person class (5 hours 🕒️ in duration), we will see how to implement in PyTorch a language model based on a feedforward neural network <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/ffnn.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>, and a transformer <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/transformer.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> following the implementations discussed in [this section](https://www.dlsi.ua.es/~japerez/materials/transformers/implementacion/#code-transformer) and the next two.

The idea is for you to study and slightly modify the notebooks we are working on. We will also present the [assignment](assignment-interpretability.md) on mechanistic interpretability you need to submit for this module of the course.

## Third session (December 19, 2025)

### Contents to prepare before the session on Jan 8 {#before-text3}

The activities to complete before this class are:

- Reading and studying the contents of [this page](https://dlsi.ua.es/~japerez/materials/transformers/en/attention2/) on the complete transformer model (with encoder and decoder) and the possible uses of an architecture that only includes the encoder. As you will see, the page indicates which contents you should read from the book. In particular, you will need to read some sections of the chapter on machine translation and others from the chapter on pretrained models, in addition to standalone sections on *beam search* and subword tokenization. After a first reading, read the professor's annotations to help you understand the key concepts of each section. Then, perform a second reading of the book's contents. In total, this part should take you about 4 hours 🕒️ of work.
- Watching and studying Jesse Mu's lecture titled “[Prompting, Reinforcement Learning from Human Feedback](https://youtu.be/SXpJ9EmG3s4?si=j4B1U2Z-JCyYJwlc)” from Stanford's CS224N course in 2023 about language models based on the transformer's decoder. This should take you about 2 hours 🕒️ of work, as you'll need to take notes so you don't have to rewatch the video when reviewing. Downloading the [slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture11-prompting-rlhf.pdf) and annotating them may be helpful. Regarding the topic discussed between minutes 39 and 46, you can simply focus on the basic ideas, as the reinforcement learning equations are not a priority topic for this course and will be covered in other courses. It's important to review what you've already studied about [transformers](https://dlsi.ua.es/~japerez/materials/transformers/en/attention/) as a language model based on the decoder before watching the video. Don't be confused by encoder-based models also sometimes being called language models. This video discusses the properties of decoder-based models initially trained to predict the next token in a sequence.
- Study the description of [multilingual models](https://dlsi.ua.es/~japerez/materials/transformers/en/attention2/#multilingual-models) in this section of one of the pages on transformers. It's a brief section that will take you about 🕒️ 15 minutes.
- After completing the previous parts, take this [assessment test](https://forms.gle/GRK5SLc3STkup8at9) on these contents. There are few questions, and it will take you a few minutes.
- If you have time left, take the opportunity to review all the contents from previous sessions.

### Contents for the in-person session on Dec 19

In the in-person class (5 hours 🕒️ in duration), we will see how to implement on top of our transformer architecture code both a language model based on a decoder <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/lmgpt.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> and a named entity recognition model <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/nerbert.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> based on an encoder.

We will take the opportunity to review some aspects of the code from previous sessions and relate theoretical aspects with practical ones.

## Fourth session (January 14, 2026)

This fourth session is actually the first and only session on the topic of speech. See the page on [speech](speech.md) to view the contents prior to this session.
