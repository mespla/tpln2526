# Architectures for Speech

[!CAUTION]
These materials are temporary and incomplete. If you choose to read them, you do so at your own risk.
    
In this module, we briefly study some neural models used to process speech. The professor for this module is Juan Antonio Pérez Ortiz.

Class materials complement the reading of some chapters from a textbook ("Speech and Language Processing" by Dan Jurafsky and James H. Martin, third edition draft, available online) with annotations made by the professor.

## First session of this module (January 15, 2025)

### Contents to prepare before the session on Jan 15 {#before-speech1}

The activities to complete before this class are:

- Reading and studying the contents of [this page](https://dlsi.ua.es/~japerez/materials/transformers/en/speech/) on speech recognition. As you will see, the page indicates which contents you should read from the book. After a first reading, read the professor's annotations to help you understand the key concepts of the chapter. Then, perform a second reading of the chapter from the book. After finishing this part, read the description of [modern architectures](https://dlsi.ua.es/~japerez/materials/transformers/en/speech/#arquitecturas-modernas-para-el-procesamiento-de-voz) specific to speech recognition. In total, this part should take you about 4 hours 🕒️ of work.
- Then, take this [assessment test](https://forms.gle/woGk9hkmepMVkrg47) on these contents. There are few questions (fewer than in previous tests, in fact), and it will only take a few minutes.

### Contents for the in-person session on Jan 15

In the in-person class (2.5 hours 🕒️ in duration), we will see how to implement a speech classification system in PyTorch. To do this, we will use the `torchaudio` library, which is part of PyTorch. Specifically, we will briefly look at this [guide to audio manipulation with torchaudio](https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html) (focusing only on waveform representation and spectrogram extraction) and the implementation of the [speech classifier](https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html). Both documents include links at the beginning to corresponding Google Colab notebooks. 

These two tutorials will only be covered in class to complement the theoretical contents, but you do not need to study them for the exam.
