# Assignment on Mechanistic Interpretability of Transformers

**Mechanistic interpretability** in the context of artificial intelligence seeks to provide a motivated explanation of how machine learning models function. It is a crucial approach to building trust in systems and inducing certain behaviors in them. Within the field of mechanistic interpretability, there are many techniques that can be applied to transformers. Here, we will focus on [activation patching][patching].

[patching]: https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=qeWBvs-R-taFfcCq-S_hgMqx

Activation patching *intervenes* in a specific model activation by replacing a *corrupted* activation with a *clean* activation. The effect of this change on the model output is then measured. This helps us identify which activations are important for the model's output and locate possible causes of prediction errors.

In this practice, you will write code to run the smallest version of GPT2 (use the string `gpt2` in the code) with two different inputs: two texts that differ by only one token. The idea is that when the corrupted input is fed to the model, we will intervene in the embedding after a certain layer (one at a time) and patch it with the corresponding embedding from the clean run. Then we will measure how much the prediction of the next token changes compared to the clean run. If the change is significant, we can be confident that the altered activation is important for the prediction. This patching process will be performed for each layer of the model and for each token in the input. With all this information, we will generate a heatmap and draw conclusions. For reasons you will soon understand, both texts must have the same number of tokens.

## Analysis Example

Here is an example to better understand the task. Consider the following input text: "Michelle Jones was a top-notch student. Michelle". If we feed it to GPT2 and study the model's emitted probability for the token following the second appearance of Michelle, we obtain the following (only the 20 most probable tokens are shown): 

| Position | Token index | Token  | Probability |
| -------- | ----------- | ------ | ----------- |
| 1        | 373         | was    | 0.1634      |
| 2        | 5437        | Jones  | 0.1396      |
| 3        | 338         | 's     | 0.0806      |
| 4        | 550         | had    | 0.0491      |
| 5        | 318         | is     | 0.0229      |
| 6        | 290         | and    | 0.0227      |
| 7        | 11          | ,      | 0.0222      |
| 8        | 531         | said   | 0.0134      |
| 9        | 468         | has    | 0.0120      |
| 10       | 635         | also   | 0.0117      |
| 11       | 1625        | came   | 0.0091      |
| 12       | 1297        | told   | 0.0084      |
| 13       | 1422        | didn   | 0.0070      |
| 14       | 2993        | knew   | 0.0067      |
| 15       | 1816        | went   | 0.0061      |
| 16       | 561         | would  | 0.0061      |
| 17       | 3111        | worked | 0.0055      |
| 18       | 750         | did    | 0.0054      |
| 19       | 2486        | Obama  | 0.0053      |
| 20       | 2492        | wasn   | 0.0050      |

As expected, the token "Jones" has a notably high probability. Now consider the corrupted input "Michelle Smith was a top-notch student. Michelle". If we provide this input to GPT2, we expect the probability of "Jones" as the next token to be much lower than before, while the probability of "Smith" will be much higher, which (you can verify) indeed happens. However, we want to go further and understand which embeddings most influence this difference. Given that both inputs have 11 tokens (we will explain how to verify this later) and the transformer in the small GPT2 model has 12 layers, if we focus on the embeddings obtained at the output of each layer, we can patch 11×12 = 132 different embeddings. Therefore, we will calculate 132 times the difference between the logit of "Smith" and the logit of "Jones" in the output of the last token of the input ("Michelle") in the corrupted model. Note that we could also calculate the differences after applying the softmax function, but we will not do so here.

A heatmap representation of the result is shown below:

![Logit difference heatmap](images/mechanistic-michelle.png)

In such a graph, due to the attention mask and the arrangement of the layers, information flows from left to right and top to bottom. You can see that intervening in the first column has no effect on the prediction of the next token, which makes sense, as the embeddings patched have exactly the same values in the clean and corrupted models, given the same preceding context. There also seem to be no changes when patching embeddings from the third to the penultimate column. However, note how intervening in the embeddings of many layers of the second token shifts the prediction towards "Jones" (the color darkens as the logit difference between "Smith" and "Jones" becomes negative because "Jones" has a higher logit). Modifying the embeddings of the last layers of the second token has much smaller effects, as the embedding barely influences the sequence's future. At the last position ("Michelle"), the embeddings of the final layers seem to anticipate the token to predict.

Some additional corrupted texts that may be interesting to explore are, for example, "Jessica Jones was a top-notch student. Michelle" or "Michelle Smith was a top-notch student. Jessica".

## Submission

In this assignment, your task is to write the code to generate graphs and probabilities like those above, propose your own clean and corrupted texts (try to be creative and avoid studying very similar texts or phenomena), perform a similar analysis, and write a report in a document of 1000–1500 words (both limits are strict). In this document, you should present and explain the implemented code, along with your approach, results, and relevant conclusions. Original ideas and additional experiments are welcome. The document in PDF format must be submitted via the UACloud *evaluation tools* **before 23:55 on Sunday, January 18, 2026** (Alicante time). The assignment must be done in pairs. Remember to include both authors' names in the document.

## Base Code 

The base code we will use is from the GPT2 implementation found in Andrej Karpathy's [minGPT][mingpt] repository. His code inspired our transformer model code, so it should not be difficult to understand. You can clone the repository on your computer or work in a Google Colab notebook as described below.

Due to changes in external elements, the current code does not work as is. To make it work, you need to change line 200 of the `mingpt/model.py` file from:

```python
assert len(keys) == len(sd)
```

to:

```python
assert len(keys) == len([k for k in sd if not k.endswith(".attn.bias")])
```

[mingpt]: https://github.com/karpathy/minGPT

## Tokenization

The GPT2 model uses a BPE-based tokenizer that segments the input text into words or smaller units depending on their frequency. The minGPT code allows downloading this tokenizer and using it to segment texts. The following code shows how to tokenize a text to obtain its indices and vice versa.

```python
from mingpt.bpe import BPETokenizer

input = "Michelle Jones was a top-notch student. Michelle"
print("Input:", input)
bpe = BPETokenizer()
# bpe() gets a string and returns a 2D batch tensor 
# of indices with shape (1, input_length)
tokens = bpe(input)[0]
print("Tokenized input:", tokens)
input_length = tokens.shape[-1]
print("Number of input tokens:", input_length)
# bpe.decode gets a 1D tensor (list of indices) and returns a string
print("Detokenized input from indices:", bpe.decode(tokens))  
tokens_str = [bpe.decode(torch.tensor([token])) for token in tokens]
print("Detokenized input as strings: " + '/'.join(tokens_str))
```

## Implementation Details

The following are some implementation details that may be useful but are not required to follow.

To write code that allows activation patching, you will need to focus on the files `mingpt/model.py` and `generate.ipynb`. If you are working locally without using a notebook (recommended), copy the code from `generate.ipynb` into a `generate.py` file that you can execute from the command line.

You can also work directly in a Google Colab session. Here is a [project][proyectocolab] (access with your `gcloud.ua.es` account) with instructions on how to use it for development. However, developing locally is much more convenient (among other things, you can work with a better text editor than Colab’s and also debug). Even if you don’t have a GPU, the code runs fine on a CPU and only takes a few seconds longer than on a GPU, as it only works with one text and a not excessively large model.

Add to the transformer’s `forward` function code that allows saving (depending on the value of a boolean *flag* passed as a parameter) the activations of each layer and each position into an instance variable. Remember to make a deep copy of the embeddings rather than only saving a reference that could be overwritten later; for this, check PyTorch’s `.detach().clone()` sequence of calls. Also, add code that allows (again based on a boolean parameter) patching the embedding of a specific layer and position.

Additionally, modify the `forward` function to store the logits of the last token, which contain the information we are interested in regarding the prediction of the next token. You can save this information in an attribute that can later be accessed from outside the class. Note that you only need the vector corresponding to the last token.

Add code to the `generate.py` file to tokenize the clean text, pass it through the model via the `generate` function (asking the model to save the intermediate embeddings), and display the most probable continuations based on the logits of the last token. Keep in mind that if you want to know the probability of a continuation like the token "Jones", for example, you need to find the index of that token in the vocabulary by prefixing it with a space (`index = bpe(' Jones')`). This is because the BPE tokenizer handles tokens at the beginning of a sequence differently from those in the middle. Once you have the token’s index, you can access the corresponding position in the logits vector and obtain the unnormalized probability of it being the continuation.

Then, you can work with the corrupted text. Include a nested loop that iterates over all layers and all positions and calls `generate` each time, passing the layer and position where the intervention should be performed. At each step, compute the appropriate logit difference and store it in a difference matrix.

Finally, use the `matshow` function from `matplotlib` to visualize the difference matrix.

[proyectocolab]: https://colab.research.google.com/drive/1dq2EClvIbEtoEnHWoAXZQTArJDHivQly?usp=sharing

## A More Informal Explanation

The following informal explanation may help you better understand the objective of the assignment.

For simplicity, consider the sequence "a b c" and its corrupted version "d e f." In general, there will be many more tokens in common, but this makes the following discussion clearer. Assume the transformer-based neural model has 5 attention layers. We want to study which embeddings are important for predicting that the token "X" follows these sequences.

First, modify the transformer’s `forward` function (in the `GPT` class) to store (e.g., in a list of lists of tensors) the 3×5=15 embeddings generated at the output of each layer when processing the sequence "a b c". The assignment provides some details because you cannot simply store a reference to the tensors, as they will be modified the next time `forward` is called. Instead, you need to clone the tensors (a "defensive copy"). This will leave you with the 15 tensors (embeddings) for the clean sequence.

Also, save the logits after the last layer. In particular, you only need the logits for the final position (i.e., those corresponding to the token "c"), which provide a measure of the probability of the next token, i.e., the token following "c". Remember that these logits are not actual probabilities (they are values like -11.1, -0.5, 0.78, or 2.32323) because the softmax function has not been applied, but working with them is more convenient due to their broader range. However, the study could equally be conducted using strict probabilities. In reality, you don’t even need to save all the logits, only the scalar corresponding to token "X," as it is the only one you will use later.

Now feed the model the corrupted version "d e f", ensuring that it does not overwrite the stored embeddings from the clean sequence. The corrupted sequence must have the same number of tokens as the clean one for the following discussion to make sense. The idea is to modify only one of the 15 embeddings generated while processing the corrupted sequence. For example, if we focus on the embedding of the first token ("d") after the first layer, the `forward` function should operate "almost" normally. However, when obtaining the output of the first layer and before passing it as input to the second layer, the embedding corresponding to the first word (and only that) should be modified and replaced with the corresponding embedding (from the same layer and position) saved for the clean sequence (in this case, the embedding saved after the first layer for the token "a"). This ensures that the second layer receives as input the embedding generated for "a" instead of "d".

After intervening in the embedding of position 1 after layer 1, the rest of the model operates without any "hiccups." As before, examine the logits for predicting the token following the last token of the corrupted sequence (i.e., "f"). Focus on the value of the logit for the prediction of token "X". The difference between this value and the one saved for the clean sequence provides insight into the relevance of the embedding at layer 1, position 1, for predicting token "X". The assignment shows that some embeddings are much more relevant than others, and you need to conduct a similar study with different sequences.

If you repeat the above operation for the other 14 embeddings (calling the `forward` function 14 more times), you will end up with 15 logit differences (15 scalar values) that can be represented in a 3×5 heatmap as seen above.

Finally, note that this discussion simplifies the task described earlier in the assignment. There, it was proposed to calculate the difference between the logit of "Smith" and the logit of "Jones" in the output of the last token in the corrupted model. This approach provides slightly more information than the difference explained here, which is the difference between the prediction of a single token ("Jones") in the clean and corrupted sequences, rather than two tokens in the corrupted sequence. Either approach is valid for arriving at the conclusions we are interested in: that in the corrupted sequence, the logit of "Jones" becomes much lower except for certain interventions. If you want your heatmap to match the one in the assignment, follow the approach based on the two tokens "Jones" and "Smith".

## Further Knowledge

The above is just one of many analyses proposed within mechanistic interpretability. For this assignment, you are not expected to go beyond this. However, if you are interested in learning about a couple more analyses, you can check out [this tutorial][lines50]. Note that although the tutorial uses a library for activation patching, in this assignment, you are not allowed to use any library for this and must implement it directly in the minGPT code. A much more detailed review of mechanistic interpretability can be found in [this work][nanda] by Neel Nanda.

[lines50]: https://www.lesswrong.com/posts/hnzHrdqn3nrjveayv/how-to-transformer-mechanistic-interpretability-in-50-lines
[nanda]: https://www.neelnanda.io/mechanistic-interpretability/glossary
