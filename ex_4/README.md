## **Task 1**
> *When creating embeddings for two different languages (e.g., English and Polish), the vectors for “women” and “kobieta” may end up unrelated. Propose a method of computing embeddings such that words that are translations of each other receive similar vectors. You may use corpora in both languages and possibly other data.*

### 1.1. The Problem: Bilingual (or Multilingual) Word Embeddings

- If we simply train two separate monolingual word-embedding models (e.g., Word2Vec or GloVe) on English and Polish, the resulting vector spaces will be in completely different coordinate systems. Even semantically identical words like *“women”* (English) and *“kobiety”* (Polish, plural) or *“kobieta”* (singular) will not align.

### 1.2. Proposed Solutions

1. **Supervised “Dictionary-based” Alignment (Mikolov et al.)**  
   - Train monolingual embeddings separately: \(\mathbf{X}\) for English, \(\mathbf{Y}\) for Polish.  
   - Collect a small bilingual dictionary or a set of known word pairs \((w_i^{\text{en}}, w_i^{\text{pl}})\).  
   - Solve for a linear transformation \(W\) that minimizes \(\sum_i \|W \mathbf{x}_i - \mathbf{y}_i\|^2\), mapping English embeddings into the Polish space.  
   - Then “women” and “kobieta” should align more closely.

2. **Shared Training with Parallel (or Comparable) Corpora**  
   - If we have parallel data (e.g., aligned sentences in English & Polish), we can train a single bilingual embedding model that enforces that translation-equivalent words share (or are close in) vector space. For instance, we can use **joint skip-gram** approaches or **fastText** approaches that link the embeddings across languages.  

3. **Use a Transformer-based approach** (like **Multilingual BERT** or **LaBSE**):
   - These are already trained to produce similar vectors for parallel sentences or words. But at the word level specifically, you can enforce subword alignment if you have a bilingual corpus.  

4. **Post-Processing**:
   - Use unsupervised methods such as **MUSE** (Conneau et al.) that align monolingual embeddings without a large bilingual dictionary, relying on shared distributional structure and a small seed or anchor points.

In short, to get “women” ~ “kobieta,” you must explicitly include either (1) a bilingual dictionary or (2) parallel corpora or (3) some cross-lingual alignment technique so that shared concepts align in the same vector space.

---

## **Task 2**
> *Propose a method for generating text (generally sequences/matrices of tokens of a fixed size) using a BERT-like model. You can invent a procedure yourself or reference a publication.*

### 2.1. The Challenge: BERT as a Masked Model

- **BERT** is typically not an autoregressive model; it’s a **masked** model. So generating text left-to-right isn’t straightforward.
- However, there are some known approaches that repurpose BERT for generation or use a **“fill-in-the-mask”** strategy.

### 2.2. Possible Procedures

1. **Iterative Masked Filling** (e.g., MaskGIT, or M. Ghazvininejad “Mask-Predict”):
   - Initialize your sequence (maybe random tokens, or partial seed).  
   - Repeatedly select some positions to mask, use BERT to fill them in, and update. Possibly mask the most uncertain positions.  
   - This process continues until convergence or a fixed number of steps, producing a coherent sequence.

2. **Using BERT as a “Plug-and-Play” LM**:
   - Combine BERT with a small autoregressive head or a two-stage approach:
     1. Sample a candidate prefix with some other model or partially.  
     2. Use BERT to fill missing parts or refine.  

3. **Publications**:
   - There are methods like **[Gibbs sampling with BERT]** (not standard, but sometimes referenced) or **Infi-Mask**. Also, works on **“Bidirectional and Auto-Regressive Transformers”** (BART) show how a masked approach can be used for generation by partially denoising.

Hence a straightforward method is:
- **Set all tokens to [MASK].**  
- In multiple rounds, predict tokens for each mask, sampling from the BERT distribution, then re-mask some subset, etc., until the text has the desired length and content. This yields a “BERT-based generation” though it’s less direct than GPT’s left-to-right approach.

---

## **Task 3**
> *Assume we have a large n-gram model \(M_{ng}\) on tokens. Propose two scenarios of training a GPT-like model that uses \(M_{ng}\). The model \(M_{ng}\) may also be used later in inference for text generation.*

We have a powerful **n-gram** model pre-trained on a big corpus. We want to integrate it into a GPT-like training scenario.

### 3.1. Scenario A: “Pre-Conditioning” or “Logits Merging”

- During GPT training, for each next token we combine the GPT’s predicted distribution \(p_{\text{GPT}}(w_t \mid w_{<t})\) with \(p_{ng}(w_t \mid w_{<t})\) from the n-gram model.
- Possibly we do a **log-linear interpolation**:
  \[
  p(w_t \mid w_{<t}) \;=\; \alpha \, p_{\text{GPT}}(w_t \mid w_{<t}) \;+\;(1-\alpha) \, p_{ng}(w_t \mid w_{<t}).
  \]
- Then we backprop through GPT’s parameters, using the combined distribution as a teacher or as a guide. This might help GPT learn from n-gram’s strengths in local patterns.

### 3.2. Scenario B: “Knowledge Distillation” from \(M_{ng}\)

- We treat the n-gram model as a teacher: for each context, we can produce “soft labels” from the n-gram distribution.  
- GPT is trained to match or approximate these distributions. This is a form of **knowledge distillation**.  
- Alternatively, we might generate “pseudo-text” from the n-gram model, then train GPT on it (though that can be less beneficial if the n-gram text is less coherent globally).

### 3.3. Later Use in Inference

- We can continue to do “ensemble inference” by mixing GPT’s next-token probabilities with the n-gram model’s probabilities. This can reduce GPT hallucination or out-of-distribution tokens that the n-gram model sees as improbable.

---

## **Task 4**
> *We return to the “careless programmer” scenario. This time, they trained a large BERT model but forgot to include positional embeddings. Could that model still be useful? Is the large vocabulary helpful or harmful?*

### 4.1. BERT Without Positional Embeddings

- Normally, BERT uses absolute or relative position embeddings so the Transformer can differentiate token order. Without them, a standard Transformer “sees” the input as a set of tokens in no particular order—**it’s basically a bag-of-tokens** with attention but no sense of sequence position.
- Such a model could still learn **co-occurrence** patterns or “which tokens often appear together,” but not **word order**.  

### 4.2. Possible Use Cases

- It might be somewhat useful for tasks that are mostly about **set membership** or “bag-of-word-like” features, e.g., some classification tasks or topic detection. For example, if you just want to detect topic or sentiment, the exact order of words may be less crucial.
- However, for tasks that rely heavily on syntax or word order, it’s likely worthless.  

### 4.3. Large Vocabulary: Pro or Con?

- On one hand, having a big vocabulary might help if the model picks up on specific tokens. On the other hand, if you can’t handle position, the big vocabulary might make it even more confusing.  
- Possibly the large vocabulary “helps” because it captures more fine-grained tokens, but typically the inability to represent word order is a major limitation. So overall, it’s probably a negative to train such a big model without positional embeddings: you wasted a lot of capacity that can’t learn order.

Thus you could argue the model isn’t entirely useless, but is **severely limited**. The large vocabulary might not really rescue it, though it could be a small silver lining for tasks that are purely bag-of-words style.

---

## **Task 5**
> *Suppose management is okay with your explanation that “positionless BERT” can do something. Now they want to train another instance for a different language. Standard Masked Language Modeling has a particular property that is especially problematic if we have no positional embeddings. What is that property?*

**Hint**: The Polish text references a rot13 message: “Cbzlśy b yvpmovr gbxraój ZNFX” (unclear if there's a direct decode). The key idea is:

- In standard **masked language modeling**, the model sees the input sentence with random tokens replaced by [MASK].  
- The big problem: **the model can “cheat” by ignoring positions** if it sees all tokens at once. Without positions, it basically sees a *multiset* of tokens plus a [MASK].
- If you can’t track position, you can’t learn which word in the text was masked. Possibly the model “just knows” which word is missing based on the presence/absence of other tokens, *but not the local context or order*.

Hence the property is:

- **The random masking is predicated on having a sense of which token is missing at which position** so that the model can place the correct word *in that specific context*.
- Without position embeddings, the model might degrade to a “fill in the missing item from the set.” Possibly it can exploit the “bag-of-words minus one” trick to guess the missing token. This yields weird or limited training signals because the model can’t learn real syntactic or sequential constraints.

---

## **Task 6**
> *Naive Bayes Classifier (NBC) is often used in text classification. Briefly explain how NBC works, and list at least two reasons why fine-tuning BERT might yield significantly better text-classification performance.*

### 6.1. How NBC Works (Naive Bayes Classifier)

1. **Assumption**: Features (like the presence of certain words) are conditionally independent given the class label.  
2. **Training**: We estimate \(P(\text{word}=w \mid \text{class}=c)\) and \(P(\text{class}=c)\) from data.  
3. **Inference**: For a new document, we compute
   \[
   P(\text{class}=c \mid \text{doc}) \propto P(\text{class}=c) \prod_{w \in \text{doc}} P(\text{word}=w \mid c).
   \]
4. That is “naive” because we multiply these conditional probabilities ignoring potential correlations among words.

### 6.2. Why Fine-Tuning BERT Can Be Much Better

1. **Contextual Understanding**: BERT captures **semantic and syntactic** context. Words are not treated as independent features. NBC’s independence assumption is simplistic.  
2. **Representation Power**: BERT’s deep transformer layers learn nuanced patterns, subword relations, even domain-specific concepts. NBC is just a simple log-prob summation.  
3. **Transfer Learning**: BERT is pretrained on massive corpora. Fine-tuning incorporates huge prior knowledge about language, which a typical NBC approach lacks.  

Hence BERT typically outperforms naive Bayes by a large margin.

---

## **Task 7**
> *Design an experiment to see if transformers can sort a sequence of natural numbers. What is your intuition about the result?*

### 7.1. The Experiment

1. We create a dataset of random sequences of integers (e.g., length 10, each up to 100).  
2. The target output is the **sorted** sequence of those integers (e.g., in ascending order).  
3. We train a transformer (with positional embeddings) in a seq2seq style (like a small encoder-decoder or possibly a single transformer that reads the input and outputs the sorted sequence token by token).  
4. Evaluate if the model can generalize to unseen sequences.

### 7.2. Intuition

- **If** the model is given enough training data and capacity, it can learn the mapping from unsorted to sorted sequences, at least up to some length (like length 10 or 20).  
- Whether it generalizes to bigger lengths might be questionable. Transformers do not inherently do classical algorithmic steps (like bubble sort or merge sort) but they can approximate them if trained carefully.
- Common experience: transformers can memorize sorting for smaller lengths, but might fail systematically for much longer sequences if not carefully designed or if they have no mechanism to handle large inputs. So the likely result is: it can do well on short sequences but might not scale algorithmically or handle large numbers.

---

## **Task 8**
> *Design an experiment to see if transformers can compute the XOR value of a sequence of bits. What is your intuition about the result?*

### 8.1. The Experiment

- Input: A sequence of bits (like `[1, 0, 1, 1, 0]`).  
- Output: The single bit that is the XOR of all input bits (in this case, 1 ^ 0 ^ 1 ^ 1 ^ 0 = 1).  
- We again use a small seq2seq approach or a classification approach that reads the entire sequence and outputs a single token `[0]` or `[1]`.

### 8.2. Intuition

- XOR is a simple function: if the number of 1’s is odd, result = 1; else 0.  
- A transformer with positional embeddings can, in principle, attend to every token. It can learn to count the 1’s mod 2.  
- For shorter sequences, it’s easy to learn. For very large sequences, it might be more tricky, but it’s still simpler than sorting. The model might do quite well with enough training, though it’s “unstructured.”  
- So we expect good performance if the training covers enough sequence lengths. This is easier than sorting, so the intuition is that the model can do it fairly reliably.

---

## **Task 9** (2 points)
> *We want to model Python code with a GPT-like autoregressive LM. We have some questions:*
> 1. *Is the standard tokenization optimal, or should we consider modifications?*  
> 2. *How to handle code indentation?*  
> 3. *What is PEP-8? Could it be useful here?*  
> 4. *Why rename variables/functions/classes while preserving code semantics?*  
> 5. *How can NLP methods help with that renaming?*  
> 6. *How can static code analysis help?*

### 9.1. (1) Standard Tokenization vs. Something Custom

- **Standard GPT Byte-Pair Encoding** or a “word-level” approach might not be optimal for code. Code often has structured tokens: `if`, `def`, `(`, `)`, `:`, indentation, variable names, etc.
- **Potential Improvement**: A dedicated code tokenizer that splits punctuation carefully, recognizes keywords, breaks up CamelCase or snake_case identifiers. This can help the LM see tokens more clearly (like `def`, `my_function`, `(`, `):` instead of lumps).

### 9.2. (2) Handling Indentation

- In Python, indentation is syntactically meaningful. We must represent it in tokens (e.g., `<INDENT>` and `<DEDENT>` or tokens that store the indentation level).
- Another approach is to convert each indentation to a symbol like `▁▁` repeated or a single `<TAB>` token. But it’s crucial to do so consistently so the model can learn code blocks.

### 9.3. (3) PEP-8

- **PEP-8** is the official style guide for Python code: it dictates recommended indentation, naming conventions, line lengths, etc.
- If we standardize all training data to PEP-8 style, the model sees more consistent examples, making learning easier. During generation, the model might produce nicely formatted code. So yes, using PEP-8 style data can help consistency.

### 9.4. (4) Renaming Variables While Preserving Semantics

- We might want to anonymize or randomize variable names so the model doesn’t overfit to specific naming. This can help the model generalize or unify the code representation.
- Also, in data augmentation, renaming (e.g., `x` → `var1`, `my_list` → `lst`) yields new code variants but the same logic, increasing training data diversity.

### 9.5. (5) NLP Methods for Renaming

1. **Synonym-like approach**: If you have an embedding space for identifiers, you might find “similar” variable names or do transformations.  
2. **Contextual embedding** (transformers) to see how an identifier is used and propose consistent renamings across the code. This ensures all references to the same variable get the same new name.  

### 9.6. (6) Static Code Analysis

- **Static analysis** can identify variable scopes, references, or unused variables. This helps us rename variables systematically across an entire code block, avoiding collisions or partial renames that break code.  
- Also can parse abstract syntax trees (AST). The AST approach clarifies code structure, letting us do a more robust tokenization and transformation.

---

## **Task 10** (⋆)
> *Which of the previous tasks could be turned into a project/thesis if expanded? What modifications would be needed?*

### 10.1. Potential Candidates

- **Task 1** (Multilingual embeddings): A full thesis could revolve around cross-lingual embedding alignment with new methods or expanded corpora.
- **Task 3** (Combining n-gram with GPT): Could become a project on “Hybrid Language Models,” exploring best ways to integrate an n-gram.  
- **Task 9** (Modeling Python code): Definitely suitable for a more extensive project, e.g., “Building a code completion model with advanced tokenization, analyzing variable renaming, etc.”

### 10.2. Required Modifications

- **Deep experiments**: collecting large corpora, systematically comparing results.  
- **Implementation**: exploring various architectural changes, ablations, metrics (e.g., BLEU for code generation or pass rates on test sets).  
- **Scalability**: dealing with big data, adding or removing features.

For example, if we pick **Task 9** for a thesis: we’d likely define a big dataset of Python code, systematically test different tokenization strategies, incorporate PEP-8 normalization, measure code perplexity, and test code generation or code completion performance. Possibly integrate static analysis to do safe variable renaming as data augmentation.

---

## **Task 11** (2 points, ⋆)
> *Propose a new task (not on this list) that could be suitable for a project. Summarize it clearly so that it’s well-defined. You’d post it on the SKOS forum, etc.*

Because this instruction says “propose a task and post it,” we can give **an example** below:

**Example Proposed Task**: *“Multimodal Summarization of Meeting Transcripts with Slide Images.”*

1. **Goal**: Summarize a meeting transcript that also includes references to slides or images. 
2. **Data**: We could create a small corpus of (transcript + slide images + short summary). 
3. **Approach**: Use a transformer that can handle text + image features (e.g., a vision–language model) to produce a textual summary focusing on key points. 
4. **Evaluation**: Compare the model’s summary to ground truth using ROUGE, etc.

We’d store the data, define the tokenization or embeddings for images, define the training process, measure results, and see if it generalizes to new meetings. This is *not* part of the current tasks, but it’s an example of a suitable project that merges language modeling with a new domain.