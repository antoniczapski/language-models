## **Task 1**

**Recall what perplexity is.** We have a language of “words” that are sequences of digits, each word consisting of *blocks* of length \(k\). In each block, the digit is repeated \(k\) times (e.g., for \(k=10\), a block might be `9999999999`). Furthermore, any block can be followed by a block of *any digit* (uniformly chosen among 10 digits).

We want to compute the perplexity (for \(k=10\)) under three models:

1. **Unigram model**  
2. **Bigram model**  
3. **Optimal \(n\)-gram model** (and what is \(n\) here?)

### **1.1. Text Distribution**

- The text is a very long sequence of digits, which can be segmented into blocks of length \(k=10\).  
- Within each block, all digits are identical.  
- For each new block, the digit is chosen uniformly from \(\{0,1,\ldots,9\}\).  

Hence the sequence of digits has strong internal correlations: within a block of length \(k=10\), the digit never changes; when moving to the next block, the digit may (uniformly) become anything.

### **1.2. Unigram Model**

A **unigram** model assumes each digit is drawn i.i.d. from some distribution \(p(\text{digit})\). Empirically, across a very large text:

- Each of the 10 digits appears equally often overall (because each chosen digit appears 10 times in its block, and blocks are chosen uniformly among the digits).  
- So the model estimate is \(p(\text{digit}=d)=1/10\).  

The **cross-entropy** of the text under this unigram model is:
\[
H_\text{unigram} = -\sum_{d=0}^{9} \bigl(\tfrac{1}{10}\bigr) \log_2\bigl(\tfrac{1}{10}\bigr) \;=\; \log_2(10)\,\approx\,3.3219\,\text{bits}.
\]

The **perplexity** is \(2^{H_\text{unigram}} = 2^{\log_2(10)}=10.\)

\[
\boxed{\text{Unigram perplexity} = 10.}
\]

### **1.3. Bigram Model**

A **bigram** model conditions on the *previous* digit. In the true text:

- With probability \(9/10\), we are still “within” the same 10-digit block, so the next digit is exactly the same as the current one.  
- With probability \(1/10\), we transition to a new block, where the next digit is uniformly among 10 possible digits.

Hence, given a current digit \(d\),
\[
P(\text{next}=d \mid \text{current}=d) \;=\; 0.9 + 0.1 \times 0.1 \;=\; 0.91,
\]
\[
P(\text{next}=e \mid \text{current}=d,\; e\neq d) \;=\; 0.1 \times 0.1 = 0.01.
\]
One can show the **cross-entropy** is:
\[
H_\text{bigram} 
= -\bigl[0.91 \log_2(0.91) + 0.09 \log_2(0.01)\bigr]
\approx 0.723\,\text{bits}.
\]
Thus the **perplexity** is about \(2^{0.723}\approx 1.64.\)

\[
\boxed{\text{Bigram perplexity} \approx 1.64.}
\]

### **1.4. Optimal \(n\)-gram Model**

If we allow an **\(n\)-gram** model with \(n\ge k\), the model can “know” exactly how many times the current digit has repeated so far. Concretely, with \(k=10\):

- If we are *not* at the boundary of a block, the next digit is 100% the same as the current one (0 bits of surprise).  
- If we are precisely at the 10th repeat, the next digit is uniform among 10 digits (\(\log_2(10)\approx 3.32\) bits).

In a block of length 10, we have 9 “within-block” transitions (0 bits each) and 1 “end-of-block” transition (\(\log_2(10)\) bits). Overall:

\[
H_\text{optimal} = 
\tfrac{1}{10}\,\log_2(10)\,\approx\,0.332\,\text{bits},
\]
\[
\text{Perplexity} \;=\; 2^{\,0.332} \;\approx\; 1.25.
\]

And **\(n=10\)** is sufficient to distinguish “I am at the boundary” from “I am in the middle.”

\[
\boxed{\text{Optimal perplexity} \approx 1.25 \quad\text{for }n=10.
}
\]

---

## **Task 2**

A company proposes using **perplexity** to detect machine-generated text—claiming “generated texts have lower perplexity than natural texts, so if perplexity is below some threshold, it’s likely unnatural.”

1. **Why might it work?**  
   - Some language models (especially if sampling with low “temperature”) produce **more predictable** text, i.e. certain token sequences are high-probability under a standard language model. This can yield a lower perplexity than more “surprising” human writing.  
   - In some real data, LLM output can be repetitively fluent and thus not so “diverse.”

2. **Why is it not ideal?**  
   - **Countermeasures**: A user can increase randomness (temperature) or use nucleus/top-\(p\) sampling, which can **raise** perplexity. The generated text can end up with perplexity in the “human” range.  
   - **False positives**: Some human texts (technical or formulaic) can be **low perplexity**.  
   - **Dynamic models**: As large language models get more sophisticated, they can approach perplexities close to real data.

3. **How can a user control perplexity?**  
   - **Sampling temperature**: By increasing temperature, the model picks less probable tokens more often, thus **raising** perplexity.  
   - **Top-k or top-p sampling**: Tuning these cutoffs can also affect perplexity.  
   - In short, a user can intentionally produce text with perplexity closer to or even higher than typical natural text, circumventing the detector.

Hence, while perplexity-based detection can sometimes flag very “predictable” text, it’s not a robust or foolproof solution.

---

## **Task 3**

We’re generating text using **2-grams or 3-grams** (bigrams/trigrams), but we must satisfy additional constraints. The “natural method” (generate left-to-right, check if the condition is met, and if not, re-generate) is often **inefficient**. We want better algorithms for:

1. **(a)** Generate text of length \(M\) where the word at position \(k\) is predefined.  
2. **(b)** Generate text of length \(M\) where **even** positions are predefined (we only get to choose the words at odd positions).  
3. **(c)** Generate a short text with a **specified first** and **last** word.  
4. **(d)** For \([s_1, s_2, \ldots, s_n]\), generate text of length \(n\) such that the \(i\)-th word ends with the suffix \(s_i\).

### Why is the natural method not effective?

- If you fix a word in the middle (or multiple positions), or a specific start and end, simply generating from left to right can often lead to dead ends:
  - You might get stuck with no valid bigram or trigram that matches the constraint late in the generation.
  - You might waste many attempts as you realize a conflict only after building most of the sequence.

### Better Approaches (Sketches)

1. **Dynamic Programming or Backtracking**  
   - For (a), if we must have a certain word \(w_k\) at position \(k\), we can **split** the generation: generate from left to \(k\) (in a way that ends with \(w_k\)) and generate from \(k\) to the end, possibly meeting in the middle.  
   - We can keep track of possible (bigram/trigram) states and only keep feasible partial paths.

2. **Constraint-Satisfying Search**  
   - For (b), if all even positions are fixed, we only fill odd positions. We can do a forward-backward approach or a “lattice” approach for bigrams: each position has a set of possible words (in your case, the even positions have exactly 1 possibility), and you find a path through the lattice that matches transitions.  
   - This significantly reduces random failures vs. naive re-generation.

3. **A* or BFS for Short Text**  
   - For (c), with a specified **first** and **last** word, you can do a BFS or A* search from “start word” to “end word” in a bigram graph (where edges connect words that can follow each other). If the text must have length \(M\), you can track partial paths of length up to \(M\).

4. **Suffix Matching**  
   - For (d), each \(i\)-th word must end in \(s_i\). Pre-filter the vocabulary to find all words that end with \(s_i\). Then link them up in a bigram/trigram graph. Use a path-finding method (DP or BFS) across positions 1..\(n\).  
   - This is much more direct than naive left-to-right generation that might produce invalid suffixes.

All these methods **avoid** the repeated “generate–fail–retry” approach by **searching or constraining** the space up front.

---

## **Task 4**

You trained a language model on texts that were all read **backward** by mistake (e.g., `['I',' like',' ice',' creams']` became `['creams',' ice',' like','I']`). Now that you realize the error, you started a correct re-training but need to justify the cost of the **first** (incorrect) training.

**Possible Arguments**:

1. **Reversed text = Right-to-left model**  
   - You effectively trained a model that predicts tokens from right to left. This can still be valuable: generating reversed text might help in certain tasks (reverse completion, or as a partial initialization for a bidirectional system).  
   - For some languages (like Hebrew/Arabic) or tasks, a right-to-left perspective can be interesting.  

2. **Transfer Learning**  
   - Some learned parameters (like embeddings of tokens) might still be partially reusable. Even if the text was reversed, the model has seen the same *vocabulary*. Token embeddings can be partially transferred or fine-tuned in the correct model.  

3. **Unique Data Insights**  
   - The experiment might yield interesting results about how robust the training is to reversed sequences. Possibly you learned about memorization, position encoding, etc.  

4. **Research & Development**  
   - The first training clarified hyperparameters, scaling, debugging processes, and improvements in your pipeline. These have intangible benefits for future training runs.

Therefore, while it was an error for your main objective, the cost wasn’t entirely wasted—there’s technical and knowledge-based value you can salvage.

---

## **Task 5**

**Steganography** scenario: You’re in prison, want to send hidden messages with minimal suspicion. You have:

- **High-quality language model** (virtually indistinguishable from natural text).  
- The adversary (Warden) sees your letters, so the text must look innocent.  
- You want to transmit a few dozen bits of hidden info in each letter.  
- Both you and the Comrades have exactly the same language model and environment.

**How to organize invisible communication?**

A classic approach is to embed bits in the **choice of tokens** while still producing a plausible sentence. For example:

1. **Shared Random Seed**: You and your Comrades share a seed or a pseudo-random generator.  
2. **Bit Extraction**:  
   - For each step of generation, you look at the top \(N\) plausible next tokens (where \(N\) is large enough to produce natural text).  
   - Convert the next few bits of your secret message into an index that selects which token (out of the plausible set) you will actually pick.  
   - The chosen token is still high-likelihood, so the final text remains coherent.  
3. **Decoding**:  
   - The Comrades run the same language model with the same context, know how you pick from the top \(N\) tokens, and re-construct the bit choices from your final word choices.  

**In essence**, you are using the language model’s probability distribution to “hide” your bit decisions among multiple plausible next words. The Warden sees only a normal-looking letter. As long as your hidden selection method is carefully done (so the text remains natural), the presence of a hidden message is very hard to detect.

---

## **Task 6**

Research **Kerckhoffs’s Principle**: “A cryptosystem should be secure even if everything about the system is known except the key.” 

1. **Does the solution from Task 5 comply?**  
   - If the Warden knows *exactly* how your steganographic scheme works (the top-\(N\) approach, etc.) and also has your same language model, they could attempt to guess your seed or intercept the distribution of token choices. Without a private key or seed, the Warden might detect anomalies or guess the hidden bits.  
   - Often, the user’s approach in Task 5 relies on a **shared secret random seed** or some hidden parameters that the Warden does **not** know.  

2. **Adjusting it to comply**  
   - If the *only* unknown is the **secret key** (like a secret seed controlling how bits map to token choices), you can argue it *does* meet Kerckhoffs’s principle. The system design is public, but the Warden does not know the specific seed you use at each step.  
   - You must ensure that, even if the adversary fully understands the method, they cannot invert or detect the hidden bits unless they guess your private key.

Hence, yes, you can incorporate Kerckhoffs’s principle by introducing a robust key-based scheme that does not fail if the method is revealed.

---

## **Task 7**

**Design a method for compressing Polish texts** using a GPT-like language model installed on every computer. 

A standard approach:

1. **Tokenize** the input text into tokens recognized by GPT-2.  
2. For each token \(w_t\), use the model to compute \(P(w_t \mid w_1,\dots,w_{t-1})\).  
3. Encode each token with an **arithmetic coder** (or another entropy coder) that uses the model’s probability distribution at each step. That is, if the next token is predicted with probability 0.001, you encode it in about \(\log_2(1000)\approx 9.97\) bits. More likely tokens require fewer bits.  
4. The final compressed output is the concatenation of all these bits.

**Decompression** is straightforward: run the same GPT-2 in “decode” mode, applying the same arithmetic decoding steps to recover the original tokens.

Key points for an implementer:

- **Model Access**: The compression algorithm must have random access to GPT-2’s next-token probabilities at each step.  
- **Implementation**: Use standard arithmetic coding or range coding.  
- **Output**: A sequence of bits. The question says we can ignore how to pack bits into bytes or handle error-robustness.

Thus you have a well-defined method to compress Polish text by exploiting the GPT-2 prior.

---

## **Task 8**

In the **Papuga** model (a GPT-like model in Polish), the prefix must **not** end with a space. Observing generations with/without a trailing space yields noticeably different completions.

**Why?**

- GPT-like models (including Papuga) often use a **Byte-Pair Encoding (BPE)** or similar subword tokenizer. A space character `' '` often signals the start of a new token or merges differently with subsequent characters.  
- If your prefix ends with “\_” (space), the next token distribution differs from the scenario where the prefix ends with a letter (no space in the token stream). The hidden states differ because the model sees a “space token.”  
- This leads to changed probabilities for the next tokens. For instance, the model might strongly prefer continuing with a “word piece” if the last character wasn’t space, or it might treat it as a brand new “word” if a space is present.

In short, **trailing space** alters how the next token is segmented and thus changes the model’s predicted distribution.

---

## **Task 9**

We want to generate **rhymed poetry** with a model like GPT-2. For example:

> I have only one burning desire  
> Let me stand next to your fire

**What makes text “rhymed”?**

- Typically, line endings (or certain positions) share similar endings or phonetic patterns.  
- Generating such lines requires controlling the final words or syllables to match a target rhyme.

**Possible algorithm**:

1. **Maintain partial generation** for a line, but **force** the last few tokens (or last syllable) to match a target rhyme pattern.  
2. **Beam search** or a **constrained search**:
   - Suppose you want each line to end with a specific rhyme like “-ire.”  
   - Generate the line token by token, but near the end you do lookahead or keep only partial expansions that can lead to a token ending in “ire” (or spelled similarly in Polish or English).  
3. **Dictionary or Phonetic Approach**:  
   - Keep a mini-dictionary or model for “which tokens rhyme with X?”  
   - Whenever the line is about to end, you pick from the short list of tokens that preserve the rhyme.  

This approach merges **search** with **LM predictions**:

- The language model ensures fluency in the middle.  
- The constraint enforces the rhyme at the end.  
- You can use backtracking or a dynamic approach if you want to guarantee the last word has the correct ending, while maximizing the probability (or sampling) for the rest of the line.

---

## **Task 10**

**Propose a non-linguistic “predict the next token” application** that could be interesting or useful. The task must *not* be purely about natural language.

### Example Idea: “Music Note Prediction”

1. **Corpus**: A large collection of **monophonic or polyphonic music** transcribed into a symbolic format (e.g., MIDI turned into tokens for pitch, duration, etc.).  
2. **Tokenization**: 
   - Each note event becomes a token: e.g., `(pitch=C4, length=quarter)` or short sub-tokens `(pitch=C4) (length=1/4)`.  
   - Alternatively, a step-based format where each time step is a token with the set of active notes.  
3. **Model**: A “Language Model” (actually a “Music Model”) is trained to predict the next note (or next chord/time slice).  
4. **Use Case**: 
   - **Generate new musical phrases** by sampling the next notes.  
   - **Fill in missing notes** (like “inpainting” for music).  
   - **Transcribe partial improvisations** and continue them in style.  

This is interesting because it’s not text: you’re predicting the *next musical event* rather than the next word. The same “predict the next token” methodology can produce coherent melodies or harmonies from large music corpora.
