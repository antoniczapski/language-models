## Task 1  
> *Słowo „tonie” has four different meanings and four lemmas. List them all, then write a short text (preferably one sentence) in which this word appears in all four meanings. Finally, check if ChatGPT can correctly interpret that sentence.*

### 1.1. Identifying the Four Meanings (and Lemmas) of “tonie”

In Polish, the surface form **“tonie”** can come from different lemmas and correspond to different meanings. Commonly:

1. **tonąć** (verb) – “(he/she/it) is sinking.”  
   - e.g., “Statek tonie” = “The ship is sinking.”

2. **ton** (noun, masculine) – “tone” or “pitch” in music or voice.  
   - The noun “ton” in Polish can appear in the locative or vocative forms as “tonie,” although it’s somewhat rare or archaic; more typical is “tonie” as a morphological variant in certain contexts (“w tonie kogoś/czegoś” – “in someone’s tone”).

3. **tona** (noun, feminine) – “a ton” (unit of mass).  
   - The form “tonie” might appear in an oblique case: for instance, in the locative or dative if spelled or declined in a particular archaic or dialect form. More commonly we’d see “tonie” if we treat “tona” in certain morphological variants (though in standard usage you might see “tonach,” “tonie” is possible in older usage or less common forms).

4. **ton** (noun, but can also mean “style” or “manner of expression”), which might appear as “tonie” in certain grammatical contexts (similar to #2 but with a nuance about style or register).

Depending on the grammar source, some dictionaries unify #2 and #4 under the same lemma “ton” (with multiple senses: musical pitch vs. manner/style). But the exercise states there are four separate lemmas, likely corresponding to:

- **(Verb)** tonąć → “(on/ona/ono) tonie” = “he/she/it is sinking.”
- **(Noun)** ton (musical pitch) → locative form: “w tonie” = “in (that) pitch” or “in that tone.”
- **(Noun)** ton (manner, style of speaking) → “w (jakimś) tonie” = “in (some) tone/manner.”
- **(Noun)** tona (mass unit) → an inflected form might produce “tonie” in archaic or specialized usage (though standard grammar might prefer other forms, but presumably the exercise says there’s such a morphological form).

Hence we have at least these four senses or lemmas that produce a surface form spelled “tonie.”

### 1.2. A Single Sentence Using “tonie” in All Four Senses

Let’s craft a (slightly playful) sentence in Polish that forces these interpretations:

> „Statek **tonie** (tonie) w ciszy, a ja wciąż mówię w wesołym **tonie** (tonie), chociaż odkryłem, że przewożę aż dwie **tonie** (tony) towaru i tym samym zmieniam **tonie** (ton) całej dyskusji.”

Breaking it down:

1. **tonie** (verb, from “tonąć”): “the ship is sinking.”
2. **tonie** (noun, from “ton” = “tone/pitch”), used in a phrase like “w wesołym tonie” = “in a cheerful tone.”
3. **tonie** (noun, from “tona” = “a ton” or “tons”), referencing quantity of cargo—some older or dialect forms might say “dwie tonie,” though standard might prefer “dwie tony.” The exercise presumably acknowledges we can use “tonie” as a morphological variant.
4. **tonie** (another sense of “ton” = “style/manner”), so “zmieniam tonie całej dyskusji” could be a stylized phrase meaning “I’m changing the overall style/tone of the entire discussion.”

### 1.3. Checking ChatGPT’s Interpretation

The second part of the task is to see if ChatGPT can identify all four meanings in that sentence. In practice, you’d feed the sentence to ChatGPT and ask it to list the sense of “tonie” each time. If ChatGPT can parse the morphological context, it might say:

1. “tonie” = third-person singular of “tonąć.”
2. “tonie” = locative of “ton” (musical pitch or manner).
3. “tonie” = inflected form for “tona” (mass unit).
4. Another “tonie” = an alternate usage referencing style or manner of expression.

## Task 2
> *What are “sparse vector representations” of words (using TF-IDF and contexts)? Why aren’t they perfect? Propose a procedure that includes clustering, potentially giving better results (and fewer dimensions).*

### 2.1. Sparse Word Vectors (TF-IDF + Context)

- **TF-IDF**: We can represent each word \(w\) by the “contexts” in which \(w\) appears. For example, we build a large vocabulary of possible context words, and for each word \(w\), the vector dimension \(i\) corresponds to the TF-IDF score of context-word \(c_i\) for \(w\). 
- These vectors are typically **very high-dimensional** and **sparse** (most entries are zero).

### 2.2. Imperfection of Such Representations

1. **Vocabulary explosion**: If the language or corpus is large, the dimension is huge.  
2. **Synonym or morphological overlap**: “żaglówka” vs “żaglowiec” might share many contexts but appear in separate dimensions that are not semantically recognized as the same root.  
3. **Lack of advanced semantic similarity**: Just because two words share many context words doesn’t necessarily capture deeper semantic or morphological relationships.

### 2.3. Proposed Procedure with Clustering + Dimensionality Reduction

A potential improvement:

1. **Gather word–context matrix** using TF-IDF or co-occurrence counts (like a standard distributional approach).  
2. **Cluster** the contexts themselves (e.g., using k-means or hierarchical clustering on columns) to group semantically similar contexts.  
3. **Aggregate** each word’s distribution across these clusters instead of across individual context words. For instance, you represent each word as a vector of length \(K\) (the number of clusters), where each component is the sum (or average) of TF-IDF scores for contexts belonging to that cluster.  
4. Optionally, do a final **dimensionality reduction** (e.g., SVD or PCA) to get a smaller vector dimension.  

This yields **fewer dimensions** (the cluster count or an SVD-truncated rank) and can group near-synonymous contexts together, hopefully improving the representation over raw, extremely sparse TF-IDF.

## Task 3  
> *Next Sentence Prediction (NSP) with two variants of negative examples: (a) negatives randomly sampled from the corpus, (b) negatives formed by swapping the order of consecutive sentences. Propose solutions with a model like Papuga and (optionally) a simpler non-neural method.*

### 3.1. Using Papuga (or GPT-like) for NSP

We can train a classifier on top of the “sentence pair” representation from the language model. For each pair \((s_1, s_2)\):

1. **Encode** the pair using the LM (Papuga GPT, or HerBERT, etc.)—perhaps by concatenating them with a separator.  
2. **Pool** the final hidden states into a single representation (like [CLS] or a special token).  
3. **Predict** whether \(s_2\) naturally follows \(s_1\).

We gather **positive** examples \((s_1, s_2)\) = real consecutive sentences, and **negative** examples either:

- **(a) Random)**: pick \(s_2\) from anywhere else in the corpus.  
- **(b) Swapped)**: take consecutive sentences but swap them, so it’s guaranteed they no longer match the correct order.

### 3.2. Non-Neural Method (Optional)

- We could use **bag-of-words or word2vec** embeddings for each sentence. Then produce a feature vector like `[similarity(s1, s2), length(s1), length(s2), overlap_of_words,…]` and train a logistic regression or SVM to classify if \(s_2\) follows \(s_1\).
- Alternatively, use word2vec to build averaged embeddings for each sentence, then measure the cosine similarity. In the random negative scenario, consecutive sentences might be more topically similar than random pairs.

## Task 4
> *Propose a method for (context-free) node embeddings in a graph (e.g., social networks, Netflix users & movies). The method should use the original Word2Vec.*

A well-known approach is **DeepWalk** or **node2vec**:

1. **Random walks** on the graph to generate “sequences of nodes.”  
2. **Treat** each walk as a “sentence” of tokens (each node = “token”).  
3. **Train** Word2Vec (CBOW or Skip-gram) on these “sentences.”  
   - This yields an embedding for each node that captures local neighborhood structure and even global community structure.

Hence we get a **word embedding** approach applied to nodes in a graph.

## Task 5
> *We have questions about Polish proverbs, like “Z czym według przysłowia porywamy się na słońce?” or “Co według przysłowia kołem się toczy?” We want an approach using information retrieval.*

A feasible solution:

1. **Collect a large corpus or database of proverbs** in Polish, possibly from an online source or dictionary of proverbs.  
2. **Use IR**: 
   - Convert the question into keywords or use advanced search like BM25 or vector-based search (e.g., with embeddings). 
   - Match it against the known text of proverbs. 
   - Identify the substring that answers the question.  

For example, if the question is “Z czym według przysłowia porywamy się na słońce?”, we might search for the key phrase “porywać się na słońce,” see the proverb “Z motyką porywać się na słońce” → answer: “motyką.”  
Similar logic for “Co według przysłowia kołem się toczy?” → “Fortuna kołem się toczy” → answer: “fortuna.”

## Task 6
> *Read the baseline solution for PolEval 2021 Task 4 (question answering). Why might it work, and propose a sensible correction.*

[[TODO]]

## Task 7
> *We know BPE (Byte-Pair Encoding) normally increases the number of tokens to a desired threshold. Propose an algorithm that does the reverse: start with a very large number of tokens and reduce it until we reach the required size, while maximizing the unigram probability of the corpus.*  
> *Constraints:  
>  1) Language-independent (no pre-search for words)  
>  2) Works on a large corpus, always retokenizing based on the current set of tokens.  
>  3) Does many steps, gradually approaching the target vocabulary size.  
>  4) Removes “less useful” tokens.  
>  5) Maximizes \(\prod p(w_i)\) or equivalently the sum of \(\log p(w_i)\) under the chosen tokenization.*

### High-Level Idea (Reverse BPE)

1. **Initial State**: We have a huge set of tokens (could be every single possible substring or starting from a more granular “character” model).  
2. **Compute Frequencies**: For each token in the current vocabulary, count how often it appears in the corpus (tokenized by the current set).  
3. **Score** or approximate the “utility” of each token (for instance, how much it contributes to reducing the overall negative log-likelihood of the corpus).  
4. **Merge or “Split Out”**:
   - Actually, we are *removing* tokens that are rarely used or that don’t help compress the text. If we remove a token, we must retokenize those occurrences into smaller or more base tokens.  
   - Each step, pick the token whose removal (and subsequent forced retokenization) yields the smallest increase (or largest net improvement) in the negative log-likelihood.  
   - Recompute or update frequencies.  
5. **Iterate** until the vocabulary is down to the target size.

**Why does maximizing \(\prod p(w_i)\) make sense?** Because we want a tokenization that best “fits” the corpus distribution under a simple unigram assumption. Minimizing the surprise of each token overall is similar to “best compression” in a naive unigram sense.

## Task 8  
> *Propose **three different scenarios** of data augmentation for **reviews** (e.g., product or movie reviews) using Papuga (GPT-like model).*

Let’s imagine we have a corpus of text reviews in Polish. We want to enlarge or diversify it using the Papuga model:

1. **Style Transfer**: For each existing review, prompt Papuga with “Rewrite this review in a slightly more informative style.” This yields new text but keeps the same sentiment.  
2. **Aspect Expansion**: If the original review is short, we can prompt Papuga: “Expand on the details of the product’s design in this review.” The model inserts more details about design. This might capture more nuanced sentences about the product’s features.  
3. **Sentiment Flip**: For data balancing, we can prompt Papuga: “Rewrite this review with the opposite sentiment while keeping the same product aspects.” This is more advanced but helps create negative-sentiment data from positive-sentiment reviews or vice versa.

## Task 9
> *Propose three scenarios for using word2vec in the “Riddles” tasks from previous lists. Assume we have access to reference definitions of words and some examples of riddles.*

Possible usage:

1. **Clue–Definition Matching**: For a given riddle clue, we extract keywords, embed them with word2vec, and find the nearest neighbor among reference definitions. The definition with the highest similarity might be the answer.  
2. **Synonym/Analogy Finder**: Some riddles rely on analogies. Word2vec can do vector arithmetic: “king - man + woman = queen.” We might adapt this for riddle-like transformations, e.g., “milk - cow + goat = ?” to guess “goatmilk.”  
3. **Contextual Scoring**: If each riddle provides a partial definition of a word, we can average the word2vec vectors of those clue words, then look up which candidate solution has the smallest distance. This is a simpler bag-of-words synergy approach.

## Task 10 
> *We want to evaluate the likelihood of a generated text in a scenario with strong constraints (like every word must start with “p”). Using the model’s probability to judge “plausibility” has a flaw. Hint: consider the word ‘przede.’ How to fix that flaw easily?*

### 10.1. The Problem

- If you force all words to start with “p,” you might artificially produce weird or partial forms like “przede,” “ppor,” or “pXYZ.” Some of these might be legitimate (like “przed e…”?), but many tokens are unnatural.  
- The model’s probability might be high if it sees “przede” as one token or merges tokens in an unnatural way. Or the model might treat “przede” (meaning “before something else” in Polish) incorrectly as “prze” + “de.”  
- In short, the raw probability of the text might not reflect that each separate “word” is forced. Some tokens might be partial merges.

### 10.2. The Easy Fix

**Token alignment**. We should ensure the segmentation of words is correct and that the model or the scoring function only considers valid lexical forms. For example, if we truly want each “word” to begin with ‘p’, we must:

- **Enforce** token boundaries so that “przede” is recognized as a single legitimate lexical item, or
- **Strip out** partial merges from the model’s vocabulary so that improbable merges like `p` + `rze` + `de` don’t get artificially high or low scores.

Alternatively, we could:

- Insert a penalty or set the model to only evaluate valid tokens that do start with ‘p’ in the lexicon. 
- Or use a specialized subword approach that ensures alignment with the constraint.

Hence we avoid illusions in the probability from subword merges that start with ‘p’ but are actually partial fragments.

## Task 11
> *We return to word embeddings tested with ABX using a BERT-like model. Suppose the embedding is the entire utterance’s [CLS] vector in HerBERT, not just a single word. Propose a method to construct such an utterance using a text corpus (or a lemma file).*

### 11.1. The Goal

We want to get a better embedding for a specific word \(w\). Instead of feeding just that single word into BERT/HerBERT, we embed it within a short “utterance” or sentence. Then we take the [CLS] vector from the last layer as the word’s representation. We hope it captures more robust context.  

### 11.2. Proposed Method

1. **Gather Real Sentences**: Search in a big text corpus for a sentence that uses \(w\). Possibly choose the most frequent or average.  
2. **Lemma Variation**: If we have a lemma file, ensure we pick the same morphological form as needed.  
3. **Slot-based Template**: If we can’t find a real usage, we create a generic template like: “Oto zdanie, w którym występuje słowo [w], tak aby [w] było w naturalnym kontekście.” Then we fill `[w]` in.  
4. **Final Embedding**: We feed that entire sentence into HerBERT, retrieve the **[CLS]** vector, and declare it the word’s representation.  

**Potential improvements**:

- Use multiple found sentences for the same word, average the resulting [CLS] vectors.  
- Or cluster the usages, to capture different senses if the word is polysemous.

### 11.3. Implementation & ABX Score

- (Implementation detail) For the ABX test, we produce these new embeddings for each word. Then we measure if the ABX accuracy is higher compared to single-word input. 
- We might see that adding a small context around \(w\) yields more stable embeddings and better ABX performance.

If the approach outperforms single-word embeddings, we get an additional point as the task states.
