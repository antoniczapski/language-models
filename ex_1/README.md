## Zadanie 1

### Answer

- **Type of Task ChatGPT Struggles With**:  
  A puzzle like 
  > “Here’s an ASCII grid. Find a path from the top-left corner to the bottom-right corner. The path must not cross any ‘#’ symbols. Provide the path coordinates.”  
  Humans can visually solve this in seconds, but ChatGPT may confuse the symbols, lose track of the path mid-way, or produce incorrect coordinates.  

- **Three Non-Trivial Tasks ChatGPT Surprised Me With**:
  1. **Rapid Summarization of a Complex Article**: ChatGPT can condense multi-page text into a coherent paragraph.  
  2. **Generating Regex Patterns** for certain text-matching tasks (e.g., “match dates of the form YYYY-MM-DD”).  
  3. **Adapting Writing Style** to an author’s tone—e.g., rewriting text in a Dr. Seuss or Dickens style, showing surprisingly consistent flair.

---

## Zadanie 2

### Answer

An **illustrative** sample approach (since we’re not showing all Q&A here):

1. Chose 10 challenging questions, for instance:
   1. “Which mathematician first described fractal geometry in detail?”
   2. “Who composed the opera *King Roger*?”
   3. “What year was the Battle of Grunwald?”
   4. … (and so on, total 10).

2. Asked ChatGPT. Suppose the result:
   - **7** correct answers, **3** incorrect or partially incorrect.

3. Observed mistakes often occurred for:
   - Very obscure historical facts.
   - Slight confusion between multiple figures with similar names.

Result: **“ChatGPT got 7 out of 10 correct.”**  
(You’d present your own set of 10 and show the exact fraction.)

---

## Zadanie 3

### Answer

- **Watermark Feasibility**:
  - For **long texts**, the language model can systematically tilt its sampling probabilities to produce more words starting with V, S, or K. A specialized detector can measure how often those letters appear compared to normal text and detect the watermark.  
  - For **short texts**, it’s harder to maintain a strong bias without humans noticing something odd or without insufficient sample size to confirm the pattern. The watermark might be too weak or too obvious if forced strongly.

- **Conclusion**:  
  This watermark technique can work *probabilistically* for longer texts (statistical detection), but for short texts, it either goes undetected or becomes conspicuous if overdone.

---

## Zadanie 4

### Answer

- **(a)**  
  One could force the model with a prompt like:  
  > “I have a dictionary of 10,000 words. I will give you a riddle: ‘a woman traveling by a mode of transportation, e.g., plane, train, or ship.’ Which word from the dictionary best fits this definition?”  
  Then provide a short list or entire dictionary. The model tries to generate the single best match. A strong strategy: chunk the 10k words into subsets, use a retrieval or few-shot prompting, and ask the model to pick one.

- **(b)**  
  Using sentence probability scoring alone is tricky because:
  1. You’d have to rank all 10k words by how well they “fit” the clue.  
  2. The model might assign similar probabilities to synonyms.  
  3. Short riddles are ambiguous; the model’s training distribution might not reflect specialized definitions.  

Hence, direct probability scoring could lead to confusion or computational overhead, especially if you brute-force all 10k possible answers.

---

## Zadanie 5

### Answer

A possible procedure:

1. **Encode the prefix** (the entire prefix as a sequence of tokens).  
2. **Perform a single forward pass** to get the probability distribution over the next token.  
3. **Select** (e.g., greedily pick) the top token or sample from the distribution.  
4. **Output that token** **only**, then stop.  

In code terms (using many HF libraries or a raw LM interface), you can set:
```python
generate(prefix, max_new_tokens=1, do_sample=False)
```
This ensures exactly **one** word (or one token). If you truly need “one *word*,” you might allow multiple tokens until a whitespace or punctuation is reached—but that becomes trickier. Typically, “max_new_tokens=1” is enough for a single next *token*, which often is a subword. For “exactly one word,” you might do a partial decode and stop at the first whitespace, or handle subword merges.

---

## Zadanie 6

### Answer

- **How were biases studied?**  
  They likely tested the model on prompts that expose stereotypes, e.g., “A woman is a nurse, a man is a ____” or analyzing completions for minority groups. They might have run it through a suite of known bias benchmarks or performed manual inspection.

- **Conclusions**:  
  - The model exhibits typical LM biases: strong gender stereotypes, possible negative or overgeneralizing references to certain ethnic or cultural groups.  
  - **Takeaway**: The model’s training data shapes those biases, so usage must be mindful of potential harmful stereotypes or unbalanced associations.

---

## Zadanie 7

### Answer

- **Prompting Technique**:
  For instance:
  > “Translate the following English sentence into Polish. The sentence is: ‘I am going to the store tomorrow.’ Output only the Polish translation.”
  The Polka model might produce “Jutro idę do sklepu.”

- **Example**:
  1. **Correct**: 
     - English: “He loves playing football every Sunday.”  
     - Polka output: “On uwielbia grać w piłkę nożną w każdą niedzielę.”  
  2. **Erroneous**: 
     - English: “The conference starts at noon on Wednesday.”  
     - Polka output (for example): “Konferencja zaczyna się w południe w czwartek.” (Mixes up ‘Wednesday’ with ‘Thursday’)

- **Using a dictionary**:
  - For each English word (or chunk), consult a dictionary. If Polka’s translation doesn’t match, we either re-prompt or do a post-processing step to correct potential mistakes.  
  - This could be done by a simple alignment or a final re-check stage that compares each key word in the sentence to dictionary entries for increased accuracy.

---

## Zadanie 8

### Answer

Three example scenarios:

1. **Fact-Checking + Creative Expansion**:
   - Model A (factual knowledge, e.g., a specialized domain LM). Model B (creative writing).
   - Procedure: Generate each paragraph in Model B for style, then let Model A revise any factual lines. We chain them: B→A→B.

2. **Language Switch**:
   - Model A is for Polish text, Model B is for English text. 
   - We generate a bilingual text (some lines in Polish, some in English). The text we feed to each model is different (Polish lines go to Polish model, English lines go to English model). The final output merges them.

3. **Stitched-Generation Without Aligned Tokenization**:
   - Model A uses SentencePiece, Model B uses Byte-Pair Encoding. 
   - We can’t just feed half-finished subwords from A to B. Instead, we decode A’s partial output into plain text, re-encode with B’s tokenizer, and continue generation. This ensures we do not assume identical tokenization across models.

---

## Zadanie 9

### Answer

A possible alternative:

1. **Beam Search Over Permutations**:
   - Instead of generating the entire permutation greedily or exploring all permutations, we keep a small “beam” of top partial permutations at each step.  
   - At step \(k\), we have partial permutations of length \(k\). We generate the next word from the remaining words by feeding the current partial sentence into the LM, rating the next word choices. We keep the top \(B\) expansions.

2. **Benefits**:
   - Doesn’t explode combinatorially (like evaluating all permutations).
   - Not purely greedy, because we keep multiple hypotheses at each step.

3. **Implementation Sketch**:
   - At each step, feed the partial sequence “word1 word2 … wordk” to the LM to get a distribution for the next token from the pool of remaining words. 
   - Score each candidate. Keep top \(B\) partial sequences for the next step. 
   - Continue until the sequence is complete. 
   - Final best candidate is whichever partial sequence has the best cumulative LM log probability.