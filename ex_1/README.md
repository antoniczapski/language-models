## Zadanie 1

**Type of Task ChatGPT Struggles With**:  
Here is an ASCII maze. Find a valid path from the Start (S) to the End (E) without crossing any walls (#). You can move up, down, left, or right, but not diagonally. Provide the path as a list of coordinates in the format (row, column), starting from (1,1) for S.

Maze Layout:
```
S . . # . . . .
# # . # . # # .
. . . . . . # .
. # # # # . . .
. . . . # . # E
```

Legend:
- `S` : Start
- `E` : End
- `.` : Open path
- `#` : Wall

Please provide the path coordinates from S to E.

Examples are in the subfolder `visualizations`
- [link] wrong solution using GPT-4o
- [link] correct solution using `01-mini`

## Zadanie 2

`5.` W jakim zespole występowała Hanka w filmie „Żona dla Australijczyka”? 
- GPT-4o: W zespole Mazowsze.
- Ground truth: Mazowsze

`8.` Co budował w Egipcie inżynier Tarkowski, ojciec Stasia? 
- GPT-4o: Kanał Sueski.
- Ground truth: Kanał Sueski

`11.` Którzy rzemieślnicy występują w ostatniej powieści Witkacego?
- GPT-4o: Szewcy.
- Ground truth: szewcy

`22.` Kto jest autorem obrazu namalowanego w 1892 r. „Kobiety z Tahiti”?
- GPT-4o: Paul Gauguin.
- Ground truth: Paul Gauguin

`47.` W którym filmie Krzysztofa Kieślowskiego zagrali Olaf Lubaszenko i Grażyna Szapołowska?
- GPT-4o: "Krótki film o miłości".
- Ground truth: „Krótki film o miłości”

`52.` W zdaniu: „Szybko przeskoczył przez płot” który wyraz jest przysłówkiem?
- GPT-4o: Szybko.
- Ground truth: szybko

`83.` Nad którą rzeką leży Kazimierz Dolny?
- GPT-4o: Nad Wisłą.
- Ground truth: nad Wisłą

`84.` Jak nazywa się najwyższy szczyt Beskidu Sądeckiego?
- GPT-4o: Radziejowa.
- Ground truth: Radziejowa

`87.` Jak nazywa się jednostka w lotnictwie licząca 2–3 eskadry?
- GPT-4o: Dywizjon.
- Ground truth: dywizjon / pułk lotniczy

`89.` Co produkowała wytwórnia astronoma Jana Heweliusza?
- GPT-4o: Piwo.
- Ground truth: piwo

**Accuracy: 100%**

## Zadanie 3

- **Watermark Feasibility**:
  - For **long texts**, the language model can systematically tilt its sampling probabilities to produce more words starting with C, S, or K. A specialized detector can measure how often those letters appear compared to normal text and detect the watermark.
  - For **short texts**, it’s harder to maintain a strong bias without humans noticing something odd or without insufficient sample size to confirm the pattern. The watermark might be too weak or too obvious if forced strongly.

- **Conclusion**:  
  This watermark technique can work *probabilistically* for longer texts (statistical detection), but for short texts, it either goes undetected or becomes conspicuous if overdone.

## Zadanie 4

- **(a)**  
  Use few-shot prompt tostear the probability distribution towards answering with one, correct word. Then sample 10 times the first word that comes after a given question. Use a majority voting to chose the winner.

  You could also restric words in teh voting to those in the given 10k set.

- **(b)**  
  Using sentence probability scoring alone is tricky because:
  1. You’d have to rank all 10k words by how well they “fit” the clue.  
  2. The model might assign similar probabilities to synonyms.  
  3. Short riddles are ambiguous; the model’s training distribution might not reflect specialized definitions.  

Hence, direct probability scoring could lead to confusion or computational overhead, especially if you brute-force all 10k possible answers.


## Zadanie 5

A possible procedure:

1. Create a binary mask of tokensfrom model dictionary that do not contain space (blank character)
2. Encode the prefix as a sequence of tokens and feed them to language model as a context.  
3. Do a beam search to find the most accurate word. The advantage over other sampling techniques is that it test words of different token length.

## Zadanie 6

### Answer

- **How were biases studied?**  
  They likely tested the model on prompts that expose stereotypes, e.g., “A woman is a nurse, a man is a ____” or analyzing completions for minority groups. They might have run it through a suite of known bias benchmarks or performed manual inspection.

- **Conclusions**:  
  - The model exhibits typical LM biases: strong gender stereotypes, possible negative or overgeneralizing references to certain ethnic or cultural groups.  
  - **Takeaway**: The model’s training data shapes those biases, so usage must be mindful of potential harmful stereotypes or unbalanced associations.

## Zadanie 7

- **Prompting Technique**:
```
prompt = (
    "Oto tłumaczenia - najpierw w języku polskim, a następnie w języku angielskim:\n\n"
    
    "PL: Dzień dobry, jak się masz?\n"
    "EN: Good morning, how are you?\n\n"
    
    "PL: Kocham programowanie.\n"
    "EN: I love programming.\n\n"
    
    "PL: Lubię pływać.\n"
    "EN: I like swimming.\n\n"
    
    "PL: Jak się nazywasz?\n"
    "EN: What is your name?\n\n"
    
    "PL: Mieszkam w Warszawie.\n"
    "EN: I live in Warsaw.\n\n"
    
    "PL: Jaka jest twoja ulubiona książka?\n"
    "EN: What is your favorite book?\n\n"
    
    "PL: Lubię jeść pizzę.\n"
    "EN: I like eating pizza.\n\n"
    
    "PL: Co robisz w wolnym czasie?\n"
    "EN: What do you do in your free time?\n\n"
    
    "PL: Słońce świeci jasno.\n"
    "EN: The sun is shining brightly.\n\n"
    
    "PL: Chciałbym kupić nowy samochód.\n"
    "EN: I would like to buy a new car.\n\n"
    
    f"PL: {given_pl_sentence}\n"
    "EN:"
)
```

- **Example** - generated in task_7.py:
```
[INFO] Loading model: eryk-mazus/polka-1.1b
=== Correct Translation Example ===
PL: Jestem studentem.
EN: I am a student.

=== Incorrect Translation Example ===
PL: Ile masz lat?
EN: I am from warsaw.
```

- **Using a dictionary**:
  - Fine-tune the language model on translations of words
  - Alterntively, for each English word (or chunk), consult a dictionary. If Polka’s translation doesn’t match, we either re-prompt or do a post-processing step to correct potential mistakes.  
  - This could be done by a simple alignment or a final re-check stage that compares each key word in the sentence to dictionary entries for increased accuracy.

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
4. **Two models using sub word tokenization**
   - We use beam search to generate one word at a time with both of the models - then we choose one that's cumulative probability from both models is the highest

## Zadanie 9

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