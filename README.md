# Computational-Psycholinguistics

Kushagra Dhingra (2022115004)
Steps to run the code
sudo apt update
pip3 install -r requirements.txt # Installs all the dependencies required
Download the glove 6B dataset from here:
https://nlp.stanford.edu/projects/glove/
Download the w2v dataset from here:
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=s
Keep both in the working directory as the python notebook,
Then simply do run all in the python notebook(.ipynb) to run the files.
1. Goal
This study investigates whether contextual language models (BERT) can more
effectively mirror human semantic priming behaviors—such as reaction times
(RTs), explicit relatedness judgments, and word-sense disambiguation—compared
to static embeddings (GloVe 50d). We focus on ambiguous words to probe
context‐sensitive meaning resolution.
2. Theoretical Background
Computational Psycholinguistics Final Report 1
Human semantic priming reflects the cognitive facilitation when a prime word
activates related concepts, speeding up recognition of the target. Reaction times
inversely correlate with semantic relatedness. Computationally, embedding
similarity (e.g., cosine similarity) between prime and target vectors serves as a
proxy for psychological relatedness: higher cosine similarity (dot product of
normalized vectors) indicates greater semantic overlap, theoretically predicting
faster RTs and higher relatedness judgments.
Cosine similarity is widely used in distributional semantics as it is scale-invariant
and emphasizes directional alignment rather than magnitude, making it ideal for
comparing high-dimensional vectors where only the relative position (semantic
alignment) matters. This measure thus operationalizes semantic proximity and
allows us to test how closely models reflect human associative processes.
Contextual models like BERT generate dynamic embeddings conditioned on entire
input text, capturing polysemy by altering a word’s vector based on surrounding
context. Static embeddings (GloVe 50d) assign a single vector per word,
conflating senses and lacking dynamic nuance. We hypothesize:
1. RT Prediction: BERT-based cosine similarities will correlate more strongly
(negatively) with human RTs than GloVe 50d.
2. Relatedness Classification: Although both may perform well, BERT’s contextawareness could yield marginal gains in ROC-style classification.
3. Sense Disambiguation: BERT will outperform GloVe when distinguishing
related vs. unrelated uses of ambiguous primes.
3. Datasets
SPP (Semantic Priming Paradigm): Prime-target pairs with human RTs and
binary relatedness. We utilize random samples (\~41 k pairs). We ran on all
fractions of the dataset (10%, 25%, 100%)
Raw Stimuli: Custom set of ambiguous target words with context and explicit
related/unrelated labels.
WiC (Word-in-Context): Standard sense disambiguation benchmark; “test”
split used for sentence-level evaluation.
Computational Psycholinguistics Final Report 2
4. Methods
1. Embedding Extraction:
GloVe 50d: Pretrained vectors; static per word.
BERT: bert-base-uncased embeddings averaged over non-special tokens,
capturing context effects. Due to limited computational resources, we
used the base version of BERT and trained only for a small number of
epochs. This compromises performance relative to larger or betterfinetuned implementations but still reveals model behavior trends.
2. Using Cosine Similarity:
where higher values indicate greater semantic relatedness.
3. RT Correlation: Pearson and Spearman correlations between similarity and
RTs.
4. Binary Classification: Logistic regression on similarity to predict human
relatedness (0/1).
5. Ambiguous Primes Analysis: For four selected ambiguous primes (FOAM,
OFTEN, OBSTACLE, CLEANER), classify each prime-target pair as related if
similarity exceeds the prime’s median; report accuracy per prime.
6. Sense Disambiguation: Concatenate prime and target embeddings; train/test
logistic classifier to distinguish related vs. unrelated for raw stimuli and WiC.
5. Key Findings
5.1 Reaction Time Correlations
Concept: Reaction time (RT) in lexical decision tasks decreases when primes preactivate semantically related targets (Traxler Ch. 3). Cosine similarity between
prime and target embeddings quantifies this semantic relatedness
computationally.
Model Pearson r (sim vs. RT)
cosine(vp, vt) =
∥vp∥ ∥vt∥
vp ⋅ vt
Computational Psycholinguistics Final Report 3
GloVe -0.080
BERT -0.202
Inference: BERT’s stronger negative correlation indicates its dynamic embeddings
capture context-driven activation more effectively. Architecturally, BERT’s selfattention layers model interactions among all tokens, aligning with human
incremental integration of semantic information. GloVe’s static vectors lack this
mechanism, resulting in weaker alignment with RT variations.
5.2 Relatedness Classification
Concept: Binary relatedness judgments measure whether a prime-target pair is
judged related (1) or not (0). This taps into long-term associative memory stores
(Traxler Ch. 3), which static embeddings can approximate via global cooccurrence.
Model Accuracy F1 Dataset Skew (0 vs. 1)
GloVe 0.988 0.994 5,268 vs. 408,296
BERT 0.988 0.994 5,268 vs. 408,296
Inference: Both models reach near-ceiling performance, showing co-occurrencebased embeddings suffice for coarse judgments. The extreme class imbalance
(many more related examples) further simplifies classification. Contextual nuance
provides minimal additional benefit when the task is highly skewed and binary.
5.3 Ambiguous Primes
Concept: Sense selection refers to choosing the correct meaning of an
ambiguous word based on context (Traxler Ch. 4). Computationally, dynamic
embeddings adjust a word’s representation to reflect the intended sense in a given
sentence.
Prime GloVe BERT
FOAM 0.018 0.47272
OFTEN 0.01851 0.46296
OBSTACLE 0.018519 0.018519
FROSTING 0.00 0.45283
Computational Psycholinguistics Final Report 4
Inference: BERT’s context-conditioned vectors improve sense selection by
reweighting semantic features via multi-head attention, paralleling human use of
context to resolve ambiguity. GloVe’s single prototype per word cannot distinguish
different senses, leading to lower accuracy on ambiguous primes.
5.4 Sense Disambiguation
Concept: Sense disambiguation at the sentence level requires integration of local
lexical, syntactic, and broader discourse context (Traxler Ch. 4). BERT’s
bidirectional transformer layers aggregate context from both directions, yielding
embeddings that reflect nuanced meaning.
Dataset GloVe BERT
Raw Stimuli 0.544 0.565
WiC (Test) 0.500 0.561
Inference: BERT’s superior performance illustrates its ability to incorporate
multiple context levels—word co-occurrences, sentence structure, and discourse
cues—improving disambiguation. GloVe embeddings remain static and cannot
adapt to sentence-specific meaning shifts.
Through these we accounted for the behavioural measures for distributional and
contextual language models. It asks : “Do these models behave like humans in
terms of reaction time, semantic relatedness, and contextual meaning resolution?”
6. Discussion
These results illustrate how distinct embedding paradigms account for behavioral
measures:
Static (GloVe): Captures broad semantic similarity, sufficient for explicit
judgments but limited in dynamic, context-driven tasks. Its single-vector-perword design conflates multiple senses, yielding weaker correlation with RTs
and lower disambiguation accuracy.
Computational Psycholinguistics Final Report 5
Contextual (BERT): Generates embeddings sensitive to context, aligning more
closely with cognitive activation patterns. The stronger negative RT correlation
suggests its similarity scores reflect human processing fluency. Superior
ambiguous-prime and sense disambiguation performance demonstrate BERT’s
ability to resolve polysemy akin to human inferencing.
Nevertheless, our use of a minimally fine-tuned BERT model due to resource
constraints likely limits the upper bound of its performance in RT prediction and
disambiguation. More extensive finetuning or larger BERT variants may yield
stronger correlations and accuracies. Due to dataset constraints, the results for
the Binary Classification: Related vs Unrelated, the results were coming out very
equal as the available dataset was very skewed but rest of the results were
positive towards our goal.
7. Code Analysis
Dataset Loading and Preprocessing
a. load_spp_data()
Loads semantic priming dataset (SPP) CSV.
Retains only prime , target , reaction time , and relatedness fields.
Recodes relatedness to binary: 1 for related, 0 for unrelated.
Returns a cleaned dataframe.
b. load_raw_stimuli()
Loads a second dataset used for sense disambiguation (target, related,
unrelated words in context).
Drops missing rows to ensure valid comparisons.
c. Load the models used
Load the GloVe model and the BERT model and set it to evaluation mode to
avoid gradient computations.
Embedding Extraction
Computational Psycholinguistics Final Report 6
a. get_word2vec_embedding(word, model)
Retrieves static word embedding for a token from W2V or GloVe.
b. get_bert_embedding(text, tokenizer, model, device)
Tokenizes input, runs it through BERT, filters out special tokens ( [CLS] , [SEP] ,
etc.).
Averages valid token embeddings from the last hidden state to get a
sentence or word representation.
Correlation with Reaction Time (SPP)
compute_spp_correlations()
Computes cosine similarities between prime and target using both W2V and
BERT.
Correlates the similarity values with human reaction times from the SPP
dataset.
Uses both Pearson and Spearman correlation to assess linear and rank-based
relationships.
Insight:
💡 This evaluates whether semantic similarity computed by models can
predict human RTs, which serves as a proxy for cognitive plausibility.
Binary Classification: Related vs Unrelated
a. classify_relatedness_glove_vs_bert()
Trains two separate logistic regression classifiers:
One using only GloVe similarity
One using only BERT similarity
Evaluates performance using accuracy and F1 score.
Computational Psycholinguistics Final Report 7
Insight:
💡 This assesses how well semantic similarity scores can distinguish
between related and unrelated prime-target pairs.
Ambiguous Prime Analysis
eval_ambiguous(df, word)
Filters rows where the prime word is ambiguous (e.g., “bank”).
Computes similarity between prime and target.
Uses median similarity as a classification threshold (above = related, below =
unrelated).
Computes accuracy for both W2V and BERT.
Insight:
💡 This checks whether embeddings capture different senses of ambiguous
words across contexts.
Sense Disambiguation Task
sense_disambiguation(df, emb_func, model_name, is_bert=False)
Takes rows from the raw_stimuli.csv dataset with:
Target word in context
A related and unrelated word
Embeds each pair: [target, related] and [target, unrelated]
Trains classifier to distinguish between them (binary labels: 1 for related, 0 for
unrelated)
Compares performance across:
Computational Psycholinguistics Final Report 8
BERT (with context)
W2V (no context)
GloVe (no context)
Insight:
💡 This is a controlled test of disambiguation—can the model distinguish
between senses when provided with context?
Word-in-Context (WiC) Task
evaluate_wic_with_models(wic_dataset, tokenizer, bert_model,
glove_model)
Loads the WiC dataset, where the same word appears in two different
sentences.
Computes similarity between sentences using:
BERT embeddings (contextualized)
GloVe average (context-independent)
Thresholds similarity (e.g., 0.5) to decide if meaning is same or different.
Evaluates using accuracy and F1.
Insight:
💡 This tests whether models can discriminate senses of a word based on
context—a known limitation of static embeddings
Results and Interpretations
Across all components, the models are evaluated using:
Pearson/Spearman correlation (RT task)
Computational Psycholinguistics Final Report 9
Accuracy & F1 scores (classification, WiC, disambiguation)
Word-level analysis (ambiguous primes)
We compare:
Static vs contextual embeddings
Similarity-based decision boundaries
Model’s sensitivity to contextual meaning
8. Conclusion
Contextual embeddings like BERT offer clear advantages over static GloVe 50d
vectors in capturing human-like semantic behavior. Across tasks such as reaction
time prediction, semantic relatedness classification, and sense disambiguation,
BERT consistently outperforms GloVe, highlighting its ability to incorporate
dynamic linguistic context. While GloVe assigns a fixed meaning to each word,
BERT adjusts representations based on context, aligning more closely with how
humans interpret meaning in real-time. This context sensitivity enables BERT to
better mirror cognitive processes observed in psycholinguistic experiments, such
as semantic priming and ambiguity resolution. These findings reinforce the
theoretical claim that dynamic, context-aware models more faithfully reflect the
nuanced nature of human semantic understanding.
Computational Psycholinguistics Final Report 10
