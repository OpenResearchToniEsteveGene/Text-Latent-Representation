# Vector Representations of Text in Classification Problems

## Introduction

This repository contains the code developed in the Bachelor’s Thesis in Mathematics devoted to the study of several text vectorization techniques for classification tasks. The main objective of the thesis is to formalize and mathematically analyze the process of **vector representation of textual documents**, comparing three fundamental methods: **TF-IDF**, **LSA**, and **LDA**. In addition to its theoretical foundation, the project applies these representations to several text datasets (spam emails, Twitter messages, and the *20 Newsgroups* collection) in order to evaluate their effectiveness in binary classification through supervised models.

Through this combined approach, the thesis provides a comparison between a model based on the original vocabulary (TF-IDF) and two latent or “thematic” models that reduce dimensionality (LSA and LDA). The available code makes it possible to reproduce the experiments conducted, including text preprocessing, vector transformation, and the training and evaluation of classifiers for each representation.

## Vector Representations Used

This section presents, in a clear and formal manner, the vector representation techniques implemented in the thesis. Each method converts text documents into numerical vectors by following different approaches, either based on the original words or on latent topics inferred from the texts.

### TF-IDF (Term Frequency - Inverse Document Frequency)

**TF-IDF** is a classical text representation technique that assigns each term a numerical weight proportional to its importance in a specific document and in the overall corpus. This importance is computed as the product of two components:

* **Term Frequency (TF):** measures the recurrence of term $t$ within document $d$:

$$
tf(t,d) = \frac{f(t,d)}{\max {f(t',d) \mid t' \in d}}
$$

where ( f(t,d) ) is the frequency of term $t$ in document $d$, and the denominator is the maximum frequency among all terms in the document.

* **Inverse Document Frequency (IDF):** measures the rarity of the term in the collection of documents $D$:

$$
\mathrm{idf}(t, D) = \log\left( \frac{|D|}{ |{ d \in D : t \in d }|} \right)
$$

where $|D|$ is the total number of documents, and the denominator is the number of documents in which term $t$ appears. The value 1 is often added in practice to avoid divisions by zero.

* **TF-IDF Weight:** is the product of the two previous components:

$$
\mathrm{tf\text{-}idf}(t, d) = \mathrm{tf}(t, d) \cdot \mathrm{idf}(t, D)
$$

In this way, highly specific terms (frequent in one document but infrequent in the overall corpus) receive a high weight. By contrast, terms that are very common across all documents are penalized.

Each document is represented as a vector in a space whose dimensionality is equal to the size of the vocabulary, where each component of the vector is the TF-IDF value associated with a term. This representation preserves the original granularity of the text and is particularly suitable for classification algorithms capable of handling high-dimensional spaces.

### LSA (Latent Semantic Analysis via Truncated SVD)

**LSA**, or *Latent Semantic Analysis*, is an algebraic method that makes it possible to obtain latent and more compact representations of textual documents. The process begins by constructing a term-document matrix $A \in \mathbb{R}^{m \times n}$, where each row represents a term, each column represents a document, and the entries are weights (typically TF-IDF values) that reflect the importance of the term in the document.

Next, a **Singular Value Decomposition (SVD)** is applied to the matrix $A$:

$$
A = U \Sigma V^T
$$

where:

* $U \in \mathbb{R}^{m \times r}$ contains the left singular vectors (associated with terms),
* $\Sigma \in \mathbb{R}^{r \times r}$ is a diagonal matrix containing the nonnegative singular values in descending order,
* $V^T \in \mathbb{R}^{r \times n}$ contains the right singular vectors (associated with documents),
* and $r$ is the rank of the original matrix $A$.

To reduce dimensionality, this decomposition is truncated by retaining only the first $k$ components, with $k \ll r$:

$$
A_k = U_k \Sigma_k V_k^T
$$

According to the **Eckart–Young theorem**, $A_k$ is the best rank-$k$ approximation to the matrix $A$ in the Frobenius norm, and it preserves the principal semantic structures of the corpus.

In this latent space of dimension $k$:

* Each document is expressed as $\hat{d}_i = \Sigma_k^{-1} U_k^T d_i$, where $d_i$ is the document representation, typically based on TF-IDF weights.

This new representation projects documents and terms into a reduced semantic space, where documents with similar meaning, even if they do not share literal words, are located closer to one another. Thus, **LSA captures latent and synonymous relationships between words and documents**, providing more compact vector representations that may improve computational efficiency and model generalization.

### LDA (Latent Dirichlet Allocation)

**LDA**, or *Latent Dirichlet Allocation*, is a probabilistic generative model that represents documents through latent topics. The fundamental idea of the model is to assume that each document $d$ is generated by a combination of $k$ latent topics, and that each topic $z_k$ is a probability distribution over the words in the vocabulary.

The model assumes the following generative process:

1. For each topic $k = 1, \dots, K$, a word distribution $\phi^k \sim \text{Dir}(\beta)$ is generated, where $\phi^k \in \mathbb{R}^V$ and $V$ is the size of the vocabulary.
2. For each document $d$:

   * A topic distribution $\theta_d \sim \text{Dir}(\alpha)$ is generated, where $\theta_d \in \mathbb{R}^K$.
   * For each word $w_n$ in document $d$:

     * A latent topic $z_n \sim \text{Multinomial}(\theta_d)$ is selected,
     * A word $w_n \sim \text{Multinomial}(\phi^{z_n})$ is then selected.

In summary, the **latent variables** are:

* $\theta_d$: vector of topic probabilities for document $d$,
* $z_n$: topic assignment for the ( n )-th word of document $d$,
* $\phi^k$: word distribution for topic $k$.

During the inference process, for example through *Gibbs sampling* or *variational methods*, the posterior distributions $p(\theta_d \mid w_d)$ and $p(\phi^k \mid w_d)$ are estimated.

Therefore, the final representation of a document is the vector $\theta_d \in \mathbb{R}^K$, which expresses the estimated proportion of each latent topic in the document. This **thematic representation** reduces the dimensionality of the document with respect to the original word space and captures **global semantic structures** of the corpus.

---

**Note:** Both **LSA** and **LDA** are dimensionality reduction methods that extract semantic factors underlying the documents. LSA uses algebraic techniques such as truncated SVD, whereas LDA uses a probabilistic generative model based on Dirichlet distributions. By contrast, **TF-IDF** represents documents directly from the observable vocabulary, without considering latent topics. According to the results obtained in this thesis, although **TF-IDF** usually provides the best predictive performance, **LSA** and **LDA** offer more compact and semantically interpretable representations, which are especially useful in datasets with redundancy or high lexical correlation.

## CODE INSTRUCTIONS

### System Requirements

* **Python 3.8+**: It is recommended to have a recent version of Python installed (3.8 or later).
* **Python libraries**: The following dependencies must be installed, with versions equivalent to those used in the thesis:

  * pandas (data manipulation)
  * numpy (numerical operations)
  * scikit-learn (TF-IDF vectorization, SVD, LDA, metrics, and validation)
  * NLTK (text processing: tokenization and lemmatization; ensure the availability of resources such as *punkt* and *wordnet*)
  * xgboost (implementation of the Gradient Boosting classifier)
* **Other requirements**: No special hardware is required; execution is feasible on CPU. An Internet connection is needed the first time some scripts are executed in order to download data:

  * The SMS *spam* dataset is loaded directly from a public URL.
  * Some *NLTK* resources, such as the tokenization model, will be automatically downloaded when `nltk.download(...)` is called within the scripts.

### Step-by-step execution of the scripts

1. **Prepare the environment:** Download or clone this repository onto your computer. Make sure that all the required packages mentioned in the requirements section have been installed. If necessary, use `pip install` to install the dependencies.
2. **Run the experiments with the *Spam Dataset*:** From the command line, navigate to the `spam/` directory and execute the three scripts in that folder:

   * `TFIDF.py`
   * `LSA.py`
   * `LDA.py`

   These scripts will automatically load the SMS spam dataset from a public URL, preprocess the messages by removing emojis, applying lemmatization, and so forth, and then train a classification model to evaluate the representation. Each script performs 5-fold cross-validation by training a Gradient Boosting model (XGBoost) and displaying on screen the AUC-ROC metric obtained for the different folds.
3. **Run the experiments with the *Twitter Dataset*:** Before executing these scripts, make sure that the data file `data/twitter_redut_Dataset.csv` is accessible. Then, from the command line, execute the three scripts in the `twitter/` directory:

   * `TFIDF.py`
   * `LSA.py`
   * `LDA.py`

   Each of these scripts reads the CSV file containing tweets, applies the corresponding vectorization method (TF-IDF, LSA with truncated SVD, or LDA with a number of topics $k$ specified in the code), and trains a supervised classifier, primarily **Logistic Regression** in these experiments. Likewise, stratified 5-fold cross-validation is performed, and the mean **AUC-ROC** is computed to compare the performance of the different methods.
4. **Run the experiments with the *20 Newsgroups Dataset*:** From the `newsgroups/` directory, execute the following scripts:

   * `TFIDF.py`
   * `LSA.py`
   * `LDA.py`

   These scripts will download the *20 Newsgroups* corpus, if it is not already cached, through *scikit-learn*. The problem will be converted into a binary classification task by defining a target label, for example, identifying documents related to sports topics. Each script will generate the corresponding vector representation of the documents, whether TF-IDF, SVD-based reduction to latent components, or topic distributions through LDA, and will then train a classification model, specifically **XGBoost** for this dataset. Finally, the mean AUC-ROC obtained under cross-validation will be displayed.
5. **Analysis of results:** Once all scripts have been executed, the AUC-ROC metrics may be compared in order to determine which vector representation performs best on each dataset. In general, it is observed that the TF-IDF-based representation tends to provide the best predictive performance, although the latent representations, namely LSA and LDA, achieve comparable results while offering the advantage of reducing dimensionality and capturing latent semantic relationships.

### References
 - Blei, DM, AYNgiMIJordan(2003). “Latent Dirichlet Allocation”. A: Journal of Machine Learning Research 3.
 - Deerwester, Scott et al. (1990). “Indexing by latent semantic analysis”. A: Journal of the American society for information science 41.6, pàg. 391-407.
 -  Valle-Lisboa, Juan C i Eduardo Mizraji (2007). “The uncovering of hidden structures by latent semantic analysis”. A: Information sciences 177.19, pàg. 4122-4147.
