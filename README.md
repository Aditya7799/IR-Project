# IR-Project
IR Project Code Semester 7

## Requirements
- Scipy - for sparse matrix manipulation/computation
- Numpy 
- gensim - importing word embeddings
- scikit_learn - test-train split and performance metrics (acc,f1 etc)

## Word Embeddings need to be put in one folder outside this folder
Download *GoogleNews-vectors-negative300.bin.gz* - https://github.com/mmihaltz/word2vec-GoogleNews-vectors


## Need to run ConstructandSave() to constrcut the Word-to-Word similarity Matirx.
- Dictionary stored as "dataset_name.pickle"
- W2W similarity matrix saved as "dataset_name_Cosine.npy" (numpy 2D matrix)


## Constructed W2W and Dictionary will be used for Document Classification