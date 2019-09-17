# Unsupervised-Document-Clustering

* The project aims to cluster the collected web tables to create data for supervised learning and to improve search performance.

* A Billion web tables was collected and stored in the MongoDB in JSON format.

* These web tables were processed by flattening the JSON, removing stop words and performing lemmatization using SpaCy and Natural Language Processing Toolkit.

* Similar words were then clustered to reduce the complexity of the data using Word2Vec and DBSCAN algorithms.

* Finally the processed web tables were clustered to find the similar web tables using TF-IDF and DBSCAN algorithms.

* The optimal parameter for Word2Vec, DBSCAN and TF-IDF was found using the Randomized Searching Algorithm.

* We also Developed a Heterogeneity based score function to score the DBSCAN clusters and find the optimal parameters.
