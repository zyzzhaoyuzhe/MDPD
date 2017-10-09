The web search data contains 5 classes denoted by 1-5. You can see a detailed description of the dataset in our NIPS 12 paper (section 6.2) and ICML 14 paper: 

D. Zhou, J. C. Platt, S. Basu, and Y. Mao. Learning from the Wisdom of Crowds by Minimax Entropy. Advances in Neural Information Processing Systems (NIPS) 25, 2204-2212, 2012.  

D. Zhou, Q. Liu, J. C. Platt, and C. Meek.  Aggregating Ordinal Labels from Crowds by Minimax Conditional Entropy. Proceedings of the 31st International Conference on Machine Learning (ICML), 2014. 

The file in “train-query-doc-ML” contains 3 columns:  item ID, worker ID, worker label 

The file in “test-query-doc-ML”  contains 2 columns:  item ID, true label 

Some items in the train file may not be in the test file and verse vise. 

Run your algorithm with the train file and evaluate the results with the test file. 
