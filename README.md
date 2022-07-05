# BERT Content Based Filtering Recommender
This project begun from DataVoucher, K-data the government org.  
I noticed that in ordinary content based filtering, the metas of item are independent so can't be correlated together.  
So just tried to entangle metas like `"meta1 and meta2 and ..."` and let the BERT to encode.  
At first, I used [CLS] token to compute similarity, but it broght rarely different results with `pooled output`.  
It's just idea, not for real service application.

--- 
## Requirements
```
python == 3.9.12  
transformers from hugging faces  
pandas  
numpy  
```
Can also see in `requirements.txt`

## Data
Actual project data are in NDA. so I used alternative data `ml-100k`

**ml-100k** : [Download Link](https://grouplens.org/datasets/movielens/100k/)

## Data Process
Transformed text to csv using `data_process.py`

## Similarity compute
Using processed data, computed correlations(Cosine similarity) and stored

## main
Recommend top-k items from correlation csv
