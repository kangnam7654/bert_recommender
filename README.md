# BERT Content Based Filtering Recommender
This project begun from DataVoucher, K-data the government org.  
I noticed that in ordinary content based filtering, the features of items unconvenient.  
So just tried to entangle metas like `"meta1 and meta2 and ..."` and let the BERT to encode.  
It's just idea, not for real service application.

--- 
## Requirements
```
python  
transformers from hugging faces  
pandas
torch 
tqdm  
```
Can also see in `requirements.txt`

## Data
Actual project data are in NDA. so I used alternative data `proudct_description.csv` that fake data

## Similarity compute
Using processed data, computed correlations(Cosine similarity) and stored
