# BERT Recommender System (This repository is for portfolio)
## Project Description
This project begun from DataVoucher, K-data the government org.  
I noticed that what can represent to vector can mearsure similarity geometrically.   
Therefore I just let the BERT to encode product's description.  


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
Actual project data are in NDA. so I used alternative data `proudct_description.csv` the fake data using `utils/make_random_data.py`

## Similarity computation
Cosine Similarity
