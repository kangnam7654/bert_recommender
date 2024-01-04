# BERT Recommender System (Portfolio Repository)
## Project Description
This project began with the Data Voucher program of K-data, a government organization. I realized that if product descriptions are represented as vectors, their similarity can be measured geometrically. Therefore, I used the BERT model to encode the descriptions of products.

---
## Requirements
- Python
- Transformers (Hugging Face)
- Pandas
- Torch
- Tqdm

The complete list can also be found in `requirements.txt`.

## Installation
```bash
pip install -r requirements.txt
```

## Data
Due to NDA (Non-Disclosure Agreement), the actual project data cannot be shared. As an alternative, I used `product_description.csv`, a mock dataset created using `utils/make_random_data.py`.

The `product_description.csv` file has the following format:
* Column 1: Product ID
* Column 2: Product Description

## Similarity Computation
Similarity is calculated using **Cosine Similarity**, which measures the cosine of the angle between two vectors in a multi-dimensional space.

## Calculating cosine similarity between vectors
Used torch's
```python
sim = torch.cosine_similarity(vector1, vector2)
```

## Contributing
Contributions to this project are welcome! Feel free to submit Pull Requests or Issues for participation.

## Contact and Support
For questions or support, please contact kkangnam7654@gmail.com.
