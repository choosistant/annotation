# Annotation

Download Amazon data:

```bash
poetry run download-amazon-data \
    --url http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz \
    --output-dir data/raw \
    --verbose
```

Sample review data:

```bash
poetry run sample-amazon-data \
    --input-path  data/raw/reviews_Musical_Instruments_5.json.gz \
    --output-dir data/processed \
    --min-word-count 5 \
    --beta 0.4 \
    --n-items 5 \
    --n-positive-reviews 2 \
    --n-negative-reviews 2 \
    --random-seed 68792281
```
