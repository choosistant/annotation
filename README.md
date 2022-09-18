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

Setup Label Studio:

```bash
docker run -it -p 8080:8080 -e EXPERIMENTAL_FEATURES=1 -v `pwd`/data/label-studio:/label-studio/data heartexlabs/label-studio:latest
```
