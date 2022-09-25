# Annotation

Download Amazon data:

```bash
poetry run download-amazon-data \
    --meta-data-path data/amazon-review-data-set.json \
    --categories "all" \
    --output-dir data/raw/amazon-data-2018 \
    --verbose
```

Sample review data:

```bash
poetry run sample-amazon-data \
    --input-path data/raw/amazon-data-2018/reviews-all-beauty-5269.json.gz \
    --output-dir data/processed/amazon-data-2018/ \
    --min-word-count 5 \
    --beta 0.4 \
    --n-items 5 \
    --n-positive-reviews 2 \
    --n-negative-reviews 2
```

Setup Label Studio:

```bash
docker run -it -p 8080:8080 -e EXPERIMENTAL_FEATURES=1 -v `pwd`/data/label-studio:/label-studio/data heartexlabs/label-studio:latest
```
