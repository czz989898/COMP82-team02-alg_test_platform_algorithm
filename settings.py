REQUIRED_SETTINGS = ["min_n", "max_n", "max_feats", "ngram_type"]
LIBRARY_METRICS = ["Cosine", "Manhattan"]   # The metrics using library funcs

settings = {
    "tokenisation": {
        "min_n": 3,
        "max_n": 4,
        "max_feats": 1000,
        "ngram_type": "char_wb",
    }
}
