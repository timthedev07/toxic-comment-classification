
def custom_standardization(text: tf.Tensor):
    # to lower case
    text = tf.strings.lower(text)
    # expand contraction
    pairs = [
        ("won't", "will not"),
        ("can't", "can not"),
        ("n't", " not"),
        ("'re", " are"),
        ("'s", " is"),
        ("'d", " would"),
        ("'ll", " will"),
        ("'t", " not"),
        ("'ve", " have"),
        ("'m", " am"),
    ]
    for contracted, replacement in pairs:
        text = tf.strings.regex_replace(text, contracted, replacement)

    # clean special symbols
    text = tf.strings.regex_replace(text, "<br />", " ")
    text = tf.strings.regex_replace(
        text, r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", " ")
    text = tf.strings.regex_replace(text, r'@([A-Za-z0-9_]+)', " ")
    text = tf.strings.regex_replace(text, r"\([^)]*\)", " ")
    text = tf.strings.regex_replace(text, r"[^A-Za-z0-9]+", " ")

    # remove stopwords
    for i in stop_words:
        text = tf.strings.regex_replace(
            text, f"[^A-Za-z0-9_]+{i}[^A-Za-z0-9_]+", " ")

    return text
