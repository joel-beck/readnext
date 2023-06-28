from transformers import logging


def suppress_transformers_logging() -> None:
    """
    Suppresses the console message `Some weights of the model checkpoint at
    bert-base-uncased were not used when initializing BertModel...`
    """
    logging.set_verbosity_error()
