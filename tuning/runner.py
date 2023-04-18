import sys

from io import StringIO
from logger import Logger

def trial(config):
    missing = (
        {"dataset", "lr", "weight_decay", "batch_size", "epochs", "trial"}
        - config.keys(),
    )
    assert not missing, f"mssing keys: {missing}"

    logger = Logger()
