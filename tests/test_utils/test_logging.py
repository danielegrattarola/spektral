import shutil

from spektral.models import GCN
from spektral.utils import logging


def test_logging_functions():
    log_dir = logging.init_logging()
    logging.log("test")
    logging.tic(message="test")
    logging.toc(message="test")

    model = GCN(1)
    model.build([(10, 2), (10, 10)])
    logging.model_to_str(model)

    shutil.rmtree(log_dir)
