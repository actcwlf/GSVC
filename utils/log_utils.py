import os
import uuid

from loguru import logger
from argparse import ArgumentParser, Namespace

from arguments import PipelineParams


def prepare_output_and_logger(args, pipe: PipelineParams):
    try:
        from torch.utils.tensorboard import SummaryWriter
        TENSORBOARD_FOUND = True
        logger.info("found tf board")
    except ImportError:
        TENSORBOARD_FOUND = False
        logger.info("not found tf board")


    if not pipe.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        pipe.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    logger.info("Output folder: {}".format(pipe.model_path))
    os.makedirs(pipe.model_path, exist_ok = True)
    with open(os.path.join(pipe.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(pipe.model_path)
    else:
        logger.info("Tensorboard not available: not logging progress")
    return tb_writer