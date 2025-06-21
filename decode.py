import os
import sys
# from argparse import ArgumentParser
from loguru import logger

import numpy as np
import torch
# import wandb
from simple_parsing import ArgumentParser


from arguments import ModelParams, PipelineParams, OptimizationParams
from pipeline.eval import eval_model
from utils.general_utils import safe_state

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(add_config_path_arg=True)
    # lp = ModelParams(parser)
    # op = OptimizationParams(parser)
    # pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--tag", type=str, default='')


    parser.add_arguments(ModelParams, dest="model")
    parser.add_arguments(PipelineParams, dest="pipeline")
    parser.add_arguments(OptimizationParams, dest="optimization")

    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.optimization.iterations)

    lp: ModelParams = args.model
    pp: PipelineParams = args.pipeline
    op: OptimizationParams = args.optimization


    model_path = pp.model_path
    os.makedirs(model_path, exist_ok=True)

    # logger = get_logger(model_path)
    #
    # logger = loguru.
    logger.add(model_path + '/decode.log')


    # rendering
    # logger.info(f'\nStarting Rendering~')
    # visible_count = render_sets(args, lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger, x_bound_min=x_bound_min, x_bound_max=x_bound_max)
    # logger.info("\nRendering complete.")
    #
    # # calc metrics
    # logger.info("\n Starting evaluation...")
    # evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    # logger.info("\nEvaluating complete.")
    logger.info("Decoding " + pp.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    args.port = np.random.randint(10000, 20000)
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)



    eval_model(
        model_params=lp,
        opt=op,
        pipe=pp,
        # dataset_name=dataset,
        checkpoint=args.checkpoint,
    )
