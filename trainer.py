import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from utils.metrics_logger import IMetrics_logger
from utils.tensorboard_logger import Tensorboard_Logger
import os


def train(args):
    seed_list = copy.deepcopy(args.seed)
    device = copy.deepcopy(args.device)

    for seed in seed_list:
        args.seed = seed
        args.device = device
        _train(args)


def _train(args):
    try:
        os.mkdir("logs/{}".format(args.model_name))
    except:
        pass
    expName = '{}_{}_{}_{}_{}_{}_{}'.format(args.prefix, args.seed, args.model_name, args.convnet_type,
                                            args.dataset, args.init_cls, args.cls_per_increment)
    logfilename = 'logs/{}/{}'.format(args.model_name, expName)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    _set_random()
    _set_device(args)
    print_args(args)


    data_manager = DataManager(args.dataset, args.shuffle, args.seed, args.init_cls, args.cls_per_increment)
    vars(args).update({'nb_tasks' : data_manager.nb_tasks})

    Tb_logger = Tensorboard_Logger('logs/tensorboard', expName) if args.tb else None
    ordered_classes_names = data_manager._classes_names[data_manager._class_order]
    metricsLog = IMetrics_logger(args, Tb_logger, ordered_classes_names)
    #add metricsLog to args for within epochs metrics
    vars(args).update({'metricsLogger' : metricsLog})

    model = factory.get_model(args.model_name, args)
    Tb_logger.setup_writer()

    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        #setup new task and train
        model.incremental_train(data_manager)
        #evaluate model
        preds_dict = model.eval_task(eval_nem=True)
        #compute metrics
        metricsLog.info("iStep_eval", **preds_dict)
        #prepare for next task
        model.after_task()
        #log iStep metrics
        metricsLog.info("iStep_end")


def _set_device(args):
    device_type = args.device
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args.device = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def print_args(args):
    for key, value in vars(args).items():
        logging.info('{}: {}'.format(key, value))
