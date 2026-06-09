import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from util.fix_env import fix_openmp_thread_env

fix_openmp_thread_env()

import argparse
import functools
import math
import pdb
import random
import traceback
from copy import deepcopy

import ignite
import ignite.utils as ignite_utils
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss
from ignite.metrics.running_average import RunningAverage
from torch.backends import cudnn

import extensions

from evaluator.classification import ClassifierEvaluator
from evaluator.contrastive import ContrastiveEvaluator
from util import train_util, yaml_utils


# 北斗
# 将 model 目录添加到 sys.path
sys.path.append('/root/autodl-tmp/Post_pre_training/data')
sys.path.append('/root/autodl-tmp/Post_pre_training/model')
# sys.path.append('/root/autodl-tmp/Post_pre_training/evaluator')
sys.path.append('/root/autodl-tmp/Post_pre_training/updater')


def log_basic_info(logger, config):
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        logger.info(f"- GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")
    logger.info("--------------")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("--------------")


def _build_paper_test_loader(logger, config, model):
    """
    目的：构建“检索数据集(dataset)”的 test split loader，用于论文三项指标(11-13)的最终评估。
    注意：这不是 dataset_cls 的 test_loader（分类任务用）。
    """
    try:
        cfg_test = deepcopy(config)

        # 复用与训练一致的 preprocess/tokenizer
        cfg_test["dataset"]["args"]["transform"] = model.module.img_preprocess
        cfg_test["dataset"]["args"]["tokenizer"] = model.module.tokenizer

        # 强制切到 test split
        cfg_test["dataset"]["args"]["test"] = True

        # 兼容不同实现：可能返回 (train, val) 或 (train, val, test) 或 (train, test)
        ret = train_util.setup_pretraining_dataloaders(cfg_test)

        paper_test_loader = None
        if isinstance(ret, (list, tuple)):
            if len(ret) == 3:
                paper_test_loader = ret[2]
            elif len(ret) == 2:
                # 绝大多数项目会把 “第二个” 当作 val/test 之类的评估 loader
                paper_test_loader = ret[1]
            else:
                paper_test_loader = None

        if paper_test_loader is None:
            logger.warning("[PaperMetrics] setup_pretraining_dataloaders(test=True) returned unexpected format; fallback to val_loader.")
        else:
            logger.info("[PaperMetrics] Built retrieval test loader via dataset.args.test=True.")
        return paper_test_loader

    except Exception as e:
        logger.warning(f"[PaperMetrics] Failed to build retrieval test loader: {e}")
        logger.warning("[PaperMetrics] Will fallback to val_loader when computing paper metrics.")
        return None


def main(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = "cuda" if (torch.cuda.is_available()) else "cpu"

    logger = ignite_utils.setup_logger(name=config["pattern"])
    log_basic_info(logger, config)

    # Create output directory
    logger.info("Creating output directory")
    out = os.path.join(config["results_dir"], config["pattern"], f"experiment{config['experiment_id']}")
    train_util.create_result_dir(out, config["config_path"])

    # Model
    logger.info("Constructing models")
    model = train_util.load_models(config["models"]["model"])
    model = torch.nn.DataParallel(model)
    model.to(device)

    # DataLoader
    logger.info("Constructing data loaders")
    config["dataset"]["args"]["transform"] = model.module.img_preprocess
    config["dataset"]["args"]["tokenizer"] = model.module.tokenizer
    train_loader, val_loader = train_util.setup_pretraining_dataloaders(config)

    config["dataset_cls"]["args"]["transform"] = model.module.img_preprocess
    test_loader = train_util.setup_test_dataloader(config)  # 这个是分类 evaluator 用的（dataset_cls）

    max_iter = int(len(train_loader) * config["epoch"])

    # ✅ 额外：为论文三指标构建“检索数据集”的 test_loader（优先），失败则后面自动用 val_loader 兜底
    paper_test_loader = _build_paper_test_loader(logger, config, model)

    # Optimizer
    logger.info("Constructing optimizers")
    opt = train_util.make_optimizer(model, config["optimizer"])

    if config["resume"]:
        logger.info("Resume training with snapshot:{}".format(config["resume"]))
        if os.path.isfile(config["resume"]):
            checkpoint = torch.load(config["resume"])
            model.module.load_state_dict(checkpoint["model"])
            opt.load_state_dict(checkpoint["optimizer"])
            resume_step = checkpoint["optimizer"]["state"][0]["step"]

    # Updater
    logger.info("Constructing updater and evaluators")
    kwargs = config["updater"]["args"] if "args" in config["updater"] else {}
    kwargs.update(
        {
            "model": model,
            "optimizer": opt,
            "device": device,
            "train_loader": train_loader,
            "max_iteration": max_iter,
        }
    )
    updater_cls = yaml_utils.load_updater_class(config)
    updater = updater_cls(**kwargs)

    # Trainer := Ignite.Engine
    trainer = Engine(updater)
    monitoring_metrics = ["train_loss", "train_feat_gap"]

    RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "train_loss")
    RunningAverage(output_transform=lambda x: x["feat_gap"]).attach(trainer, "train_feat_gap")
    if "log_metrics" in config:
        for key in config["log_metrics"]:
            monitoring_metrics.append(key)
            RunningAverage(output_transform=lambda x, key=key: x[key]).attach(trainer, key)

    logger.info(f"Monitoring Metrics: {monitoring_metrics}")
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    # Contrastive Evaluator
    cont_evaluator = Engine(ContrastiveEvaluator(model=model, device=device))
    RunningAverage(output_transform=lambda x: x["loss"]).attach(cont_evaluator, "val_loss")
    RunningAverage(output_transform=lambda x: x["feat_gap"]).attach(cont_evaluator, "val_gap")
    RunningAverage(output_transform=lambda x: x["alignment"]).attach(cont_evaluator, "val_alignment")
    RunningAverage(output_transform=lambda x: x["uniformity"]).attach(cont_evaluator, "val_uniformity")

    # Classification Evaluator
    cls_evaluator = Engine(
        ClassifierEvaluator(
            model=model,
            device=device,
            classname_file=config["dataset_cls"]["classname_file"],
            tokenizer=model.module.tokenizer,
        )
    )
    RunningAverage(output_transform=lambda x: x["accuracy"]).attach(cls_evaluator, "accuracy")
    RunningAverage(output_transform=lambda x: x["loss"]).attach(cls_evaluator, "loss")

    # Event Handlers
    logger.info("Constructing event handlers")
    log = {"running": [{"epoch": "init"}], "best_val_loss": 10000000000000.0}
    log = log if not config["resume"] else extensions.load_log(out)

    logger_train = functools.partial(extensions.log_training_results_mps, log=log, pbar=pbar)
    logger_val = functools.partial(
        extensions.log_pretraining_validation_results,
        cont_evaluator=cont_evaluator,
        val_loader=val_loader,
        cls_evaluator=cls_evaluator,
        test_loader=test_loader,  # 分类用
        log=log,
        pbar=pbar,
        dist=str(out),
    )

    # Check Point Handler
    check_pointer = ModelCheckpoint(str(out), filename_prefix="model", n_saved=1, require_empty=not config["resume"])
    best_check_pointer = ModelCheckpoint(
        str(out),
        filename_prefix="best",
        score_function=extensions.check_loss,
        n_saved=1,
        score_name="val_loss",
        require_empty=not config["resume"],
    )

    # Learning Rate Schedule Handler
    lr_scheduler = train_util.set_learning_rate_scheduler(trainer, opt, config["optimizer"], max_iter)

    if config["resume"]:
        lr_scheduler.event_index = checkpoint["lr_scheduler"]["event_index"]

    # Append handlers to trainer/evaluator engine
    trainer.add_event_handler(Events.EPOCH_COMPLETED, logger_train)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, logger_val)

    cont_evaluator.add_event_handler(
        Events.COMPLETED,
        best_check_pointer,
        {
            "model": model,
        },
    )

    if config["resume"]:
        config["start_epoch"] = math.ceil(resume_step * config["batchsize"] / len(train_loader.dataset))
        resumer = functools.partial(extensions.resume_training, resume_epoch=config["start_epoch"])
        trainer.add_event_handler(Events.STARTED, resumer)

    if config["evaluate"]:
        trainer.add_event_handler(Events.STARTED, logger_val)

    # Run the training
    logger.info("Running train script")
    try:
        trainer.run(train_loader, max_epochs=config["epoch"])
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
    finally:
        # ✅ 训练结束后：按论文公式计算三项指标，只执行一次（优先 paper_test_loader，失败则 val_loader）
        try:
            if hasattr(updater, "evaluate_paper_metrics_on_test"):
                eval_loader = paper_test_loader if paper_test_loader is not None else val_loader
                metrics = updater.evaluate_paper_metrics_on_test(
                    eval_loader,
                    chunk_size=1024,
                    uniformity_max_M=20000,
                )
                log.update({"paper_metrics": metrics})
            else:
                logger.warning("[FINAL][PAPER] Updater has no method evaluate_paper_metrics_on_test(...).")
        except Exception as e:
            logger.error(f"[FINAL][PAPER] Failed to compute paper metrics: {e}")
            logger.error(traceback.format_exc())

        # 仍然保存 best checkpoint 路径和日志
        log.update({"best_model": str(best_check_pointer.last_checkpoint)})
        extensions.dump_log(log, str(out))

    # Evaluation（你原来这里是空的）


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="../config/01_post-pre-training/clip-refine.yaml", help="path to config file")
    parser.add_argument("--results_dir", type=str, default="./result", help="directory to save the results to")
    parser.add_argument("--resume", type=str, default="", help="path to the snapshot")
    parser.add_argument("--experiment_id", type=int, default=0)
    parser.add_argument("--num_worker", type=int, default=16)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action="store_true")

    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.SafeLoader))
    config.config_dict["pattern"] = yaml_utils.make_pattern(config)
    config.config_dict.update(vars(args))
    main(config.config_dict)


if __name__ == "__main__":
    run()
