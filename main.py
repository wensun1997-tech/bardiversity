import argparse
import os
import yaml
import shutil
import subprocess

import extorch.utils as utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-file", type = str, required = True, help = "configuration file")
    parser.add_argument("--data-dir", type = str, default = "./")
    parser.add_argument("--train-dir", type = str, default = None)
    parser.add_argument("--save-every", type = int, default = 50)
    parser.add_argument("--gpu", type = int, default = 0, help = "gpu device id")
    
    parser.add_argument("--num-workers", default = 2, type = int)
    parser.add_argument("--report-every", default = 100, type = int)
    parser.add_argument("--seed", default = None, type = int)
    args = parser.parse_args()

    with open(args.cfg_file, "r") as rf:
        cfg = yaml.load(rf, Loader = yaml.FullLoader)
    
    if args.train_dir:
        utils.makedir(args.train_dir, remove = True)
        shutil.copyfile(args.cfg_file, os.path.join(args.train_dir, "train_config.yaml"))

    base_command = "--data-dir {} --gpu {} --num-workers {} --save-every {} --report-every {}".\
        format(args.data_dir, args.gpu, args.num_workers, args.save_every, args.report_every)
    if args.seed:
        base_command += " --seed {}".format(args.seed)

    # Step 2: Pruning
    epochs = cfg["prune_cfg"]["epochs"]
    batch_size = cfg["prune_cfg"]["batch_size"]
    weight_decay = cfg["prune_cfg"]["weight_decay"]
    lr = cfg["prune_cfg"]["lr"]
    grad_clip = cfg["prune_cfg"]["grad_clip"]
    budget = cfg["prune_cfg"]["budget"]
    progress_func = cfg["prune_cfg"]["progress_func"]
    _lambda = cfg["prune_cfg"]["_lambda"]
    distillation_temperature = cfg["prune_cfg"]["distillation_temperature"]
    distillation_alpha = cfg["prune_cfg"]["distillation_alpha"]
    tolerance = cfg["prune_cfg"]["tolerance"]
    margin = cfg["prune_cfg"]["margin"]
    sigmoid_a = cfg["prune_cfg"]["sigmoid_a"]
    upper_bound = cfg["prune_cfg"]["upper_bound"]
    pretrain_path = '.................'
    load = os.path.join(pretrain_path, "final.ckpt")

    command = "python prune_train.py --epochs {} --batch-size {} --weight-decay {}" \
              " --lr {} --budget {} --progress-fun {} --_lambda {}"\
              " --distillation-temperature {} --distillation-alpha {} --tolerance {}"\
              " --margin {} --sigmoid-a {} --upper-bound {} --load {}".format(
                epochs, batch_size, weight_decay, lr, budget, progress_func, _lambda,
                distillation_temperature, distillation_alpha, tolerance,
                margin, sigmoid_a, upper_bound, load)

    if grad_clip:
        command += " --grad-clip {}".format(grad_clip)

    if args.train_dir:
        prune_path = os.path.join(args.train_dir, "prune")
        command += " --train-dir {}".format(prune_path)

    command += " " + base_command
    subprocess.check_call(command, shell = True)

    # Step 3: Finetune the network
    epochs = cfg["finetune_cfg"]["epochs"]
    batch_size = cfg["finetune_cfg"]["batch_size"]
    weight_decay = cfg["finetune_cfg"]["weight_decay"]
    grad_clip = cfg["finetune_cfg"]["grad_clip"]
    lr = cfg["finetune_cfg"]["lr"]
    gamma = cfg["finetune_cfg"]["gamma"]
    milestones = ""
    for milestone in cfg["finetune_cfg"]["milestones"]:
        milestones += " {}".format(milestone)
    load = os.path.join(prune_path , "hard_pruned.ckpt")

    command = "python normal_train.py --epochs {} --batch-size {} --weight-decay {}" \
              " --lr {} --milestones {} --gamma {} --load {}".format(
                epochs, batch_size, weight_decay, lr, milestones, gamma, load)

    if grad_clip:
        command += " --grad-clip {}".format(grad_clip)

    if args.train_dir:
        finetune_path = os.path.join(args.train_dir, "finetune")
        command += " --train-dir {}".format(finetune_path)

    command += " " + base_command
    subprocess.check_call(command, shell = True)


if __name__ == "__main__":
    main()
