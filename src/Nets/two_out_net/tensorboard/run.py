from os import system, path


logdir_name = "./logs"
logdir = path.abspath(path.join(path.dirname(__file__), logdir_name))

system(f"tensorboard --logdir={logdir}")
