import sys
import json
import os
os.environ["WANDB_DIR"] = "/dev/shm/wandb"
# os.environ["WANDB_MODE"] = "offline"
import wandb


def save_dict_log_wandb(d):
    index = d.pop('Index')
    log = {'survived': d.pop('survived'),
           'eaten': d.pop('eaten')
           }
    run = wandb.init(sync_tensorboard=False, project="eaten_survived_not_random_again", entity='deepcharles',
                     reinit=True,  # Allow multiple wandb.init() calls in the same process
                     # mode="offline",
                     config=d
                     )
    # wandb.config.update(d)
    wandb.log(log)
    run.finish()

    print(f"\033[1;32mprogress[{index}]...")


if __name__ == '__main__':
    # data = sys.stdin.readlines()
    data = sys.argv[1]

    d = json.loads(data)


    save_dict_log_wandb(d)

# python /root/redflags-honza/external_utils/process_shell_output.py  | xargs -0  --max-procs=79 -I CMD  python /root/redflags-honza/external_utils/save_wandb.py 'CMD'