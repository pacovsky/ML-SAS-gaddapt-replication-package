from itertools import permutations


def get_commands(t, string):
    commands = [string + ''.join(x) for x in set(permutations('0' * t + '1' * (9-t)))]
    commands.sort()
    return commands


def print_commands(t, string):
    commands = get_commands(t, string)
    for c in commands:
        print(c)


def print_multiple(fromm, too, seed):
    string = fr'C:\Users\X\Pycharms\milad\en2-drone-charging\VENV_DIR\Scripts\python.exe C:/Users/X/Pycharms/milad/en2-drone-charging/drone_charging_example/run.py experiments/8drones.yaml --seed {seed} --subfolder'
    for i in range(fromm, too + 1):
        print_commands(9 - i, f"{string} ones{i}_")


if __name__ == '__main__':
    print_multiple(0, 9, seed=44)
