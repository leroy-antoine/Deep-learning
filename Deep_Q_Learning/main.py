##
## EPITECH PROJECT, 2026
## START_TREK
## File description:
## main
##

import subprocess
import sys

def main():
    python = sys.executable

    subprocess.run([python, 'Deep_Q_Learning/train.py'], check=True)
    subprocess.run([python, 'Deep_Q_Learning/eval.py'], check=True)

if __name__ == '__main__':
    main()