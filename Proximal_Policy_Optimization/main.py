##
## EPITECH PROJECT, 2026
## START_TREK
## File description:
## main
##

import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt

def create_plot():
    df = pd.read_csv('eval.csv')
    window = 100
    df['percentage_crashed'] = df['crashed ?'].astype(int).rolling(window=window, min_periods=1).mean() * 100
    plt.plot(df.index, df['percentage_crashed'], color='crimson')
    plt.title(f'Crash rate')
    plt.xlabel('Episode')
    plt.ylabel('% Crashed')
    plt.ylim(0, 100)
    plt.xlim(0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/crash_rate.png', dpi=150)
    plt.show()

def main():
    python = sys.executable
    subprocess.run([python, 'train.py'], check=True)
    subprocess.run([python, 'eval.py'], check=True)
    create_plot()

if __name__ == '__main__':
    main()






