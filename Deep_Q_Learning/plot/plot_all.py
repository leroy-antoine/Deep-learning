import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('Deep_Q_Learning/eval.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    next(lines)
    i = 1
    for row in lines:
        x.append(i)
        y.append(float(row[1]))
        i += 1

plt.plot(x, y, color = 'g', linestyle = 'dashed',
         marker = 'o',label = "Performance Data")

plt.xticks(rotation = 0)
plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title("Training Performance")
plt.grid()
plt.legend()
plt.savefig("Deep_Q_Learning/plot/plot_all.png")

