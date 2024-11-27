import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

folder_path = 'result'

# files_and_folders = sorted(os.listdir(folder_path))[::-1]

files_and_folders = ['Models_5_105', 'Models_55_105', 'Models_105_105', 'Models_5_55', 'Models_55_55', 'Models_105_55', 'Models_5_5', 'Models_55_5', 'Models_105_5']
# files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

subfolder = [os.path.join(folder_path, f) for f in files_and_folders]

csv_files = [[f for f in os.listdir(_) if f.endswith('.csv')] for _ in subfolder]

train_csv_path = [os.path.join(subfolder[i], csv_files[i][0]) for i in range(len(subfolder))]

print(files_and_folders)
print(len(files_and_folders))
print(subfolder)
# print(csv_files)
print(train_csv_path)


def plot_and_fit_curve(data, ax, column, title, y_label):
    line_color = 'lightblue'
    fit_color = 'blue'

    sns.lineplot(x='Episode', y=column, data=data, ax=ax, label=column, color=line_color, alpha=0.6)

    fitted = lowess(data[column], data['Episode'], frac=0.3)
    ax.plot(fitted[:, 0], fitted[:, 1], color=fit_color, label=f'Fitted {column}', alpha=1.0)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Episode')
    ax.legend()


files = train_csv_path

fig1, axs1 = plt.subplots(3, 3, figsize=(15, 15))
fig2, axs2 = plt.subplots(3, 3, figsize=(15, 15))
fig3, axs3 = plt.subplots(3, 3, figsize=(15, 15))

for i, file in enumerate(files):
    data = pd.read_csv(file)

    data.columns = data.columns.str.strip()

    ax1 = axs1[i // 3, i % 3]
    plot_and_fit_curve(data, ax1, 'Agent1_Total_Reward', f'N_target, N_apple {files_and_folders[i][7:]}', 'Agent1 Total Reward')

    ax2 = axs2[i // 3, i % 3]
    plot_and_fit_curve(data, ax2, 'Agent2_Total_Reward', f'N_target, N_apple {files_and_folders[i][7:]}', 'Agent2 Total Reward')

    ax3 = axs3[i // 3, i % 3]
    plot_and_fit_curve(data, ax3, 'Attack_count', f'N_target, N_apple {files_and_folders[i][7:]}', 'Attack Count')

    avg_attack_count = data['Attack_count'].mean()
    ax3.axhline(avg_attack_count, color='dodgerblue', linestyle='--', label='Avg Attack Count', alpha=0.7)
    ax3.legend()

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()

plt.show()