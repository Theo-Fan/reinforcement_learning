import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

folder_path = 'result'

files_and_folders = ['Models_5_105', 'Models_55_105', 'Models_105_105', 'Models_5_55', 'Models_55_55', 'Models_105_55', 'Models_5_5', 'Models_55_5', 'Models_105_5']

subfolder = [os.path.join(folder_path, f) for f in files_and_folders]

csv_files = [[f for f in os.listdir(_) if f.endswith('.csv')] for _ in subfolder]

test_csv_path = [os.path.join(subfolder[i], csv_files[i][1]) for i in range(len(subfolder))]

print(files_and_folders)
print(len(files_and_folders))
print(subfolder)
print(test_csv_path)

def plot_data(data, ax, title):
    sns.lineplot(x='Episode', y='Agent1 Reward', data=data, ax=ax, label='Agent1 Reward', alpha=0.8)  # Transparent blue
    sns.lineplot(x='Episode', y='Agent2 Reward', data=data, ax=ax, label='Agent2 Reward', alpha=0.8)  # Transparent orange
    sns.lineplot(x='Episode', y='Attack_count', data=data, ax=ax, label='Attack Count', alpha=0.8)  # Transparent green

    avg_attack_count = data['Attack_count'].mean()
    ax.axhline(avg_attack_count, color='red', linestyle='--', label='Avg Attack Count')

    # Set a fixed y-axis limit
    ax.set_ylim(0, 800)

    ax.set_title(title)
    ax.legend()

files = test_csv_path
print(len(files))

fig, axs = plt.subplots(3, 3, figsize=(15, 15))

for i, file in enumerate(files):
    data = pd.read_csv(file)
    print(file, files_and_folders[i])
    ax = axs[i // 3, i % 3]
    plot_data(data, ax, f'N_target, N_apple {files_and_folders[i][7:]}')

plt.tight_layout()
plt.show()