
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from pathlib import (
    Path,
)
from gzip import (
    open as gzip_open,
)
from pickle import (
    load as pickle_load,
)

from src.config.config import Config
from src.config.plot_config import (
    Config as PlotConfig,
    pt_to_inches
)


def get_metrics(file_paths):
    metrics: dict = {}
    for file_path in file_paths:
        with gzip_open(file_path, 'rb') as file:
            data = pickle_load(file)
            for key in data.keys():
                if key not in metrics:
                    metrics[key] = [data[key]]
                else:
                    metrics[key].append(data[key])
    return metrics


config = Config()
plot_config = PlotConfig()

# Setup-----------------------------------------------------------------------------------------------------------------
# TODO: ONLY WORKS FOR UP TO 9 ITERATIONS
log_dirs: dict = {}
last_new_dir: str = ''
for log_dir_path in Path(Path.cwd(), '..', 'logs').iterdir():
    if not log_dir_path.stem[:-1] == last_new_dir:
        last_new_dir = log_dir_path.stem[:-1]
        log_dirs[log_dir_path.stem[:-2]] = [log_dir_path]
    else:
        log_dirs[log_dir_path.stem[:-2]].append(log_dir_path)

# One file per Run
file_paths_baseline = [
    Path(dirs[0], 'testing_0.0001_crit_events_base_per_episode_metrics.gzip')
    for dirs in log_dirs.values()]
metrics_baseline = get_metrics(file_paths_baseline)

file_paths_random = [
    Path(dirs[0], 'testing_random_scheduler_per_episode_metrics.gzip')
    for dirs in log_dirs.values()]
metrics_random = get_metrics(file_paths_random)

file_paths_twenty_percent_critical = [
    Path(dirs[0], 'testing_0.2_crit_events_base_per_episode_metrics.gzip')
    for dirs in log_dirs.values()]
metrics_twenty_percent_critical = get_metrics(file_paths_twenty_percent_critical)

file_paths_fifty_percent_critical = [
    Path(dirs[0], 'testing_0.5_crit_events_base_per_episode_metrics.gzip')
    for dirs in log_dirs.values()]
metrics_fifty_percent_critical = get_metrics(file_paths_fifty_percent_critical)

file_paths_critical = [
    Path(dirs[0], 'testing_1.0_crit_events_base_per_episode_metrics.gzip')
    for dirs in log_dirs.values()]
metrics_critical = get_metrics(file_paths_critical)

file_paths_continued_twenty = [
    Path(dirs[0], 'testing_continued_0.0_crit_events_pretrained_twenty_per_episode_metrics.gzip')
    for dirs in log_dirs.values()]
metrics_continued_twenty = get_metrics(file_paths_continued_twenty)

# # All repeats per run
# metrics_anchored = {}
# for run in log_dirs.keys():
#     run_paths = [
#         Path(path, 'testing_anchored_per_episode_metrics.gzip')
#         for path in log_dirs[run]]
#     metrics_anchored[run] = get_metrics(run_paths)

metrics_anchored_pretrained = {}
metrics_continued_anchored = {}
for run in log_dirs.keys():
    run_paths = [
        Path(path, 'testing_0.0001_crit_events_anchored_1.0_crit_events_base_pretrained_per_episode_metrics.gzip')
        for path in log_dirs[run]]
    metrics_anchored_pretrained[run] = get_metrics(run_paths)

    run_paths_continued = [
        Path(path, 'testing_continued_0.0_crit_events_anchored_1.0_per_episode_metrics.gzip')
        for path in log_dirs[run]]
    metrics_continued_anchored[run] = get_metrics(run_paths_continued)

bar_plots = {
    # 'Critical': metrics_critical,
    'Zufällig': metrics_random,
    'Kein Anchoring': metrics_baseline,
    'Anchoring': metrics_anchored_pretrained['anchoring_1e5'],
    'cont. anchoring': metrics_continued_anchored['anchoring_1e5'],
    'Vergleichsmethode': metrics_fifty_percent_critical,
    # 'Anchored_0': metrics_anchored['anchoring_0'],
    # 'Anchored_Pretrained_0': metrics_anchored_pretrained['anchoring_0'],
    # 'Anchored_1e1': metrics_anchored['anchoring_1e1'],
    # 'Anchored_Pretrained_1e1': metrics_anchored_pretrained['anchoring_1e1'],
    # 'Anchored_1e2': metrics_anchored['anchoring_1e2'],
    # 'Anchored_Pretrained_1e2': metrics_anchored_pretrained['anchoring_1e2'],
    # 'Anchored_1e3': metrics_anchored['anchoring_1e3'],
    # 'Anchored_Pretrained_1e3': metrics_anchored_pretrained['anchoring_1e3'],
    # 'Anchored_1e4': metrics_anchored['anchoring_1e4'],
    # 'Anchored_Pretrained_1e4': metrics_anchored_pretrained['anchoring_1e4'],
    # 'Anchored_1e5': metrics_anchored['anchoring_1e5'],
    # 'Anchored_1e6': metrics_anchored['anchoring_1e6'],
    # 'Anchored_Pretrained_1e6': metrics_anchored_pretrained['anchoring_1e6'],
    # 'Anchored_Pretrained_1e7': metrics_anchored_pretrained['anchoring_1e7'],
    'cont. 20': metrics_continued_twenty,
}

color = [
    # plot_config.cp4['grey'],
    # plot_config.cp4['red'],
    # plot_config.cp4['orange'],
    # plot_config.cp4['vanilla'],
    # plot_config.cp4['white'],
    # plot_config.cp4['mint'],
]

color = [
    plot_config.cp4['vanilla'],
    plot_config.cp4['orange'],
    plot_config.cp4['mint'],
    'white',
    plot_config.cp4['red'],
    'black',
]


# plot------------------------------------------------------------------------------------------------------------------
width = pt_to_inches(500)
height = 0.5 * width
padding = 0

# -------------------------------------------------------------------------------------------------
metric = 'reward_per_step'
plot_name = metric
fig, ax = plt.subplots(figsize=(width, height))

bar_plot_args = {
    'edgecolor': 'black',
    'linewidth': .8,
}

baseline_mean = np.mean(metrics_baseline[metric])

for entry_id, entry_key in enumerate(bar_plots):
    ax.barh(
        y=entry_id,
        width=np.mean(bar_plots[entry_key][metric]) / baseline_mean,
        xerr=np.var(bar_plots[entry_key][metric] / baseline_mean),
        color=color[entry_id],
        **bar_plot_args,
        **plot_config.bar_plot_args,
    )

arrow_pos = [.3, 1.4]
arrow_len = [.6, 0]
ax.arrow(arrow_pos[0], arrow_pos[1], arrow_len[0], arrow_len[1], width=.005, head_width=.03, facecolor='black')
ax.text((2 * arrow_pos[0] + arrow_len[0]) / 2 + 0.11, arrow_pos[1] + 0.04, 'besser')

ax.set_yticks(range(len(bar_plots.keys())))
ax.set_yticklabels(bar_plots.keys())
# ax.set_xlim([0.0, 1.02])
ax.set_xlabel('Zielmetrik 1')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0, is_latex=True))

ax.grid(which='major')
ax.grid(which='minor', alpha=0.3 * plt.rcParams['grid.alpha'], linewidth=0.8 * plt.rcParams['grid.linewidth'])
ax.tick_params(axis='y', which='minor', left=False)
fig.tight_layout(pad=0)

plot_config.save_figures(plot_name=plot_name, padding=padding)

# -------------------------------------------------------------------------------------------------
metric = 'priority_timeouts_per_occurrence'
plot_name = metric
fig, ax = plt.subplots(
    figsize=(width, height),
)

baseline_mean = np.mean(metrics_baseline[metric])

print(baseline_mean)
for entry_id, entry_key in enumerate(bar_plots):
    print(entry_key, np.mean(bar_plots[entry_key][metric]))
    ax.barh(
        y=entry_id,
        width=np.mean(bar_plots[entry_key][metric]) / baseline_mean,
        xerr=np.var(bar_plots[entry_key][metric] / baseline_mean),
        color=color[entry_id],
        **bar_plot_args,
        **plot_config.bar_plot_args,
    )

arrow_pos = [.7, 1.4]
arrow_len = [-.6, 0]
ax.arrow(arrow_pos[0], arrow_pos[1], arrow_len[0], arrow_len[1], width=.005, head_width=.03, facecolor='black')
ax.text((2 * arrow_pos[0] + arrow_len[0]) / 2 + 0.11, arrow_pos[1] + 0.04, 'besser')

ax.set_yticks(range(len(bar_plots.keys())))
ax.set_yticklabels(bar_plots.keys())
ax.set_xlabel('Zielmetrik 2')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0, is_latex=True))

ax.grid(which='major')
ax.grid(which='minor', alpha=0.3 * plt.rcParams['grid.alpha'], linewidth=0.8 * plt.rcParams['grid.linewidth'])
ax.tick_params(axis='y', which='minor', left=False)
fig.tight_layout(pad=0)

plot_config.save_figures(plot_name=plot_name, padding=padding)

# ----------------------------------------------------------------------------------------------------------------------
plt.show()
