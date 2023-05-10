
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

from src.config.plot_config import (
    Config as PlotConfig,
    pt_to_inches,
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
    'BS': metrics_baseline,
    'AU20': metrics_fifty_percent_critical,
    'AU100': metrics_critical,
    'AN1': metrics_anchored_pretrained['anchoring_1e5'],
    'AN2': metrics_anchored_pretrained['anchoring_1e6'],
    'AN3': metrics_anchored_pretrained['anchoring_1e7'],
    'AU20+': metrics_continued_twenty,
    'AN1+': metrics_continued_anchored['anchoring_1e5'],
    # 'Zufällig': metrics_random,
    # 'Kein Anchoring': metrics_baseline,
    # 'Anchoring': metrics_anchored_pretrained['anchoring_1e5'],
    # 'cont. anchoring': metrics_continued_anchored['anchoring_1e5'],
    # 'Vergleichsmethode': metrics_fifty_percent_critical,
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
    # 'cont. 20': metrics_continued_twenty,
}

# color = [
#     plot_config.cp4['mint'],
#     plot_config.cp4['vanilla'],
#     plot_config.cp4['mint'],
#     plot_config.cp4['mint'],
#     plot_config.cp4['mint'],
#     plot_config.cp4['vanilla'],
#     plot_config.cp4['vanilla'],
#     plot_config.cp4['blue'],
# ]

color = [
    plot_config.cp3['blue1'],
    plot_config.cp3['red2'],
    plot_config.cp3['blue1'],
    plot_config.cp3['blue1'],
    plot_config.cp3['blue1'],
    plot_config.cp3['red2'],
    plot_config.cp3['red2'],
    plot_config.cp3['white'],
]

# plot------------------------------------------------------------------------------------------------------------------
width = pt_to_inches(252)
height = 0.35 * width
padding = 0

bar_plot_args = {
    'edgecolor': 'black',
    'linewidth': .8,
    'alpha': 0.82,
}

plot_config.bar_plot_args['height'] = 1.0

# -------------------------------------------------------------------------------------------------
plot_name = 'paper_results'
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(width, height), sharey=True)

metric = 'reward_per_step'
baseline_mean = np.mean(metrics_baseline[metric])

for entry_id, entry_key in enumerate(reversed(bar_plots)):
    ax[0].barh(
        y=entry_id,
        width=np.mean(bar_plots[entry_key][metric]) / baseline_mean,
        xerr=np.var(bar_plots[entry_key][metric] / baseline_mean),
        color=color[entry_id],
        **bar_plot_args,
        **plot_config.bar_plot_args,
    )

# arrow_pos = [.92, 7-0.2]
# arrow_len = [.06, 0]
# ax[0].arrow(arrow_pos[0], arrow_pos[1], arrow_len[0], arrow_len[1],
#             head_length=0.1*abs(arrow_len[0]), head_width=.15, facecolor='black', length_includes_head=True,)
# ax[0].text((2 * arrow_pos[0] + arrow_len[0]) / 2 - 0.005, arrow_pos[1] + 0.06, 'better',)

ax[0].set_yticks(range(len(bar_plots.keys())))
ax[0].set_yticklabels(reversed(bar_plots.keys()), horizontalalignment='right')
ax[0].set_xlim([0.9, 1.04])
ax[0].set_xlabel('Avg. Reward per Step')
ax[0].xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0, is_latex=True))

ax[0].grid(which='major')
ax[0].grid(which='minor', alpha=0.3 * plt.rcParams['grid.alpha'], linewidth=0.8 * plt.rcParams['grid.linewidth'])
ax[0].tick_params(axis='y', which='minor', left=False)


metric = 'priority_timeouts_per_occurrence'
baseline_mean = np.mean(metrics_baseline[metric])

for entry_id, entry_key in enumerate(reversed(bar_plots)):
    ax[1].barh(
        y=entry_id,
        width=np.mean(bar_plots[entry_key][metric]) / baseline_mean,
        xerr=np.var(bar_plots[entry_key][metric] / baseline_mean),
        color=color[entry_id],
        **bar_plot_args,
        **plot_config.bar_plot_args,
    )

# arrow_pos = [.75, 7-.2]
# arrow_len = [-0.55, 0]
# ax[1].arrow(arrow_pos[0], arrow_pos[1], arrow_len[0], arrow_len[1],
#             head_length=0.1*abs(arrow_len[0]), head_width=.15, facecolor='black', length_includes_head=True,)
# ax[1].text((2 * arrow_pos[0] + arrow_len[0]) / 2 - 0.005, arrow_pos[1] + 0.06, 'better',)

ax[1].set_yticks(range(len(bar_plots.keys())))
ax[1].set_yticklabels(reversed(bar_plots.keys()))
ax[1].set_xlabel('Avg. Priority Timeouts')
ax[1].xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0, is_latex=True))

ax[1].grid(which='major')
ax[1].grid(which='minor', alpha=0.3 * plt.rcParams['grid.alpha'], linewidth=0.8 * plt.rcParams['grid.linewidth'])
ax[1].tick_params(axis='y', which='minor', left=False)

fig.align_labels()
fig.tight_layout(pad=0)

plot_config.save_figures(plot_name=plot_name, padding=padding)

# # -------------------------------------------------------------------------------------------------
# metric = 'priority_timeouts_per_occurrence'
# plot_name = metric
# fig, ax = plt.subplots(
#     figsize=(width, height),
# )
#
# baseline_mean = np.mean(metrics_baseline[metric])
#
# print(baseline_mean)
# for entry_id, entry_key in enumerate(bar_plots):
#     print(entry_key, np.mean(bar_plots[entry_key][metric]))
#     ax.barh(
#         y=entry_id,
#         width=np.mean(bar_plots[entry_key][metric]) / baseline_mean,
#         xerr=np.var(bar_plots[entry_key][metric] / baseline_mean),
#         color=color[entry_id],
#         **bar_plot_args,
#         **plot_config.bar_plot_args,
#     )
#
# arrow_pos = [.7, 1.4]
# arrow_len = [-.6, 0]
# ax.arrow(arrow_pos[0], arrow_pos[1], arrow_len[0], arrow_len[1], width=.005, head_width=.03, facecolor='black')
# ax.text((2 * arrow_pos[0] + arrow_len[0]) / 2 + 0.11, arrow_pos[1] + 0.04, 'besser')
#
# ax.set_yticks(range(len(bar_plots.keys())))
# ax.set_yticklabels(bar_plots.keys())
# ax.set_xlabel('Zielmetrik 2')
# ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0, is_latex=True))
#
# ax.grid(which='major')
# ax.grid(which='minor', alpha=0.3 * plt.rcParams['grid.alpha'], linewidth=0.8 * plt.rcParams['grid.linewidth'])
# ax.tick_params(axis='y', which='minor', left=False)
# fig.tight_layout(pad=0)
#
# plot_config.save_figures(plot_name=plot_name, padding=padding)

# ----------------------------------------------------------------------------------------------------------------------
plt.show()
