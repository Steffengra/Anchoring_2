

import matplotlib.pyplot as plt
import numpy as np
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
from src.plotting.plot_configurations import (
    PlotContentConfigs,
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


def get_paths(log_path, configuration):
    return [
        log_dir_path
        for log_dir_path in log_path.iterdir()
        if log_dir_path.stem[:-2] == configuration
           and log_dir_path.is_dir()
    ]


def plot():
    # SETUP-------------------------------------------------------------------------------------------------------------
    plot_config = PlotConfig()
    cnt_cfg = PlotContentConfigs()
    log_folder = Path(Path.cwd(), '..', 'logs').resolve()
    plt.rc('font', family='sans-serif')
    plt.rc('axes', labelsize=8.33)  # fontsize of the axes labels in pt, default: 10
    plt.rc('xtick', labelsize=8.33)  # fontsize of the x tick labels in pt, default: 10
    plt.rc('ytick', labelsize=8.33)  # fontsize of the y tick labels in pt, default: 10
    plt.rc('legend', fontsize=8.33)  # fontsize of the legend in pt, default: 10

    width = pt_to_inches(340)
    height = pt_to_inches(160)
    padding = 0

    norm_baseline = 'Baseline'
    palette = plot_config.cp3

    plot_contents = {
        'Baseline': cnt_cfg.contents_baseline,
        'Critical': cnt_cfg.contents_critical,
        # 'Random': cnt_cfg.contents_random,

        'SotA': cnt_cfg.contents_aug20,
        # 'Aug 0.5': cnt_cfg.contents_aug50,
        'GEM 1': cnt_cfg.contents_gem512,
        'GEM 2': cnt_cfg.contents_gem8138,
        # 'GEM K=2^15': cnt_cfg.contents_gem32768,
        'GEM 3': cnt_cfg.contents_gem65536,
        'EWC 1': cnt_cfg.contents_ewce5,
        'EWC 2': cnt_cfg.contents_ewce6,
        'EWC 3': cnt_cfg.contents_ewce7,
    }

    plot_contents2 = {
        'Aug 0.2+': cnt_cfg.contents_aug20cont,
        # 'GEM K=2^9+': cnt_cfg.contents_gem512cont,
        # 'GEM K=2^13+': cnt_cfg.contents_gem8138cont,
        # 'GEM K=2^15+': cnt_cfg.contents_gem32768cont,
        'GEM K=2^16+': cnt_cfg.contents_gem65536cont,
        # 'EWC w=1e-5+': cnt_cfg.contents_ewce5cont,
        # 'EWC w=1e-6+': cnt_cfg.contents_ewce6cont,
        'EWC w=1e-7+': cnt_cfg.contents_ewce7cont,
    }

    # revert order so that listing matches the plot
    plot_contents = dict(reversed(plot_contents.items()))
    plot_contents2 = dict(reversed(plot_contents2.items()))

    # GATHER STUFF------------------------------------------------------------------------------------------------------
    mode = 'testing'
    for plot_key in plot_contents.keys():
        # add all related folder paths
        plot_contents[plot_key]['paths'] = get_paths(
            log_path=log_folder,
            configuration=plot_contents[plot_key]['stem']
        )

        # add metrics
        plot_contents[plot_key]['metrics'] = get_metrics(
            [
                Path(folder_path, f'{mode}_{plot_contents[plot_key]["affix"]}_per_episode_metrics.gzip')
                for folder_path in plot_contents[plot_key]['paths']
                if Path(folder_path, f'{mode}_{plot_contents[plot_key]["affix"]}_per_episode_metrics.gzip').exists()
            ]
        )
    for plot_key in plot_contents2.keys():
        # add all related folder paths
        plot_contents2[plot_key]['paths'] = get_paths(
            log_path=log_folder,
            configuration=plot_contents2[plot_key]['stem']
        )

        # add metrics
        plot_contents2[plot_key]['metrics'] = get_metrics(
            [
                Path(folder_path, f'{mode}_{plot_contents2[plot_key]["affix"]}_per_episode_metrics.gzip')
                for folder_path in plot_contents2[plot_key]['paths']
                if Path(folder_path, f'{mode}_{plot_contents2[plot_key]["affix"]}_per_episode_metrics.gzip').exists()
            ]
        )

    # PLOTTING----------------------------------------------------------------------------------------------------------
    def generic_plot_beautify():
        ax[axis].set_axisbelow(True)
        ax[axis].grid(which='major')
        ax[axis].grid(
            which='minor',
            linewidth=0.8 * plt.rcParams['grid.linewidth'],
            color='whitesmoke',
        )
        ax[axis].tick_params(axis='y', which='minor', left=False)

    bar_plot_args = {
        'edgecolor': 'black',
        'linewidth': .8,
    }

    plot_name = 'paper_results_wsascc'
    fig, ax = plt.subplots(
        nrows=2, ncols=2,
        figsize=(width, height),
        sharey='row',
        sharex='col',
        height_ratios=[1, len(plot_contents2.keys()) / len(plot_contents.keys())]
    )

    metric = 'reward_per_step'
    axis = (0, 0)
    baseline_mean = np.mean(plot_contents[norm_baseline]['metrics'][metric])
    for entry_id, entry_key in enumerate(plot_contents.keys()):
        # if palette[plot_contents[entry_key]['color']] == '#f7a600' or palette[plot_contents[entry_key]['color']] == '#008878':
        #     ax[axis].barh(y=entry_id, width=0, **bar_plot_args, **plot_config.bar_plot_args)
        # else:
        ax[axis].barh(
            y=entry_id,
            width=np.mean(plot_contents[entry_key]['metrics'][metric]) / baseline_mean,
            xerr=np.var(plot_contents[entry_key]['metrics'][metric]) / baseline_mean,
            color=palette[plot_contents[entry_key]['color']],
            **bar_plot_args,
            **plot_config.bar_plot_args,
        )
    ax[axis].set_yticks(range(len(plot_contents.keys())))
    ax[axis].set_yticklabels([
        plot_contents[entry_key]['plot_title_alt']
        # if not 'GEM' in plot_contents[entry_key]['plot_title_alt']
        #     and not 'EWC' in plot_contents[entry_key]['plot_title_alt']
        # else ''
        for entry_key in plot_contents.keys()
    ])
    # ax[axis].spines['bottom'].set_visible(False)
    generic_plot_beautify()

    axis = (1, 0)
    for entry_id, entry_key in enumerate(plot_contents2.keys()):
        # if palette[plot_contents2[entry_key]['color']] == '#f7a600' or palette[plot_contents2[entry_key]['color']] == '#008878':
        #     ax[axis].barh(y=entry_id, width=0, **bar_plot_args, **plot_config.bar_plot_args)
        # else:
        ax[axis].barh(
            y=entry_id,
            width=np.mean(plot_contents2[entry_key]['metrics'][metric]) / baseline_mean,
            xerr=np.var(plot_contents2[entry_key]['metrics'][metric]) / baseline_mean,
            color=palette[plot_contents2[entry_key]['color']],
            **bar_plot_args,
            **plot_config.bar_plot_args,
        )
    ax[axis].set_yticks(range(len(plot_contents2.keys())))
    ax[axis].set_yticklabels([
        plot_contents2[entry_key]['plot_title_alt']
        # if not 'GEM' in plot_contents2[entry_key]['plot_title_alt']
        #     and not 'EWC' in plot_contents2[entry_key]['plot_title_alt']
        # else ''
        for entry_key in plot_contents2.keys()
    ])
    # ax[axis].set_yticks([])
    generic_plot_beautify()
    ax[axis].set_xlim([0.75, 1.04])
    ax[axis].set_xlabel('Overall Performance\n(maximize)')

    metric = 'priority_timeouts_per_occurrence'
    axis = (0, 1)
    baseline_mean = np.mean(plot_contents[norm_baseline]['metrics'][metric])
    for entry_id, entry_key in enumerate(plot_contents.keys()):
        # if palette[plot_contents[entry_key]['color']] == '#f7a600' or palette[plot_contents[entry_key]['color']] == '#008878':
        #     ax[axis].barh(y=entry_id, width=0, **bar_plot_args, **plot_config.bar_plot_args)
        # else:
        ax[axis].barh(
            y=entry_id,
            width=np.mean(plot_contents[entry_key]['metrics'][metric]) / baseline_mean,
            xerr=np.var(plot_contents[entry_key]['metrics'][metric]) / baseline_mean,
            color=palette[plot_contents[entry_key]['color']],
            **bar_plot_args,
            **plot_config.bar_plot_args,
        )
    # ax[axis].set_yticks(range(len(plot_contents.keys())))
    # ax[axis].set_yticklabels([
    #     plot_contents[entry_key]['plot_title']
    #     for entry_key in plot_contents.keys()
    # ])
    # ax[axis].spines['bottom'].set_visible(False)
    generic_plot_beautify()

    axis = (1, 1)
    for entry_id, entry_key in enumerate(plot_contents2.keys()):
        # if palette[plot_contents2[entry_key]['color']] == '#f7a600' or palette[plot_contents2[entry_key]['color']] == '#008878':
        #     ax[axis].barh(y=entry_id, width=0, **bar_plot_args, **plot_config.bar_plot_args)
        # else:
        ax[axis].barh(
            y=entry_id,
            width=np.mean(plot_contents2[entry_key]['metrics'][metric]) / baseline_mean,
            xerr=np.var(plot_contents2[entry_key]['metrics'][metric]) / baseline_mean,
            color=palette[plot_contents2[entry_key]['color']],
            **bar_plot_args,
            **plot_config.bar_plot_args,
        )
    # ax[axis].set_yticks(range(len(plot_contents2.keys())))
    # ax[axis].set_yticklabels([
    #     plot_contents2[entry_key]['plot_title']
    #     for entry_key in plot_contents2.keys()
    # ])
    ax[axis].set_xlim([0.0, 1.17])
    generic_plot_beautify()
    ax[axis].set_xlabel('Ambulance Timeouts\n(minimize)')

    fig.tight_layout(pad=0)

    plot_config.save_figures(plot_name=plot_name, padding=padding)

    # ------------------------------------------------------------------------------------------------------------------
    plt.show()


if __name__ == '__main__':
    plot()
