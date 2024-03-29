
from os import (
    system,
)
from pathlib import (
    Path,
)
from matplotlib.pyplot import (
    show as plt_show,
)

from src.config.config import Config
from src.models.runner_GEM import Runner


def main():
    def train():
        pass

        train_on_hundred_percent_crit_events()
        train_on_normal_crit_events_gem_hundred_percent_crit_events_pretrained()
        # train_on_no_crit_events_gem_hundred_percent_crit_events_pretrained_twice()

        # train_on_normal_crit_events()

    def test():
        pass
        # runner.config.update_num_episodes(new_num_episodes=5)
        # runner.config.update_num_steps_per_episode(new_steps_per_episode=200_000)

        test_hundred_percent_crit_events_scheduler_on_normal()
        test_normal_gem_hundred_pretrained_on_normal()
        # test_zero_gem_hundred_pretrained_twice_on_normal()

        # test_normal_pretrained_on_normal()

    # TRAINING WRAPPERS-------------------------------------------------------------------------------------------------
    def _train_network(
            probability_crit_events: float,
            prior_task_sample_names: None or str = None,
            preload_value_parameters: None or str = None,
            preload_policy_parameters: None or str = None,
            name_extra: None or str = None,
    ):

        training_name = f'training_{probability_crit_events}_{name_extra}'

        value_network_path = None
        if preload_value_parameters:
            value_network_path = Path(config.model_path, f'critic_allocation_training_{preload_value_parameters}')
        policy_network_path = None
        if preload_policy_parameters:
            policy_network_path = Path(config.model_path, f'actor_allocation_training_{preload_value_parameters}')

        prior_task_samples_path = [
            Path(config.model_path, f'experiences_training_{prior_task_sample_name}.gzip')
            for prior_task_sample_name in prior_task_sample_names
        ]

        runner.train(
            training_name=training_name,
            probability_critical_events=probability_crit_events,
            value_network_path=value_network_path,
            policy_network_path=policy_network_path,
            prior_task_samples_path=prior_task_samples_path,
        )

    def train_on_hundred_percent_crit_events():
        _train_network(
            probability_crit_events=1.0,
            prior_task_sample_names=[],
            name_extra='critical',
        )

    def train_on_normal_crit_events_gem_hundred_percent_crit_events_pretrained():
        _train_network(
            probability_crit_events=config.normal_priority_job_probability,
            prior_task_sample_names=['1.0_critical'],
            preload_value_parameters='1.0_critical',
            preload_policy_parameters='1.0_critical',
            name_extra='gem'
        )

    def train_on_no_crit_events_gem_hundred_percent_crit_events_pretrained_twice():
        _train_network(
            probability_crit_events=0.0,
            prior_task_sample_names=['1.0_critical'],
            preload_policy_parameters='0.0001_gem',
            preload_value_parameters='0.0001_gem',
            name_extra='twice',
        )

    def train_on_normal_crit_events():
        _train_network(
            probability_crit_events=config.normal_priority_job_probability,
            prior_task_sample_names=[],
            name_extra='base',
        )

    # TESTING WRAPPERS--------------------------------------------------------------------------------------------------
    def _test_network(network_name: str):
        runner.test(
            probability_critical_events=config.normal_priority_job_probability,
            policy_network_path=Path(config.model_path, f'actor_allocation_training_{network_name}'),
            name=network_name,
        )

    def test_hundred_percent_crit_events_scheduler_on_normal():
        _test_network('1.0_critical')

    def test_normal_gem_hundred_pretrained_on_normal():
        _test_network('0.0001_gem')

    def test_zero_gem_hundred_pretrained_twice_on_normal():
        _test_network('0.0_twice')

    def test_normal_pretrained_on_normal():
        _test_network('0.0001_base')

    # BODY--------------------------------------------------------------------------------------------------------------
    config = Config()
    runner = Runner()

    train()
    test()

    if config.shutdown_on_complete:
        system('shutdown /h')

    if config.show_plots:
        plt_show()


if __name__ == '__main__':
    main()
