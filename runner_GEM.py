
from numpy import (
    ndarray,
    newaxis,
    concatenate,
    infty,
    ones,
)
from datetime import (
    datetime,
)
from pathlib import (
    Path,
)
from gzip import (
    open as gzip_open,
)
from pickle import (
    dump as pickle_dump,
    # load as pickle_load,
)
from shutil import (
    copy2,
)

from config import Config
from anchoring_2_imports.simulation import Simulation
from anchoring_2_imports.gem_actor_critic import GEMActorCritic
from anchoring_2_imports.plotting_functions import (
    plot_scatter_plot,
)


class Runner:
    def __init__(
            self,
    ) -> None:
        self.config = Config()
        self.rng = self.config.rng

        if self.config.toggle_profiling:
            import cProfile
            self.profiler: cProfile.Profile = cProfile.Profile()

    def train(
            self,
            training_name: str,
            probability_critical_events: float,
            value_network_path: None or str = None,  # for loading from pretrained
            policy_network_path: None or str = None,  # for loading from pretrained
            prior_task_samples_path: list = (),  # for loading samples from prior tasks
    ) -> None:

        def progress_print() -> None:
            progress = (episode_id * self.config.num_steps_per_episode + step_id + 1) / self.config.steps_total
            timedelta = datetime.now() - real_time_start
            finish_time = real_time_start + timedelta / progress

            print(f'\rSimulation completed: {progress:.2%}, '
                  f'est. finish {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}'
                  , end='')

        def save_networks() -> None:
            state = sim.gather_state()
            action = allocator.get_action(state)

            allocator.networks['policy']['target'](state[newaxis])
            allocator.networks['policy']['target'].save(
                Path(self.config.model_path, f'actor_allocation_{training_name}'))

            allocator.networks['value1']['target'](concatenate([state, action])[newaxis])
            allocator.networks['value1']['target'].save(
                Path(self.config.model_path, 'critic_allocation_' + training_name))

            copy2(Path(Path.cwd(), 'config.py'), self.config.model_path)

        def anneal_parameters() -> tuple:
            if simulation_step > self.config.exploration_noise_step_start_decay:
                exploration_noise_momentum_new = max(
                    0.0,
                    exploration_noise_momentum - self.config.exploration_noise_linear_decay_per_step
                )
            else:
                exploration_noise_momentum_new = exploration_noise_momentum

            return (
                exploration_noise_momentum_new,
            )

        # TRAINING SETUP------------------------------------------------------------------------------------------------
        # general
        if self.config.verbosity == 1:
            print('\n' + training_name)
        real_time_start = datetime.now()

        if self.config.toggle_profiling:
            self.profiler.enable()

        # initialize sim
        sim = Simulation(
            config=self.config,
            rng=self.rng,
        )
        if self.config.verbosity == 1:
            print(f'Expected load: {sim.get_expected_load():.2f}')

        # initialize allocator agent
        allocator = GEMActorCritic(
            name='allocator',
            rng=self.rng,
            path_prior_task_samples=prior_task_samples_path,
            dummy_input_actor=sim.gather_state(),
            dummy_input_critic=concatenate([sim.gather_state(), self.rng.normal(size=self.config.num_actions_policy)]),
            **self.config.gem_args,
        )

        # load pretrained networks
        if value_network_path:
            allocator.load_pretrained_networks(value_path=value_network_path,
                                               policy_path=policy_network_path)

        # initialize exploration noise
        exploration_noise_momentum = self.config.exploration_noise_momentum_initial

        # initialize global training metrics
        per_episode_metrics: dict = {
            'reward_per_step': -infty * ones(self.config.num_episodes),
            'priority_timeouts_per_occurrence': +infty * ones(self.config.num_episodes),
        }

        # MAIN LOOP-----------------------------------------------------------------------------------------------------
        for episode_id in range(self.config.num_episodes):

            # initialize per episode metrics
            episode_metrics: dict = {
                'rewards': -infty * ones(self.config.num_steps_per_episode),
                'priority_timeouts': +infty * ones(self.config.num_steps_per_episode),
            }

            # initialize per step buffers
            step_experience: dict = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}
            state_next: ndarray = sim.gather_state()

            # initialize network
            if episode_id == 0:
                if not policy_network_path:  # not pretrained
                    allocator.networks['policy']['primary'].initialize_inputs(state_next[newaxis])

            # step loop
            for step_id in range(self.config.num_steps_per_episode):

                # global step counter
                simulation_step = episode_id * self.config.num_steps_per_episode + step_id

                # determine current state & add to buffer
                state_current = state_next
                step_experience['state'] = state_current

                # find allocation action based on state & add to buffer
                bandwidth_allocation_solution = allocator.get_action(state_current)
                noisy_bandwidth_allocation_solution = self.add_random_distribution(
                    action=bandwidth_allocation_solution,
                    tau_momentum=exploration_noise_momentum)
                step_experience['action'] = noisy_bandwidth_allocation_solution

                # step simulation based on action & add resulting reward to buffer
                (
                    step_reward,
                    unweighted_step_reward_components,
                ) = sim.step(
                    percentage_allocation_solution=noisy_bandwidth_allocation_solution,
                    critical_events_chance=probability_critical_events,
                )
                step_experience['reward'] = step_reward

                # determine resulting new state & add to buffer
                state_next = sim.gather_state()
                step_experience['next_state'] = state_next

                # save buffer tuple (S, A, r, S_{new})
                allocator.add_experience(**step_experience)

                # train the allocator agent off-policy
                allocator.train(
                    update_ratio_tau=self.config.tau_target_update
                )

                # anneal parameters
                (
                    exploration_noise_momentum,
                ) = anneal_parameters()

                # log step results
                episode_metrics['rewards'][step_id] = step_experience['reward']
                episode_metrics['priority_timeouts'][step_id] = unweighted_step_reward_components['sum_priority_timeouts']

                # progress print
                if self.config.verbosity == 1:
                    if step_id % 50 == 0:
                        progress_print()

            # calculate & log episode metrics averages
            per_episode_metrics['reward_per_step'][episode_id] = (
                    sum(episode_metrics['rewards']) / self.config.num_steps_per_episode
            )
            per_episode_metrics['priority_timeouts_per_occurrence'][episode_id] = (
                    sum(episode_metrics['priority_timeouts']) / self.config.num_steps_per_episode / (
                        probability_critical_events + self.config.tiny_numerical_value)
            )

            # print episode results
            if self.config.verbosity == 1:
                results_print: str = (
                    '\n'
                    f'  {"episode per step reward:":<45}{per_episode_metrics["reward_per_step"][episode_id]:.2f}\n'
                    f'  {"episode per occurrence priority timeouts:":<45}{per_episode_metrics["priority_timeouts_per_occurrence"][episode_id]:.2f}\n'
                    f'  {"current exploration noise momentum epsilon:":<45}{exploration_noise_momentum:.2f}\n'
                )
                print(results_print)

            # reset simulation for next episode
            sim.reset()

        # POST LOOP-----------------------------------------------------------------------------------------------------
        # save trained networks locally
        save_networks()

        # save logged results
        with gzip_open(Path(self.config.log_path, f'{training_name}_per_episode_metrics.gzip'), 'wb') as file:
            pickle_dump(per_episode_metrics, file=file)
        allocator.dump_experiences(
            title=training_name,
            dump_path=self.config.model_path,
            num_experiences=self.config.num_experiences_dump
        )

        # end compute performance profiling
        if self.config.toggle_profiling:
            self.profiler.disable()
            if self.config.verbosity == 1:
                self.profiler.print_stats(sort='cumulative')
            self.profiler.dump_stats(Path(self.config.performance_profile_path, f'{training_name}.profile'))

        # plots
        plot_scatter_plot(per_episode_metrics['reward_per_step'],
                          title='Per Step Reward')
        plot_scatter_plot(per_episode_metrics['priority_timeouts_per_occurrence'],
                          title='Per Occurrence Priority Timeouts')

    def test(
            self,
            probability_critical_events: float,
            policy_network_path: None or str = None,
            name: str = '',
    ) -> None:

        def progress_print() -> None:
            progress = (episode_id * self.config.num_steps_per_episode + step_id + 1) / self.config.steps_total
            timedelta = datetime.now() - real_time_start
            finish_time = real_time_start + timedelta / progress

            print(f'\rSimulation completed: {progress:.2%}, '
                  f'est. finish {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}', end='')

        def allocate(st):
            return allocator_network.call(st[newaxis]).numpy().squeeze()


        # GENERAL SETUP-------------------------------------------------------------------------------------------------
        testing_name = f'testing_{name}'
        if self.config.verbosity == 1:
            print('\n' + testing_name)

        # log start time
        real_time_start = datetime.now()

        # enable compute profiling
        if self.config.toggle_profiling:
            self.profiler.enable()

        # set up testing environment
        sim = Simulation(
            config=self.config,
            rng=self.rng,
        )
        if self.config.verbosity == 1:
            print(f'Expected load: {sim.get_expected_load():.2f}')

        # set up allocator
        allocator_network = load_model(policy_network_path)

        # set up global metrics
        per_episode_metrics: dict = {
            'reward_per_step': -infty * ones(self.config.num_episodes),
            'priority_timeouts_per_occurrence': +infty * ones(self.config.num_episodes),
        }

        # MAIN LOOP-----------------------------------------------------------------------------------------------------
        for episode_id in range(self.config.num_episodes):

            # set up episode metrics
            episode_metrics: dict = {
                'rewards': -infty * ones(self.config.num_steps_per_episode),
                'priority_timeouts': +infty * ones(self.config.num_steps_per_episode),
            }

            # determine first state
            state_next = sim.gather_state()

            # episode step loop
            for step_id in range(self.config.num_steps_per_episode):
                # simulation_step = episode_id * self.config.num_steps_per_episode + step_id

                # determine state
                state_current = state_next

                # find allocation action based on state
                bandwidth_allocation_solution = allocate(state_current)
                noisy_bandwidth_allocation_solution = bandwidth_allocation_solution

                # step simulation based on action
                (
                    step_reward,
                    unweighted_step_reward_components,
                ) = sim.step(
                    percentage_allocation_solution=noisy_bandwidth_allocation_solution,
                    critical_events_chance=probability_critical_events,
                )

                # determine new state
                state_next = sim.gather_state()

                # log step results
                episode_metrics['rewards'][step_id] = step_reward
                episode_metrics['priority_timeouts'][step_id] = unweighted_step_reward_components['sum_priority_timeouts']

                # progress print
                if self.config.verbosity == 1:
                    if step_id % 50 == 0:
                        progress_print()

            # log episode results
            per_episode_metrics['reward_per_step'][episode_id] = (
                    sum(episode_metrics['rewards']) / self.config.num_steps_per_episode
            )
            per_episode_metrics['priority_timeouts_per_occurrence'][episode_id] = (
                    sum(episode_metrics['priority_timeouts']) / self.config.num_steps_per_episode / (
                        probability_critical_events + self.config.tiny_numerical_value)
            )

            # print episode results
            if self.config.verbosity == 1:
                per_step_reward = sum(episode_metrics['rewards']) / self.config.num_steps_per_episode
                relative_timeouts = (sum(episode_metrics['priority_timeouts'])
                                     / self.config.num_steps_per_episode
                                     / (probability_critical_events + self.config.tiny_numerical_value))
                print('\n', end='')
                print(f'episode per step reward: {per_step_reward:.2f}')
                print(f'episode per occurrence priority timeouts: {relative_timeouts:.2f}')

            # reset simulation for next episode
            sim.reset()

        # TEARDOWN------------------------------------------------------------------------------------------------------
        # save logged results
        with gzip_open(Path(self.config.log_path, f'{testing_name}_per_episode_metrics.gzip'), 'wb') as file:
            pickle_dump(per_episode_metrics, file=file)

        # end compute performance profiling
        if self.config.toggle_profiling:
            self.profiler.disable()
            if self.config.verbosity == 1:
                self.profiler.print_stats(sort='cumulative')
            self.profiler.dump_stats('train_critical_events.profile')

        # plots
        plot_scatter_plot(per_episode_metrics['reward_per_step'],
                          title='Per Step Reward')
        plot_scatter_plot(per_episode_metrics['priority_timeouts_per_occurrence'],
                          title='Per Occurrence Priority Timeouts')

    def add_random_distribution(
            self,
            action: ndarray,  # turns out its much faster to numpy the tensor and then do operations on ndarray
            tau_momentum: float,  # tau * random_distribution + (1 - tau) * action
    ) -> ndarray:
        """
        Mix an action vector with a random_uniform vector of same length
        by tau * random_distribution + (1 - tau) * action
        """
        if tau_momentum == 0.0:
            return action

        # create random action
        random_distribution = self.rng.random(size=self.config.num_actions_policy, dtype='float32')
        random_distribution = random_distribution / sum(random_distribution)

        # combine
        noisy_action = tau_momentum * random_distribution + (1 - tau_momentum) * action

        # normalize
        sum_noisy_action = sum(noisy_action)
        if sum_noisy_action != 0:
            noisy_action = noisy_action / sum_noisy_action

        return noisy_action
