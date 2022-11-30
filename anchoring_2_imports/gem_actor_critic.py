
from numpy import (
    ndarray,
    ndim,
    array,
    newaxis,
    concatenate,
    reshape,
    ones,
    diag,
    matmul,
    dot,
)
from numpy.random import (
    default_rng,
)
from tensorflow import (
    convert_to_tensor,
    function as tf_function,
    GradientTape as tf_GradientTape,
    concat as tf_concat,
    squeeze as tf_squeeze,
    reduce_mean as tf_reduce_mean,
    ones as tf_ones,
)
from keras.models import (
    load_model,
)
from pathlib import (
    Path,
)
from gzip import (
    open as gzip_open,
)
from pickle import (
    dump as pickle_dump,
    load as pickle_load,
)

from anchoring_2_imports.experience_buffer import (
    ExperienceBuffer,
)
from anchoring_2_imports.network_models import (
    ValueNetwork,
    PolicyNetwork,
)
from anchoring_2_imports.dl_internals_with_expl import (
    mse_loss,
    huber_loss,
)

import cvxopt


class GEMActorCritic:
    def __init__(
            self,
            name: str,
            rng: default_rng,
            num_actions: int,
            path_prior_task_samples: list[Path],  # contains paths to experiences from prior tasks
            batch_size: int,  # for current task
            num_max_experiences: int,  # size of experience buffer for current task
            num_hidden_critic: list[int],  # number of units per hidden layer in critic networks
            num_hidden_actor: list[int],  # number of units per hidden layer in actor networks
            future_reward_discount_gamma: float,  # for loss
            optimizer_critic,
            optimizer_critic_args: dict,
            optimizer_actor,
            optimizer_actor_args: dict,
            hidden_layer_args: dict,
            dummy_input_critic: ndarray,  # for initialization
            dummy_input_actor: ndarray,  # for initialization
            priority_scale_alpha: float = 1.0,  # for experience buffer of current task
    ) -> None:
        """
        GEM is an algorithm for continual learning/lifelong learning.
        GEM proposes the following:
        - Keep a memory buffer of samples for each prior task
        - When a gradient update on the current task is performed, compare this
            gradient update to prior tasks:
            a) The gradient update leads to a change <=0 in the Loss of prior tasks. Proceed.
            b) The gradient update leads to an increase in the loss on prior tasks. In
                this case, perform a geometric transform on the gradient vector to the
                least perturbed alternative gradient vector g_tilde that does not
                increase loss on any prior task.
        - Loss on prior tasks is approximated by performing scalar product on proposed gradient
            update g and gradients g_i obtained from the memory buffer. If they match direction,
            increase in loss on prior tasks is unlikely.
        - Geometric projection is solved via Quadratic Program optimization

        Lopez-Paz, David, and Marc'Aurelio Ranzato.
          "Gradient episodic memory for continual learning."
          NeurIPS (2017).
        """

        self.name: str = name
        self.rng: default_rng = rng
        self.batch_size: int = batch_size
        self.future_reward_discount_gamma: float = future_reward_discount_gamma

        # INITIALIZE EXP BUFFER
        self.exp_buffer = ExperienceBuffer(
            rng=self.rng,
            buffer_size=num_max_experiences,
            priority_scale_alpha=priority_scale_alpha,
        )

        # INITIALIZE NETWORKS
        self.networks = {}
        networks = {
            'value1': 'critic',
            # 'value2': 'critic',
            'policy': 'actor',
        }
        for network_name, network_type_string in networks.items():
            if network_type_string == 'critic':
                network_type = ValueNetwork
                init_arguments = {
                    'num_hidden': num_hidden_critic,
                }
                optimizer = optimizer_critic
                optimizer_args = optimizer_critic_args
                dummy_input = dummy_input_critic
            elif network_type_string == 'actor':
                network_type = PolicyNetwork
                init_arguments = {
                    'num_hidden': num_hidden_actor,
                    'num_actions': num_actions,
                }
                optimizer = optimizer_actor
                optimizer_args = optimizer_actor_args
                dummy_input = dummy_input_actor
            else:
                raise ValueError('Unknown network type')

            self.networks[network_name] = {
                'primary': network_type(
                    name=f'{name}_{network_name}_primary',
                    **hidden_layer_args,
                    **init_arguments,
                ),
                'target': network_type(
                    name=f'{name}_{network_name}_target',
                    **hidden_layer_args,
                    **init_arguments,
                ),
            }

            # compile model with optimizer and loss
            self.networks[network_name]['primary'].compile(
                optimizer=optimizer(**optimizer_args),
                loss=mse_loss,
            )

            # initialize weights and input layer with a dummy input
            for network in self.networks[network_name].values():
                network.call(dummy_input[newaxis].astype('float32'))

        # 100% copy primary params onto target at init
        self.update_target_networks(update_ratio_tau=1.0)

        # LOAD PRIOR EXPERIENCES
        self.prior_experiences = []
        for experience_path in path_prior_task_samples:
            with gzip_open(experience_path, 'rb') as file:
                # transform to tensors for later use
                task_experiences = pickle_load(file)
                self.prior_experiences.append(
                    {
                        'states': convert_to_tensor(
                            [experience['state'] for experience in task_experiences], dtype='float32'),
                        'actions': convert_to_tensor(
                            [experience['action'] for experience in task_experiences], dtype='float32'),
                        'rewards': convert_to_tensor(
                            [experience['reward'] for experience in task_experiences], dtype='float32'),
                        'next_states': convert_to_tensor(
                            [experience['next_state'] for experience in task_experiences], dtype='float32'),
                    }
                )

    def load_pretrained_networks(
            self,
            value_path: str,
            policy_path: str,
    ) -> None:
        # Value networks
        for network in [
            'value1',
            # 'value2'
        ]:
            optimizer = self.networks[network]['primary'].optimizer
            loss = self.networks[network]['primary'].loss

            self.networks[network]['primary'] = load_model(value_path)
            self.networks[network]['target'] = load_model(value_path)

            self.networks[network]['primary'].compile(
                optimizer=optimizer,
                loss=loss,
            )

        # Policy networks
        for network in ['policy']:
            optimizer = self.networks[network]['primary'].optimizer
            loss = self.networks[network]['primary'].loss

            self.networks[network]['primary'] = load_model(policy_path)
            self.networks[network]['target'] = load_model(policy_path)

            self.networks[network]['primary'].compile(
                optimizer=optimizer,
                loss=loss,
            )

    def dump_experiences(
            self,
            title: str,
            num_experiences: int,
            dump_path: Path,
    ) -> None:

        if self.exp_buffer.get_len() < num_experiences:
            raise ValueError('num_experiences > experiences in buffer, cannot save')

        (
            sample_experiences,
            _,
            _,
        ) = self.exp_buffer.sample(
            batch_size=num_experiences,
            importance_sampling_correction_beta=1.0
        )

        with gzip_open(Path(dump_path, f'experiences_{title}.gzip'), 'wb') as file:
            pickle_dump(sample_experiences, file=file)

    @tf_function
    def update_target_networks(
            self,
            update_ratio_tau: float,
    ) -> None:
        """
        Performs a soft update theta_target_new = tau * theta_primary + (1 - tau) * theta_target_old
        """
        for network_pair in self.networks.values():
            # trainable variables are a list of tf variables
            for v_primary, v_target in zip(network_pair['primary'].trainable_variables,
                                           network_pair['target'].trainable_variables):
                v_target.assign(update_ratio_tau * v_primary + (1 - update_ratio_tau) * v_target)

    def get_action(
            self,
            state: ndarray,
    ) -> ndarray:
        """
        Wrapper to evaluate target policy network
        """
        state = state.astype('float32')
        if ndim(state) == 1:
            state = state[newaxis]

        return self.networks['policy']['target'].call(state).numpy().squeeze()

    def add_experience(
            self,
            state,
            action,
            reward,
            next_state
    ) -> None:
        """
        Wrapper to add experience to buffer
        """
        self.exp_buffer.add_experience(
            {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
            }
        )

    @tf_function
    def apply_gradients_actor(
            self,
            grads,
    ):
        self.networks['policy']['primary'].optimizer.apply_gradients(
            zip(grads, self.networks['policy']['primary'].trainable_variables)
        )

    @tf_function
    def apply_gradients_critic(
            self,
            grads,
    ):
        self.networks['value1']['primary'].optimizer.apply_gradients(
            zip(grads, self.networks['value1']['primary'].trainable_variables)
        )

    def train(
            self,
            update_ratio_tau: float,
    ) -> None:

        def grads_to_vector(
                grads,
        ) -> ndarray:
            return concatenate([reshape(layer_grads, [-1])
                                for layer_grads in grads], 0)

        def vector_to_grads(
                grads_vector,
                shapes,
        ) -> list:
            grads = []
            index_pointer = 0
            for layer_id in range(len(shapes)):
                if len(shapes[layer_id]) == 1:
                    num_parameters_layer = shapes[layer_id][0]
                else:
                    num_parameters_layer = shapes[layer_id][0] * shapes[layer_id][1]
                layer_grads = grads_vector[index_pointer:index_pointer+num_parameters_layer]
                layer_grads = reshape(layer_grads, shapes[layer_id])
                index_pointer += num_parameters_layer
                grads.append(layer_grads)
            return grads

        def test_gradient_conflicts(
                grad_vec,
                pt_grad_vecs,
        ) -> bool:
            for pt_grad_vec in pt_grad_vecs:
                dot_product = dot(grad_vec, pt_grad_vec)
                # print(dot_product)
                if dot_product < 0.0:
                    # print('dot product', dot_product)
                    return True
            return False

        def transform_gradient(
                grad_vec,
                pt_grad_vecs,
        ) -> list:
            grad_layer_shapes = []
            for layer_gradient in gradients:
                grad_layer_shapes.append(layer_gradient.shape)
            grad_vec_opt = solve_qp_gradient(
                grad_vec=grad_vec,
                pt_grad_vecs=pt_grad_vecs,
            )
            return vector_to_grads(
                grads_vector=grad_vec_opt,
                shapes=grad_layer_shapes,
            )

        def solve_qp_gradient(
                grad_vec,
                pt_grad_vecs,
        ) -> ndarray:
            """
            quadprog
            Minimize     1/2 x^T G x - a^T x
            Subject to   C.T x >= b

            quadprog Notation | GEM Paper Notation
            G | G G^T
            a^T | g^T G^T
            C | No constraint matrix
            b=0
            """
            # # construct G
            # new_gradient_vector_g = convert_to_tensor(grad_vec)[tf_newaxis]  # 1 x m
            # pt_gradient_vectors_G = -1 * convert_to_tensor(pt_grad_vecs)
            #
            # # print(pt_gradient_vectors_G)
            # QP_G = tf_matmul(pt_gradient_vectors_G, tf_transpose(pt_gradient_vectors_G))
            # # print(QP_G.numpy().astype('double'))
            # QP_aT = -1 * tf_matmul(new_gradient_vector_g, tf_transpose(pt_gradient_vectors_G))
            # # print(tf_transpose(QP_aT).numpy().astype('double'))
            # QP_C = diag(ones(QP_G.shape[0])).astype('double')
            # # print(QP_C)
            # v_opt = quadprog.solve_qp(
            #     G=QP_G.numpy().astype('double'),
            #     a=tf_transpose(QP_aT).numpy().astype('double')[0],
            #     C=QP_C,
            #     b=array([0.0], dtype='double'),
            # )[0]

            """
            CVXOPT
            minimize_x 1/2 x^T P x + q^T x
            s.t. Gx <= h
            """
            cvxopt.solvers.options['show_progress'] = False  # reduce verbosity
            # cvxopt.solvers.options['abstol'] = 1e-20  # default 1e-7
            # cvxopt.solvers.options['reltol'] = 1e-20  # default 1e-6
            # cvxopt.solvers.options['feastol'] = 1e-20  # default 1e-7
            new_gradient_vector_g = grad_vec[newaxis]  # 1 x m
            pt_gradient_vectors_G = -1 * array(pt_grad_vecs)
            QP_P = matmul(pt_gradient_vectors_G, pt_gradient_vectors_G.T)
            QP_qT = matmul(new_gradient_vector_g, pt_gradient_vectors_G.T)
            # By notation, QP_G should be negative to be equivalent to GEM paper QP. However, correct results
            #  are produced with positive sign.
            QP_G = 1 * diag(ones(QP_P.shape[0]))
            optimization = cvxopt.solvers.qp(
                P=cvxopt.matrix(QP_P.astype('double')),
                q=cvxopt.matrix(QP_qT.astype('double')),
                G=cvxopt.matrix(QP_G),
                h=cvxopt.matrix([0.0]),
            )
            if optimization['status'] != 'optimal':
                raise ValueError(f'Optimization status: {optimization["status"]}')
            v_opt = optimization['x']

            # add a small positive constant to v:
            #  From GEM Paper "In practice, we found that adding a small constant \gamma > 0 to v
            #  biased the gradient projection to updates that favoured benefitial backwards transfer."
            # v_opt = v_opt * 1.0
            # small_positive_constant = -1e3  # -1e3 leads to large gain in overall reward
            # v_opt = v_opt + small_positive_constant

            gradient_vector_opt = matmul(pt_gradient_vectors_G.T, array(v_opt)).squeeze() + grad_vec

            # This check is for debugging purposes only and can be removed to improve performance
            # for pt_gradient_vector in pt_grad_vecs:
            #     dot_product = dot(gradient_vector_opt, pt_gradient_vector)
            #     print('dot_after', dot_product)

            return gradient_vector_opt

        if self.exp_buffer.get_len() < self.batch_size:
            return

        # Sample from experience buffer---------------------------------------------
        (
            sample_experiences,
            experience_ids,
            sample_importance_weights,
        ) = self.exp_buffer.sample(
            batch_size=self.batch_size,
            importance_sampling_correction_beta=1.0
        )

        states = convert_to_tensor([experience['state'] for experience in sample_experiences], dtype='float32')
        actions = convert_to_tensor([experience['action'] for experience in sample_experiences], dtype='float32')
        rewards = convert_to_tensor([experience['reward'] for experience in sample_experiences], dtype='float32')
        next_states = convert_to_tensor([experience['next_state'] for experience in sample_experiences], dtype='float32')
        # --------------------------------------------------------------------------

        # TRAIN CRITIC--------------------------------------------------------------
        gradients, pt_gradients = self.train_graph_critic(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            sample_importance_weights=sample_importance_weights,
        )

        gradient_vector = grads_to_vector(grads=gradients)
        pt_gradient_vectors = [grads_to_vector(pt_gradient)
                               for pt_gradient in pt_gradients]

        # test if gradient update would worsen performance on prior tasks
        violation_flag = test_gradient_conflicts(
            grad_vec=gradient_vector,
            pt_grad_vecs=pt_gradient_vectors,
        )
        if violation_flag:
            # print('critic')
            # if so, transform gradients to the least perturbed other vector that does not worsen performance
            gradients = transform_gradient(
                grad_vec=gradient_vector,
                pt_grad_vecs=pt_gradient_vectors,
            )
            gradients = [gradient.astype('float32')
                         for gradient in gradients]

        # apply gradient update
        self.apply_gradients_critic(
            grads=gradients,
        )
        # --------------------------------------------------------------------------

        # TRAIN ACTOR---------------------------------------------------------------
        gradients, pt_gradients = self.train_graph_actor(
            states=states,
        )
        gradient_vector = grads_to_vector(grads=gradients)
        pt_gradient_vectors = [grads_to_vector(pt_gradient)
                               for pt_gradient in pt_gradients]

        # Test if gradient update would worsen performance on prior tasks
        violation_flag = test_gradient_conflicts(
            grad_vec=gradient_vector,
            pt_grad_vecs=pt_gradient_vectors,
        )
        if violation_flag:
            # print('actor')
            # if so, transform gradients to the least perturbed other vector that does not worsen performance
            gradients = transform_gradient(
                grad_vec=gradient_vector,
                pt_grad_vecs=pt_gradient_vectors,
            )
            gradients = [gradient.astype('float32')
                         for gradient in gradients]

        # apply gradient update
        self.apply_gradients_actor(
            grads=gradients,
        )
        # --------------------------------------------------------------------------

        # UPDATE TARGET NETWORKS
        self.update_target_networks(update_ratio_tau=update_ratio_tau)

    @tf_function
    def train_graph_critic(
            self,
            states: ndarray,
            actions: ndarray,
            rewards: ndarray,
            next_states: ndarray,
            sample_importance_weights: ndarray,
    ):
        """
        Wraps as much as possible of the training process into a graph for performance
        """
        def calculate_target_q(
                rew,  # rewards
                nx_st,  # next_states
        ):
            trg_q = rew
            if self.future_reward_discount_gamma > 0:  # future rewards estimate
                nx_ac = self.networks['policy']['target'].call(nx_st)
                in_vec = tf_concat([nx_st, nx_ac], axis=1)
                q_estimate_1 = self.networks['value1']['target'].call(in_vec)
                # q_estimate_2 = self.networks['value2']['target'].call(in_vec)
                # conservative_q_estimate = tf_squeeze(tf_minimum(q_estimate_1, q_estimate_2))
                # trg_q = trg_q + self.future_reward_discount_gamma * conservative_q_estimate
                trg_q = trg_q + self.future_reward_discount_gamma * q_estimate_1
            return trg_q

        def calculate_value_net_gradient(
                value_net,
                st,  # states
                ac,  # actions
                trg_q,  # target_q
                smp_imp_wgh,  # sample importance weights
        ):
            in_vec = tf_concat([st, ac], axis=1)
            with tf_GradientTape() as tape:  # autograd
                estimate = tf_squeeze(value_net.call(in_vec))
                td_error = trg_q - estimate
                loss_estimation = value_net.loss(td_error, smp_imp_wgh)
            grad = tape.gradient(target=loss_estimation,  # d_loss / d_parameters
                                 sources=value_net.trainable_variables)
            return grad

        # calculate gradient on current task
        target_q = calculate_target_q(rew=rewards, nx_st=next_states)
        gradients = calculate_value_net_gradient(value_net=self.networks['value1']['primary'],
                                                 st=states,
                                                 ac=actions,
                                                 trg_q=target_q,
                                                 smp_imp_wgh=sample_importance_weights)

        # Calculate gradients on experiences from prior tasks:
        pt_gradients = []
        for task_experiences in self.prior_experiences:
            pt_states = task_experiences['states']
            pt_actions = task_experiences['actions']
            pt_rewards = task_experiences['rewards']
            pt_next_states = task_experiences['next_states']

            target_q = calculate_target_q(rew=pt_rewards, nx_st=pt_next_states)
            pt_gradient = calculate_value_net_gradient(value_net=self.networks['value1']['primary'],
                                                       st=pt_states,
                                                       ac=pt_actions,
                                                       trg_q=target_q,
                                                       smp_imp_wgh=tf_ones(pt_rewards.shape))
            pt_gradients.append(pt_gradient)

        return gradients, pt_gradients

    @tf_function
    def train_graph_actor(
            self,
            states: ndarray,
    ):

        def calculate_policy_net_gradient(
                policy_net,
                st,  # states

        ):
            in_vec = st
            with tf_GradientTape() as tape:  # autograd
                # loss value network
                actor_actions = policy_net.call(in_vec)
                value_network_input = tf_concat([in_vec, actor_actions], axis=1)
                # Original Paper, DDPG Paper and other implementations train on primary network. Why?
                #  Because otherwise the value net is always one gradient step behind
                value_network_score = tf_reduce_mean(self.networks['value1']['primary'].call(value_network_input))
                target = -1 * value_network_score
            grad = tape.gradient(target=target,  # d_loss / d_parameters
                                 sources=policy_net.trainable_variables)
            return grad

        # calculate gradient on current task
        gradients = calculate_policy_net_gradient(policy_net=self.networks['policy']['primary'],
                                                  st=states)

        # Calculate gradients on experiences from prior tasks:
        pt_gradients = []
        for task_experiences in self.prior_experiences:
            pt_states = task_experiences['states']

            pt_gradient = calculate_policy_net_gradient(policy_net=self.networks['policy']['primary'],
                                                        st=pt_states)
            pt_gradients.append(pt_gradient)

        return gradients, pt_gradients
