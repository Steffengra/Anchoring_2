
class GEM:
    def __init__(
            self,
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