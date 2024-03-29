
from numpy.random import (
    default_rng,
)

from src.data.job import Job


class UserEquipment:
    user_id: int
    position: dict
    channel_fading: float
    max_job_size_rb: int
    jobs: list
    rng: default_rng

    def __init__(
            self,
            user_id: int,
            pos_initial: dict,
            max_job_size_rb: int,
            rayleigh_fading_scale: float,
            rng: default_rng,
    ) -> None:
        self.rng = rng

        self.user_id = user_id
        self.position = pos_initial
        self.rayleigh_fading_scale = rayleigh_fading_scale

        self.channel_fading = 0
        self.update_channel_fading()

        self.max_job_size_rb = max_job_size_rb
        self.jobs = []

    def update_channel_fading(
            self
    ) -> None:
        rayleigh_fading = self.rng.rayleigh(scale=self.rayleigh_fading_scale)

        self.channel_fading = rayleigh_fading**2

    def generate_job(
            self
    ) -> None:
        size = self.rng.integers(low=1, high=self.max_job_size_rb + 1)
        job = Job(size_rb=size)
        self.jobs.append(job)
