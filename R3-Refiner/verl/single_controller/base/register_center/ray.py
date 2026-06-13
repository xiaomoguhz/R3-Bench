

import ray


@ray.remote
class WorkerGroupRegisterCenter:
    def __init__(self, rank_zero_info):
        self.rank_zero_info = rank_zero_info

    def get_rank_zero_info(self):
        return self.rank_zero_info


def create_worker_group_register_center(name, info):
    return WorkerGroupRegisterCenter.options(name=name).remote(info)
