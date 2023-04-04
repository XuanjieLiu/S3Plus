import sys
import os
from typing import List
from loss_counter import read_record

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))


class ExpGroup:
    def __init__(
            self,
            exp_name: str,
            exp_alias: str,
            sub_exp: str,
            record_name: str,
            exp_root: str = '../VQ/exp',
            check_point: str = ''
            ):
        self.exp_name = exp_name
        self.sub_exps = sub_exp
        self.check_point = check_point
        self.exp_alias = exp_alias
        self.record_name = record_name
        self.file_name = os.path.join(exp_name, sub_exp, check_point)
        self.check_point_path = os.path.join(exp_root, self.file_name)


