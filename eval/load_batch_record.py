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
            sub_exp: List,
            record_name: str,
            exp_root: str = '../VQ/exp',
            ):
        self.exp_name = exp_name
        self.sub_exps = [str(n) for n in sub_exp]
        self.exp_path = os.path.join(exp_root, exp_name)
        self.exp_alias = exp_alias
        self.record_name = record_name
        self.sub_record = self.load_record()

    def load_record(self):
        sub_record = []
        for sub_exp in self.sub_exps:
            sub_record_path = os.path.join(self.exp_path, sub_exp, self.record_name)
            sub_record.append(read_record(sub_record_path))
        return sub_record



if __name__ == '__main__':
    eg = ExpGroup(
        exp_name="2023.03.19_10vq_Zc[2]_Zs[0]_edim1_singleS",
        exp_alias='test',
        sub_exp=[1,2,3,4,5],
        record_name="plus_eval.txt"
    )
    print('aaa')