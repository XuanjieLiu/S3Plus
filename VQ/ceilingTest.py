def decimal_to_base(n, base):
    if not 2 <= base <= 36:
        raise ValueError("Base must be between 2 and 36")
    if n == 0:
        return "0"

    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""
    while n > 0:
        remainder = n % base
        result = digits[remainder] + result
        n //= base
    return result


def add_len(n: str, len_to_be: int):
    s = n
    while len(s) < len_to_be:
        s = f'0{s}'
    return s


decimal_number = 10
base_5_number = decimal_to_base(decimal_number, 4)
print(base_5_number)


class DiyCodebook:
    def __init__(self, n_dim, dim_size, v_range=(-1., 1.)):
        self.n_dim = n_dim
        self.dim_size = dim_size
        self.v_range = v_range
        self.dim_points = self.init_dim_points()

    def init_dim_points(self):
        interval = (self.v_range[1] - self.v_range[0]) / (self.dim_size - 1)
        points = [self.v_range[0]]
        base = self.v_range[0]
        for i in range(0, self.dim_size-1):
            base += interval
            points.append(base)
        return points

    def linear_book(self):
        for i in range(0, pow(self.dim_size, self.n_dim)):
            trans_num = decimal_to_base(i, self.dim_size)
            num_str = add_len(trans_num, self.n_dim)
            



        return


diy_codebook = DiyCodebook(3, 4)
print(diy_codebook.dim_points)
