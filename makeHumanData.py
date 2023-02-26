import random

TOKEN_LIST = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'o', 'p', 'q']
MAX_PLUS_NUM = 6
INIT_NUM = 2

def gen_plus_pairs(token_list, init_num, max_plus_num):
    rand_token = random.sample(token_list, len(token_list))
    token_pair_list = []
    num_pair_list = []
    for i in range(init_num, max_plus_num+1):
        for j in range(i, max_plus_num+1):
            if random.random() > 1:
                a = i
                b = j
            else:
                a = j
                b = i
            num_pair_list.append(f'{a}, {b} => {a+b}')
            token_a = rand_token[a-1]
            token_b = rand_token[b-1]
            token_c = rand_token[a+b-1]
            token_pair_list.append(f'{token_a}, {token_b} => {token_c}')
    return num_pair_list, token_pair_list


def print_list(l):
    for i in l:
        print(i)

if __name__ == "__main__":
    num_pair_list, token_pair_list = gen_plus_pairs(TOKEN_LIST, INIT_NUM, MAX_PLUS_NUM)
    print_list(token_pair_list)
    print_list(num_pair_list)
    #print_list(random.sample(token_pair_list, len(token_pair_list)))

