import os


DATA_ROOT = 'dataset/single_style_pairs(0,20)_tripleSet/'
TRAIN_DATA_PATH = f'{DATA_ROOT}/train'
TEST_1_DATA_PATH = f'{DATA_ROOT}/test_1'
TEST_2_DATA_PATH = f'{DATA_ROOT}/test_2'


def data_name2pair(data_name):
    data_name = data_name.split('-')
    return int(data_name[0]), int(data_name[1])


def load_data_list(data_path):
    data_names = os.listdir(data_path)
    data_pairs = [data_name2pair(data_name) for data_name in data_names]
    return data_pairs


def detect_repeat_pair(data_pairs):
    for i in range(0, len(data_pairs)):
        for j in range(i + 1, len(data_pairs)):
            if data_pairs[i][0] == data_pairs[j][1] and data_pairs[i][1] == data_pairs[j][0]:
                print(f'Repeat pair: {data_pairs[i]} and {data_pairs[j]}')
    print('Done')


def detect_none_repeat_pair(data_pairs):
    for i in range(0, len(data_pairs)):
        has_pair = False
        for j in range(0, len(data_pairs)):
            if data_pairs[i][0] == data_pairs[j][1] and data_pairs[i][1] == data_pairs[j][0]:
                has_pair = True
                break
        if has_pair:
            continue
        else:
            print(f'None repeat pair: {data_pairs[i]}')
    print('Done')


def validate_train_set(start, end):
    print(f'Validate train set from {start} to {end}')
    data_pairs = load_data_list(TRAIN_DATA_PATH)
    no_zero_pairs = [data_pair for data_pair in data_pairs if data_pair[0] != 0 and data_pair[1] != 0]
    pair_sums = [data_pair[0] + data_pair[1] for data_pair in no_zero_pairs]
    for i in range(start, end+1):
        if i not in pair_sums:
            print(f'No pair sum: {i}')
    print('Done')


def detect_repeat_pair_from_subset(data_path):
    print(f'Detect repeat pair from {data_path}')
    data_pairs = load_data_list(data_path)
    detect_repeat_pair(data_pairs)


def detect_none_repeat_pair_from_subset(data_path):
    print(f'Detect none repeat pair from {data_path}')
    data_pairs = load_data_list(data_path)
    detect_none_repeat_pair(data_pairs)




if __name__ == "__main__":
    print("Hello World")
    detect_repeat_pair_from_subset(TRAIN_DATA_PATH)
    detect_repeat_pair_from_subset(TEST_2_DATA_PATH)
    detect_none_repeat_pair_from_subset(TEST_1_DATA_PATH)
    validate_train_set(2, 20)
