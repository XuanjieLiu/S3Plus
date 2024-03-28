from dataloader_plus import Dataset

def remove_duplicate_elements(a_list):
    new_list = []
    for e in a_list:
        if e not in new_list:
            new_list.append(e)
    return new_list

def from_dataset_find_pairs_and_style(dataset: Dataset):
    pairs = [('-').join(f.split('-')[0:2]) for f in dataset.f_list]
    styles = [('-').join(f.split('-')[2:]) for f in dataset.f_list]
    pairs = remove_duplicate_elements(pairs)
    styles = remove_duplicate_elements(styles)
    return pairs, styles
