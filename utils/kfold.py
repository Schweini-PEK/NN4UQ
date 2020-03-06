def k_fold_index_gen(l, k=5):
    split = len(l) // k
    milestone = [0]
    for i in range(1, k):
        milestone.append(i * split)
    milestone.append(len(l))
    return milestone
