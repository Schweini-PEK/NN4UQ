import pickle
import csv
import random
from os import listdir
from os.path import join


def preprocessor(src, trg, test=False):
    samples = []
    n_a, n_x = 0, 0
    for f_path in listdir(src):
        alpha = list(map(float, f_path[:-4].split('_')[2::2]))
        n_a = len(alpha)

        with open(join(src, f_path), 'r') as f:
            reader = csv.reader(f)

            if test:
                sample = []
                for line in reader:
                    sample.append(line[1:])
                samples.append([alpha, sample])

            else:
                x_prior = next(reader)[1:]
                n_x = len(x_prior)
                for line in reader:
                    x_future = line[1:]
                    samples.append([x_prior + alpha, x_future])
                    x_prior = x_future

    if test:
        print("TODO")

    else:
        path = trg + '/data_{}_x{}a{}.pkl'.format(len(samples), n_x, n_a)
        with open(path, 'wb') as f_save:
            pickle.dump(samples, f_save)
        print('Save data at {}'.format(path))


# def re_loader(src, trg, test=False):


if __name__ == '__main__':

    dir = "/Users/schweini/Downloads/DataFengzhe"
    # dir = "/Users/schweini/Downloads/test"
    # target = 'dataset/data_72000_x3a5.pkl'
    target = 'dataset'

    # preprocessor(dir, target)
    f = open('dataset/data_72000_x3a5.pkl', 'rb')
    data = pickle.load(f)
    new_data = []
    for i, d in enumerate(data):
        # j = random.randint(0, 400)
        if i % 400 == 177:
            new_data.append(d)

    print(len(new_data))
