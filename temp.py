from utils.data import generator

if __name__ == '__main__':
    # Init a generator to load/generate record.
    data_generator = generator.Generator()
    # Generate record with scipy's solver.
    # data = data_generator.generate(n_data=1000, shuffle=False, name='g_ns')

    # Load from csv file.
    data = data_generator.load_from_csv(x_path='dataset/X.csv', y_path='dataset/Y.csv', save=True, shuffle=False)
    print(len(data))
