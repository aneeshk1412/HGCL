import scipy.sparse as sp
import numpy as np
import csv
import pickle
import datetime

if __name__ == "__main__":
    print(datetime.datetime.now())

    # Create trust csr from user_graph.npz
    user_graph = sp.load_npz('./user_graph.npz')
    num_users = user_graph.shape[0]
    trustMat = sp.csr.csr_matrix(user_graph, dtype=np.int64)
    with open('./trust.csv', 'wb') as f:
        pickle.dump(trustMat, f)

    # Create category csr from item_categories.csv
    categories = np.loadtxt('item_categories.csv', skiprows=1, dtype=np.int32, delimiter=',')
    num_items = categories.shape[0]
    catMat = sp.csr.csr_matrix(categories, dtype=np.int32)
    with open('./category.csv', 'wb') as f:
        pickle.dump(catMat, f)

    # Create ratings csr from data.csv
    user_ids = []
    item_ids = []
    ratings = []

    with open('./data.csv', 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            user_ids.append(int(row['user_id']))
            item_ids.append(int(row['item_id']))
            ratings.append(int(row['rating']))

    ratings_matrix = sp.csr.csr_matrix((ratings, (user_ids, item_ids)), shape=(num_users, num_items), dtype=np.int64)
    with open('./ratings.csv', 'wb') as f:
        pickle.dump(ratings_matrix, f)

    print(datetime.datetime.now())
