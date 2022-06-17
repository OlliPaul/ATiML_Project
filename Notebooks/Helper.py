import cv2 as cv
import numpy as np
import tensorflow as tf
from IPython.core.display import clear_output
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans, COPKMeans, MPCKMeansMF, MPCKMeans
from active_semi_clustering.active.pairwise_constraints import ExampleOracle, ExploreConsolidate, MinMax
import active_semi_clustering.active.pairwise_constraints.random
import imagehash
from PIL import Image
from operator import itemgetter


# hyperparamets
def create_zigzag_flat(rows, columns):
    a = np.arange(0, rows * columns, 1, dtype=int).reshape((rows, columns))
    # needed to extact the diagonals because they can only go left to right
    a = np.flipud(a)
    flip = 1
    diagonal = []
    for i in range(1 - rows, columns):
        diagonal.append(np.diagonal(a, i)[::flip])
        flip = -flip
    return np.concatenate(diagonal)


def color_layout_descriptor(im, rows_cld, columns_cld):
    # numpy is a bit faster than using tensors
    im = np.array(im)
    # could do these as a hyperparameter
    rows = rows_cld
    columns = columns_cld
    zigzag_flat = create_zigzag_flat(rows_cld, columns_cld)
    small_image = np.zeros((rows, columns, 3))
    height, width = im.shape[:2]
    percentage_w = width / columns
    percentage_h = height / rows
    for row in range(rows):
        for column in range(columns):
            # Note this partion should propably be checked against a reference implementation. However the difference should be minimal
            portion = im[int(percentage_h * row):int(percentage_h * (row + 1)),
                      int(percentage_w * column):int(percentage_w * (column + 1))]
            small_image[row, column] = np.mean(np.mean(portion, axis=0), axis=0)

    small_image = cv.cvtColor(small_image.astype(np.uint8), cv.COLOR_RGB2BGR)
    small_image = cv.cvtColor(small_image.astype(np.uint8), cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(small_image)
    dct_y = cv.dct(y.astype(np.float32))
    dct_cb = cv.dct(cb.astype(np.float32))
    dct_cr = cv.dct(cr.astype(np.float32))
    return np.concatenate((dct_y.flatten()[zigzag_flat], dct_cb.flatten()[zigzag_flat], dct_cr.flatten()[zigzag_flat]))


# 0.01203213019504501  for 64 ahash
# 0.011844196784313301 for 32 ahash
# 0.007797600576972058 for 16 ahash
# performs bettetter even if euclidian distance is technically wrong here
def average_hashing(im):
    hashed_img = imagehash.average_hash(Image.fromarray(im.numpy().astype(np.uint8)), hash_size=32)

    imhash = np.array(list(str(hashed_img)))

    converted_hash = []
    n = 0
    for i in imhash:
        converted_hash.append(float.fromhex(imhash[n]))
        n = n + 1

    out_hash = np.array(converted_hash) / 16.0
    # print("image hash:", imhash)
    # print("hash converted to array:", out_hash)

    return out_hash


# helper functions
def calculate_label_area(objects, labels):
    obj_labels = objects["label"]
    obj_boundingbox = objects["bbox"]
    return calculate_label_area_helper(obj_labels, obj_boundingbox, labels)


def calculate_label_area_helper(obj_labels, obj_boundingbox, labels):
    areas = []
    for label in labels:
        area_estimate = 0
        foundbbox = []
        for k, boundinglabel in enumerate(obj_labels):
            if (boundinglabel != label): continue
            y_min, x_min, y_max, x_max = obj_boundingbox[k]
            # Here we could calulate the overlap to the other boxes
            area_estimate += (y_max - y_min) * (x_max - x_min)
        areas.append(area_estimate)
    return areas


def get_max_label_helper(obj_labels, obj_boundingbox, labels):
    areas = calculate_label_area_helper(obj_labels, obj_boundingbox, labels)
    max_arg, _ = max(enumerate(areas), key=itemgetter(1))
    return labels[max_arg]


def get_max_label(objects, labels):
    obj_labels = objects["label"]
    obj_boundingbox = objects["bbox"]
    return get_max_label_helper(obj_labels, obj_boundingbox, labels)


from active_semi_clustering.active.pairwise_constraints.example_oracle import MaximumQueriesExceeded


class LabelOracle:
    def __init__(self, data, max_queries_cnt=100, max_querry=False, area_difference_weight=1.0,
                 different_labels_weight=2):
        self.data = data
        self.queries_cnt = 0
        self.max_queries_cnt = max_queries_cnt
        self.different_labels_weight = different_labels_weight
        self.area_difference_weight = area_difference_weight
        if (max_querry):
            self.query = self.query_max
        else:
            self.query = self.query_heuristic

    def query_heuristic(self, i, j):
        "Query the oracle to find out whether i and j should be must-linked"
        if self.queries_cnt < self.max_queries_cnt:
            self.queries_cnt += 1
            set_a = set(self.data[i]["labels"])
            set_b = set(self.data[j]["labels"])
            sameclasses = set_a.intersection(set_b)
            only_a = set_a - set_b
            only_b = set_b - set_a
            if (len(sameclasses) == 0): return False
            same_area_i = sum(calculate_label_area(self.data[i]["objects"], sameclasses))
            same_area_j = sum(calculate_label_area(self.data[j]["objects"], sameclasses))

            diff_area_i = sum(calculate_label_area(self.data[i]["objects"], only_a))
            diff_area_j = sum(calculate_label_area(self.data[j]["objects"], only_b))
            areaquot = same_area_i / same_area_j
            if (areaquot < 1):
                areaquot = 1.0 / areaquot
            score1 = 1 / (1 + self.area_difference_weight * areaquot)
            # heuristics 2 depends on share of similiar clases to difference clasesses
            difference_adjusted_i = diff_area_i / same_area_i
            difference_adjusted_j = diff_area_j / same_area_j
            # heuristics are multiplied
            score2 = 1 / (1 + self.different_labels_weight * (difference_adjusted_j + difference_adjusted_i))
            score12 = score1 * score2
            if (score12 < 0.1):
                if (score12 > 0.05):
                    # print(set_a)
                    # print(set_b)
                    # print(score1)
                    # print(difference_adjusted_i)
                    # print(difference_adjusted_j)
                    # print(score2)
                    # print(score12)
                    # f, ax = plt.subplots(1,2)
                    # ax[0].imshow(self.data[i]["image"])
                    # ax[1].imshow(self.data[j]["image"])
                    # plt.show()
                    pass
                return False
            return True

        else:
            raise MaximumQueriesExceeded

    # helps with evaluation because the result depends only on one label so labeled cluster evaultion methods can be used
    def query_max(self, i, j):
        "Query the oracle to find out whether i and j should be must-linked"
        if self.queries_cnt < self.max_queries_cnt:
            self.queries_cnt += 1
            max_label_i = get_max_label(self.data[i]["objects"], self.data[i]["labels"])
            max_label_j = get_max_label(self.data[j]["objects"], self.data[j]["labels"])
            # if((max_label_i == self.data[j]["labels"][max_arg_j])):
            # f, ax = plt.subplots(1,2)
            # ax[0].imshow(self.data[i]["image"])
            # ax[1].imshow(self.data[j]["image"])
            # plt.show()
            return max_label_i == max_label_j
        else:
            raise MaximumQueriesExceeded


# y for must link, everything else is cannot link
class ExpertOracle:
    def __init__(self, data, max_queries_cnt=100):
        self.data = data
        self.queries_cnt = 0
        self.max_queries_cnt = max_queries_cnt

    def query(self, i, j):
        "Query the oracle to find out whether i and j should be must-linked"
        if self.queries_cnt < self.max_queries_cnt:
            self.queries_cnt += 1
            # return len(set(self.labels[i]).intersection(set(self.labels[j])))>0

            # The synchronised content blocks while waiting for unsynchronized one, not sure if there is a way around this
            # x = asyncio.run(f(self.buttonAcc,self.buttonRej))

            # now i use busy waiting which needs another thread
            f, ax = plt.subplots(1, 2)
            ax[0].imshow(self.data[i]["image"])
            ax[1].imshow(self.data[j]["image"])
            plt.show()
            x = input()
            clear_output(wait=True)
            if x == "y":
                return True
            else:

                return False

        else:
            raise MaximumQueriesExceeded


from scipy.spatial import distance
import time


class KMajority:
    def __init__(self, n_clusters=8, max_iter=100, ):

        self.n_cluster = n_clusters
        self.max_iter = max_iter

    def fit(self, X, y=None, centroid=None):
        X = np.array(X, dtype=bool)
        # random centriods
        if centroid is None:
            centroids = np.random.randint(2, size=(self.n_cluster, X.shape[1]))
            self.centroids = np.array(centroids, dtype=bool)
        else:
            self.centroids = centroid
        centroids = None

        for i in range(self.max_iter):
            # distance from each point in X and the centroids so ouput is shape (number_samples, number_clusters)
            number_samples = X.shape[0]
            number_features = X.shape[1]

            dist = distance.cdist(X, self.centroids, 'hamming')
            # shape (number_samples,)
            cluster_assigned = np.argmin(dist, axis=1)

            # we calculate one hot encodings to select the rows we want to sum over using matrix multiplication
            # test=np.zeros((X.shape[0],self.n_cluster))
            # test[np.arange(X.shape[0]),cluster_assigned]=1
            # out=test.T@X

            row_sum = np.array([X[cluster_assigned == i].sum(axis=0) for i in range(self.n_cluster)])
            row_count = np.array([(cluster_assigned == i).sum(axis=0) for i in range(self.n_cluster)])

            # print(row_sum)
            # print(row_count)
            new_centroids = row_sum > (row_count / 2).reshape(-1, 1)
            # remove empty clusering init randomly again nets bad results but setting it to actual points helps
            nullcheck = row_count == 0
            if (np.any(nullcheck)):
                for k in np.argwhere(nullcheck):
                    # does not work to bad performance in comparision to already fitted clusters
                    # zw=np.random.randint(2, size=(X.shape[1]))

                    index = np.random.choice(X.shape[0], 1, replace=False)
                    new_centroids[k[0]] = X[index]

            # print(centroids[0])
            # print(new_centroids[0])
            print(i)
            if (np.array_equal(self.centroids, new_centroids)):
                self.labels = cluster_assigned
                self.centroids = new_centroids
                return self

            self.labels = cluster_assigned
            self.centroids = new_centroids

        return self

    def predict(self, X):
        X = np.array(X, dtype=bool)
        # distance from each point in X and the centroids so ouput is shape (number_samples, number_clusters)
        number_samples = X.shape[0]
        number_features = X.shape[1]
        dist = distance.cdist(X, self.centroids, 'hamming')
        cluster_assigned = np.argmin(dist, axis=1)
        return cluster_assigned


def filter_dataset(dataset, query_point, number_neighboors=500):
    train_knn_arr = np.array(
        list(dataset.map(lambda x: tf.reshape(tf.image.resize(x["image"], [200, 200]), [-1])).as_numpy_iterator()))

    def get_KNN_points(numb_neigh, query_point, X):
        neigh = NearestNeighbors(n_neighbors=numb_neigh).fit(X)
        dist_arr, ind_array = neigh.kneighbors(X=X[query_point].reshape(1, -1))
        return ind_array.flatten('c')

    filter_ind = get_KNN_points(number_neighboors, query_point, train_knn_arr)
    filter_arr = np.zeros((train_knn_arr.shape[0]))
    filter_arr[filter_ind] = 1
    filter_arr = tf.convert_to_tensor(filter_arr, np.uint8)
    return dataset.enumerate().filter(lambda x, y: filter_arr[x] == 1). \
        map(lambda x, y: y)


def cluster_histogramm(data, predicted_values, y, use_all=True):
    unique, counts = np.unique(predicted_values, return_counts=True)
    n_clusters = len(unique)
    counter = np.zeros((n_clusters, 20))
    for i in range(len(predicted_values)):
        if (use_all):
            for k in data[i]["labels"]:
                counter[predicted_values[i], k] += 1
        else:
            counter[predicted_values[i], y[i]] += 1

    return counter


def calculate_score_per_query(data,filtered_list,n_cluster,y,power_of_query_count=5,tries=1,use_explore_consolidate=False):
    values_random=[]
    values_sil=[]
    querry_counts=[]
    for i in range (power_of_query_count):
        cnt=100*(2**(i))
        querry_counts.append(cnt)
        values_random_per_try=[]
        values_sil_per_try=[]

        for k in range(tries):

            #Gives way more constraints by trying to find neighboorhoods, but ammount is random
            #active_learner = active_semi_clustering.active.pairwise_constraints.explore_consolidate.ExploreConsolidate(n_clusters=20)
            fitted=False
            while (not fitted):
                oracle = LabelOracle(filtered_list, max_queries_cnt=cnt,max_querry=True)
                if(use_explore_consolidate):
                     active_learner = active_semi_clustering.active.pairwise_constraints.explore_consolidate.ExploreConsolidate(n_clusters=n_cluster)
                else:
                    active_learner = active_semi_clustering.active.pairwise_constraints.random.Random(n_clusters=n_cluster)
                active_learner.fit(data, oracle)
                pairwise_constraints = active_learner.pairwise_constraints_
                pck = PCKMeans(n_clusters=n_cluster,max_iter=100,w=500)
                try:
                    pck.fit(data, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
                    values_random_per_try.append(metrics.adjusted_rand_score(y, pck.labels_))
                    values_sil_per_try.append(metrics.silhouette_score(data, pck.labels_, metric='euclidean'))
                    fitted=True
                except Exception as e:
                    if str(e) == "KeyboardInterrupt":
                        raise
                    else:
                        print(str(e))
                    pass
        values_random.append(values_random_per_try)
        values_sil.append(values_sil_per_try)
        print(np.mean(values_random_per_try))
        print(np.mean(values_sil_per_try))

    return values_random,values_sil,querry_counts

def gsf():
    pass