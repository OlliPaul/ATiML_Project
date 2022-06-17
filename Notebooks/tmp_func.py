import active_semi_clustering
from active_semi_clustering.active.pairwise_constraints.example_oracle import MaximumQueriesExceeded
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans,COPKMeans, MPCKMeansMF, MPCKMeans
import numpy as np
from sklearn import metrics
import active_semi_clustering.active.pairwise_constraints.random

def calculate_label_area(objects, labels):
    obj_labels = objects["label"]
    obj_boundingbox = objects["bbox"]
    return calculate_label_area_helper(obj_labels, obj_boundingbox, labels)

from operator import itemgetter
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




from functools import partial
import inspect
import multiprocessing
from multiprocessing import Pool

