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


def task(data,filtered_list,n_cluster,y,power_of_query_count=5,tries=1,use_explore_consolidate=False):
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

from functools import partial
import inspect
import multiprocessing
from multiprocessing import Pool

