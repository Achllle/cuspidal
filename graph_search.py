from queue import PriorityQueue
import numpy as np


class FullyConnectedDirectionalGraph(object):

    def __init__(self, cost_metric):
        self.edges = []
        self.layers = []
        self.cost_metric = cost_metric

    def add_layer(self, layer):
        if not self.edges:  # initialize first edges with zero cost from single virtual node
            self.edges.append(len(layer) * [0])
        self.layers.append(layer)


def dijkstra(graph: FullyConnectedDirectionalGraph):
    pass