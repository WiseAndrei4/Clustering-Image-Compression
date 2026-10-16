from PIL import Image
import numpy as np
from math import sqrt
import random


def calc_center(points):
    n_dim = len(points[0].coordinates)
    vals = [0.0 for n in range(n_dim)]
    for p in points:
        for i in range(n_dim):
            vals[i] += p.coordinates[i]
    return Point([(v / len(points)) for v in vals])

def distance(a: Point, b: Point) -> float:
    n_dim = len(a.coordinates)
    return sqrt(float(sum([
       (a.coordinates[i] - b.coordinates[i]) ** 2 for i in range(n_dim)
    ])))

class Point:
    def __init__(self, coordinates):
        self.coordinates=coordinates


class Cluster:
    def __init__(self, center: Point, points):
        self.center=center
        self.points=points


class KMeans:
    def __init__(self, n_clusters, epsilon=1e-3):
        self.n_clusters=n_clusters
        self.epsilon=epsilon # If the centers of the clusters change less than epsilon, we stop

    def assign_points(self, clusters, points):
        plists=[[] for i in range(len(clusters))]
        for p in points:
            mini=float('inf')
            idx=-1
            for i in range(len(clusters)):
                dist=distance(p, clusters[i].center)
                if dist<mini:
                    mini=dist
                    idx=i
            plists[idx].append(p)
        return plists

    def fit(self, points):
        clusters=[Cluster(center=p, points=[p]) for p in random.sample(points, self.n_clusters)]
        while 1:
            plists=self.assign_points(clusters, points)
            max_diff=0
            for i in range(len(plists)):
                if not plists[i]:
                    continue
                old=clusters[i]
                center=calc_center(plists[i])
                new=Cluster(center=center, points=plists[i])
                clusters[i]=new
                d=distance(old.center, new.center)
                if d>max_diff:
                    max_diff=d
            if max_diff<=self.epsilon:
                break
        return clusters


def get_points(image_path):
    img=Image.open(image_path)
    img=img.convert("RGB")
    img.thumbnail(size=[1000,16*1000//9])

    w,h=img.size

    points=[]

    for count, color in img.getcolors(w*h):
        for _ in range(count):
            points.append(Point(color)) # We transform each pixel's color into a point to do K-Means on the set ** points **
    return points

def rgb_to_hex(rgb):
  return '#%s' % ''.join(('%02x' % p for p in rgb))

def get_colors(filename, n_colors=5):
    points=get_points(filename)
    clusters=KMeans(n_clusters=n_colors,epsilon=1e-3).fit(points)
    clusters.sort(key=lambda c: len(c.points), reverse=True)
    #print([clusters[i].center.coordinates for i in range(len(clusters))])
    return [clusters[i] for i in range(len(clusters))]
    #rgbs=[map(int, c.center.coordinates) for c in clusters]
    #return list(map(rgb_to_hex, rgbs))
