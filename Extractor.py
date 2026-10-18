import random
from math import sqrt
import numpy as np
from PIL import Image


class Point:
    def __init__(self, coordinates):
        self.coordinates = list(coordinates)

    def __repr__(self):
        return f"Point({self.coordinates})"


class Cluster:
    def __init__(self, center: Point, points=None):
        self.center = center
        self.points = points if points is not None else []

    def __repr__(self):
        return f"Cluster(center={self.center}, n_points={len(self.points)})"

class KMeans:
    # A demonstrative custom KMeans class, that is used in the get_colors function
    def __init__(self, n_clusters, max_iter=100, epsilon=1e-3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.epsilon = epsilon  # If the centers of the clusters change less than epsilon, we stop

    def assign_points(self, clusters, points):
        plists = [[] for _ in range(len(clusters))]
        for p in points:
            mini = float('inf')
            idx = 0
            for i in range(len(clusters)):
                dist = distance(p, clusters[i].center)
                if dist < mini:
                    mini = dist
                    idx = i
            plists[idx].append(p)
        return plists

    def fit(self, points):
        if len(points) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} points, but got {len(points)}.")
        clusters = [Cluster(center=p, points=[p]) for p in random.sample(points, self.n_clusters)]
        for _ in range(self.max_iter):
            plists = self.assign_points(clusters, points)
            max_diff = 0.0
            for i in range(len(plists)):
                if not plists[i]:
                    # Reinitialize empty cluster with a random point to prevent dead clusters
                    clusters[i] = Cluster(center=random.choice(points), points=[])
                    continue
                old = clusters[i]
                center = calc_center(plists[i])
                new = Cluster(center=center, points=plists[i])
                clusters[i] = new
                d = distance(old.center, new.center)
                if d > max_diff:
                    max_diff = d
            if max_diff <= self.epsilon:
                break
        return clusters


def calc_center(points):
    if not points:
        return Point([0.0, 0.0, 0.0])
    n_dim = len(points[0].coordinates)
    vals = [0.0 for _ in range(n_dim)]
    for p in points:
        for i in range(n_dim):
            vals[i] += p.coordinates[i]
    n = len(points)
    return Point([(v / n) for v in vals])


def distance(a: Point, b: Point) -> float:
    n_dim = len(a.coordinates)
    return sqrt(float(sum([
        (a.coordinates[i] - b.coordinates[i]) ** 2 for i in range(n_dim)
    ])))


def get_points(image_path, thumbnail_size=(150, 150)):
    img = Image.open(image_path).convert("RGB")
    if thumbnail_size is not None:
        img.thumbnail(size=thumbnail_size)

    w, h = img.size
    points = []
    colors = img.getcolors(w * h)
    if colors is not None:
        for count, color in colors:
            for _ in range(count):
                points.append(Point(color))
    else:
        for color in img.getdata():
            points.append(Point(color))
    return points


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(int(round(p)) for p in rgb)


def get_colors(filename, n_colors=5, max_iter=30, epsilon=1.0, thumbnail_size=(150, 150)):
    # For comparison, a function that uses manually implemented KMeans
    points = get_points(filename, thumbnail_size=thumbnail_size)
    clusters = KMeans(n_clusters=n_colors, max_iter=max_iter, epsilon=epsilon).fit(points)
    clusters.sort(key=lambda c: len(c.points), reverse=True)
    return clusters


def get_colors_numpy(filename, n_colors=5, max_iter=100, tol=1e-3, return_all=False):
    # A faster method than the other ( get_colors ) using NumPy vectorization
    img = Image.open(filename).convert("RGB")
    img.thumbnail(size=[1000, 16 * 1000 // 9])
    data = np.array(img).reshape(-1, 3).astype(np.float32)

    if data.shape[0] < n_colors:
        raise ValueError(f"Image has fewer pixels ({data.shape[0]}) than requested colors ({n_colors})")

    indices = np.random.choice(data.shape[0], n_colors, replace=False)
    centers = data[indices].copy()

    for _ in range(max_iter):
        distances = np.sum((data[:, np.newaxis] - centers) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centers = np.empty_like(centers)
        for i in range(n_colors):
            pts = data[labels == i]
            if len(pts) > 0:
                new_centers[i] = pts.mean(axis=0)
            else:
                new_centers[i] = data[np.random.choice(data.shape[0])]

        shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
        centers = new_centers
        if shift < tol:
            break

    if return_all:
        return centers, labels, data
    return centers


def extract_palette(clusters_or_centers):
    """
    Extracts a numpy array of shape (K, 3) with dtype uint8 from:
    - a list of Cluster objects (from manual get_colors)
    - a numpy array of centers (from get_colors_numpy)
    - a tuple (centers, labels, data)
    """
    if isinstance(clusters_or_centers, tuple):
        clusters_or_centers = clusters_or_centers[0]
    if isinstance(clusters_or_centers, list) and len(clusters_or_centers) > 0 and hasattr(clusters_or_centers[0], 'center'):
        return np.array([np.round(c.center.coordinates) for c in clusters_or_centers], dtype=np.uint8)
    return np.round(np.array(clusters_or_centers)).astype(np.uint8)


def apply_palette(image_source, palette):
    """
    Applies a color palette to an image, mapping every pixel to the nearest palette color.

    Args:
        image_source: The image's path (string)
        palette: Cluster list, centers array, or (K, 3) array.

    Returns:
        PIL.Image.Image: The image reconstructed with the reduced color palette.
    """

    img=Image.open(image_source).convert("RGB")

    palette_arr = extract_palette(palette).astype(np.float32)
    img_arr = np.array(img)
    flat_pixels = img_arr.reshape(-1, 3).astype(np.float32)

    distances = np.sum((flat_pixels[:, np.newaxis] - palette_arr) ** 2, axis=2)
    nearest = np.argmin(distances, axis=1)

    final_flat = palette_arr[nearest].astype(np.uint8)
    return Image.fromarray(final_flat.reshape(img_arr.shape))