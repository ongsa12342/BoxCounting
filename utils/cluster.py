import numpy as np

def extract_corners(box):
    return [
        (box['x1'], box['y1']),
        (box['x1'], box['y2']),
        (box['x2'], box['y1']),
        (box['x2'], box['y2'])
    ]

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def is_point_inside_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    return x1 <= x <= x2 and y1 <= y <= y2

def is_any_point_inside(box1, box2):
    corners1 = extract_corners(box1)
    return any(is_point_inside_box(corner, box2) for corner in corners1)


def are_boxes_near_or_inside(box1, box2, threshold=10):
    corners1 = extract_corners(box1)
    corners2 = extract_corners(box2)
    near = any(distance(c1, c2) <= threshold for c1 in corners1 for c2 in corners2)
    inside = is_any_point_inside(box1, box2) or is_any_point_inside(box2, box1)
    return near or inside

def merge_clusters(clusters, threshold):
    merged = True
    while merged:
        merged = False
        new_clusters = []
        while clusters:
            cluster = clusters.pop(0)
            for i, other_cluster in enumerate(clusters):
                if any(are_boxes_near_or_inside(box, other_box, threshold) for box in cluster for other_box in other_cluster):
                    cluster.extend(other_cluster)
                    clusters.pop(i)
                    merged = True
                    break
            new_clusters.append(cluster)
        clusters = new_clusters
    return clusters

def cluster_boxes(boxes, threshold=10):
    clusters = []
    for box in boxes:
        added_to_cluster = False
        for cluster in clusters:
            if any(are_boxes_near_or_inside(box, other_box, threshold) for other_box in cluster):
                cluster.append(box)
                added_to_cluster = True
                break
        if not added_to_cluster:
            clusters.append([box])

    clusters = merge_clusters(clusters, threshold)
    return clusters