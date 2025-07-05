import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from yolox.tracker import kalman_filter
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1, dtype=np.float32)
    m2 = np.asarray(m2,dtype=np.float32)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches,dtype=int)
    return matches, unmatched_a, unmatched_b

def python_bbox_ious(boxes1, boxes2):
    """Pure Python implementation of IoU calculation as a fallback"""
    boxes1 = np.ascontiguousarray(boxes1, dtype=np.float32)
    boxes2 = np.ascontiguousarray(boxes2, dtype=np.float32)
    
    # Calculate areas for all boxes in boxes1
    areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    
    # Calculate areas for all boxes in boxes2
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    ious = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    
    # Calculate IoUs
    for i in range(len(boxes1)):
        x11, y11, x12, y12 = boxes1[i]
        for j in range(len(boxes2)):
            x21, y21, x22, y22 = boxes2[j]
            
            # Calculate intersection area
            xA = max(x11, x21)
            yA = max(y11, y21)
            xB = min(x12, x22)
            yB = min(y12, y22)
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            unionArea = areas1[i] + areas2[j] - interArea
            
            if unionArea > 0:
                ious[i, j] = interArea / unionArea
    
    return ious

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    # Convert lists to numpy arrays if needed
    if isinstance(atlbrs, list):
        atlbrs = np.array(atlbrs,dtype=np.float32)
    if isinstance(btlbrs, list):
        btlbrs = np.array(btlbrs,dtype=np.float32)
    
    # Handle empty arrays
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    
    # Make contiguous float32 arrays for Cython
    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float32)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float32)
    
    # Reshape if needed - Cython expects 2D arrays
    if len(atlbrs.shape) == 1:
        atlbrs = atlbrs.reshape(1, -1)
    if len(btlbrs.shape) == 1:
        btlbrs = btlbrs.reshape(1, -1)
    
    # Compute IoUs
    ious = python_bbox_ious(atlbrs, btlbrs)
    
    return ious


# filepath: c:\Users\smabbas\miniconda3\envs\collision-detector\Lib\site-packages\yolox\tracker\matching.py
def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    # Force convert to float32 arrays for ANY input type
    if len(atracks) > 0 and len(btracks) > 0:
        if isinstance(atracks[0], np.ndarray) or not hasattr(atracks[0], 'tlbr'):
            atlbrs = np.asarray(atracks, dtype=np.float32)
        else:
            atlbrs = np.array([track.tlbr for track in atracks], dtype=np.float32)
            
        if isinstance(btracks[0], np.ndarray) or not hasattr(btracks[0], 'tlbr'):
            btlbrs = np.asarray(btracks, dtype=np.float32)
        else:
            btlbrs = np.array([track.tlbr for track in btracks], dtype=np.float32)
    else:
        atlbrs = []
        btlbrs = []
        
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs =  np.asarray(atracks, dtype=np.float32) 
        btlbrs =  np.asarray(btracks, dtype=np.float32) 
    else:
        atlbrs = [np.asarray(track.tlwh_to_tlbr(track.pred_bbox), dtype=np.float32) for track in atracks]  
        btlbrs = [np.asarray(track.tlwh_to_tlbr(track.pred_bbox), dtype=np.float32) for track in btracks] 
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements =np.asarray([det.to_xyah() for det in detections], dtype=np.float32) 
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix.astype(np.float32)


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections], dtype=np.float32)
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix.astype(np.float32)


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections], dtype=np.float32)
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost =(1 - fuse_sim).astype(np.float32) 
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections], dtype=np.float32)
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = (1 - fuse_sim).astype(np.float32) 
    return fuse_cost