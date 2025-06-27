import numpy as np

def estimate_velocity(pos_now, pos_prev):
    """Estimate velocity vector given current and previous positions."""
    return np.array(pos_now) - np.array(pos_prev)

def are_approaching(pos1_now, pos1_prev, pos2_now, pos2_prev):
    """Check if two objects are moving towards each other using dot product."""
    v1 = estimate_velocity(pos1_now, pos1_prev)
    v2 = estimate_velocity(pos2_now, pos2_prev)
    rel_pos = np.array(pos2_now) - np.array(pos1_now)
    rel_vel = v2 - v1
    # Negative dot product means approaching
    return np.dot(rel_pos, rel_vel) < 0

def collision_prob(dist, threshold=50, scale=10):
    """Sigmoid-based collision probability based on distance."""
    return 1 / (1 + np.exp((dist - threshold) / scale))

def estimate_collision(obj1, obj2):
    """
    Estimate collision probability between two tracked objects.

    Args:
        obj1, obj2: dicts with keys:
            - 'pos_now': (x, y)
            - 'pos_prev': (x, y)

    Returns:
        probability: float in [0, 1]
        approaching: bool
        distance: float
    """
    pos1_now, pos1_prev = obj1['pos_now'], obj1['pos_prev']
    pos2_now, pos2_prev = obj2['pos_now'], obj2['pos_prev']

    distance = np.linalg.norm(np.array(pos1_now) - np.array(pos2_now))
    approaching = are_approaching(pos1_now, pos1_prev, pos2_now, pos2_prev)
    prob = collision_prob(distance) if approaching else 0.0
    return prob, approaching, distance