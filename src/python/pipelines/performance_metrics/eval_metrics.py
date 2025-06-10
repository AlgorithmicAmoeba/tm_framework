import numpy as np


random_seed = 42
np.random.seed(random_seed)


T, W, E = 7, 11, 13


def sim(v1: np.ndarray, v2: np.ndarray) -> float:
    return - np.dot(v1, v2) 


def weps_score_slow(topics_tensor: np.ndarray) -> float:

    score = 0.0
    count = 0
    for t_i in range(T):
        for t_j in range(t_i + 1, T):
            for w_i in range(W):
                for w_j in range(W):
                    score += sim(topics_tensor[t_i, w_i, :], topics_tensor[t_j, w_j, :])
                    count += 1

    return score, count, score / count


def weps_score_fast(topics_tensor: np.ndarray) -> float:
    v_1 = np.einsum('twe,sve->', topics_tensor, topics_tensor)
    v_2 = np.einsum('twe,tve->', topics_tensor, topics_tensor)
    score = -(v_1 - v_2) / 2
    count = T * (T - 1) * W * W / 2
    return score, count, score / count


def weps_equality(topics_tensor: np.ndarray):
    print(weps_score_slow(topics_tensor))
    print(weps_score_fast(topics_tensor))


def wecs_score_slow(topics_tensor: np.ndarray) -> float:
    score = 0.0
    count = 0
    for t in range(T):
        for w_i in range(W):
            for w_j in range(w_i + 1, W):
                score += sim(topics_tensor[t, w_i, :], topics_tensor[t, w_j, :])
                count += 1
    return score, count, score / count


def wecs_score_fast(topics_tensor: np.ndarray) -> float:
    v_1 = np.einsum('twe,tve->', topics_tensor, topics_tensor)
    v_2 = np.einsum('twe,twe->', topics_tensor, topics_tensor)
    score = -(v_1 - v_2) / 2
    count = T * W * (W - 1) / 2
    return score, count, score / count


def wecs_equality(topics_tensor: np.ndarray):
    print(wecs_score_slow(topics_tensor))
    print(wecs_score_fast(topics_tensor))


if __name__ == '__main__':
    topics_tensor = np.random.rand(T, W, E)
    wecs_equality(topics_tensor)
    weps_equality(topics_tensor)