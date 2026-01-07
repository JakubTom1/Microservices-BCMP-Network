import math
import numpy as np
from BCMP import BcmpNetworkClosed

def get_microservices_scenario():
    N = 5 # Systems
    R = 4 # Classes

    # Populations [Read, Update, Batch, Critical]
    K = [6, 4, 2, 2]

    # Service Rates (mu) - higher = faster
    # S1: API (FIFO), S2: Auth (FIFO), S3: Data (FIFO), S4: Logic (FIFO, m=2), S5: Logs (IS)
    mi = np.array([
        [100.0, 100.0, 100.0, 100.0],   # S1
        [0.0, 50.0, 0.0, 60.0],         # S2
        [200.0, 30.0, 0.0, 0.0],        # S3
        [0.0, 0.0, 5.0, 40.0],          # S4
        [200.0, 200.0, 200.0, 200.0]    # S5
    ])

    # Transition Matrices (Routing)
    # Class 1 (Read): S1 -> S3 -> S5 -> S1
    p1 = np.zeros((N, N))
    p1[0, 2] = 1.0
    p1[2, 4] = 1.0
    p1[4, 0] = 1.0

    # Class 2 (Update): S1 -> S2 -> S3 -> S5 -> S1
    p2 = np.zeros((N, N))
    p2[0, 1] = 1.0
    p2[1, 2] = 1.0
    p2[2, 4] = 1.0
    p2[4, 0] = 1.0

    # Class 3 (Batch): S1 -> S4 -> S5 -> S1
    p3 = np.zeros((N, N))
    p3[0, 3] = 1.0
    p3[3, 4] = 1.0
    p3[4, 0] = 1.0

    # Class 4 (Critical): S1 -> S2 -> S4 -> S5 -> S1
    p4 = np.zeros((N, N))
    p4[0, 1] = 1.0
    p4[1, 3] = 1.0
    p4[3, 4] = 1.0
    p4[4, 0] = 1.0

    p_matrices = [p1, p2, p3, p4]

    m = [4, 1, 4, 2, 1]
    types = [1, 1, 1, 1, 3]

    return N, R, K, mi, p_matrices, m, types


if __name__ == "__main__":
    # Setup
    N, R, K, mi, p_matrices, m, types = get_microservices_scenario()

    network = BcmpNetworkClosed(R, N, K, mi, p_matrices, m, types)
    network.run_sum_method()
    network.report_results()