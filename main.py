import math
import numpy as np
from BCMP import BcmpNetworkClosed

def get_microservices_scenario():
    # N=5 Systems, R=4 Classes
    # S1: API (FIFO), S2: Auth (FIFO), S3: Data (FIFO), S4: Logic (FIFO, m=2), S5: Logs (IS)
    N, R = 5, 4

    # 1. Populations [Read, Update, Batch, Critical]
    # Keep K low enough to ensure stability for this example
    K = [10, 5, 2, 2]

    # 2. Service Rates (mu) - higher = faster
    # S1(API), S2(Auth), S3(Data), S4(Logic), S5(Logs)
    # Note: 0 indicates the class does not visit this node (or minimal time)
    mi = np.array([
        [100.0, 100.0, 100.0, 100.0],  # S1: Very fast routing
        [0.0, 50.0, 0.0, 60.0],  # S2: Auth (Class 2 & 4 only)
        [40.0, 30.0, 0.0, 0.0],  # S3: Data (Class 1 & 2 only)
        [0.0, 0.0, 5.0, 40.0],  # S4: Logic (Batch is slow, Critical is fast)
        [200.0, 200.0, 200.0, 200.0]  # S5: Logs (IS - infinite capacity)
    ])

    # 3. Transition Matrices (Routing)
    # Nodes: 0=S1, 1=S2, 2=S3, 3=S4, 4=S5

    # Class 1 (Read): S1 -> S3 -> S5 -> S1
    p1 = np.zeros((N, N))
    p1[0, 2] = 1.0;
    p1[2, 4] = 1.0;
    p1[4, 0] = 1.0

    # Class 2 (Update): S1 -> S2 -> S3 -> S5 -> S1
    p2 = np.zeros((N, N))
    p2[0, 1] = 1.0;
    p2[1, 2] = 1.0;
    p2[2, 4] = 1.0;
    p2[4, 0] = 1.0

    # Class 3 (Batch): S1 -> S4 -> S5 -> S1
    p3 = np.zeros((N, N))
    p3[0, 3] = 1.0;
    p3[3, 4] = 1.0;
    p3[4, 0] = 1.0

    # Class 4 (Critical): S1 -> S2 -> S4 -> S5 -> S1
    p4 = np.zeros((N, N))
    p4[0, 1] = 1.0;
    p4[1, 3] = 1.0;
    p4[3, 4] = 1.0;
    p4[4, 0] = 1.0

    p_matrices = [p1, p2, p3, p4]

    # 4. Servers (m) and Types
    m = [1, 1, 1, 2, 1]  # S4 has 2 instances (Load Balanced)
    types = [1, 1, 1, 1, 3]  # S5 is IS (Type 3)

    return N, R, K, mi, p_matrices, m, types


if __name__ == "__main__":
    # Setup
    N, R, K, mi, p_matrices, m, types = get_microservices_scenario()

    # Initialize Solver
    network = BcmpNetworkClosed(R, N, K, mi, p_matrices, m, types)

    # Run Calculation
    network.run_sum_method()

    # Output
    network.report_results()