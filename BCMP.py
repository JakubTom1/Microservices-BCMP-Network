import math
import numpy as np

# Set print options for readability
np.set_printoptions(suppress=True, precision=4)


class BcmpNetworkClosed:
    def __init__(self, R, N, K, mi_matrix, p_matrices, m, types, epsilon=1e-5):
        """
        R: Number of classes
        N: Number of systems (nodes)
        K: Vector of customer populations for each class
        mi_matrix: Service rates (mu) - shape (N, R)
        p_matrices: List of transition matrices for each class
        m: Number of servers at each node
        types: Type of each node (1=FIFO, 3=IS)
        epsilon: Convergence threshold for SUM method
        """
        self.R = R
        self.N = N
        self.K = np.array(K)
        self.K_sum = sum(K)
        self.mi = mi_matrix
        self.m = np.array(m)
        self.types = types
        self.epsilon = epsilon

        # Calculate visit ratios (e) based on transition matrices
        self.e = self._solve_visit_ratios(p_matrices)

        # Initialize throughputs (lambdas) with a small value
        self.lambdas = np.full(self.R, epsilon)

    def _solve_visit_ratios(self, p_matrices):
        """Solves the system of linear equations to find visit ratios (e_ir)."""
        e_matrix = np.zeros((self.N, self.R))

        for r in range(self.R):
            # Formulate A * x = b for class r
            # equation: e_r = e_r * P_r
            P = p_matrices[r]
            A = P.T - np.eye(self.N)

            # Replace last equation with normalization (e.g., e[0] = 1) to solve singularity
            A[-1] = np.zeros(self.N)
            A[-1, 0] = 1.0  # Reference node 0 has visit ratio 1
            b = np.zeros(self.N)
            b[-1] = 1.0

            try:
                e_r = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fallback for singular matrices (using least squares)
                e_r, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            e_matrix[:, r] = e_r

        return e_matrix

    def _calculate_rho(self, i, r):
        """Calculates relative utilization for class r at node i."""
        if self.mi[i, r] == 0:
            return 0

        # Formula depends on node type
        # For IS (Type 3) or FIFO (Type 1) in SUM method context:
        return (self.lambdas[r] * self.e[i, r]) / (self.m[i] * self.mi[i, r])

    def _calculate_rho_total(self, i):
        """Sum of utilizations for all classes at node i."""
        return sum(self._calculate_rho(i, r) for r in range(self.R))

    def _calculate_p_m_i(self, i, rho_i):
        """Calculates marginal probability P(m, i) for multi-server nodes[cite: 311]."""
        m = int(self.m[i])
        if m == 1: return 1.0  # Simplified for m=1
        if rho_i >= 1: return 0.0  # Ergodicity violation check within formula

        # Erlang-C like components
        mr = m * rho_i
        term1 = (mr ** m) / (math.factorial(m) * (1 - rho_i))
        term2 = sum([(mr ** k) / math.factorial(k) for k in range(m)])
        return term1 / (term2 + term1)

    def _get_fix_value(self, i, r, rho_i):
        """Calculates the auxiliary function 'fix' used in SUM method iteration[cite: 331]."""
        # Node Type 3 (Infinite Server)
        if self.types[i] == 3:
            return self.e[i, r] / self.mi[i, r]

        # Node Type 1 (FIFO)
        m = self.m[i]

        # Single Server (m=1)
        if m == 1:
            if self.mi[i, r] == 0: return 0
            # Denominator modification for closed networks
            correction = 1.0 - ((self.K_sum - 1) / self.K_sum) * rho_i
            if correction <= 0: return 1e9  # Prevent division by zero
            return (self.e[i, r] / self.mi[i, r]) / correction

        # Multi Server (m>1)
        else:
            if self.mi[i, r] == 0: return 0
            term1 = self.e[i, r] / self.mi[i, r]
            L = self.e[i, r] / (m * self.mi[i, r])
            correction = 1.0 - ((self.K_sum - m - 1) / (self.K_sum - m)) * rho_i
            if correction <= 0: correction = 1e-9

            p_mi = self._calculate_p_m_i(i, rho_i)
            return term1 + (L / correction) * p_mi

    def run_sum_method(self, max_iter=1000):
        """Executes the iterative SUM method to find class throughputs (lambdas)."""
        print(f"Starting SUM Method (Epsilon: {self.epsilon})...")

        for iteration in range(max_iter):
            prev_lambdas = np.copy(self.lambdas)

            # Recalculate lambdas for each class
            for r in range(self.R):
                denom = 0
                for i in range(self.N):
                    rho_i = self._calculate_rho_total(i)
                    denom += self._get_fix_value(i, r, rho_i)

                if denom > 0:
                    self.lambdas[r] = self.K[r] / denom
                else:
                    self.lambdas[r] = 0

            # Check convergence
            error = np.sqrt(np.sum((self.lambdas - prev_lambdas) ** 2))
            if error < self.epsilon:
                print(f"Converged in {iteration + 1} iterations.")
                return

        print("Warning: Max iterations reached without full convergence.")

    def check_ergodicity(self):
        """Validates if total utilization < number of servers for FIFO nodes."""
        print("\n--- Ergodicity Check ---")
        is_stable = True
        for i in range(self.N):
            if self.types[i] == 3: continue  # IS nodes are always stable

            rho = self._calculate_rho_total(i)
            limit = 1.0 if self.m[i] == 1 else 1.0  # Logic: rho is relative to m in formula

            # Interpretation: rho calculated here is relative intensity.
            # Real load factor is rho. If rho >= 1, queue grows infinitely.
            status = "OK" if rho < 1.0 else "UNSTABLE!"
            if rho >= 1.0: is_stable = False

            print(f"System {i + 1} Load: {rho:.4f} (Limit: 1.0) -> {status}")
        return is_stable

    def report_results(self):
        """Prints formatted results."""
        if not self.check_ergodicity():
            print("\nWARNING: RESULTS MAY BE INVALID DUE TO UNSTABLE NODES.")

        print("\n--- Final Results ---")
        print(f"Throughput (Lambdas): {self.lambdas}")

        print("\nPer-Class Metrics at Systems:")
        print(f"{'Sys':<5} {'Type':<5} {'Class':<5} {'VisitRatio':<12} {'SvcTime(1/u)':<15} {'Utilization':<12}")
        for i in range(self.N):
            t_str = "IS" if self.types[i] == 3 else "FIFO"
            for r in range(self.R):
                svc_time = 0 if self.mi[i, r] == 0 else 1.0 / self.mi[i, r]
                rho = self._calculate_rho(i, r)
                print(f"S{i + 1:<4} {t_str:<5} C{r + 1:<5} {self.e[i, r]:<12.4f} {svc_time:<15.4f} {rho:<12.4f}")