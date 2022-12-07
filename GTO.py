import numpy as np
import matplotlib.pyplot as plt


class GTO:
    def __init__(self, fitness, D=30, P=20, G=500, ub=1, lb=0,
                 b=1, a_max=2, a_min=0, a2_max=-1, a2_min=-2, l_max=1, l_min=-1):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.ub = ub * np.ones([self.P, self.D])
        self.lb = lb * np.ones([self.P, self.D])
        self.p = 0.03
        self.beta = 3
        self.w = 0.8

        self.pbest_X = np.zeros([self.P, self.D])
        self.pbest_F = np.zeros([self.P]) + np.inf
        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)

    def opt(self):
        # initialization

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        # Fitness value calculation
        F = self.fitness(self.X)

        # Update the local best position & local fitness value
        fil = F < self.pbest_F
        self.pbest_X[fil] = self.X[fil].copy()
        self.pbest_F[fil] = F[fil].copy()

        # Update the global best position & global fitness value
        if np.amin(F) < self.gbest_F:
            idx = F.argmin()
            self.gbest_X = self.X[idx].copy()
            self.gbest_F = F.min()

        self.loss_curve[0] = self.gbest_F

        # Exploration Phase Loop
        for g in range(1, self.G):
            r1 = np.random.uniform()
            r2 = np.random.uniform()
            F_rule = (np.cos(2 * r1) + 1)  # Equation 3
            C = F_rule * (1 - g / self.G)  # Equation 2
            L = C * (2 * r2 - 1)   # Equation 4  (but  L = C * l ?? why (2 * r2 - 1) ??)

            for i in range(self.P):
                r3 = np.random.uniform()
                # Mitigate to unknown location
                if r3 < self.p:
                    self.X[i] = np.random.uniform(low=self.lb[0], high=self.ub[0])  # Equation 1.1   (but GX(t+1) = (UB - LB) * r1 + LB ?? why there is no (* r1 + LB) ??)
                # Mitigate to other gorillas
                elif r3 >= 0.5:
                    r4 = np.random.uniform()
                    rand_X = self.pbest_X[np.random.randint(low=0, high=self.P)]
                    Z = np.random.uniform(low=-C, high=C, size=[self.D])  # Equation 6
                    H = Z * self.pbest_X[i]  # Equation 5
                    self.X[i] = ((r4 - C) * rand_X + L * H)  # Equation 1.2
                # Mitigate to a known location
                else:
                    r5 = np.random.uniform()
                    rand_X1 = self.X[np.random.randint(low=0, high=self.P)]
                    rand_X2 = self.X[np.random.randint(low=0, high=self.P)]
                    self.X[i] = self.pbest_X[i] - L * (
                                L * (self.pbest_X[i] - rand_X1) + r5 * (self.pbest_X[i] - rand_X2))  # Equation 1.3

            # Boundary processing (Clipping the X values) "Group Formation"
            self.X = np.clip(self.X, self.lb, self.ub)

            # Fitness value calculation
            F = self.fitness(self.X)

            # Update the local best position & local fitness value
            fil = F < self.pbest_F
            self.pbest_X[fil] = self.X[fil].copy()
            self.pbest_F[fil] = F[fil].copy()

            # Update the global best position & global fitness value
            if np.amin(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()

            # Exploitation Phase Loop
            for i in range(self.P):
                # Follow the Silverback
                if C >= self.w:
                    g_var = 2 ** L  # Equation 9
                    M = (np.abs(self.X.mean()) ** g_var) ** (1 / g_var)  # Equation 8
                    self.X[i] = L * M * (self.pbest_X[i] - self.gbest_X) + self.pbest_X[i]  # Equation 7
                # Compete for the adult females (if C < self.w)
                else:
                    r6 = np.random.uniform()
                    if r6 >= 0.5:
                        E = np.random.normal(size=[self.D])  # Equation 13
                    else:
                        E = np.random.normal()

                    r7 = np.random.uniform()
                    Q = (2 * r7 - 1)  # Equation 11
                    A = (self.beta * E)  # Equation 12
                    self.X[i] = self.gbest_X - (self.gbest_X * Q - self.pbest_X[i] * Q) * A  # Equation 10

            # Boundary processing (Clipping the X values) "Group Formation"
            self.X = np.clip(self.X, self.lb, self.ub)

            # Fitness value calculation
            F = self.fitness(self.X)

            # Update the local best position & local fitness value
            fil = F < self.pbest_F   # Filter or Mask
            self.pbest_X[fil] = self.X[fil].copy()
            self.pbest_F[fil] = F[fil].copy()

            # Update the global best position & global fitness value
            if np.amin(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()

            # The Best Fitness
            self.loss_curve[g] = self.gbest_F