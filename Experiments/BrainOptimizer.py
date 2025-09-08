""" Brain Optimizer (Differential Evolution) """

import config
import numpy as np
import numpy.typing as npt
from typing import Any, Tuple
from tqdm import tqdm
from database_components import (
    Base,
    Experiment,
    Generation,
    Genotype,
    Individual,
    Population,
)
from evaluator_brain_targeted_locomotion import Evaluator

from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg import CpgNetworkStructure
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)
from revolve2.experimentation.evolution.abstract_elements import Learner


class BrainOptimizerDE(Learner):
    """Optimizer class (Differential Evolution)"""

    def __init__(
        self,
        bounds: tuple[float, float],
        use_state_reset: bool,
        inherit: bool,
    ) -> None:
        """
        :param bounds: Target spawn bounds (deprecated).
        :param use_state_reset: Bool to toggle using state reset between rollouts.
        :param inherit: Bool to inherit weights from parents.
        """
        self.bounds = bounds
        self.use_state_reset = use_state_reset
        self.inherit = inherit

    def learn(self, population: Population, **kwargs: Any) -> Population:
        """
        Generate individual robots from the population and optimize their weights
        for a targeted locomotion task.

        :param population: Population to go through DE.
        """
        # Generate children bodies and brains
        bodies, brains, solution_sizes = self.setupLearner(population)

        # Initialize child weights if inheritance is NOT chosen
        if not self.inherit:
            population = self.initialSolutionsChildren(population)

        # Make sure solution vectors match current brain sizes
        population = self.setSolutionSizes(population, solution_sizes)

        # Keep a copy of pre-learning fitness (if present)
        for individual in population.individuals:
            individual.old_fitness = individual.fitness

        # Learning loop
        for idx, body in enumerate(tqdm(bodies, leave=False, position=0)):
            cpg_network_structure, output_mapping = brains[idx]
            individual = population.individuals[idx]

            if not hasattr(individual, "fitness_history") or individual.fitness_history is None:
                individual.fitness_history = []

            # Robots with no connections cannot learn here
            if cpg_network_structure.num_connections == 0:
                individual.fitness = -10.0
                pre_learning_fitness = float(individual.old_fitness) if individual.old_fitness is not None else -10.0
                post_learning_fitness = float(individual.fitness)
                individual.learning_delta = post_learning_fitness - pre_learning_fitness
                individual.fitness_history.append(
                    [pre_learning_fitness, post_learning_fitness, individual.learning_delta]
                )
                continue

            # Ensure we have a valid nose orientation on the individual
            nose = getattr(individual, "nose", None)
            assert nose is not None and nose >= 0, (
                "No nose orientation. Call morpho.findNose() on population to set individual.nose."
            )

            # Build evaluator once per individual
            evaluator = Evaluator(
                headless=True,
                num_simulators=config.NUM_SIMULATORS_BRAIN,
                cpg_network_structure=cpg_network_structure,
                body=body,
                output_mapping=output_mapping,
                nose=nose,
                targets=config.TARGETS.copy(),
            )

            # --- Pre-learning evaluation (untrained / current weights) ---
            # Ensure solutions is a 1D vector with correct length (3 * num_connections)
            solutions_vec = np.array(individual.solutions, dtype=float).reshape(1, -1)
            fitnesses, _ = evaluator.evaluate(
                solutions=solutions_vec,
                sim_time=config.SIM_TIME,
                use_state_reset=self.use_state_reset,
            )
            pre_learning_fitness = float(fitnesses[0])
            individual.fitness = pre_learning_fitness  # keep DB in sync

            # --- Differential Evolution training loop ---
            solutions = np.array(individual.solutions, dtype=float)
            sol_t, sol_c = self.generate_T_C(solutions)
            max_fit = pre_learning_fitness
            best_vec = solutions.copy()

            for _ in tqdm(range(config.NUM_GENERATIONS_BRAIN), leave=False):
                sol_next_gen, gen_max_fit = self.optimize(sol_t, sol_c, evaluator)
                # Prepare next generation's T/C
                sol_t, sol_c = self.generate_T_C(sol_next_gen)
                # Track best-so-far (use the true best of this generation)
                if gen_max_fit > max_fit:
                    max_fit = gen_max_fit
                    best_vec = sol_next_gen[0].copy()

            individual.solutions = best_vec.flatten("C").tolist()
            individual.fitness = float(max_fit)

            # --- Post-learning metrics ---
            post_learning_fitness = float(individual.fitness)
            individual.learning_delta = post_learning_fitness - pre_learning_fitness
            individual.fitness_history.append(
                [pre_learning_fitness, post_learning_fitness, individual.learning_delta]
            )

        return population

    def generateTargets(self) -> npt.NDArray[np.float_]:
        """
        [DEPRECATED] Generate list of target coordinates for robots to navigate to.
        """
        targets = np.random.randint(low=self.bounds[0], high=self.bounds[1], size=(20, 2)).astype(float)
        return targets

    def generate_T_C(
        self, T: npt.NDArray[np.float_]
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """
        Generates target and candidate tensors for Differential Evolution.

        Accepts:
        - 1D: a single flattened solution vector (len == 3 * n_conn)
        - 2D: (pop, 3 * n_conn) flattened population
        - 3D: (pop, 3, n_conn) population tensor (already shaped)

        Returns:
        T, C as (pop, 3, n_conn)
        """
        T = np.array(T, dtype=float)
        if T.ndim == 1:
            # Expand single vector into (pop, 3, n_conn)
            assert T.size % 3 == 0, f"Solution length must be multiple of 3, got {T.size}."
            T3 = np.stack([T.reshape(3, T.size // 3)] * config.NUM_POPULATION_BRAIN, axis=0)
            # Add small perturbation to seed the population
            P_pop = np.random.normal(loc=0.0, scale=0.05, size=T3.shape)
            T3 = T3 + P_pop
        elif T.ndim == 2:
            # Reshape flattened population into (pop, 3, n_conn)
            assert T.shape[1] % 3 == 0, f"Second dimension must be multiple of 3, got {T.shape[1]}."
            T3 = T.reshape(T.shape[0], 3, T.shape[1] // 3)
        elif T.ndim == 3:
            # Already in (pop, 3, n_conn)
            assert T.shape[1] == 3, f"Middle dimension must be 3, got {T.shape[1]}."
            T3 = T
        else:
            raise AssertionError(f"Incorrect target matrix shape: {T.shape}")

        pop = T3.shape[0]

        # Mutation: M = T[a] + F * (T[b] - T[c])
        m_1, m_2, m_3 = self.mutationIndices(pop)
        M = T3[m_1] + config.F * (T3[m_2] - T3[m_3])

        # Crossover with binary mask, then clip to [-1, 1]
        cr_mask = np.random.choice([0, 1], size=T3.shape, p=[1 - config.P_CR, config.P_CR])
        C = np.where(cr_mask == 1, M, T3)

        C = np.clip(C, a_min=-1.0, a_max=1.0)
        T3 = np.clip(T3, a_min=-1.0, a_max=1.0)
        return T3, C


    def optimize(
        self,
        T: npt.NDArray[np.float_],
        C: npt.NDArray[np.float_],
        evaluator: Evaluator,
    ) -> Tuple[npt.NDArray[np.float_], float]:
        """
        Compare target vectors with candidate vectors for the next generation.

        :param T: Target vectors (pop_size, 3, num_connections).
        :param C: Candidate solutions (pop_size, 3, num_connections).
        :returns: (Top pop_size solutions for next gen, best fitness)
        """
        # Flatten matrices into solution vectors
        T_flat = np.reshape(T, (len(T), T.shape[1] * T.shape[2]))
        C_flat = np.reshape(C, (len(C), C.shape[1] * C.shape[2]))
        assert T_flat.ndim == 2, f"Incorrect target matrix shape: {T_flat.shape}"

        # Evaluate both T and C together
        solutions = np.vstack((T_flat, C_flat))
        fitnesses, _ = evaluator.evaluate(
            solutions=solutions,
            sim_time=config.SIM_TIME,
            use_state_reset=self.use_state_reset,
        )

        # Sort by fitness (desc)
        sort_idx = np.flip(np.argsort(fitnesses))
        solutions = solutions[sort_idx]

        # Keep top NUM_POPULATION_BRAIN as new T; report best fitness
        next_T = solutions[: config.NUM_POPULATION_BRAIN]
        best_fitness = float(np.max(fitnesses))

        # Reshape back to (pop_size, 3, num_connections)
        next_T = np.reshape(next_T, (config.NUM_POPULATION_BRAIN, 3, int(next_T.shape[1] / 3)))
        return next_T, best_fitness

    def mutationIndices(self, t_pop: int) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """
        Generate the indices for the mutation arrays.

        :param t_pop: No. of target vectors to choose from.
        """
        assert t_pop > 3, f"Need at least 4 vectors to choose 3 mutation vectors. {t_pop} given."

        base = np.arange(0, t_pop, 1)
        m1 = np.random.permutation(t_pop)
        while np.any(m1 == base):
            m1 = np.random.permutation(t_pop)

        m2 = np.random.permutation(t_pop)
        while np.any(m2 == m1) or np.any(m2 == base):
            m2 = np.random.permutation(t_pop)

        m3 = np.random.permutation(t_pop)
        while np.any(m3 == m1) or np.any(m3 == m2) or np.any(m3 == base):
            m3 = np.random.permutation(t_pop)

        return m1, m2, m3

    def setupLearner(
        self, children: Population
    ) -> tuple[
        list[Body],  # Bodies
        list[tuple[CpgNetworkStructure, list[tuple[int, ActiveHinge]]]],  # Brains
        list[int],  # Solution sizes
    ]:
        """
        Generate lists containing the bodies and brains of the population.

        :param children: Population of children.
        """

        bodies = [body.genotype.develop().body for body in children.individuals]
        brains: list[tuple[CpgNetworkStructure, list[tuple[int, ActiveHinge]]]] = []
        sol_sizes: list[int] = []

        for body in bodies:
            active_hinges = body.find_modules_of_type(ActiveHinge)
            cpg_network_structure, output_mapping = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
            brains.append((cpg_network_structure, output_mapping))
            sol_sizes.append(cpg_network_structure.num_connections)

        return bodies, brains, sol_sizes

    def initialSolutions(self, population: Population) -> Population:
        """
        Generate random weights for the initial population.
        """
        _, _, sol_sizes = self.setupLearner(population)

        for idx, sol_size in enumerate(sol_sizes):
            population.individuals[idx].solutions = np.random.uniform(low=-1.0, high=1.0, size=sol_size * 3).tolist()

        return population

    def initialSolutionsChildren(self, population: Population) -> Population:
        """
        Generate new solution vectors for children only.
        """
        _, _, sol_sizes = self.setupLearner(population)

        for idx, sol_size in enumerate(sol_sizes):
            if not population.individuals[idx].solutions:
                population.individuals[idx].solutions = np.random.uniform(
                    low=-1.0, high=1.0, size=sol_size * 3
                ).tolist()

        return population

    def setSolutionSizes(self, children: Population, sol_sizes: list[int]) -> Population:
        """
        Reformat solution vectors to the right sizes.
        This is done by either concatenating weights to the right dimension
        or 'cutting off' unnecessary weights.

        :param children: Population of children.
        :param sol_sizes: Correct sizes of the solution vectors.
        """
        for idx, sol_size in enumerate(sol_sizes):
            if sol_size == 0:
                continue  # Robots with no connections are skipped in learn()

            solutions = np.array(children.individuals[idx].solutions, dtype=float)
            if solutions.size == 0:
                # fill fresh vector if somehow missing
                solutions = np.random.uniform(low=-1.0, high=1.0, size=sol_size * 3)

            assert solutions.size % 3 == 0, f"Solution vector must be multiple of 3. Got {solutions.size}."
            solutions = np.reshape(solutions, (3, int(len(solutions) / 3)))

            # If solutions are too long -> cut off unnecessary part
            if solutions.shape[1] >= sol_size:
                solutions = np.hsplit(solutions, np.array([sol_size, solutions.shape[1] - sol_size]))[0]
            else:
                # If too short -> sample necessary weights and add
                samples = np.random.uniform(low=-1.0, high=1.0, size=(3, sol_size - solutions.shape[1]))
                solutions = np.hstack((solutions, samples))

            children.individuals[idx].solutions = solutions.flatten("C").tolist()

        return children

    def _dummyFitnesses(self, population: Population) -> Population:
        """
        Generate dummy fitness values for a population.
        Only for testing.
        """
        for p in population.individuals:
            p.fitness = float(np.random.normal())

        return population
