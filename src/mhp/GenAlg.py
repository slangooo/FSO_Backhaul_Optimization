import time

from src.mhp.pygad import GA
from src.mhp.DroneNet import DroneNet
from src.mhp.GASolution import GASolution

class FitnessMode:
    def __init__(self, mode, penalty_value):
        if mode == 'ENP':
            # Edge Surplus Without Penalty (ENP:EST): fitness function based on edge surplus without penalty.
            self.fitness_fitness_mode  = 'edgeSurplus'
            self.fitness_penalty_mode  = 'same'
            self.fitness_penalty_value = 0
            self.use_solution_func     = True
        elif mode == 'EVP':
            # Edge Surplus with Value Penalty (EVP:EVT): fitness function based on edge surplus with a penalty value.
            self.fitness_fitness_mode  = 'edgeSurplus'
            self.fitness_penalty_mode  = 'same'
            self.fitness_penalty_value = penalty_value
            self.use_solution_func     = True
        elif mode == 'EEP':
            # Edge Surplus with Edge Deficits Penalty (EEP:EPT): fitness function based on edge surplus with a penalty corresponding to deficits on edge flows.
            self.fitness_fitness_mode  = 'edgeSurplus'
            self.fitness_penalty_mode  = 'penalty'
            self.fitness_penalty_value = 0
            self.use_solution_func     = True
        elif mode == 'NNP':
            # Node Surplus Without Penalty (NNP:NST): fitness function based on node surplus without penalty.
            self.fitness_fitness_mode  = 'nodeSurplus'
            self.fitness_penalty_mode  = 'same'
            self.fitness_penalty_value = 0
            self.use_solution_func     = True
        elif mode == 'NVP':
            # Node Surplus with Value Penalty (NVP:NVF): fitness function based on node surplus with a penalty value.
            self.fitness_fitness_mode  = 'nodeSurplus'
            self.fitness_penalty_mode  = 'same'
            self.fitness_penalty_value = penalty_value
            self.use_solution_func     = False
        elif mode == 'NEP':
            # Node Surplus with Edge Deficits Penalty (NEP:NPF): fitness function based on node surplus with a penalty corresponding to deficits on edge flows.
            self.fitness_fitness_mode  = 'nodeSurplus'
            self.fitness_penalty_mode  = 'penalty'
            self.fitness_penalty_value = 0
            self.use_solution_func     = False
        
        self.solution_fitness_mode  = 'nodeSurplus'
        self.solution_penalty_mode  = 'penalty'
        self.solution_penalty_value = 0


class GenAlg:
    def __init__(
            self,
            time_limit   =  120, # in seconds, for 0 one run with other conditions, for < 0 run without time limit
            any_solution = True,
            
            num_generations    = 1000,
            num_parents_mating =  200,
            sol_per_pop        = 1000,
            keep_elitism       =  100,
            
            # parameters for fitness function
            fitness_fitness_mode  = 'nodeSurplus', # any : for all correct return 0; edgeSurplus : sum of surplus on edges; nodeSurplus : sum of surplus on nodes
            fitness_penalty_mode  = 'penalty', # penalty_mode : same - same as fitness; penalty - sum of all deficits in edge flows
            fitness_penalty_value = 0, # value to decrease fitness value if result is incorrect

            # parameters for solution function
            use_solution_func      = False,
            solution_fitness_mode  = 'nodeSurplus', # any : for all correct return 0; edgeSurplus : sum of surplus on edges; nodeSurplus : sum of surplus on nodes
            solution_penalty_mode  = 'penalty', # penalty_mode : same - same as fitness; penalty - sum of all deficits in edge flows
            solution_penalty_value = 0, # value to decrease fitness value if result is incorrect

            parent_selection_type = 'sss', # sss : steady_state_selection, rws : roulette_wheel_selection, sus : stochastic_universal_selection, random, tournament, rank
            crossoverFix          = 'best', # best random
            
            mutation_type             = 'random', # random swap inversion scramble adaptive
            mutation_probability_norm = 0.2,
            mutation_num_genes        = 2, # not used if propability set
            
            saturate_stop = 20,
            
            fitness_draw = None):
        
        self.time_limit                = time_limit
        self.any_solution              = any_solution
        self.num_generations           = num_generations
        self.num_parents_mating        = num_parents_mating
        self.sol_per_pop               = sol_per_pop
        self.keep_elitism              = keep_elitism
        self.fitness_fitness_mode      = fitness_fitness_mode
        self.fitness_penalty_mode      = fitness_penalty_mode
        self.fitness_penalty_value     = fitness_penalty_value
        self.use_solution_func         = use_solution_func
        self.solution_fitness_mode     = solution_fitness_mode
        self.solution_penalty_mode     = solution_penalty_mode
        self.solution_penalty_value    = solution_penalty_value
        self.parent_selection_type     = parent_selection_type
        self.crossoverFix              = crossoverFix
        self.mutation_type             = mutation_type
        self.mutation_probability_norm = mutation_probability_norm
        self.mutation_num_genes        = mutation_num_genes
        
        self.stopCriteria = []
        self.stopCriteria.append('saturate_' + str(saturate_stop))
        if (any_solution):
            self.stopCriteria.append('reach_0')
        
        self.fitness_draw = fitness_draw
        
        
    def run(self, dn: DroneNet) -> GASolution:
        start = time.time()
        
        solution = []
        solutionFitness = -1
        executionTime = -1
        total_runs = 0;
        generations_runs = 0;
        positive_runs = 0;
        best_ga = None;
        
        fitness_func = lambda solution, solution_idx: dn.fitnessChain(
            solution, solution_idx, self.fitness_fitness_mode, self.fitness_penalty_mode, self.fitness_penalty_value)
        
        if self.use_solution_func:
            solution_func = lambda solution, solution_idx: dn.fitnessChain(
                solution, solution_idx, self.solution_fitness_mode, self.solution_penalty_mode, self.solution_penalty_value)
        else:
            solution_func = None
        
        firstExecutionTime   =  0
        firstSolutionFitness = -1
        firstSolution        = None

        runs = []

        while ((self.time_limit < 0) or (self.time_limit > executionTime)) and ((not self.any_solution) or (solutionFitness < 0)):
            run_time = time.time()
            ga = GA(
                num_generations         = self.num_generations,
                num_parents_mating      = self.num_parents_mating,
                sol_per_pop             = self.sol_per_pop,
                keep_elitism            = self.keep_elitism,

                initial_population      = dn.randomSolutions(self.sol_per_pop),
                
                num_genes               = dn.node_number()-1,
                gene_type               = int,
                gene_space              = range(dn.node_number()-1),
                allow_duplicate_genes   = False,
                
                fitness_func            = fitness_func,
                solution_func           = solution_func,
                
                parent_selection_type   = self.parent_selection_type,
                crossover_type          = lambda parents, offspring_size, ga: dn.singleChainCrossover(parents, offspring_size, ga, self.crossoverFix), # single_point two_points uniform scattered
                
                mutation_type           = self.mutation_type, # random swap inversion scramble adaptive
                mutation_probability    = self.mutation_probability_norm / dn.drone_number,
                mutation_num_genes      = self.mutation_num_genes, # or mutation_percent_genes
                mutation_by_replacement = True,
                
                #on_crossover = on_crossover,
                #on_mutation  = on_mutation ,
                
                stop_criteria           = self.stopCriteria)
                
            #print(ga_instance.initial_population)
            
            ga.run()

            if self.fitness_draw:
                ga.plot_fitness(show=False, save_dir= self.fitness_draw + '_fitness_' + str(total_runs + 1))

            executionTime = time.time() - start
            
            currentSolutionFitness, currentSolution, currentSolutionFitnessByFitness, currentSolutionByFitness = ga.result_solution()

            generations_runs += ga.generations_completed

            runs.append((run_time - time.time(), currentSolutionFitness, currentSolution, currentSolutionFitnessByFitness, currentSolutionByFitness))

            total_runs += 1
            if currentSolutionFitness >= 0:
                positive_runs += 1

                if firstExecutionTime == 0:
                    firstExecutionTime   = executionTime
                    firstSolutionFitness = currentSolutionFitness
                    firstSolution        = currentSolution
                
            if (total_runs == 1) or (currentSolutionFitness > solutionFitness):
                solutionFitness = currentSolutionFitness
                solution        = currentSolution
                best_ga         = ga
        
        return GASolution(
            best_ga, generations_runs, total_runs, positive_runs, solutionFitness >= 0,
            executionTime, solutionFitness, solution,
            firstExecutionTime, firstSolutionFitness, firstSolution,
            runs)
