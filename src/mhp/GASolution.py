from src.mhp.pygad import GA
from src.mhp.ExactSolution import ExactSolution 

# nano to mili
N2M = 10**6

class GASolution:
    def __init__(self, ga: GA,
                 generations_runs, total_runs, positive_runs, found,
                 time, score, result,
                 first_time, first_score, first_result,
                 runs):
        self.ga               = ga
        self.generations_runs = generations_runs
        self.total_runs       = total_runs
        self.positive_runs    = positive_runs
        self.found            = found
        self.time             = time
        self.score            = score
        self.result           = result
        self.first_time       = first_time
        self.first_score      = first_score
        self.first_result     = first_result
        self.runs             = runs
        
        self.exact                 = False
        self.eaBest                = 0
        self.eaAll                 = 0
        self.gaTop                 = 0
        self.gaScoreFactor         = 0
        self.gaPositionFactor      = 0
        self.gaFirstTop            = 0
        self.gaFirstScoreFactor    = 0
        self.gaFirstPositionFactor = 0
    
    def applyExact(self, exact : ExactSolution = None):
        if exact is not None and exact.found:
            self.exact = True
            
            self.eaBest = exact.bestScore
            self.eaAll  = len(exact.results)

            if self.found:
                self.gaScoreFactor      = 1 if exact.bestScore == 0 else self.score       / exact.bestScore
                self.gaFirstScoreFactor = 1 if exact.bestScore == 0 else self.first_score / exact.bestScore
                
                self.gaTop = 0
                self.gaFirstTop = 0
                for result in exact.results:
                    score = result[1]
                    if score > self.score      : self.gaTop      += 1
                    if score > self.first_score: self.gaFirstTop += 1
                self.gaPositionFactor      = self.gaTop      / self.eaAll
                self.gaFirstPositionFactor = self.gaFirstTop / self.eaAll
            else:
                self.gaScoreFactor         = ''
                self.gaFirstScoreFactor    = ''
                self.gaTop                 = ''
                self.gaFirstTop            = ''
                self.gaPositionFactor      = ''
                self.gaFirstPositionFactor = ''

    def print(self, mode = 'full', plotFitness = False, printDroneNet = False, show_exact = True):
        if mode == 'full':
            print('= GENETIC ================================')
            print("Time : {first_time}/{execution_time} ms".format(first_time=int(round(self.first_time / N2M), round(execution_time=self.time / N2M))))
            print("Runs : {positive_runs}/{total_runs}".format(positive_runs=self.positive_runs, total_runs=self.total_runs))
            print("Generations : {generations_runs}".format(generations_runs=self.generations_runs))
            if self.found:
                print("Score 1   : {score}".format(score=self.first_score))
                print("Score Top : {score}".format(score=self.score))
                print("Solution  : ", self.result)
                
                if show_exact and self.exact:
                    print('First GA versus Exact')
                    print('  val {score} / {best} = {gaSol}'.format(score = self.first_score, best = self.eaBest, gaSol = int(10000*self.gaFirstScoreFactor   )/100))
                    print('  top {gaTop} / {all} = {gaPos}' .format(gaTop = self.gaFirstTop, all  = self.eaAll , gaPos = int(10000*self.gaFirstPositionFactor)/100))
                    print('GA versus Exact')
                    print('  val {score} / {best} = {gaSol}'.format(score = self.score, best = self.eaBest, gaSol = int(10000*self.gaScoreFactor   )/100))
                    print('  top {gaTop} / {all} = {gaPos}' .format(gaTop = self.gaTop, all  = self.eaAll , gaPos = int(10000*self.gaPositionFactor)/100))
                
                if printDroneNet:
                    printDroneNet()
            else:
                print('Genetic algorithm fail')
                for run in self.runs:
                    print('    {:10.3f} : {:6}'.format(run[0], run[1]))
            
            # if plotFitness:
            #    self.ga.plot_fitness(show=False, save_dir='.')

            print('==========================================')
        elif mode == 'csv':
            print(self.first_time / N2M, end=';')
            print(self.time / N2M, end=';')
            print(self.positive_runs, end=';')
            print(self.total_runs, end=';')
            print(self.generations_runs, end=';')
            if self.found:
                print('T', end=';')
                print(self.first_score, end=';')
                print(self.score, end=';')
                
                if show_exact:
                    if self.exact:
                        print('T', end=';')
                        print(self.eaBest, end=';')
                        print(self.gaFirstTop, end=';')
                        print(self.gaTop, end=';')
                        print(self.eaAll, end=';')
                    else:
                        print('F', end=';')
                        print('' , end=';')
                        print('' , end=';')
                        print('' , end=';')
                        print('' , end=';')
            else:
                print('F', end=';')
                print('' , end=';')
                print(self.score, end=';')
                if show_exact:
                    if self.exact:
                        print('T', end=';')
                        print(self.eaBest, end=';')
                        print('', end=';') # self.gaFirstTop
                        print('', end=';') # self.gaTop
                        print(self.eaAll, end=';')
                    else:
                        print('F', end=';')
                        print('' , end=';')
                        print('' , end=';')
                        print('' , end=';')
                        print('' , end=';')
    
    @staticmethod
    def printEmpty(mode, show_exact = True):
        if mode == 'csv':
            print('-', end=';')
            print('-', end=';')
            print('' , end=';')
            print('' , end=';')
            print('' , end=';')
            print('' , end=';')
            print('' , end=';')
            print('' , end=';')
            if show_exact:
                print('' , end=';')
                print('' , end=';')
                print('' , end=';')
                print('' , end=';')
                print('' , end=';')
