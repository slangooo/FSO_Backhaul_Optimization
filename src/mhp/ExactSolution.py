import random

# nano to mili
N2M = 10**6

class ExactSolution:
    def __init__(self, stopType, found, firstTime, fullTime, firstCorrectInstance, processedInstances, results, mode = 'EXACT'):
        self.stopType             = stopType
        self.found                = found
        self.firstTime            = firstTime
        self.fullTime             = fullTime
        self.firstCorrectInstance = firstCorrectInstance
        self.processedInstances   = processedInstances
        self.results              = results
        self.mode                 = mode
        
        self.bestResults = []
        self.bestTime    = -1
        self.bestScore   = -1
        self.scoreHisto  = []
        
        if self.found:
            for result in self.results: # (solution, fitness, time)
                score = result[1]
                time  = result[2]
                if score >= self.bestScore:
                    if score > self.bestScore:
                        self.bestScore = score
                        self.bestTime  = time
                        self.bestResults = []
                    self.bestResults.append(result)
                    if self.bestTime > time:
                        self.bestTime = time

            
            prevHistoValue = -1
            for histoRange in range(10):
                histoValue = (histoRange+1) * self.bestScore / 10
                histoCount = 0
                histoTime = -1
                for result in self.results:
                    score = result[1]
                    if (score > prevHistoValue) and (score <= histoValue):
                        histoCount += 1
                    if score > prevHistoValue:
                        time = result[2]
                        if histoTime == -1 or histoTime > time:
                            histoTime = time
                if prevHistoValue == -1:
                    prevHistoValue = 0
                self.scoreHisto.append((histoRange, prevHistoValue, histoValue, histoCount, int(1000 * histoTime) if histoTime > 0 else histoTime))
                prevHistoValue = histoValue
        
    def print(self, mode = 'full', showAll = False, printDroneNet = False):
        if mode == 'full':
            print('= ' + self.mode + ' ==================================')
            print('stop  : ', self.stopType)
            if self.found:
                print('time F : ', round(self.firstTime / N2M), 'ms')
                print('time B : ', round(self.bestTime  / N2M), 'ms')
                print('time T : ', round(self.fullTime  / N2M), 'ms')
                print('inst T : ', self.firstCorrectInstance)
                print('inst F : ', self.processedInstances)
                print('total  : ', len(self.results))
                print('best   : ', len(self.bestResults))
                print('score 1: ', self.results[0][1])
                sample = random.choice(self.bestResults)
                print('score  : ', sample[1])
                print('histo  : ', self.scoreHisto)
                print('best result : ', sample[0])
                
                if printDroneNet:
                    printDroneNet()

                if showAll:
                    print('------------------------------------------')
                    for result in self.results:
                        print(result)
            else:
                print('Fail')
            
            print('==========================================')
        elif mode == 'csv':
            print(self.stopType, end=';')
            if self.found:
                print('T', end=';')
                print(self.firstTime / N2M, end=';')
                print(self.fullTime  / N2M, end=';')
                print(self.firstCorrectInstance, end=';')
                print(self.processedInstances, end=';')
                print(len(self.results), end=';')
                print(len(self.bestResults), end=';')
                sample = random.choice(self.bestResults)
                print(sample[1], end=';')
            else:
                print('F', end=';')
                print('' , end=';')
                print(self.fullTime  / N2M, end=';')
                print('' , end=';')
                print(self.processedInstances, end=';')
                print('0' , end=';')
                print('0' , end=';')
                print('' , end=';')
    
    @staticmethod
    def printEmpty(mode):
        if mode == 'csv':
            print('-', end=';')
            print('-', end=';')
            print('' , end=';')
            print('' , end=';')
            print('' , end=';')
            print('' , end=';')
            print('' , end=';')
            print('' , end=';')
            print('' , end=';')
