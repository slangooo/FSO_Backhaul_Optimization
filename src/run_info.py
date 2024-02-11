from datetime import datetime
from src.mhp.DroneNet import DroneNet
from src.mhp.ExactSolution import ExactSolution
from src.mhp.GASolution import GASolution

class Info:

    def __init__(self,
                 iteration, clustering_method, n_drones, ue_rate, max_fso_distance, fso_transmit_power, em_n_iters,
                 dn : DroneNet, mbs_list, dbs_list, fitness_mode):
        bases_bandwidth_max, bases_bandwidth_total = dn.calc_bases_bandwidth()
        edge_number = dn.edge_number()
        self.iteration              = iteration
        self.clustering_method      = clustering_method
        self.n_drones               = n_drones
        self.ue_rate                = ue_rate
        self.max_dist               = max_fso_distance
        self.power                  = fso_transmit_power
        self.em_n_iters             = em_n_iters
        self.mbs                    = len(mbs_list)
        self.dbs                    = len(dbs_list)
        self.fitness_mode           = fitness_mode
        self.edges                  = edge_number
        self.avg_degree             = 2 * edge_number / (len(mbs_list) + len(dbs_list))
        self.total_node_bandwidth   = sum(dn.bandwidth)
        self.bases_bandwidth_max    = bases_bandwidth_max
        self.bases_bandwidth_total  = bases_bandwidth_total
        self.avg_node_bandwidth     = sum(dn.bandwidth) / len(dbs_list)
        self.avg_edge_bandwidth     = (sum(sum(node_net) for node_net in dn.net) / 2) / edge_number if edge_number else 0
        self.avg_bandwidth_per_base = sum(dn.bandwidth) / len(mbs_list)

    def appendExactSolution(self, exactSolution : ExactSolution): self.exactSolution = exactSolution
    def appendGASolution   (self,    gaSolution :    GASolution): self.   gaSolution =    gaSolution

    def exact_solutions (self): return len(self.exactSolution.results)
    def exact_score     (self): return self.exactSolution.bestScore
    def exact_time_first(self): return self.exactSolution.firstTime
    def exact_time_full (self): return self.exactSolution.fullTime
    def ga_score        (self): return self.gaSolution.score
    def ga_time         (self): return self.gaSolution.time

    def found(self):
        return self.exactSolution.found or self.gaSolution.found

    def score_percentage(self):
        if self.gaSolution.score > 0:
            if self.exactSolution.bestScore > 0:
                return 100 * self.gaSolution.score / self.exactSolution.bestScore
            else:
                return 999
        else:
            if self.exactSolution.bestScore > 0:
                return 0
            else:
                return 100

    def msg(self):
        msg = '.' if self.found() else 'X'
        msg += ' ' + 'iteration'              + ' = ' + str(      self.iteration                                 ).rjust(2)
        msg += ' ' + 'clustering_method'      + ' = ' + str(      self.clustering_method                         ).rjust(1)
        msg += ' ' + 'n_drones'               + ' = ' + str(      self.n_drones                                  ).rjust(2)
        msg += ' ' + 'ue_rate'                + ' = ' + str(      self.ue_rate                                   )
        msg += ' ' + 'max_dist'               + ' = ' + str(      self.max_dist                                  ).rjust(5)
        msg += ' ' + 'power'                  + ' = ' + str(      self.power                                     )
        msg += ' ' + 'em_n_iters'             + ' = ' + str(      self.em_n_iters                                ).rjust(2)
        msg += ' ' + 'mbs'                    + ' = ' + str(      self.mbs                                       )
        msg += ' ' + 'dbs'                    + ' = ' + str(      self.dbs                                       ).rjust(2)
        msg += ' ' + 'fitness_mode'           + ' = ' + str(      self.fitness_mode                              )
        msg += ' ' + 'total_node_bandwidth'   + ' = ' + str(      self.total_node_bandwidth                      ).rjust(7)
        msg += ' ' + 'bases_bandwidth_max'    + ' = ' + str(      self.bases_bandwidth_max                       ).rjust(7)
        msg += ' ' + 'bases_bandwidth_total'  + ' = ' + str(      self.bases_bandwidth_total                     ).rjust(7)
        msg += ' ' + 'avg_node_bandwidth'     + ' = ' + str(round(self.avg_node_bandwidth                      ) ).rjust(6)
        msg += ' ' + 'avg_degree'             + ' = ' + str(   f"{self.avg_degree                          :.1f}").rjust(3)
        msg += ' ' + 'edges'                  + ' = ' + str(      self.edges                                     ).rjust(4)
        msg += ' ' + 'avg_edge_bandwidth'     + ' = ' + str(round(self.avg_edge_bandwidth                      ) ).rjust(5)
        msg += ' ' + 'avg_bandwidth_per_base' + ' = ' + str(round(self.avg_bandwidth_per_base                  ) ).rjust(5)
        msg += ' ' + 'exFullTime'             + ' = ' + str(   f"{self.exactSolution.fullTime              :.3f}").rjust(8)
        msg += ' ' + 'exFirstTime'            + ' = ' + str(   f"{self.exactSolution.firstTime             :.3f}").rjust(8)
        msg += ' ' + 'exProcessedInstances'   + ' = ' + str(      self.exactSolution.processedInstances          ).rjust(5)
        msg += ' ' + 'exFirstCorrectInstance' + ' = ' + str(      self.exactSolution.firstCorrectInstance        ).rjust(5)
        msg += ' ' + 'exSolutions'            + ' = ' + str(      self.exact_solutions                   ()      ).rjust(4)
        msg += ' ' + 'exScore'                + ' = ' + str(      self.exact_score                       ()      ).rjust(7)
        msg += ' ' + 'exMode'                 + ' = ' + str(      self.exactSolution.mode                        ).rjust(6)
        msg += ' ' + 'exStopType'             + ' = ' + str(      self.exactSolution.stopType                    ).rjust(6)
        msg += ' ' + 'gaTime'                 + ' = ' + str(   f"{self.ga_time                           ():.3f}").rjust(8)
        msg += ' ' + 'gaGenerations'          + ' = ' + str(      self.gaSolution.generations_runs               ).rjust(8)
        msg += ' ' + 'gaScore'                + ' = ' + str(      self.ga_score                          ()      ).rjust(7)
        msg += ' ' + 'score_percentage'       + ' = ' + str(      self.score_percentage                  ()      ).rjust(3)
        return msg

    def log(self):
        logFile = open('out/info.log', 'a')
        logFile.write(datetime.now().strftime('%Y-%m-%d_%H-%M-%S ') + self.msg() + '\n')
        logFile.close()
    
    @staticmethod
    def csv_header():
        row  =  'found'
        row += ';iteration'
        row += ';clustering_method'
        row += ';n_drones'
        row += ';ue_rate'
        row += ';max_dist'
        row += ';power'
        row += ';em_n_iters'
        row += ';mbs'
        row += ';dbs'
        row += ';fitness_mode'
        row += ';total_node_bandwidth'
        row += ';bases_bandwidth_max'
        row += ';bases_bandwidth_total'
        row += ';avg_node_bandwidth'
        row += ';avg_degree'
        row += ';edges'
        row += ';avg_edge_bandwidth'
        row += ';avg_bandwidth_per_base'
        row += ';exFullTime'
        row += ';exFirstTime'
        row += ';exProcessedInstances'
        row += ';exFirstCorrectInstance'
        row += ';exSolutions'
        row += ';exScore'
        row += ';exMode'
        row += ';exStopType'
        row += ';gaTime'
        row += ';gaGenerations'
        row += ';gaScore'
        row += ';score_percentage'

        csvFile = open('out/info.csv', 'a')
        csvFile.write(row + '\n')
        csvFile.close()

    def csv(self, header=False):
        row = '1' if self.found() else '0'
        row += ';' + str(self.iteration                           )
        row += ';' + str(self.clustering_method                   )
        row += ';' + str(self.n_drones                            )
        row += ';' + str(self.ue_rate                             )
        row += ';' + str(self.max_dist                            )
        row += ';' + str(self.power                               )
        row += ';' + str(self.em_n_iters                          )
        row += ';' + str(self.mbs                                 )
        row += ';' + str(self.dbs                                 )
        row += ';' + str(self.fitness_mode                        )
        row += ';' + str(self.total_node_bandwidth                )
        row += ';' + str(self.bases_bandwidth_max                 )
        row += ';' + str(self.bases_bandwidth_total               )
        row += ';' + str(self.avg_node_bandwidth                  )
        row += ';' + str(self.avg_degree                          )
        row += ';' + str(self.edges                               )
        row += ';' + str(self.avg_edge_bandwidth                  )
        row += ';' + str(self.avg_bandwidth_per_base              )
        row += ';' + str(self.exactSolution.fullTime              )
        row += ';' + str(self.exactSolution.firstTime             )
        row += ';' + str(self.exactSolution.processedInstances    )
        row += ';' + str(self.exactSolution.firstCorrectInstance  )
        row += ';' + str(self.exact_solutions                   ())
        row += ';' + str(self.exact_score                       ())
        row += ';' + str(self.exactSolution.mode                  )
        row += ';' + str(self.exactSolution.stopType              )
        row += ';' + str(self.ga_time                           ())
        row += ';' + str(self.gaSolution.generations_runs         )
        row += ';' + str(self.ga_score                          ())
        row += ';' + str(self.score_percentage                  ())

        csvFile = open('out/info.csv', 'a')
        csvFile.write(row + '\n')
        csvFile.close()
