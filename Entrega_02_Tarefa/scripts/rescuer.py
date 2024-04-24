import os
import csv
from map import Map
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from abc import ABC
from genetic_algorithm import genetic_algorithm
from a_star import AStar
from IA_classifier import classify_vital_signals
from IA_cluster import clusterize_victims
from genetic_algorithm import genetic_algorithm
import pandas as pd

class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers=1, clusters=[]):
        print("Iniciando rescuer.")

        super().__init__(env, config_file)

        self.nb_of_explorers = nb_of_explorers
        self.received_maps = 0
        self.map = Map()
        self.victims = {}
        self.plan = []
        self.plan_x = 0
        self.plan_y = 0
        self.plan_visited = set()
        self.plan_rtime = self.TLIM
        self.plan_walk_time = 0.0
        self.x = 0
        self.y = 0
        self.clusters = clusters
        self.sequences = clusters

        self.set_state(VS.IDLE)

    def save_sequence_txt(self, sequence, sequence_id):
        print(f"Salvando {sequence_id}")
        filename = f"seq{sequence_id}.txt"  # Nome do arquivo na mesma pasta dos códigos
        with open(filename, 'w') as file:
            for index, row in sequence.iterrows():
                # Adiciona uma coluna zerada antes da última coluna
                row_with_zero = row.tolist()
                row_with_zero.insert(-1, 0)  # Insere o valor zero antes da última coluna

                # Escreve os valores das colunas na ordem determinada pela sequência
                file.write(f"{', '.join(str(value) for value in row_with_zero)}\n")

    def sync_explorers(self, explorer_map, victims):
        
        self.received_maps += 1

        print(f"{self.NAME} Map received from the explorer")
        self.map.update(explorer_map)
        self.victims.update(victims)

        if self.received_maps == self.nb_of_explorers:
            print(f"{self.NAME} all maps received from the explorers")

            classify_vital_signals(self.victims)

            clusters_of_vic = clusterize_victims(self.victims)
  
            # Instancia os outros socorristas
            rescuers = [None] * 4
            rescuers[0] = self # O 0 é o master_rescuer

            self.clusters = [clusters_of_vic[0]]

            for i in range(1, 4):    

                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                rescuers[i] = Rescuer(self.get_env(), config_file, 4, [clusters_of_vic[i]]) 
                rescuers[i].map = self.map  

            # Calcular sequências usando algoritmo genético
            for i, cluster in enumerate(clusters_of_vic):
                best_sequence = genetic_algorithm(cluster)
                self.sequences.append(best_sequence)

            for i, rescuer in enumerate(rescuers):
                for j, sequence in enumerate(rescuer.sequences):
                    sequence_id = i * len(rescuer.sequences) + j + 1
                    self.save_sequence_txt(sequence, sequence_id)

                rescuer.planner()
                rescuer.set_state(VS.ACTIVE)

    def planner(self):
        # Criar objeto A* com os custos especificados
        a_star_instance = AStar(self.map, self.COST_LINE, self.COST_DIAG)

        # Inicializar a posição inicial como (0, 0)
        start = (0, 0)

        # Para cada sequência de vítimas na lista de sequências
        for sequence in self.sequences:
            # Para cada vítima na sequência de vítimas
            for index, row in sequence.iterrows():
                x, y = row['X'], row['Y']
                # Calcular o caminho entre a posição atual e a vítima usando A*
                path, time = a_star_instance.search_astar(start, (x, y), self.plan_rtime)

                # Adicionar o caminho ao plano de salvamento do agente
                self.plan += path
                self.plan_rtime -= time  # Atualizar o tempo restante após calcular o caminho

                # Atualizar a posição inicial para a próxima vítima
                start = (x, y)

        # Calcular o caminho de volta para a base após atender todas as vítimas
        path_back, time_back = a_star_instance.search_astar(start, (0, 0), self.plan_rtime)
        self.plan += path_back
        self.plan_rtime -= time_back

    def deliberate(self) -> bool:

        if self.plan == []:  # empty list, no more actions to do
           print(f"{self.NAME} has finished the plan [ENTER]")
           return False

        dx, dy = self.plan.pop(0)

        walked = self.walk(dx, dy)

        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy

            if self.map.in_map((self.x, self.y)):
                vic_id = self.map.get_vic_id((self.x, self.y))
                if vic_id != VS.NO_VICTIM:
                    self.first_aid()
                 
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        return True