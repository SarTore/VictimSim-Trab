from collections import deque
from vs.constants import VS
from map import Map
import math

class AStar:
    def __init__(self, map, cost_line=1.0, cost_diag=1.5):
        self.map = map
        self.frontier = None
        self.cost_line = cost_line
        self.cost_diag = cost_diag
        self.tlim = float('inf')
        
        self.incr = {
            0: (0, -1),  # u: Cima
            1: (1, -1),  # ur: Diagonal superior direita
            2: (1, 0),   # r: Direita
            3: (1, 1),   # dr: Diagonal inferior direita
            4: (0, 1),   # d: Baixo
            5: (-1, 1),  # dl: Diagonal inferior esquerda
            6: (-1, 0),  # l: Esquerda
            7: (-1, -1)  # ul: Diagonal superior esquerda
        }

    def get_possible_actions(self, pos):
        x, y = pos
        actions = []

        if self.map.in_map(pos):
            incr = 0
            for key in self.incr:
                possible_pos = self.map.get_actions_results(pos)
                if possible_pos[incr] == VS.CLEAR:
                    actions.append((self.incr[key][0], self.incr[key][1]))

                incr += 1

        return actions

    def in_the_frontier(self, pos):
        for node in self.frontier:
            frontier_pos, _, _ = node
            if pos == frontier_pos:
                return True
            
        return False

    def heuristic(self, pos, goal):
        # Usar a distância euclidiana como heurística
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def search_astar(self, start, goal, tlim=float('inf')):
        self.tlim = tlim
        selected = set()
        self.frontier = deque([(start, [], 0)])
        if start == goal:
            return [], 0
        
        while self.frontier:
            current_pos, plan, acc_cost = self.frontier.popleft()
            selected.add(current_pos)
            possible_actions = self.get_possible_actions(current_pos)

            for action in possible_actions:
                child = (current_pos[0] + action[0], current_pos[1] + action[1])
                
                if self.map.in_map(child) and child not in selected and not self.in_the_frontier(child):
                    difficulty = self.map.get_difficulty(child)
                    if action[0] == 0 or action[1] == 0:
                        new_acc_cost = acc_cost + self.cost_line * difficulty
                    else:
                        new_acc_cost = acc_cost + self.cost_diag * difficulty

                    new_plan = plan + [action]
                    h_cost = self.heuristic(child, goal)  # Calcular custo heurístico
                    f_cost = new_acc_cost + h_cost       # Calcular custo total
                    
                    if child == goal:
                        if new_acc_cost > self.tlim:
                            return [], -1    # Tempo excedido

                        return new_plan, new_acc_cost

                    self.frontier.append((child, new_plan, new_acc_cost))

            # Ordenar a fronteira com base no custo total
            self.frontier = deque(sorted(self.frontier, key=lambda x: x[2] + self.heuristic(x[0], goal)))

        return None, 0  # Caminho não encontrado