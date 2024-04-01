import sys
import os
import random
import math
import heapq
from abc import ABC
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc):
        super().__init__(env, config_file)
        self.walk_stack = Stack()
        self.set_state(VS.ACTIVE)
        self.resc = resc
        self.x = 0
        self.y = 0
        self.map = Map()
        self.victims = {}

        # Put the current position (base) in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    def get_next_position(self):
        obstacles = self.check_walls_and_lim()
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Clockwise starting from up
        visited = set()
        stack = [(self.x, self.y)]

        while stack:
            x, y = stack.pop()
            visited.add((x, y))
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and self.is_clear_position(obstacles, nx, ny):
                    return dx, dy  # Found a clear position, return the direction
                    stack.append((nx, ny))

        return random.choice([(0, -1), (1, 0), (0, 1), (-1, 0)])  # Default to random if DFS fails

    def is_clear_position(self, obstacles, x, y, can_jump=False):
        if isinstance(obstacles, list) and obstacles:
            if isinstance(obstacles[0], list) and obstacles[0]:
                if 0 <= x < len(obstacles) and 0 <= y < len(obstacles[0]):
                    if can_jump:
                        return True  # Consider the position as clear if the agent can jump over obstacles
                    else:
                        return obstacles[x][y] == 0  # Consider the position as clear only if there is no obstacle
        return False

    def explore(self):
        dx, dy = self.get_next_position()
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        if result == VS.EXECUTED:
            self.walk_stack.push((dx, dy))
            self.x += dx
            self.y += dy

            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")

            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())

    def come_back(self):
        start = (self.x, self.y)
        end = (0, 0)
        path = self.a_star(start, end)

        if path is None:
            print("Error: A* failed to find a path back to the base")
            return 

        for position in path:
            dx = position[0] - self.x
            dy = position[1] - self.y
            result = self.walk(dx, dy)
            if result != VS.EXECUTED:
                print(f"Error: Unable to execute step ({dx}, {dy}) in A* path")
                return

            # Debugging: Print map state after each step
            print("Current Map State:")
            self.map.print_map()  # Assuming a method like this exists in your Map class

        self.x, self.y = end
        self.resc.go_save_victims(self.map, self.victims)

    def a_star(self, start, end):
        def heuristic(pos):
            return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            current_cost, current_pos = heapq.heappop(open_set)

            if current_pos == end:
                path = []
                while current_pos in came_from:
                    path.append(current_pos)
                    current_pos = came_from[current_pos]
                return path[::-1]

            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                neighbor = (current_pos[0] + dx, current_pos[1] + dy)
                new_cost = g_score[current_pos] + 1

                if self.is_clear_position([], neighbor[0], neighbor[1]):
                    if neighbor not in g_score or new_cost < g_score[neighbor]:
                        g_score[neighbor] = new_cost
                        f_score = new_cost + heuristic(neighbor)
                        heapq.heappush(open_set, (f_score, neighbor))
                        came_from[neighbor] = current_pos

        return None

    def deliberate(self) -> bool:
        consumed_time = self.TLIM - self.get_rtime()
        if consumed_time < self.get_rtime():
            self.explore()
            return True

        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the rescuer")
            self.resc.go_save_victims(self.map, self.victims)
            return False

        self.come_back()
        return True
