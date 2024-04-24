import random
import pandas as pd

def genetic_algorithm(cluster):
    population_size = 20
    num_generations = 100
    mutation_rate = 0.1

    def fitness(sequence):
        # Avalia a aptidão de uma sequência (número de vítimas graves socorridas)
        num_grave_victims = sum(1 for index, row in sequence.iterrows() if row.iloc[-1] == 1)
        return num_grave_victims

    def generate_initial_population():
        # Gera a população inicial respeitando as prioridades das classes das vítimas
        population = []
        for _ in range(population_size):
            # Ordena os clusters com base na última coluna (classe)
            sorted_cluster = cluster.copy()
            sorted_cluster.sort_values(by=sorted_cluster.columns[-1], ascending=True, inplace=True)
            population.append(sorted_cluster)
        return population
    
    def mutate(sequence):
        if random.random() < mutation_rate:
            # Realiza uma mutação na sequência
            index1, index2 = random.sample(range(len(sequence)), 2)
            # Extraia as vítimas da sequência
            victims = [row for index, row in sequence.iterrows()]
            # Troca as posições das vítimas nos índices index1 e index2
            victims[index1], victims[index2] = victims[index2], victims[index1]
            # Atualize a sequência com as vítimas mutadas
            sequence = pd.DataFrame(victims, columns=sequence.columns)

    population = generate_initial_population()

    for generation in range(num_generations):
        population = sorted(population, key=fitness, reverse=True)

        # Seleciona os melhores indivíduos para reprodução
        selected = population[:population_size // 2]

        # Reprodução e mutação
        offspring = []
        for _ in range(population_size // 2):
            parent1, parent2 = random.choices(selected, k=2)
            child = parent1[:]
            mutate(child)
            offspring.append(child)

        population = selected + offspring

    best_sequence = max(population, key=fitness)
    return best_sequence