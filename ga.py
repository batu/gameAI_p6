import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

ELITISM = True
#Do you want elitism?

SUCCESSION_METHOD = 3
# 1 for Roulette Wheel
# 2 for Tournament
# 0 for just mutation

CROSSOVER_METHOD = 1
# 1 for Single Point Crossover

# the multipliers change the frequiency.
# the example level is used as a rough boilerplate
"""
options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin

    #Do not randomly generate a pipe segment. Just do it in conjunction with T
    #"|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]
"""
options = \
    ["X"] * 2 +\
    ["-"] * 300 +\
    ["?"] * 2 +\
    ["M"] * 1 +\
    ["B"] * 5 +\
    ["o"] * 10 +\
    ["T"] * 1 +\
    ["E"] * 2

    #Do not randomly generate a pipe segment. Just do it in conjunction with T
    #"|",  # a pipe segment
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate

# The level as a grid of tiles


class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=1.2,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, genome):
        # STUDENT implement a mutation operator, also consider not mutating this individual
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc

        evolution_rate = .025

        left = 1
        right = width - 3

        # Higher level mutation options
        # increase or decrease the width of the hole
        # Increase or decrease the height of the pipe
        # Increase the flock of coins

        for y in range(height):
            for x in range(left, right):
                if random.random() < evolution_rate:
                    if genome[y][x] in ["B", "?", "B", "M"]:
                        try:
                            rand_int = random.randint(-1,2)
                            if genome[y + rand_int][x + rand_int] not in ["T","|"]:
                                genome[y + rand_int][x + rand_int] = random.choice(["B", "?", "B", "M", "-"])
                        except:
                            pass
                    elif genome[y][x] == "o":
                        try:
                            rand_int = random.randint(-1,2)
                            if genome[y + rand_int][x + rand_int] not in ["T","|"]:
                                genome[y + rand_int][x + rand_int] = random.choice(["o", "o", "o", "-"])
                        except:
                            pass

                    elif y == height:
                        if genome[y + rand_int][x + rand_int] not in ["T","|"]:
                            genome[y][x] = random.choice("-","X")

                    elif genome[y][x] not in ["|", "T"]:
                        genome[y][x] = random.choice(["B", "?", "B", "M", "-"])
                pass

        genome[7][-1] = "v"
        genome[8:14][-1] = ["f"] * 6
        genome[14:16][-1] = ["X", "X"]
        return genome

    # Create zero or more children from self and other
    def generate_children(self, other):

        new_genome = copy.deepcopy(self.genome)
        #new_genome_2
        # Leaving first and last columns alone...
        # do crossover with other
        this_genome = self.genome
        other_genome = other.genome

        if CROSSOVER_METHOD == 1:
            left = 1
            right = width - 3
            for y in range(height - 1):
                cross_over_point = random.randint(left, right)
                for x in range(left, right):
                    if x < cross_over_point:
                        new_genome[y][x] = this_genome[y][x]
                        #new_genome_2[x][y] = other_genome[x][y]
                    else:
                        new_genome[y][x] = other_genome[y][x]
                        #new_genome_2[x][y] = this_genome[x][y]
                    # STUDENT Which one should you take?  Self, or other?  Why?
                    # STUDENT consider putting more constraints on this to prevent pipes in the air, etc

                    pass
        # do mutation; note we're returning a one-element tuple here
        mutated_genome = self.mutate(new_genome)
        return (Individual_Grid(mutated_genome),)

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"


        #Make sure the pipes are always connected to the ground
        left = 1
        right = width - 3
        count_t = 0
        for y in range(height):
            for x in range(left, right):
                if g[y][x] == "T":
                    if y < 10:
                        g[y][x] = "-"
                    else:
                        count_t += 1
                        try:
                            i = 0
                            while True:
                                i += 1
                                g[y + i][x] = "|"
                        except IndexError:
                            pass

        fall_percentage = 0.05
        for x in range(left, right):
            if random.random() < fall_percentage:
                g[height - 1][x] = "-"

        g[7][-1] = "v"
        g[8:14][-1] = ["f"] * 6
        g[14:16][-1] = ["X", "X"]

        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    #add to the value a value between 0 and sqrt(variance)
    #capped by min and max values
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level","applied_Fitness"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None
        applied_Fitness=0


    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Add more metrics?
        # STUDENT Improve this with any code you like
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.9,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        penalties = 0
        # STUDENT For example, too many stairs are unaesthetic.  Let's penalize that
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 5:
            penalties -= 2

        # Penalize 1 width gaps
        if len(list(filter(lambda de: de[1] == "0_holes" and de[2] == 1, self.genome))) > 1:
            penalties -= 0.5

        one_wide_gap_count = len(list(filter(lambda de: de[1] == "0_holes" and de[2] == 1, self.genome)))
        penalties -= min(1, one_wide_gap_count * 0.1)

        #Coins are fun! Add more of them
        coin_count = len(list(filter(lambda de: de[1] == "3_coin", self.genome)))
        penalties += coin_count * 0.05 if coin_count < 30 else -1



        # STUDENT If you go for the FI-2POP extra credit, you can put constraint calculation in here too and cache it in a new entry in __slots__.
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients)) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        # STUDENT How does this work?  Explain it in your writeup.
        # STUDENT consider putting more constraints on this, to prevent generating weird things

        # It has a mutation chance of 15%.
        # and ensures the genome has length.

        # A genome has at most:
        # x_position
        # type
        # width
        # y_position
        # a bool
        # a choice

        #Increased mutation chance
        if random.random() < 0.15 and len(new_genome) > 0:

            # Decides what gene in genome to change.
            to_change = random.randint(0, len(new_genome) - 1)

            # Saves the genome and makes a copy of it.
            de = new_genome[to_change]
            new_de = de

            #Grabs the x component and the type.
            x = de[0]
            de_type = de[1]
            additional_de = None

            #Now gets another random value.
            choice = random.random()

            #Does something different based on what the type is.
            # For allmost all types, the x or y values is slightly tweaked.
            # For certain DE's there is a possibility that something else changes
            # such as whether the block is breakable, or what kind of platform something is
            # 66% chance the position of the block moves
            # remaning 33% something specific happens, explained near the code

            # BLOCK
            # Toggles whether a block is breakable or not
            if de_type == "4_block":
                # Grab the y value and whether it is breakable.
                y = de[2]
                breakable = de[3]
                # Changes the x or y position a little bit, or makes it not breakable?
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    breakable = not de[3]
                new_de = (x, de_type, y, breakable)

            # Question Block
            # Toggles whether the block has power up or not
            elif de_type == "5_qblock":
                y = de[2]
                has_powerup = de[3]  # boolean
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    has_powerup = not de[3]
                new_de = (x, de_type, y, has_powerup)

            # Coin
            # No special case
            elif de_type == "3_coin":
                y = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                if random.random() < 0.3:
                    try:
                        additional_de = new_genome[to_change + 1]
                        additional_de = (additional_de[0], "3_coin", additional_de[2] )
                    except:
                        pass
                new_de = (x, de_type, y)

            # Pipe
            # No special case
            elif de_type == "7_pipe":
                h = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    h = offset_by_upto(h, 2, min=2, max=height - 4)
                new_de = (x, de_type, h)

            # Hole
            # No special case,
            # can toggle with
            elif de_type == "0_hole":
                w = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    w = offset_by_upto(w, 4, min=1, max=width - 2)
                new_de = (x, de_type, w)

            # Stairs
            # Toggles the ?alignment? of the stairs?
            elif de_type == "6_stairs":
                h = de[2]
                dx = de[3]  # -1 or 1
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    h = offset_by_upto(h, 8, min=1, max=height - 4)
                else:
                    dx = -dx
                new_de = (x, de_type, h, dx)

            # Platforms
            # Toggles the type of platforms
            elif de_type == "1_platform":
                w = de[2]
                y = de[3]
                madeof = de[4]  # from "?", "X", "B"
                if choice < 0.25:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.5:
                    w = offset_by_upto(w, 8, min=1, max=width - 2)
                elif choice < 0.75:
                    y = offset_by_upto(y, height, min=0, max=height - 1)
                elif choice:
                    madeof = random.choice(["?", "X", "B"])
                new_de = (x, de_type, w, y, madeof)

            # Does not mutate the enemy
            elif de_type == "2_enemy":
                pass

            new_genome.pop(to_change)
            heapq.heappush(new_genome, new_de)

            if additional_de:
                new_genome.pop(to_change + 1)
                heapq.heappush(new_genome, additional_de)

        return new_genome

    def generate_children(self, other):
        # STUDENT How does this work?  Explain it in your writeup.

        # Gets the a point in both parents.
        pa = random.randint(0, len(self.genome) - 1) if len(self.genome) > 0 else 0
        pb = random.randint(0, len(other.genome) - 1) if len(other.genome) > 0 else 0

        # combine for one child using the first point
        # In this child the first part of the child is the part before pb
        # (point selected in the other) and the latter part is the part of genome
        # that is further away than point pa (the point selected in this genome)
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part

        # combine for the other child
        # The same process as above, only switched ordering
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part

        print(len(ga))
        print(len(gb))
        print()
        # do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(32, 96)
        g = [random.choice([
            # X position, width of hole
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),

            #X position, width, height, choice
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)


#Individual = Individual_Grid
Individual = Individual_DE

def roulette_succession(pop):

    generation_size = len(pop)
    new_generation = []

    if ELITISM:
        ELITISIM_PERCENTAGE = 5
        elitism_count = int(generation_size / (100 / ELITISIM_PERCENTAGE))
        new_generation = heapq.nlargest(elitism_count, pop, key = lambda genome: genome.fitness())


    min_fitness = min(chromosome.fitness() for chromosome in pop)
    sum_fitness = sum(chromosome.fitness() for chromosome in pop)

    last_pick = max(pop, key = lambda genome: genome.fitness())

    while len(new_generation) < generation_size:
        pick = random.uniform(0, sum_fitness)
        current = 0
        for chromosome in pop:
            if len(new_generation) == generation_size:
                break
            #The addition of min fitness is a regularizer to ensure
            # the negative values are not messing with the calculations
            current += (chromosome.fitness() +  min_fitness)

            if current > pick and len(chromosome.genome) != 0 and len(last_pick.genome) != 0:
                new_child = chromosome.generate_children(last_pick)
                last_pick = chromosome
                new_generation += new_child
                continue

    new_sum_fitness = sum(chromosome.fitness() for chromosome in new_generation)
    mean_fitness = new_sum_fitness / generation_size
    print("The mean fitness is : {}".format(mean_fitness))
    return new_generation

def tournament_succession(pop):

    #binary Torny

    '''
    select two randoms
    find fitness of randoms
    choose highest fitness
    return results
    '''
    new_generation=[]
    while(len(new_generation)<len(pop)):
        loopOnce=True
        Parent_B = pop[0]
        Parent_A=Parent_B
        while(loopOnce):

            loopOnce=False
            chosen_A=random.randint(0,len(pop)-1)
            chosen_B = random.randint(0, len(pop)-1)

            chosen_A=pop[chosen_A]
            chosen=chosen_A
            chosen_B=pop[chosen_B]


            if(chosen_B.fitness()>chosen_A.fitness()):
                chosen=chosen_B
            '''
            print(chosen_A.fitness())
            print(chosen_B.fitness())
            print(chosen.fitness())
            '''
            if(loopOnce):
                Parent_A=chosen
            else:
                Parent_B=chosen
        if(len(Parent_B.genome)!=0 and Parent_A.genome!=0):
            kid=Parent_A.generate_children(Parent_B)
            new_generation+=kid
    return new_generation

def rank_succession(pop):
    pop=rank_merge_sort(pop)
    generation_size = len(pop)
    new_generation = []
    for x in new_generation:
        print(x.fitness())


    if ELITISM:
        ELITISIM_PERCENTAGE = 5
        elitism_count = int(generation_size / (100 / ELITISIM_PERCENTAGE))
        new_generation = heapq.nlargest(elitism_count, pop, key = lambda genome: genome.applied_Fitness)


    min_fitness = min(chromosome.applied_Fitness for chromosome in pop)
    sum_fitness = sum(chromosome.applied_Fitness for chromosome in pop)

    last_pick = max(pop, key = lambda genome: genome.applied_Fitness)
    print("here")
    while len(new_generation) < generation_size:
        pick = random.uniform(0, sum_fitness)
        current = 0
        for chromosome in pop:
            if len(new_generation) == generation_size:
                break
            #The addition of min fitness is a regularizer to ensure
            # the negative values are not messing with the calculations
            current += (chromosome.applied_Fitness +  min_fitness)

            if current > pick and len(chromosome.genome) != 0 and len(last_pick.genome) != 0:
                new_child = chromosome.generate_children(last_pick)
                last_pick = chromosome
                new_generation += new_child
                continue


    new_sum_fitness = sum(chromosome.fitness() for chromosome in new_generation)
    mean_fitness = new_sum_fitness / generation_size
    print("after")
    print("The mean fitness is : {}".format(mean_fitness))
    return new_generation

def rank_merge_sort(pop):
    if(len(pop)>1):
        '''
        popIndex=[]
        for x in range(0,len(pop)):
            print(x)
            popIndex.append(int(x))
        '''
        mid=len(pop)/2
        left=pop[:int(mid)]
        right=pop[int(mid):]

        rank_merge_sort(left)
        rank_merge_sort(right)

        i=0
        j=0
        k=0
        while i<len(left)and j<len(right):

            if(left[i].fitness())<right[j].fitness():
                pop[k]=left[i]
                i=i+1
            else:
                pop[k]=right[j]
                j=j+1
            k=k+1
        while i<len(left):
            pop[k]=left[i]
            i=i+1
            k=k+1
        while j<len(right):
            pop[k]=right[j]
            j=j+1
            k=k+1
    i=1
    for x in pop:
        x.applied_Fitness=i
        i+=1

    return pop

def mutation_succession(pop):
    mutated_gen = []
    for dude in pop:
        mutated_gen += dude.mutate(dude)
    return  mutated_gen


# List of individual grid objects
def generate_successors(population):
    #pop.sort(key=lambda x: x.fitness(), reverse=True)

    results = []
    if SUCCESSION_METHOD == 1:
        results = roulette_succession(population)
    elif SUCCESSION_METHOD == 2:
        results = tournament_succession(population)
    elif SUCCESSION_METHOD == 0:
        results = mutation_succession(population)
    elif SUCCESSION_METHOD ==3:
        results = rank_succession(population)




    # STUDENT Design and implement this
    # Hint: Call generate_children() on some individuals and fill up results.
    return results


def ga():
    # STUDENT Feel free to play with this parameter
    pop_limit = 240
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    print()
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1
                # STUDENT Determine stopping condition
                stop_condition = False
                if stop_condition:
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population


if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")
