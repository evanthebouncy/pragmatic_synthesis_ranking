# Rank aggregation

import numpy
import random

def generate_dicts(pages, ranks):
    '''
        Generate the dictionaries that map pages available in the partial rankings to their indices
    '''
    page_to_lists = {}

    for page in pages:
        page_to_lists[page] = []
        for r in ranks:
            if(page in r):
                page_to_lists[page].append(ranks.index(r))

    return page_to_lists


def generate_MC3_transition_matrix(pages, ranks):
    '''
        Generate the transition matrix for the MC3 model

        If the current state is P, then the next state is chosen as follows:
            1. pick a ranking t uniformly at random from all the partial rankings containing P
            2. pick a page Q uniformly at random from the ranking t
            3. if t(Q) < t(P), then the next state is Q, otherwise the next state is P
    '''
    num_pages = len(pages)
    num_ranks = len(ranks)

    page_to_lists = generate_dicts(pages, ranks)

    # transition matrix
    T = numpy.zeros((num_pages, num_pages))
    for i in pages:
        i_ranks = page_to_lists[i] # List of rankings that contain page i
        for j in pages:
            # find the probability of going from page i to page j
            # find all rankings that contain page i
            # for each ranking, check if page j is in the ranking
            # if page j is in the ranking, check if the rank of page j is less than the rank of page i
            # if the rank of page j is less than the rank of page i, then increment the count
            for r in i_ranks:
                if(j in ranks[r]): # if page j is in the ranking
                    prob = 1 / len(ranks[r])
                    if(ranks[r].index(j) < ranks[r].index(i)):
                        T[pages.index(i)][pages.index(j)] += prob
                    else:
                        T[pages.index(i)][pages.index(i)] += prob
        # divide the count by the number of rankings that contain page i
        for j in pages:
            T[pages.index(i)][pages.index(j)] = T[pages.index(i)][pages.index(j)] / len(i_ranks)
    
    # print(T)
    return T

def generate_MC4_transition_matrix(pages, ranks):
    '''
        Generate the transition matrix for the MC4 model

        If the current state is P, then the next state is chosen as follows:
            1. pick a page Q uniformly from the union of all the pages ranked by the engines
            2. if t(Q) < t(P) for the majority of the list, then the next state is Q, otherwise the next state is P
    '''
    num_pages = len(pages)
    num_ranks = len(ranks)

    page_to_lists = generate_dicts(pages, ranks)

    # Transition matrix
    T = numpy.zeros((num_pages, num_pages))
    for i in pages:
        i_ranks = page_to_lists[i]
        for j in pages:
            # find the probability of page going from i to j
            # find the intersection of lists that rank i and j
            # if t(Q) < t(P) for majority of the list, then increment the count
            prob = 1/len(pages)
            j_ranks = page_to_lists[j]
            both_ranks = list(set(i_ranks) & set(j_ranks))

            # count the number of times j is ranked higher
            count = 0
            for ridx in both_ranks:
                rlst = ranks[ridx]
                if(rlst.index(j) < rlst.index(i)):
                    count += 1
            
            if count > len(both_ranks)/2:
                T[pages.index(i)][pages.index(j)] += prob
            else:
                T[pages.index(i)][pages.index(i)] += prob
    
    return T


def find_stationary_distribution(T, num_iter):
    '''
        Find the stationary distribution of the transition matrix T
    '''
    # initial state
    state = numpy.zeros(len(T))
    state[0] = 1

    # stationary distribution using power iteration
    for i in range(num_iter):
        state = numpy.dot(state, T)
    
    return state

def MC1_Simulation(pages, ranks):
    '''
        Simulate the MC1 model
    '''
    # generate a random ranking
    ranking = list(numpy.random.permutation(pages))

    # Simulation:
    transisions = [1]*1000
    iter = 0
    sliding_counts = []
    while True:
        # pick a rank uniformly at random
        r = random.sample(ranks, 1)[0]
        # check if the rank at least has two pages
        if(len(r) < 2):
            continue
        iter += 1
        # pick two pages uniformly at random from the rank
        p1, p2 = random.sample(r, 2)
        p1_index = r.index(p1)
        p2_index = r.index(p2)
        if(p1_index < p2_index):
            # p1 is ranked higher than p2 in this rank
            # if p1 is not ranked higher than p2 in the random ranking
            if(ranking.index(p1) > ranking.index(p2)):
                # swap the pages in the random ranking
                ranking[ranking.index(p1)], ranking[ranking.index(p2)] = ranking[ranking.index(p2)], ranking[ranking.index(p1)]
                transisions.append(1)
                # print the transition
                # print("Transition: ", p1, " -> ", p2)
            else:
                transisions.append(0)
        else:
            # p2 is ranked higher than p1 in this rank
            # if p2 is not ranked higher than p1 in the random ranking
            if(ranking.index(p2) > ranking.index(p1)):
                # swap the pages in the random ranking
                ranking[ranking.index(p1)], ranking[ranking.index(p2)] = ranking[ranking.index(p2)], ranking[ranking.index(p1)]
                transisions.append(1)
                # print the transition
                # print("Transition: ", p1, " -> ", p2)
            else:
                transisions.append(0)
        
        # check the number of transitions in the last 1000 iterations
        num_sliding_transitions = sum(transisions[-1000:])
        sliding_counts.append(num_sliding_transitions)
        print('sliding count:', num_sliding_transitions)
        # check the sliding transition count across the last 10 iterations, if they are the same, then stop
        if(num_sliding_transitions < 20 and all(x == sliding_counts[-1] for x in sliding_counts[-10:])):
            # reached the stationary distribution, write the ranking to a file
            print('reached stationary distribution')
            print('sliding count:', num_sliding_transitions)
            # print the trainsitions
            print('transitions')
            print(transisions)
            print('---------')
            save_ranking(ranking, 'data/l3500/MC1_ranking.txt')
            break
        
        if(iter%10000 == 0):
            print('saving and purging')
            # save the current ranking
            save_ranking(ranking, 'data/l3500/MC1_ranking.txt')
            # Purge the counts
            transisions = transisions[-1000:]
            sliding_counts = sliding_counts[-10:]
    
    return ranking, iter

def save_ranking(ranking, filename):
    '''
        Save the ranking to a file
    '''
    with open(filename, 'w') as f:
        f.write(','.join([str(i) for i in ranking]))
    


def main():
    # pages
    pages = [i for i in range(0, 3500)]

    # get ranks from training data file
    ranks = []
    
    f_lines = open('data/l3500/training_data_full.txt', 'r').readlines()
    for line in f_lines:
        chunks = line.split(';')
        ranks.append([int(i) for i in chunks[2].split(',')])
    
    
    # simulate MC1
    print('Simulating MC1')
    MC1_ranking, MC1_iter = MC1_Simulation(pages, ranks)
    print('MC1 ranking: ', MC1_ranking)
    print('MC1 iterations: ', MC1_iter)
    return

    # find the stationary distribution for MC3
    T3 = generate_MC3_transition_matrix(pages, ranks)
    print('generated MC3 transition matrix')
    final_rank_mc3 = find_stationary_distribution(T3, 150)
    print('found stationary distribution for MC3')
    _, l1 = zip(*sorted(zip(final_rank_mc3, pages)))
    MC3_Rank = list(reversed(l1))

    # print(T3)
    print('MC3: ', MC3_Rank)
    # print('MC3 scores', final_rank_mc3)

    # save MC3_Rank to file
    with open('data/l3500/MC3_Rank.txt', 'w') as f:
        f.write(",".join([str(i) for i in MC3_Rank]))

    # find the stationary distribution for MC4
    T4 = generate_MC4_transition_matrix(pages, ranks)
    final_rank_mc4 = find_stationary_distribution(T4, 150)
    _, l2 = zip(*sorted(zip(final_rank_mc4, pages)))
    MC4_Rank = list(reversed(l2))
    
    # print(T4)
    print('MC4: ', MC4_Rank)
    # print('MC4 scores', final_rank_mc4)

    # save MC4_Rank to file
    with open('data/l3500/MC4_Rank.txt', 'w') as f:
        f.write(",".join([str(i) for i in MC4_Rank]))

    return


# main() # run the main function