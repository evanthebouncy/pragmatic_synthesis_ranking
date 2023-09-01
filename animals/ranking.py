from joint import LiteralListener, LiteralSpeaker, PragmaticListener, PragmaticSpeaker
from grammar import ShapeGridGrammar
from tqdm import tqdm
import json
import numpy as np
from copy import deepcopy
import sys
import random
import time

def generate_ranking_dataset(program_set, spec_len, grammar, version_space):
    dataset = []
    literal_listener = LiteralListener(version_space)
    pragmatic_speaker = PragmaticSpeaker(grammar, version_space)
    pragmatic_listener = PragmaticListener(grammar, version_space)

    for pidx, p in enumerate(tqdm(program_set, desc=f"{sys.argv[1]}")):
        spec = []
        for k in range(1, spec_len + 1):
            program_id = version_space['programs'].index(grammar.hash(p))
            spec_space = grammar.unhash(version_space['programs'][program_id])

            us, scores = pragmatic_speaker.step(spec_space, spec)
            spec.append(us[np.argmax(scores)])

            output = pragmatic_listener.get_distribution(spec)
            dataset.append((p, deepcopy(spec), sorted(zip(*output), key=lambda x: x[1], reverse=True)))
    
    return dataset

def rank_programs(rankings_data, vs, max_iters=1000000, num_swaps_per_program=1, validate_every=10000, tolerance=10):
    rankings = {i: i for i, p in enumerate(vs['programs'])}
    swaps = []
    validations = []
    for i in tqdm(range(max_iters)):
        for _ in range(num_swaps_per_program):
            prog, spec, r = rankings_data[np.random.randint(len(rankings_data))]
            if len(r) < 2:
                continue
            (p1, s1), (p2, s2) = random.sample(r, 2)
            if (s1 > s2 and rankings[p1] > rankings[p2]):
                rankings[p1], rankings[p2] = rankings[p2], rankings[p1]
                swaps.append(i)
            elif (s2 > s1 and rankings[p2] > rankings[p1]):
                rankings[p1], rankings[p2] = rankings[p2], rankings[p1]
                swaps.append(i)
            
        if i % validate_every == 0:
            swaps_in_last_n = len([s for s in swaps if s > i - validate_every])
            if len(validations) > 2:
                if (max(validations[-5:]) - min(validations[-5:])) < tolerance:
                    print(f"Converged after {i} iterations")
                    break
            validations.append(swaps_in_last_n)
    
    print(len(swaps))
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.ecdfplot(x=swaps, stat='count')
    plt.savefig('swaps.png')
    return list(map(lambda x: x[0], sorted(rankings.items(), key=lambda x: x[1])))

def ranking_inference(dataset, vs, program_rankings):
    literal_listener = LiteralListener(vs)
    pragmatic_listener = PragmaticListener(ShapeGridGrammar(7), vs)

    logs = []
    for prog, spec in tqdm(dataset):
        log = []
        for i in range(len(spec)):
            start = time.time()
            progs, scores = literal_listener.get_distribution(spec[:i+1])
            best_program_l0 = sorted(zip(progs, scores), key=lambda x: x[1], reverse=True)[0]
            end = time.time()
            time_l0 = end - start

            start = time.time()
            pragmatic_progs, pragmatic_scores = pragmatic_listener.get_distribution(list(map(tuple, spec[:i+1])))
            best_program_l1 = sorted(zip(pragmatic_progs, pragmatic_scores), key=lambda x: x[1], reverse=True)[0]
            end = time.time()
            time_l1 = end - start
            
            start = time.time()
            progs, scores = literal_listener.get_distribution(spec[:i+1])
            ranked_programs = [(p, program_rankings[p]) for p in progs]
            best_program_lr = sorted(ranked_programs, key=lambda x: x[1])[0]
            end = time.time()
            time_lr = end - start
            
            log.append({
                'l0': best_program_l0[0],
                'time_l0': time_l0,
                'l1': best_program_l1[0],
                'time_l1': time_l1,
                'lr': best_program_lr[0],
                'time_lr': time_lr
            })

        logs.append({"prog": prog, "spec": spec,"log": log})

    return logs


if __name__ == "__main__":
    g = ShapeGridGrammar(7)
    
    with open("version_spaces/version_space.json", 'r') as f:
        vs = json.load(f)
    # print(sys.argv[1])
    # with open(f"ranking_programs_{sys.argv[1]}.json") as f:
    #     programs = json.load(f)
    
    # program_set = [p for idx, p in programs]

    # dataset = generate_ranking_dataset(program_set, 10, g, vs)

    # with open(f"rankings_{sys.argv[1]}.json", 'w') as f:
    #     json.dump(dataset, f)

    # with open('rankings/full_rankings_S1_L1.json') as f:
    #     rankings_data = json.load(f)
    
    # rankings = rank_programs(rankings_data, vs, max_iters=10000000, num_swaps_per_program=1)

    # with open('ranked_programs.json', 'w') as f:
    #     json.dump(rankings, f)

    
    with open('ranked_programs.json') as f:
        rankings = json.load(f)
        program_rankings = {v: k for k, v in enumerate(rankings)}
    
    with open('version_spaces/prog2idx.json') as f:
        prog2idx = json.load(f)

    with open(sys.argv[1]) as f:
        specs = json.load(f)

    stats = ranking_inference(specs, vs, program_rankings)

    with open(sys.argv[2], 'w') as f:
        json.dump(stats, f)