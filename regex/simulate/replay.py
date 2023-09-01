import numpy as np
from src.utils import text_from_file
from src.incre_prag import run_inc, getL1, getL1L0, make_S1_sampler
from more_itertools import unique_everseen
from src.utils import write_text_to_file
import random
import time


M = None
H = []
U = []
L1 = None
L0 = None
S1 = None
S1_Sampler = None
S0_Sampler = None
Selected_H = []
valid_utterance_count = {}
MC3 = []
MC4 = []
MC1 = []


def loadData(mm_file, regex_file, sentences_file):
    
    mm = np.load(mm_file)
    regexes = text_from_file(regex_file).splitlines()
    sentences = text_from_file(sentences_file).splitlines()

    return mm, regexes, sentences

def initValidUtteranceCount():
    columns = M.shape
    for i in range(columns[1]):
        count = 0
        for j in range(columns[0]):
            if M[j, i] == 1:
                count += 1
        valid_utterance_count[i] = count

def make_S0_sampler():
    global M, U
    # S0: Find all the valid utterances for the selected hypothesis, and return a random permutation of them
    def S0(h_index, k):
        valid_utterances = []
        for i in range(len(U)):
            if M[i, h_index] == 1:
                valid_utterances.append(i)
        return random.sample(valid_utterances, min(k, len(valid_utterances)))
    return S0

def make_S1_sampler(L0, U_prior, U):
    # take in a h and a number k, sample best utterance of size k
    def S1(h, k):
        prob = 1
        utts_sofar = []

        ret = []
        for _ in range(k):
            # compute all alternatives
            alt_terms = []
            for uu in range(len(U)):
                uu_prior = U_prior[uu]
                u_together = utts_sofar + [uu]
                pl0 = L0(u_together)[h]
                term = pl0 * uu_prior
                alt_terms.append(term)
            best_u = np.argmax(alt_terms)
            utts_sofar.append(best_u)
        return utts_sofar
    return S1

def init_training():
    global M, H, U, L1, L0, S1, Selected_H, S1_Sampler, S0_Sampler
    M, H, U = loadData('data/m350/mm.npy', 'data/m350/regexes.txt', 'data/m350/sentences.txt')
    # M, H, U = loadData('data/l3500/mm.npy', 'data/l3500/regexes.txt', 'data/l3500/sentences.txt')
    # M, H, U = convert_for_negative_examples(M, H, U)
    # Selected_H = text_from_file('data/m350/selected.txt').splitlines()
    h_prior = [1/(len(H))] * len(H)
    u_prior = [1/(len(U))] * len(U)
    print('initializing inc prag')
    initValidUtteranceCount()
    L1, L0, S1 = getL1L0(H, U, M, h_prior, u_prior)
    S1_Sampler = make_S1_sampler(L0, u_prior, U)
    S0_Sampler = make_S0_sampler()

def init_MC():
    global MC3, MC4, MC1
    MC3 = [int(i) for i in text_from_file('data/m350/Rank.txt').split(',')]
    # MC1 = [int(i) for i in text_from_file('data/l3500/MC1_Rank.txt').split(',')]

def guessLMC(MC, utt_idxs):
    # go through the list of hypotheses and return the one which satisfies all the utterances
    for h_idx in MC:
        if all(M[utt_idx][h_idx] == 1 for utt_idx in utt_idxs):
            return h_idx
    return 0

def guessL0(utt_idxs):
    l0_res_list = list(L0(utt_idxs))
    max_res = max(l0_res_list)
    # Finding the indexes of the max results
    all_l0_res = [i for i, x in enumerate(l0_res_list) if x == max_res]

    l0_res_idx = random.choice(all_l0_res)
    return l0_res_idx

def guessL1(utt_idxs):
    l1_res_list = L1(utt_idxs)
    max_res = max(l1_res_list)
    l1_res = l1_res_list.index(max_res)
    return l1_res

def guessLMC3(utt_idxs):
    return guessLMC(MC3, utt_idxs)

def guessLMC4(utt_idxs):
    return guessLMC(MC4, utt_idxs)

def parse_user_replay_data():
    replay_file = 'data/m350/replay.csv'
    replay_data = []
    for line in text_from_file(replay_file).splitlines():
        items = [i.strip() for i in line.split(',')]
        obj = {}
        obj['p'] = items[0]
        hyp = '^' + items[1] + '$'
        obj['h'] = H.index(hyp)
        obj['r'] = items[2]
        # get the rest of the utterances
        utts = [U.index(u) for u in items[3:] if u != '' and u in U]
        # validate the utterances
        obj['u'] = [u for u in utts if M[u][obj['h']] == 1]
        replay_data.append(obj) 

    # Create the replay dataset from the replay data
    replay_dataset = []
    utts_count = []
    # choose 2 objs at a time
    for i in range(0, len(replay_data), 2):
        obj1 = replay_data[i]
        obj2 = replay_data[i+1]
        assert obj1['p'] == obj2['p']
        assert obj1['h'] == obj2['h']
        robj = {}
        utts = obj1['u'] + obj2['u']
        # robj['u'] = list(unique_everseen(utts))
        utts = obj1['u'] if obj1['r'] == 'g' else obj2['u']
        utts_count.append(len(utts))
        robj['u'] = utts
        robj['h'] = obj1['h']
        replay_dataset.append(robj)
    # print('utts count: ', utts_count)
    return replay_dataset

def replay_sumilation(MC, replay_dataset, verbose=False):
    utts_counts = []
    pred_times = []
    for rd in replay_dataset:
        times = []
        # print(len(utts_counts))
        utts = rd['u']
        h = rd['h']
        count = -1
        for j in range(len(utts)):
            partial_utts = utts[:j+1]
            if verbose:
                print('partial utts: ', partial_utts)
            
            start = time.time()
            mc_guess = guessLMC(MC, partial_utts)
            end = time.time()
            times.append(end - start)
            if verbose:
                print('guess LMC: ', mc_guess)
            if mc_guess == h:
                count = len(partial_utts)
                break
        
        pred_times.append(times)
        
        utts_counts.append(count)
    
    print('utts counts: ', utts_counts)
    # Calculate the average number of utterances for successful guesses
    success_utts = [i for i in utts_counts if i != -1]
    print('avg utts for success: ', sum(success_utts)/len(success_utts))
    # number of unsuccessful guesses
    print('number of unsuccessful guesses: ', len(utts_counts) - len(success_utts), 'out of: ', len(utts_counts))

    print('\n\n')
    # convert the list of lists pred_time to string
    pred_times_str = ''
    for i in pred_times:
        pred_times_str += str(i) + ',\n'
    
    pred_times_str = '[' + pred_times_str + ']'
    write_text_to_file('pred_times.txt', pred_times_str)
    return utts_counts

def replay_sumilation_L(guess_func, replay_dataset, verbose=False):
    utts_counts = []
    pred_times = []
    for rd in replay_dataset:
        times = []
        # print(len(utts_counts))
        utts = rd['u']
        h = rd['h']
        count = -1
        for j in range(len(utts)):
            partial_utts = utts[:j+1]
            if verbose:
                print('partial utts: ', partial_utts)
            
            start = time.time()
            mc_guess = guess_func(partial_utts)
            end = time.time()
            times.append(end - start)
            if verbose:
                print('guess_l: ', mc_guess)
            if mc_guess == h:
                count = len(partial_utts)
                break
        
        pred_times.append(times)
        
        utts_counts.append(count)
    
    print('utts counts: ', utts_counts)
    # Calculate the average number of utterances for successful guesses
    success_utts = [i for i in utts_counts if i != -1]
    print('avg utts for success: ', sum(success_utts)/len(success_utts))
    # number of unsuccessful guesses
    print('number of unsuccessful guesses: ', len(utts_counts) - len(success_utts), 'out of: ', len(utts_counts))

    print('\n\n')
    # convert the list of lists pred_time to string
    pred_times_str = ''
    for i in pred_times:
        pred_times_str += str(i) + ',\n'
    
    pred_times_str = '[' + pred_times_str + ']'
    write_text_to_file('pred_times.txt', pred_times_str)
    return utts_counts

def replay_testing():
    init_training()
    init_MC()
    replay_dataset = parse_user_replay_data()

    # run the replay simulation for Prag ranking
    print('running replay for Pragmatic Ranking')
    replay_sumilation(MC3, replay_dataset)

    # run the replay simulation for L0
    print('running replay L0')
    replay_sumilation_L(guessL0, replay_dataset)

    print('running replay L1')
    replay_sumilation_L(guessL1, replay_dataset)

def consoleInterface():
    init_training()
    init_MC()
    
    utts = []
    
    target_h = random.choice(H)
    target_idx = H.index(target_h)
    print('Target hypothesis: ', target_h)

    while True:
        print('Enter an utterance: ')
        utt = input()
        if utt == ';':
            break
        if utt == 'h':
            # clear utts
            utts = []
            target_h = random.choice(H)
            target_idx = H.index(target_h)
            print('Target hypothesis: ', target_h)
            continue
        if utt not in U:
            print('Utterance not found')
            continue
        utts.append(U.index(utt))
        print('Utterances: ', [U[u] for u in utts])
        g_hyp = guessLMC(MC1, utts)
        print('LMC1 guess: ', H[g_hyp])
        if g_hyp == target_idx:
            print('LMC1 guessed correctly')
        

def run():
    replay_testing()
    # consoleInterface()