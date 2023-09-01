import random
import numpy as np
from src.utils import text_from_file
from src.incre_prag import run_inc, getL1, getL1L0, make_S1_sampler
import json
import os
import sys
import gc
from nltk import CFG
from src.utils import text_from_file, write_text_to_file
from nltk.parse.generate import generate, demo_grammar
from src.app import writeMMToFile
import re
from more_itertools import unique_everseen
import time

M = None
H = []
U = []
L1 = None
L0 = None
S1 = None
S1_Sampler = None
S0_Sampler = None
u_prior = None
h_prior = None
Selected_H = []
valid_utterance_count = {}

class Stats:
    def __init__(self, h, valid_utts, utts, correct):
        self.regex = h
        self.valid_utterances = valid_utts
        self.utterances = utts
        self.correct = correct
    def num_utts(self):
        return len(self.utterances)

def loadData(mm_file, regex_file, sentences_file):
    
    mm = np.load(mm_file)
    regexes = text_from_file(regex_file).splitlines()
    sentences = text_from_file(sentences_file).splitlines()

    return mm, regexes, sentences

def convert_for_negative_examples(M_old, H_old, U_old):
    # Convert it to negative examples
    U_new = []
    for u in U_old:
        U_new.append((u, '+'))
        U_new.append((u, '-'))
    old_row_count = M_old.shape[0]
    M_new = np.zeros((old_row_count * 2, M_old.shape[1]))

    new_row_count = 0
    for i in range(old_row_count):
        M_new[new_row_count] = M_old[i]
        new_row_count += 1
        M_new[new_row_count] = [ (0 if v == 1 else 1) for v in M_old[i]]
        new_row_count += 1
    return M_new, H_old, U_new

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


def init():
    global M, H, U, L1, L0, S1, Selected_H, S1_Sampler, S0_Sampler
    M, H, U = loadData('data/medium/mm.npy', 'data/medium/regexes.txt', 'data/medium/sentences.txt')
    M, H, U = convert_for_negative_examples(M, H, U)
    Selected_H = text_from_file('data/medium/selected.txt').splitlines()
    h_prior = [1/(len(H))] * len(H)
    u_prior = [1/(len(U))] * len(U)
    print('initializing inc prag')
    initValidUtteranceCount()
    L1, L0, S1 = getL1L0(H, U, M, h_prior, u_prior)
    S1_Sampler = make_S1_sampler(L0, u_prior, U)
    S0_Sampler = make_S0_sampler()
    
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

def L1Order(utt_idxs):
    # probability of each hypothesis given the utterances
    l1_res_list = L1(utt_idxs)
    # Sort the hypotheses by probability
    l1_res_list_sorted = sorted(range(len(l1_res_list)), key=lambda k: l1_res_list[k], reverse=True)
    # skip the ones with -inf
    l1_res_list_sorted = [x for x in l1_res_list_sorted if l1_res_list[x] != float('-inf')]
    return l1_res_list_sorted

def L0Order(utt_idxs):
    # probability of each hypothesis given the utterances
    l0_res_list = list(L0(utt_idxs))
    # Sort the hypotheses by probability
    l0_res_list_sorted = sorted(range(len(l0_res_list)), key=lambda k: l0_res_list[k], reverse=True)
    return l0_res_list_sorted

def simulate(speaker_sampler, listener, num_simulations = 5, max_utts = 8):
    # select num_simulations random hypotheses
    rand_h = random.sample(Selected_H, num_simulations)
    statistics = []
    all_utts = []
    for i, h in enumerate(rand_h):
        print('iteration: ', i)
        # get the index of the hypothesis
        h_index = H.index(h)
        # get the utterances that are valid for the hypothesis
        valid_utterances = speaker_sampler(h_index, max_utts)
        valid_utterances = list(unique_everseen(valid_utterances))
        
        # Selected regex
        print('Selected regex: ', h)
        # Selected utterances
        print('Selected utterances: ', [U[i] for i in valid_utterances])
        utt_count = -1
        partial_utterances = []
        # Guess L1
        for j in range(len(valid_utterances)):
            partial_utterances = valid_utterances[:j+1]
            guess_h = listener(partial_utterances)
            print('utterances: ', [U[i] for i in partial_utterances])
            print('Listener guess for ', j, ' utterances: ', H[guess_h])
            if guess_h == h_index:
                utt_count = j+1
                all_utts.append(utt_count)
                break
        
        if utt_count == -1:
            all_utts.append(max_utts+1)

        s = Stats(H[h_index], valid_utterances, partial_utterances, utt_count!= -1)
        statistics.append(s)
        print('Utterance count: ', utt_count)
        print('--------------------------------')

    print('Average utterance count: ', sum(all_utts)/len(all_utts))
    return statistics, all_utts

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def runSimulation():
    init()
    # blockPrint()
    _, all_utts1 = simulate(S0_Sampler, guessL0, 100, 10)
    _, all_utts2 = simulate(S0_Sampler, guessL1, 100, 10)
    _, all_utts3 = simulate(S1_Sampler, guessL0, 100, 10)
    _, all_utts4 = simulate(S1_Sampler, guessL1, 100, 10)
    # enablePrint()
    print('--------------------------------')
    print('stats')
    print('S0 L0: ')
    print([len([j for j in all_utts1 if j <= i]) for i in range(1, 11)])
    print('S0 L1: ')
    print([len([j for j in all_utts2 if j <= i]) for i in range(1, 11)])
    print('S1 L0: ')
    print([len([j for j in all_utts3 if j <= i]) for i in range(1, 11)])
    print('S1 L1: ')
    print([len([j for j in all_utts4 if j <= i]) for i in range(1, 11)])

    
def generate_random_sentence(max_length = 10):
    ALPHABETS = ['0', '1']
    len = random.randint(1, max_length)
    chars = []
    for i in range(len):
        chars.append(random.choice(ALPHABETS))
    return ''.join(chars)


def generate_meaning_matrix(num_regexes = 100, num_sentences = 1000, max_sentence_length = 10):
    
    
    grammar_str = text_from_file('src/regex_grammar')
    grammar = CFG.fromstring(grammar_str)
    
    # Generate 10000 Regexes - we will selectively sample 1000 regexes from this set.
    regexes = []
    for regex in generate(grammar, n=20000):
        regexes.append( '^' + ''.join(regex) + '$')

    # exisiting regexes
    existing_regexes = text_from_file('data/medium/regexes.txt').splitlines()
    
    # dedup
    regexes = list(set(regexes) - set(existing_regexes))

    # Randomly shuffling the list of regexes
    random.shuffle(regexes)

    regexes = existing_regexes + regexes

    # Sampling sentences
    sentenceFile = 'data/m350/sentences.txt'
    
    # read all sentences from file
    sentences = text_from_file(sentenceFile).splitlines()

    meaning_matrix_columns = []
    selected_regexes = []
    for r in regexes:
        reg = re.compile(r)
        col = [(1 if reg.match(s) else 0) for s in sentences]
        if(meaning_matrix_columns.count(col) == 0):
            meaning_matrix_columns.append(col)
            selected_regexes.append(r)
            print('total columns added: ', len(meaning_matrix_columns))
        if(len(selected_regexes) == num_regexes):
            break


    meaning_matrix = np.array(meaning_matrix_columns).T
    return meaning_matrix, selected_regexes, list(sentences)

def generate_data_user_study():
    M, H, U = generate_meaning_matrix(3500, 2000)
    old_regexes = text_from_file('data/medium/regexes.txt').splitlines()
    for i in range(len(old_regexes)):
        if old_regexes[i] not in H:
            print('error')
            break
    writeMMToFile(M, H, U, 'data/l3500/mm.csv', 'data/l3500/mm.npy', 'data/l3500/regexes.txt', 'data/l3500/sentences.txt')

def init_training():
    global M, H, U, L1, L0, S1, Selected_H, S1_Sampler, S0_Sampler, u_prior, h_prior
    M, H, U = loadData('data/l3500/mm.npy', 'data/l3500/regexes.txt', 'data/l3500/sentences.txt')
    # M, H, U = convert_for_negative_examples(M, H, U)
    # Selected_H = text_from_file('data/m350/selected.txt').splitlines()
    h_prior = [1/(len(H))] * len(H)
    u_prior = [1/(len(U))] * len(U)
    print('initializing inc prag')
    initValidUtteranceCount()
    L1, L0, S1 = getL1L0(H, U, M, h_prior, u_prior)
    S1_Sampler = make_S1_sampler(L0, u_prior, U)
    S0_Sampler = make_S0_sampler()

def crap():
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

def generate_training_data(speaker_sampler, listener_order, num_simulations = 5, max_utts = 8):
    # select num_simulations random hypotheses
    rand_h = random.sample(H, num_simulations)
    # rand_h = H[:num_simulations] # go in order of regexes
    statistics = []
    all_utts = []
    f = open('data/l3500/training_neg_data.txt', 'w')
    curr_time = time.time()
    for i, hyp in enumerate(rand_h):
        print('iteration: ', i)
        print('time elapsed: ', time.time() - curr_time)
        curr_time = time.time()
        if i == 1000:
            break
        # get the index of the hypothesis
        h_index = H.index(hyp)
        # print('Selected regex: ', hyp)
        utt_count = 0

        prob = 1
        utts_sofar = []
        for j in range(max_utts):
            # compute all alternatives
            alt_terms = []
            for uu in range(len(U)):
                uu_prior = u_prior[uu]
                u_together = utts_sofar + [uu]
                pl0 = L0(u_together)[h_index]
                term = pl0 * uu_prior
                alt_terms.append(term)
            best_u = np.argmax(alt_terms)
            if(M[best_u][h_index] == 1):
                if best_u not in utts_sofar:
                    utts_sofar.append(best_u)
                else:
                    continue
            else:
                break
            # get the listener order of the hypothesis
            h_order = listener_order(utts_sofar)
            guess_h = h_order[0]
            # print('utterances: ', [U[i] for i in utts_sofar])
            # print('Listener guess for ', j, ' utterances: ', H[guess_h])
            write_string = "{};{};{}\n".format(h_index, ",".join([str(i) for i in utts_sofar]), ",".join([str(i) for i in h_order]))
            f.write(write_string)
            if guess_h == h_index:
                utt_count = j+1
                all_utts.append(utt_count)
                # print('guessed correctly')
                break
        
        if utt_count == -1:
            all_utts.append(max_utts+1)

        
        f.flush()
        # print('Utterance count: ', utt_count)
        # print('--------------------------------')

    f.close()
    # print('Average utterance count: ', sum(all_utts)/len(all_utts))
    return all_utts

MC3 = []
MC4 = []
utts_dict = {}

def init_MC():
    global MC3, MC4
    MC3 = [int(i) for i in text_from_file('data/m350/MC3_Rank.txt').split(',')]
    MC4 = [int(i) for i in text_from_file('data/m350/MC4_Rank.txt').split(',')]


def guessLMC(MC, utt_idxs):
    # go through the list of hypotheses and return the one which satisfies all the utterances
    for h_idx in MC:
        if all(M[utt_idx][h_idx] == 1 for utt_idx in utt_idxs):
            return h_idx
    return 0

def guessLMC3(utt_idxs):
    return guessLMC(MC3, utt_idxs)

def guessLMC4(utt_idxs):
    return guessLMC(MC4, utt_idxs)

def test_MC(MC, max_utts = 8):
    # for each hypothesis sample n utterances using S1_Sampler
    # And guess using LMC3 and LMC4
    counts = []
    for h in H:
        h_idx = H.index(h)
        print('Selected regex: ', h, ' index: ', h_idx)
        if h_idx not in utts_dict:
            utts = S1_Sampler(h_idx, max_utts)
            utts = list(unique_everseen(utts))
            utts_dict[h_idx] = utts
        else:
            utts = utts_dict[h_idx]
        print('Selected utterances: ', [U[i] for i in utts])
        utts_count = -1
        # select increasing number of utterances
        for i in range(len(utts)):
            partial_utts = utts[:i+1]
            guess_h = guessLMC(MC, partial_utts)
            print('Utterances: ', [U[i] for i in partial_utts])
            print('Guess LMC: ', H[guess_h])
            print('-----------')
            if guess_h == h_idx:
                print('Guessed correctly with ', i+1, ' utterances')
                utts_count = i+1
                break
        
        if utts_count == -1:
            utts_count = max_utts+1
        counts.append(utts_count)
        print()
        print('--------------------------------')
        print()
    print('Average utterance count: ', sum(counts)/len(counts))
    print('counts:')
    print(counts)
    return counts

def run_testing():
    global MC3, MC4
    init_training()
    init_MC()
    print('Testing MC3')
    test_MC(MC3)
    print('Testing MC4')
    test_MC(MC4)
    



def run():
    global U, H, M
    # generate_data_user_study()
    init_training()
    generate_training_data(S1_Sampler, L1Order, 800, 8)
    # run_testing()