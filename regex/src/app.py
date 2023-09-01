from itertools import repeat
from os import write
from nltk.parse.generate import generate, demo_grammar
from nltk import CFG
from src.utils import text_from_file, write_text_to_file
import random
import re
import numpy as np
from src.prag_utils import draw, make_listener, normalise, comm_acc
from src.incre_prag import make_L0
from src.incre_prag import run_inc, getL1, getL1L0
import math
from collections import defaultdict
import statistics

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
    for regex in generate(grammar, n=300):
        regexes.append( '^' + ''.join(regex) + '$')

    # Randomly shuffling the list of regexes
    random.shuffle(regexes)

    # Sampling sentences
    sentences = set()
    while len(sentences) < num_sentences:
        r_sentence = generate_random_sentence(max_sentence_length)
        sentences.add(r_sentence)

    meaning_matrix_columns = []
    selected_regexes = []
    for r in regexes:
        reg = re.compile(r)
        col = [(1 if reg.match(s) else 0) for s in sentences]
        if(meaning_matrix_columns.count(col) == 0):
            meaning_matrix_columns.append(col)
            selected_regexes.append(r)
        if(len(selected_regexes) == num_regexes):
            break


    meaning_matrix = np.array(meaning_matrix_columns).T
    return meaning_matrix, selected_regexes, list(sentences)

def print_meaning_matrix_as_csv(meaning_matrix, selected_regexes, sentences):
    # print(meaning_matrix.shape)
    print('_, ', ', '.join(selected_regexes))
    scount = 0
    sentences_list = list(sentences)
    for row in meaning_matrix:
        print( sentences_list[scount] + ', ' , ', '.join([str(l) for l in list(row)]))
        lines.append(sentences_list[scount] + ', ' + ', '.join([str(l) for l in list(row)]))
        scount+=1

def writeMMToFile(meaning_matrix, selected_regexes, sentences, csv_file, mm_file, regex_file, sentences_file):
    # Saving MM file
    with open(mm_file, 'wb') as mmf:
        np.save(mmf, meaning_matrix)
    
    write_text_to_file(regex_file, "\n".join(selected_regexes))
    write_text_to_file(sentences_file, "\n".join(sentences))
    # writing csv file
    lines = []
    lines.append('_, ' + ', '.join(selected_regexes))
    scount = 0
    sentences_list = list(sentences)
    for row in meaning_matrix:
        lines.append(sentences_list[scount] + ', ' + ', '.join([str(l) for l in list(row)]))
        scount+=1
    with open(csv_file, 'w') as csvf:
        csv_text = '\n'.join(lines)
        csvf.write(csv_text)

def loadData(mm_file, regex_file, sentences_file):
    
    mm = np.load(mm_file)
    regexes = text_from_file(regex_file).splitlines()
    sentences = text_from_file(sentences_file).splitlines()

    return mm, regexes, sentences

def CheckL1():
    # M, H, U = generate_meaning_matrix()


    # writeMMToFile(M, H, U, 'mm3.csv', 'mm3.npy', 'regexes3.txt', 'sentences3.txt')

    M, H, U = loadData('mm3.npy', 'regexes3.txt', 'sentences3.txt')
    
    h_prior = [1/(len(H))] * len(H)
    u_prior = [1/(len(U))] * len(U)

    print('running inc prag')

    L1, L0, S1 = getL1L0(H, U, M, h_prior, u_prior)
    input_str = ''

    while True:
        input_str = input("enter utterences (or q to quit):  ")
        if input_str == 'q':
            break
        utt = list( map( lambda x: x.strip(), input_str.split(',') ) )
        print('Utterences: ', utt)
        utt_idx = list(map(lambda x: U.index(x), utt))

        # Getting L1 result
        l1_res_list = L1(utt_idx)
        max_res = max(l1_res_list)
        l1_res = H[l1_res_list.index(max_res)]

        # Getting L0 result
        l0_res_list = L0(utt_idx).tolist()
        max_l0_res = max(l0_res_list)
        l0_res = H[l0_res_list.index(max_l0_res)]
        print('Best match: ', l1_res)
        print('L0 match:', l0_res)
        print(l0_res_list.count(max_l0_res))
        print('\n\n')
        print('Matching results: ', l1_res_list.count(max_res))
        # all_results = [str(t[1]) + ': ' + str(t[0]) for t in zip(l1_res_list, H) if t[0] > float('-inf')]
        # print('\n'.join(all_results))
        # Find the number of best matches

    
    # utt = ['0110001', '10', '0']

    # run_inc(H, U, M, h_prior, u_prior)
    # L_0 = make_L0(M, h_prior)
    # print(M.shape)
    # print(L_0)

    # Regular Pragmatics
    # for i in range(100):
    #     S = normalise(L, 1)
    #     L = normalise(S, 0)
    #     end_acc = comm_acc(S, L)
    #     print (end_acc)


def CheckL0L1():
    # M, H, U = generate_meaning_matrix()


    # writeMMToFile(M, H, U, 'mm3.csv', 'mm3.npy', 'regexes3.txt', 'sentences3.txt')

    M, H, U = loadData('mm3.npy', 'regexes3.txt', 'sentences3.txt')
    
    h_prior = [1/(len(H))] * len(H)
    u_prior = [1/(len(U))] * len(U)

    print('running inc prag')

    L1, L0, S1 = getL1L0(H, U, M, h_prior, u_prior)
    input_str = ''

    while True:
        input_str = input("enter utterences (or q to quit):  ")
        if input_str == 'q':
            break
        utt = list( map( lambda x: x.strip(), input_str.split(',') ) )
        print('Utterences: ', utt)
        utt_idx = list(map(lambda x: U.index(x), utt))

        # Getting L1 result
        l1_res_list = L1(utt_idx)
        max_res = max(l1_res_list)
        l1_res = H[l1_res_list.index(max_res)]

        # Getting L0 result
        l0_res_list = L0(utt_idx).tolist()
        max_l0_res = max(l0_res_list)
        l0_res = H[l0_res_list.index(max_l0_res)]
        print('Best match: ', l1_res)
        print('L0 match:', l0_res)
        print('L0 Mathcing results', l0_res_list.count(max_l0_res))
        print('\n\n')
        print('Matching results: ', l1_res_list.count(max_res))
        print(S1(utt_idx, l1_res_list.index(max_res)))
        # all_results = [str(t[1]) + ': ' + str(t[0]) for t in zip(l1_res_list, H) if t[0] > float('-inf')]
        # print('\n'.join(all_results))
        # Find the number of best matches

def getS1Result(S1, utts, hyp_idx):
    return S1(utts, hyp_idx)[0][0]

def sampleS1(M, H, U, S1, L0, hyp):
    
    # Get the list of valid utterances
    # Get probability for each utterance given the hypothesis
    all_valid_utts = []
    for i in range(1000):
        if(M[i, hyp] == 1):
            all_valid_utts.append(i)
    
    valid_utts = all_valid_utts.copy()

    # Find the best utterance for the hypothesis
    # print(probs)
    # print(len(valid_utts), valid_utts)
    utts = []
    flag = True
    hyp_count = 0
    # probs = [getS1Result(S1, [ut], hyp) for ut in valid_utts]
    while(len(valid_utts) > 0):
        # find the maximum prob utterance
        probs = [getS1Result(S1, utts + [ut], hyp) for ut in valid_utts]
        max_prob = max(probs)
        idx_max = probs.index(max_prob)
        utt_max = valid_utts[idx_max]
        utts.append(utt_max)
        del valid_utts[idx_max]
        del probs[idx_max]
        
        # Find if the hypothesis has the max prob
        hProbDist = list(L0(utts))
        max_hyp_prob = max(hProbDist)
        # print(max_hyp_prob)
        # print(hProbDist)
        hyp_count = hProbDist.count(max_hyp_prob)
        # print(len(all_valid_utts), len(utts), hyp_count)
        if(max_hyp_prob == hProbDist[hyp] and hProbDist.count(max_hyp_prob) == 1):
            # The set of utterances is enough
            return (all_valid_utts, utts, hyp_count)
        if(len(utts) > 30):
            # print('oops: ', list(L0(all_valid_utts)))
            return (all_valid_utts, all_valid_utts, -1)

    # print('-----------------------------------------')
    # print(H[hyp], hyp)
    # print(list(L0(all_valid_utts)))
    return (all_valid_utts, utts, hyp_count)

    
def S0(us, hyp):
    return [[1]]

def avg(lst):
    return sum(lst)/len(lst)

def expr0():
    # Comparing S0 L1
    M, H, U = loadData('mm3.npy', 'regexes3.txt', 'sentences3.txt')
    
    h_prior = [1/(len(H))] * len(H)
    u_prior = [1/(len(U))] * len(U)

    print('running inc prag')

    L1, L0, S1 = getL1L0(H, U, M, h_prior, u_prior)
    # print (L0([1,2,3]))
    print('Getting S1')

    sampled_hypothesis = random.sample(range(len(H)), 30)
    # sampled_hypothesis = list(range(len(H)))
    sample_len = len(sampled_hypothesis)
    utts_len = []
    hcounts = []
    skipped = 0
    for i in range(sample_len):
        avu, utts, hcount = sampleS1(M, H, U, S1, L1, sampled_hypothesis[i])
        # print("final utts: ", utts)
        print(len(avu), "vs", len(utts), "|", hcount)
        if(hcount != -1):
            utts_len.append(len(utts))
            hcounts.append(hcount)
        else:
            skipped += 1
    
    print('average utts length: ', avg(utts_len))
    print('average hcount: ', avg(hcounts))
    print('skipped: ', skipped)


def get_valid_utts(M, hyp):
    u_count, h_count = M.shape
    valid_utts = []
    for i in range(u_count):
        if(M[i, hyp] == 1):
            valid_utts.append(i)
    return valid_utts

def get_top_k_utterances(M, H, U, S, k, hyp):
    valid_utts = get_valid_utts(M, hyp)
    utt = []
    min_range = min(len(valid_utts), k)
    if(S.__name__ == "S0"):
        return random.sample(valid_utts, min_range)
    for i in range(min_range):
        probs = [ S(utt + [u], hyp)[0][0] for u in valid_utts]
        max_prob = max(probs)
        max_idx = probs.index(max_prob)
        # print(U[valid_utts[max_idx]])
        utt.append(valid_utts[max_idx])
        del valid_utts[max_idx]
    return utt

def get_graph_points_expr1(M, H, U, L, S):
    k = 8
    prob_dict = defaultdict(list)
    # do it for all the hyp

    for hyp in range(len(H)):
        print('calculating for hyp ', H[hyp], hyp)
        topKUtt = get_top_k_utterances(M, H, U, S, k, hyp)
        print('top K:', [U[i] for i in topKUtt])
        # Get top k utterances
        # For |u| = 1, 2, ... k
        for i in range(len(topKUtt)):
            h_dist = L(topKUtt[:i+1])
            h_dist = [math.exp(p) for p in h_dist]
            sum_h_dist = sum(h_dist)
            norm_h_dist = [float(p)/sum_h_dist for p in h_dist]
            print(norm_h_dist[hyp])
            prob_dict[i].append(norm_h_dist[hyp])

    for j in range(k):
        print(j, statistics.mean(prob_dict[j]))
    return prob_dict

def expr1():
    # Comparing S0 L1
    M, H, U = loadData('mm3.npy', 'regexes3.txt', 'sentences3.txt')

    h_prior = [1/(len(H))] * len(H)
    u_prior = [1/(len(U))] * len(U)

    print('Getting S1')
    L1, L0, S1 = getL1L0(H, U, M, h_prior, u_prior)
    
    k = 5
    # hyp = random.randint(0, len(H) - 1)
    # print('for hyp ', H[hyp], ':')
    get_graph_points_expr1(M, H, U, L0, S0)
    

def generate_data_user_study():
    M, H, U = generate_meaning_matrix(150, 2000)
    writeMMToFile(M, H, U, 'mm4.csv', 'mm4.npy', 'regexes4.txt', 'sentences4.txt')

def expr2():
    M, H, U = loadData('mm4.npy', 'regexes4.txt', 'sentences4.txt')

    h_prior = [1/(len(H))] * len(H)
    u_prior = [1/(len(U))] * len(U)

    print('Getting S1')
    L1, L0, S1 = getL1L0(H, U, M, h_prior, u_prior)

def run():
    # expr1()
    generate_data_user_study()
    # expr2()