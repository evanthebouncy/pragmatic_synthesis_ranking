import grammar
import tqdm
import json
import numpy as np
import random
import heapq
import time
import torch
import argparse
from grammar import ShapeGridGrammar

class LiteralListener:
    """
        Global literal listener that reasons over the joint program space.
    """
    def __init__(self, version_space):
        self.version_space = version_space
    
    def get_distribution(self, spec):
        """
            Get distributions over programs given spec

            :param spec: list of spec elements (i, j, object)
        """

        # Start with version space as the set of programs that satisfy the first
        # element of the spec and incrementally find intersection with programs 
        # that satisfy the rest of the spec
        programs = set(self.version_space['spec'][','.join(map(str, spec[0]))])
        for u in spec[1:]:
            programs = programs.intersection(self.version_space['spec'][','.join(map(str, u))])

        # Uniform distribution over satisfying programs
        P = np.ones(len(programs))
        return programs, P / np.sum(P)
    
    def synthesize_programs(self, spec):
        """
            Find best program that satisfies the spec.

            :param spec: list of spec elements (i, j, object)
        """
        prog, score = self.get_distribution(spec)
        best_prog, _ = max(zip(prog, list(score)), key=lambda x: x[1])
        return self.version_space['targets'][best_prog][0]

class PragmaticListener:
    """
        Global pragmatic listener that recursively reasons over joint program space.
    """
    def __init__(self, grammar, version_space):
        self.pragmatic_speaker = PragmaticSpeaker(grammar, version_space)

    def get_distribution(self, spec, precomputed_listener=None, precomputed_speaker=None):
        """
            Get distributions over programs given spec

            :param spec: list of spec elements (i, j, object)
        """

        # Find programs that satisfy the spec using literal listener (ignores 
        # the probabilities)
        
        programs = list(self.pragmatic_speaker.literal_listener.get_distribution(spec)[0])
        scores = list()
        # Recursively reason over all possible referent programs
        for program_id, prog in enumerate(programs):
            spec_space = self.pragmatic_speaker.grammar.unhash(self.pragmatic_speaker.version_space['programs'][prog])
            score = 0
            for i in range(len(spec)):
                u, w = self.pragmatic_speaker.step(spec_space, spec[:i])
                j = u.index(spec[i])
                score += w[j]
            scores.append(np.exp(score))
        
        
        return programs, scores / sum(scores)
    
    def synthesize_programs(self, spec):
        """
            Find best program that satisfies the spec.

            :param spec: list of spec elements (i, j, object)
        """
        prog, score = self.get_distribution(spec)
        best_prog, _ = max(zip(prog, list(score)), key=lambda x: x[1])
        return self.pragmatic_speaker.version_space['targets'][best_prog][0]

class LiteralSpeaker:
    """
        Literal speaker that generates random spec elements that satisfy the 
        program.
    """
    def __init__(self, grammar, version_space):
        self.grammar = grammar
        self.version_space = version_space
    
    def step(self, spec_space, us):
        """
            Given a space of possible spec elements and a history of chosen 
            spec elements, generate the next spec element.

            :param spec_space: Set of possible spec elements that satisfy the 
                                program
            :param us: Set of spec elements already presented
        """

        # Find spec elements that have not been presented already
        u_news = list(spec_space.difference(us))
        P = np.ones(len(u_news))
        return u_news, P / P.sum()
    
    def generate_spec(self, program, length, n_specs=1):
        """
            Generate a specs for a program.

            :param program: Program for which spec is required
            :param length: Number of spec elements per spec generated
            :param n_specs: Number of specs to generate
        """
        program_id = self.version_space['programs'].index(self.grammar.hash(program))
        spec_space = self.grammar.unhash(self.version_space['programs'][program_id])
        if n_specs == 1:
            return random.sample(spec_space, length)
        else:
            return [random.sample(spec_space, length) for i in range(n_specs)]

class PragmaticSpeaker:
    """
        Pragmatic speaker that generates specs by reasoning about a literal 
        listener.
    """
    def __init__(self, grammar, version_space):
        self.grammar = grammar
        self.version_space = version_space
        self.literal_listener = LiteralListener(version_space)
        self.vs_cache = dict()

    def step(self, spec_space, us):
        """
            Given a space of possible spec elements and a history of chosen 
            spec elements, generate the next spec element.

            :param spec_space: Set of possible spec elements that satisfy the 
                                program
            :param us: Set of spec elements already presented
        """

        u_news = list(spec_space.difference(us))
        us_hash = ','.join(map(str, us))
        if us_hash in self.vs_cache:
            vs_us = self.vs_cache[us_hash]
        else:
            vs_us = self.literal_listener.get_distribution(us)[0] if len(us) > 0 else set()
            self.vs_cache[us_hash] = vs_us

        u_weights = []
        for u_new in u_news:
            if len(vs_us) > 0:
                vs_us_new = set.intersection(vs_us, self.literal_listener.get_distribution([u_new])[0])  
            else:
                vs_us_new = self.literal_listener.get_distribution([u_new])[0]
            u_weights.append(1 / len(vs_us_new))
        u_weights = np.array(u_weights)
        return (u_news, np.log(u_weights / np.sum(u_weights)))
    
    def generate_spec(self, program, length, n_specs=1):
        """
            Generate a specs for a program using beam search.

            :param program: Program for which spec is required
            :param length: Number of spec elements per spec generated
            :param n_specs: Number of specs to generate
        """
        program_id = self.version_space['programs'].index(self.grammar.hash(program))
        beam = [(0, list()) for i in range(n_specs)]
        spec_space = self.grammar.unhash(self.version_space['programs'][program_id])

        # Beam search for exactly `length` steps
        for _ in range(length):
            frontier = [self.step(spec_space, samples) for score, samples in beam]
            new_beam = list()
            for i, element in enumerate(beam):
                for sample, score in zip(*frontier[i]):
                    new_beam.append((element[0] + score, element[1] + [sample]))
            beam = heapq.nlargest(n_specs, new_beam, key=lambda x: x[0])

        if n_specs == 1:
            return beam[0][1]
        else:
            return [spec for score, spec in beam]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--listener", 
        type=str, 
        choices=["literal", "pragmatic"],
        help="Type of listener"
        )
    parser.add_argument(
        "--specs", 
        type=str,
        help="Path to JSON file containing pairs of (program, spec) as a list"
        )
    parser.add_argument(
        "--outputs", 
        type=str,
        help="Path to JSON file to write outputs"
        )
    parser.add_argument(
        "--write_every", 
        type=int,
        default=20,
        help="Write to JSON file every N times"
        )
    parser.add_argument(
        "--version_space", 
        type=str, 
        default='version_spaces/version_space.json',
        help="Path to version space file"
        )

    args = parser.parse_args()
    with open(args.version_space, 'r') as f:
        vs = json.load(f)

    g = ShapeGridGrammar(7)
    if not args.listener is None:
        with open(args.specs, 'r') as f:
            data = json.load(f)
        
        specs = [[tuple(u) for u in s] for p, s in data]
        programs = [p for p, s in data]
        
        if args.listener == "literal":
            listener = LiteralListener(vs)
        else:
            listener = PragmaticListener(g, vs)

        outputs = list()
        for i, spec in tqdm.tqdm(enumerate(specs)):
            expt = {
                'ground_truth_program': programs[i],
                'spec': spec,
                'synthesis': list()
            }
            for j in range(1, len(spec) + 1):
                p, s = listener.get_distribution(spec[:j])
                best_prog = max(zip(p, list(s)), key=lambda x: x[1])
                expt['synthesis'].append({
                    "distribution": list(zip(p, s)),
                    "top_program": best_prog
                })
            outputs.append(expt)

            # Write results to file at intermediate points
            if i % args.write_every:
                with open(args.outputs, 'w') as f:
                    json.dump(outputs, f)

        with open(args.outputs, 'w') as f:
            json.dump(outputs, f)