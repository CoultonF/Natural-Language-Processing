import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10

def memo(f):
    "Memoize function f."
    table = {}
    def fmemo(*args):
        if args not in table:
            table[args] = f(*args)
        return table[args]
    fmemo.memo = table
    return fmemo

class Segment:

    def __init__(self, Pw, P2w=None):
        self.Pw = Pw
        self.P2w = P2w

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: return []
        heap = []
        chart = {}
        # Initialize the heap
        text_size = len(text)


        for i in range(1, text_size):
            word = text[:i]
            try:
                heapq.heappush(heap,(word, 0, log10(self.Pw(word)), None))
            except:
                heapq.heappush(heap,(word, 0, log10(float(1e-300)), None))

            # print("pushing1:", word)
        
        while heap:
            entry = heapq.heappop(heap)
            # print("selecting: ", entry)
            end_index = entry[1] + len(entry[0]) - 1
            go_further = True
            if end_index in chart:
                prev_entry = chart[end_index]
                if entry[2] > prev_entry[2]:
                    chart[end_index] = entry
                else:
                    go_further = False
                    continue
            else:
                chart[end_index] = entry

            if go_further:
                for i in range(1, text_size-end_index):
                    new_word = text[end_index+1:end_index+1+i]
                    try:
                        new_entry = (new_word, end_index+1, entry[2]+log10(self.Pw(new_word)), end_index)
                    except:
                        new_entry = (new_word, end_index+1, entry[2]+log10(float(1e-300)), end_index)
                    cur_end_idx = end_index + len(new_word)
                    add = True
                    if add:
                        heapq.heappush(heap, new_entry)
                        # print("pushing2: ",new_word)
            # input()
        
        entry = chart[len(text)-1]
        prev = entry[3]
        segmentation = []
        segmentation.append(entry[0])
        while prev != None:
            entry = chart[prev]
            prev = entry[3]
            segmentation.append(entry[0])

        return reversed(segmentation)


    def segment2(self, text):
        best_candidate = self.segment_bigram(text)
        return best_candidate[1]

    @memo
    def segment_bigram(self, text, prev='<S>'):   
        "Return (log P(words), words), where words is the best segmentation."    
        if not text: return 0.0, []    
        candidates = [self.combine(log10(self.cPw(first, prev)), first, self.segment_bigram(rem, first))                  
                    for first,rem in self.splits(text)]   
                    
        return max(candidates)

    def combine(self, Pfirst, first, segment2_ret):    
        "Combine first and rem results into one (probability, words) pair."    
        Prem, rem = segment2_ret
        return Pfirst+Prem, [first]+rem

    def splits(self, text, L=20):
        "Return a list of all possible (first, rem) pairs, len(first)<=L."
        return [(text[:i+1], text[i+1:]) 
                for i in range(min(len(text), L))]

    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        return sum(log10(self.Pw(w)) for w in words)

    def cPw(self, word, prev):
        # JM = .7 #.0.9338
        JM = .8 #.0.9349
        # JM = .75 #.0.9337
        # JM = .9 #.0.9323

        try:
            # scores before penalize tuning
            # return self.Pw[word] #.64
            # return self.P2w[prev + ' ' + word] # .71
            # return (self.P2w[prev + ' ' + word])/float(self.Pw(prev)) #.77
            # return (JM*self.P2w[prev + ' ' + word] + (1-JM)*self.Pw[word]) #.84
            return (JM*(self.P2w[prev + ' ' + word]/self.Pw[prev]) + (1-JM)*self.Pw(word)) #.87
        except KeyError:  
            return self.Pw(word)


#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
    def __call__(self, key): 
        if key in self: return self[key]/self.N  
        else: return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)

def penalize_long_unknown(key, N): 
    # return (1/N) * 100**-(len(key)-1) #.79
    # return (1/N) * 1000000**-(len(key)-1) #.87
    # return (1/N) * 100000**-(len(key)-1) #.92
    # return (1/N) * 10000**-(len(key)-1) #.9345
    return (1/N) * 12500**-(len(key)-1) #.9349
    # return (1/N) * 1000**-(len(key)-1) #.92
    # return (1/N) * 5000**-(len(key)-1) #.93
    # return (1/N) * 7500**-(len(key)-1) #.9334
    # return (1/N) * 15000**-(len(key)-1) #.9341

def penalize_long_unknown_baseline(key, N):
    return (1/N) * 1000000**-(len(key)-1)
    


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    ########################## For running baseline ####################################################
    # Pw = Pdist(data=datafile(opts.counts1w), missingfn=penalize_long_unknown_baseline)
    # segmenter = Segment(Pw)
    # with open(opts.input) as f: 
    #     for line in f:
    #         print(" ".join(segmenter.segment(line.strip())))

    ########################## For running bigram model ####################################################

    Pw = Pdist(data=datafile(opts.counts1w), missingfn=penalize_long_unknown) 
    P2w = Pdist(data=datafile(opts.counts2w), missingfn=penalize_long_unknown)


    segmenter2 = Segment(Pw,P2w)
    with open(opts.input) as f: 
        for line in f:
            print(" ".join(segmenter2.segment2(line.strip()))) 




