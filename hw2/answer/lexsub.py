import os, sys, optparse
import tqdm
import pymagnitude
import numpy as np

# pcos = {}

class LexSub:

    def __init__(self, wvec_file, topn=10):
        self.wvecs = pymagnitude.Magnitude(wvec_file)
        self.topn = topn

    # def pcos(self,w1,w2):
    #     cos_val = np.dot(w1,w2)/(np.linalg.norm(w1, axis=1)*np.linalg.norm(w2))
    #     return (cos_val+1)/2

    def substitutes(self, index, sentence):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        
        # if (index-1) < 0:
        #     c = sentence[index+1]
        # elif (index+1) > (len(sentence)-1):
        #     c = sentence[index-1]
        # else:
        #     c = sentence[index-1]
        # target = sentence[index]

        # c = self.wvecs.query(self.wvecs.most_similar(sentence[index], topn=1)[0][0])

        # mmap = self.wvecs.get_vectors_mmap()
        # pcos_target = self.pcos(mmap,self.wvecs.query(sentence[index]))
        # pcos_c = self.pcos(mmap, c)

        # if c in pcos:
        #     pcos_c =  pcos[c]
        # else:
        #     pcos[c] = np.dot(mmap,self.wvecs.query(c))
        #     pcos_c =  pcos[c]

        # if target in pcos:
        #     pcos_target = pcos[target]
        # else:
        #     pcos[target] = np.dot(mmap,self.wvecs.query(c))
        #     pcos_target = pcos[target]


        # word_tokens = word_tokenize(sentence) 
        # res = 0
        # for word in sentence:
        #     res += self.wvecs.query(word)
        # res = np.add(self.wvecs.query(target),self.wvecs.query(c))
        # val = ((pcos_target)*(pcos_c))**(1/2)
        # ind = np.argpartition(val,-10)[-10:]
        # top_10 = []
        # for i in ind:
        #     top_10.append(self.wvecs[i][0])
        # return top_10
        return list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn)))

        # return list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn)))
        
if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="input file with target word in context")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.retrofit.magnitude'), help="word vectors file")
    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    lexsub = LexSub(opts.wordvecfile, int(opts.topn))
    num_lines = sum(1 for line in open(opts.input,'r'))
    with open(opts.input) as f:
        for line in tqdm.tqdm(f, total=num_lines):
            fields = line.strip().split('\t')
            print(" ".join(lexsub.substitutes(int(fields[0].strip()), fields[1].strip().split())))
