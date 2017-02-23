"#!/usr/bin/python3"
from nltk import word_tokenize, sent_tokenize
import math

#
# PRE-PROCESSING
#

def flatten(listoflists):
   return [elt for l in listoflists for elt in l]

def get_sentences(file):
    f_in = open(file, 'r')
    return flatten([sent_tokenize(line) for line in f_in.readlines()])

def get_words(sentences):
    return flatten([word_tokenize(sentence) for sentence in sentences])

def get_freqs(file):
    
    freqs = {}
    words = get_words(get_sentences(file))
    for word in words:
        freqs[word] = freqs.get(word, 0) + 1
    return freqs

def file_to_freqmodel(infile, outfile):

    freqs = get_freqs(infile)
    f_out = open(outfile, 'w+')

    for word in freqs:
        f_out.write(word+','+str(freqs[word])+'\n')

    return

#
# Part 1.1
#

class UnigramModel:
    
    def __init__(self, freqmodel):

        counts = {}
        total = 0
        
        f = open(freqmodel, 'r')
        for line in f.readlines():
            s = line.rsplit(',',1)
            counts[s[0]] = int(s[1])
            total += int(s[1])

        self.counts = counts
        self.total = total

        return
    
    def logprob(self, target_word):

        if(target_word in self.counts):
            return math.log(self.counts[target_word]/self.total,2)
        else:
            return float('-inf')

#
# Part 1.2
#

#helper function
def get_count_counts(frequency_model):

    counts = {}

    f = open(frequency_model, 'r')
    for line in f.readlines():
        s = line.rsplit(',',1)
        count = int(s[1])
        if(count <= 6):
            counts[count] = counts.get(count, 0) + 1

    for i in range(0,6):
        if i not in counts:
            counts[i] = 0

    return counts


def get_good_turing(frequency_model):

    counts = get_count_counts(frequency_model)

    good_turing = {0: counts[1]}

    for i in range(1,6):
        good_turing[i] = (i+1)*(counts[i+1])/(counts[i])

    return good_turing

#
# Part 2.2
#

def get_type_token_ratio(counts_file):
    return

#
# Part 2.3
#

def get_entropy(unigram_counts_file):
    return

#
# Part 3
#

class BigramModel:
    def __init__(self, trainfiles):
        return
    def logprob(self, prior_context, target_word):
        return

#
# Part 4
#

def srilm_bigram_models(input_file, output_dir):
    return

def srilm_ppl(model_file, raw_text):
    return

#
# Running code
#

def make_freqmodels():
    file_to_freqmodel('data/train/nytimes.txt', 'nyt_freqmodel.txt')
    file_to_freqmodel('data/train/cancer.txt', 'cancer_freqmodel.txt')
    file_to_freqmodel('data/train/obesity.txt', 'obesity_freqmodel.txt')

def part_1():

    with open('hw2_1_2_nytimes.txt', 'w+') as f:
        gt = get_good_turing('nyt_freqmodel.txt')
        cc = get_count_counts('nyt_freqmodel.txt')

        for i in range(0,6):
            f.write(str(cc[i]) + '  ' + str(gt[i]) + '\n')

    with open('hw2_1_2_obesity.txt', 'w+') as f:
        gt = get_good_turing('obesity_freqmodel.txt')
        cc = get_count_counts('obesity_freqmodel.txt')

        for i in range(0,6):
            f.write(str(cc[i]) + '  ' + str(gt[i]) + '\n')

    with open('hw2_1_2_cancer.txt', 'w+') as f:
        gt = get_good_turing('cancer_freqmodel.txt')
        cc = get_count_counts('cancer_freqmodel.txt')

        for i in range(0,6):
            f.write(str(cc[i]) + '  ' + str(gt[i]) + '\n')

    return