"#!/usr/bin/python3"
from nltk import word_tokenize, sent_tokenize
import math
import os
import statistics
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

# same as get_freqs but for a string instead of a file
def get_string_freqs(s):
    freqs = {}
    words = get_words(sent_tokenize(s))
    for word in words:
        freqs[word] = freqs.get(word, 0) + 1
    return freqs

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

        # internally store counts as a dict and total number of tokens
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

# helper function
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

    with open(counts_file, "r") as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    tokens = 0
    types = 0
    for l in content:
        types+=1
        tokens+= int(l.rsplit(",", 1)[1])
    return float(types)/float(tokens)

#
# Part 2.3
#

def get_entropy(unigram_counts_file):

    uModel = UnigramModel(unigram_counts_file)

    entropy = 0

    for word in uModel.counts:
        entropy -= (uModel.counts[word]/uModel.total)*uModel.logprob(word)

    return entropy

#
# Part 3
#

class BigramModel:

    def __init__(self, trainfiles):

        sentences = []

        # get sentences
        for trainfile in trainfiles:
            s = get_sentences(trainfile)
            sentences.extend(s)

        words = get_words(sentences)

        word_counts = {'<UNK>': 0}

        # get word counts
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        one_count_words = []

        # convert all words appearing once to <UNK>
        for word in word_counts:
            if(word_counts[word] == 1):
                word_counts['<UNK>'] += 1
                one_count_words.append(word)

        for word in one_count_words:
            del word_counts[word]

        # store individual word counts
        self.word_counts = word_counts

        # store sentences
        self.sentences = sentences

        word_pair_counts = {}

        # get word pair counts
        for sentence in self.sentences:
            words = ["<s>"]
            words.extend(word_tokenize(sentence))
            words.append("</s>")
            for i in range(1, len(words)-1):

                # first word in a sentence
                if(i == 1):
                    word_pair_counts[("<s>", words[i])] = word_pair_counts.get(
                        ("<s>", words[i]), 0) + 1

                # last word in a sentence
                if (i == len(words) -2):
                    word_pair_counts[(words[i], "</s>")] = word_pair_counts.get(
                        (words[i], "</s>"), 0) + 1

                # in the middle of a sentence
                else:
                    word_pair_counts[(words[i], words[i+1])] = word_pair_counts.get(
                        (words[i], words[i+1]), 0) + 1

        # store word pair counts
        self.word_pair_counts = word_pair_counts

        # store the vocab size
        self.V = len(word_counts)

        return

    def logprob(self, prior_context, target_word):

        c_pair = self.word_pair_counts.get((prior_context, target_word), 0)
        c_context = self.word_counts.get(prior_context, self.word_counts.get("<UNK>"))

        prob = (c_pair + 0.25)/(c_context + 0.25*self.V)

        return math.log(prob,2)

#
# Part 4
#
def srilm_ppl(model_file, raw_text):
    file_name = os.path.basename(raw_text)
    test_file = srilm_preprocess(raw_text, "temp"+file_name)
    cmd = '/home1/c/cis530/srilm/ngram -lm %s -ppl %s' % ( model_file, test_file)
    os.system(cmd)

def srilm_preprocess(raw_text, temp_file):
    f = open(temp_file, "w+")
    list_of_sentences = get_sentences(raw_text)
    for sentence in list_of_sentences:
        f.write(sentence+"\n")
    return temp_file


def srilm_bigram_models(input_file, output_dir):
    file_name = os.path.basename(input_file)
    temp = "temp"+file_name
    temp_input_file = srilm_preprocess(input_file, temp)
    
    cmd1 = '/home1/c/cis530/srilm/ngram-count -text %s -lm %s -order 1 -addsmooth 0.25' % ( temp_input_file, file_name+'.uni.lm' )
    cmd2 = '/home1/c/cis530/srilm/ngram-count -text %s -lm %s -order 2 -addsmooth 0.25' % ( temp_input_file, file_name+'.bi.lm' )
    cmd3 = '/home1/c/cis530/srilm/ngram-count  -text %s -lm %s -order 2 -kndiscount' % ( temp_input_file, file_name+'.bi.kn.lm' )
    
    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)

    return


#
# Part 5
#


def get_file_info(excerpt):

    freq = get_string_freqs(excerpt)
    ny = list(get_freqs("nytimes.txt"))
    
    #nyt_fract
    diff = list(set(freq) - set(ny))
    nyt_fract = len(diff)/len(freq)
    
    #vocab_size
    vocab_size = len(freq)
    
    
    #Getting frac_freq
    count = 0
    
    for i in freq.keys():
        if freq[i] > 5:
            count = count + 1
    frac_freq = float(count)/vocab_size

    #frac_rare
    frac_rare = 0
    word_lengths = 0
    for word in freq:
        word_lengths+= len(word)
        if freq[word] == 1:
            frac_rare+=1

    frac_rare /= vocab_size

    #average_word
    average_word = float(word_lengths)/vocab_size
    
    #median_word length
    median_word = statistics.median([len(word) for word in freq])

    # type_token ratio
    type_token = float(vocab_size)/len(get_words(sent_tokenize(excerpt)))

    #entropy
    total_freqs = 0
    for word in freq:
        total_freqs += freq[word]
    entropy = 0
    for word in freq:
        prob = freq[word]/total_freqs
        entropy -= prob*math.log(prob,2)
    
    return (vocab_size, frac_freq, frac_rare, average_word, median_word, nyt_fract, type_token, entropy)

def get_readability_scores(excerpts):

    attributes = {}

    # calculate attributes
    for i in range(0, len(excerpts)):
        attributes[i] = get_file_info(excerpts[i])

    vals = attributes.values()

    # get max and min
    
    vs_max = max([val[0] for val in vals])
    vs_min = min([val[0] for val in vals])

    ff_max = max([val[1] for val in vals])
    ff_min = min([val[1] for val in vals])

    fr_max = max([val[2] for val in vals])
    fr_min = min([val[2] for val in vals])

    aw_max = max([val[3] for val in vals])
    aw_min = min([val[3] for val in vals])

    mw_max = max([val[4] for val in vals])
    mw_min = min([val[4] for val in vals])

    fn_max = max([val[5] for val in vals])
    fn_min = min([val[5] for val in vals])

    tt_max = max([val[6] for val in vals])
    tt_min = min([val[6] for val in vals])

    e_max = max([val[7] for val in vals])
    e_min = min([val[7] for val in vals])

    # calculate scores
    scores = {}

    for i in range(0, len(excerpts)):
        scores[i] = 0
        scores[i] += 2*(attributes[i][6]-tt_min)/(tt_max-tt_min)
        scores[i] += 2*(attributes[i][7]-e_min)/(e_max-e_min)
        scores[i] += 0.5*(attributes[i][0]-vs_min)/(vs_max-vs_min)
        scores[i] += (ff_max-attributes[i][1])/(ff_max-ff_min)
        scores[i] += (attributes[i][2]-fr_min)/(fr_max-fr_min)
        scores[i] += (attributes[i][3]-aw_min)/(aw_max-aw_min)
        scores[i] += 0.5*(attributes[i][4]-mw_min)/(mw_max-mw_min)
        scores[i] += 2*(fn_max-attributes[i][5])/(fn_max-fn_min)


    return scores

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

def calculate_entropy():
    print("NY Times: " + str(get_entropy('nyt_freqmodel.txt')))
    print("obesity: " + str(get_entropy('obesity_freqmodel.txt')))
    print("cancer: " + str(get_entropy('cancer_freqmodel.txt')))

def readability():
    with open('data/test/cloze.txt') as f:
        scores = get_readability_scores(f.readlines())
        with open('readability.txt', 'w+') as f_out:
            for i in range(0, len(scores)):
                f_out.write(str(scores[i])+'\n')
    return