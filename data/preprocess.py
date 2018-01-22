import glob
from nltk import word_tokenize
from gensim.models import KeyedVectors
from os.path import exists
import string
from random import shuffle
import pickle

MIN_TOKENS = 4
WORDS_EMBEDDINGS = 'GoogleNews-vectors-negative300.txt'
VAL_SIZE, TEST_SIZE = 100000, 100000
OUTPUT_FILE = 'bookcorpus_1M.p'

files = glob.glob('texts/*.story')

corpus = []
for file in files:
	with open(file, 'r', encoding='utf-8') as fp:
		for line in fp:
			line = line.strip().lower()
			if line != "":
				tokens = word_tokenize(line)
				if len(tokens) >= MIN_TOKENS:
					corpus.append(tokens)

if not exists(WORDS_EMBEDDINGS):
	model = Word2Vec.load_word2vec_format(WORDS_EMBEDDINGS.replace('.txt', '.bin.gz'), binary=True)
	model.save_word2vec_format(WORDS_EMBEDDINGS, binary=False)

# If we use word embeddings
#words = set()
#with open(WORDS_EMBEDDINGS, 'r', encoding='utf-8') as fp:
#	for line in fp:
#		line = line.strip()
#		words.add(line.split()[0])
#
#for c in string.punctuation:
#	words.add(c)

word_to_index = {'<END>':0, '<UNK>':1}
word_freq = {'<END>':0, '<UNK>':0}
next_index = 2

corpus_cleaned = []
for sentence in corpus:
	clean_sentence = []
	for token in sentence:
		# If we use word embedding
		#clean_sentence.append(token if token in words else '<UNK>')
		
		clean_sentence.append(token)
		if token not in word_to_index:
			word_to_index[token] = next_index
			next_index += 1
		if token not in word_freq:
			word_freq[token] = 0
		word_freq[token] += 1
	
	clean_sentence.append('<END>')
	corpus_cleaned.append(clean_sentence)

word_freq = dict(list(sorted(word_freq.items(), key=lambda x:x[1], reverse=True))[:100000])

corpus_cleaned_index = []
for sentence in corpus_cleaned:
	clean_sentence_index = [(word_to_index[token] if token in word_freq else word_to_index['<UNK>']) for token in sentence]
	corpus_cleaned_index.append(clean_sentence_index)

index_to_word = {v:k for k,v in word_to_index.items()}

corpus_final = list(zip(corpus, corpus_cleaned, corpus_cleaned_index))

shuffle(corpus_final)

val_set = corpus_final[:VAL_SIZE]
test_set = corpus_final[VAL_SIZE:VAL_SIZE+TEST_SIZE]
train_set = corpus_final[VAL_SIZE+TEST_SIZE:]

train, train_text = [x[2] for x in train_set], [x[0] for x in train_set]
val, val_text = [x[2] for x in val_set], [x[0] for x in val_set]
test, test_text = [x[2] for x in test_set], [x[0] for x in test_set]

output = [train, val, test, train_text, val_text, test_set, word_to_index, index_to_word]

with open(OUTPUT_FILE, 'wb') as fp:
	pickle.dump(output, fp, 2)
