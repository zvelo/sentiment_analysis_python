import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

POLARITY_DATA_DIR = os.path.join('polarityData', 'rt-polaritydata')
RT_POLARITY_POS_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-pos.txt')
RT_POLARITY_NEG_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-neg.txt')

#this function takes a feature selection mechanism and returns its performance in a variety of metrics
def evaluate_features(feature_select):
	posFeatures = []
	negFeatures = []
	#http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
	#breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
	with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
		for i in posSentences:
			posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			posWords = [feature_select(posWords), 'pos']
			posFeatures.append(posWords)
	with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
		for i in negSentences:
			negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			negWords = [feature_select(negWords), 'neg']
			negFeatures.append(negWords)

	posCutoff = int(math.floor(len(posFeatures)*3/4))
	negCutoff = int(math.floor(len(negFeatures)*3/4))
	trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]

	global CLASSIFIER
	CLASSIFIER = NaiveBayesClassifier.train(trainFeatures)	

#creates a feature selection mechanism that uses all words
def make_full_dict(words):
	return dict([(word, True) for word in words])

#scores words based on chi-squared test to show information gain (http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/)
def create_word_scores():
	#creates lists of all positive and negative words
	posWords = []
	negWords = []
	with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
		for i in posSentences:
			posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			posWords.append(posWord)
	with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
		for i in negSentences:
			negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			negWords.append(negWord)
	posWords = list(itertools.chain(*posWords))
	negWords = list(itertools.chain(*negWords))

	#build frequency distibution of all words and then frequency distributions of words within positive and negative labels
	word_fd = FreqDist()
	cond_word_fd = ConditionalFreqDist()
	for word in posWords:
		word_fd.inc(word.lower())
		cond_word_fd['pos'].inc(word.lower())
	for word in negWords:
		word_fd.inc(word.lower())
		cond_word_fd['neg'].inc(word.lower())

	#finds the number of positive and negative words, as well as the total number of words
	pos_word_count = cond_word_fd['pos'].N()
	neg_word_count = cond_word_fd['neg'].N()
	total_word_count = pos_word_count + neg_word_count

	#builds dictionary of word scores based on chi-squared test
	word_scores = {}
	for word, freq in word_fd.iteritems():
		pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
		neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
		word_scores[word] = pos_score + neg_score

	return word_scores

#finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
	best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
	best_words = set([w for w, s in best_vals])
	return best_words

#creates feature selection mechanism that only uses best words
def best_word_features(words):
	return dict([(word, True) for word in words if word in best_words])

# returns sentiment rating for a given phrase
def sentiment(phrase):
	return CLASSIFIER.classify(make_full_dict(phrase))

word_scores = create_word_scores()
best_words = find_best_words(word_scores, 15000)
evaluate_features(best_word_features)

while True:
    try:
        line = raw_input();
    except EOFError:
        break
    print sentiment(line)
