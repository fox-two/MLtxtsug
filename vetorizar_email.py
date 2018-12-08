from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser

from gensim.utils import simple_preprocess
from gensim.models import LsiModel


def convert2dense(sparse, size):
	vect = [0]*size

	for item in sparse:
		vect[item[0]] = item[1]
	
	return vect

class vectorizer:
	num_topics = 500
	def __init__(self):
		pass
	
	def fit_transform(self, data):
		data = [simple_preprocess(x, deacc=True) for x in data]

		phrases = Phrases(data, min_count=1, threshold=10)
		self.phraser = Phraser(phrases)
		data = self.phraser[data]

		self.dct = Dictionary(data) 
		docs_bow = [self.dct.doc2bow(line) for line in data]

		self.tfidf = TfidfModel(docs_bow)
		vectors = list(self.tfidf[docs_bow])

		self.lsimodel = LsiModel(corpus=vectors, num_topics=self.num_topics)


		retorno = [convert2dense(x, self.num_topics) for x in self.lsimodel[vectors]]
		
		return retorno


	def transform(self, text):
		text = simple_preprocess(text, deacc=True)
		palavras = self.phraser[text]
		bow = self.dct.doc2bow(palavras)
		return convert2dense(self.lsimodel[self.tfidf[bow]], 500)