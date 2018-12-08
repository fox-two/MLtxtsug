from gensim.models import KeyedVectors
import gensim
import string
import numpy
from keras.models import Sequential, load_model
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, SpatialDropout1D
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import Callback
import csv
import json 
import time
from attention_decoder import Attention
from vetorizar_email import vectorizer

tabela_pontuacao = str.maketrans(dict.fromkeys(string.punctuation))
tamanho_maximo_texto = 100 #limite de palavras
tamanho_vetor_palavra = 300

tamanho_maximo_dicionario = 99999

def preprocessaTextoEmail(texto):
	return [x for x in gensim.utils.simple_preprocess(texto, deacc=False)]

def getDadosTreino():
	dataset = []
	
	#ler todos os emails com suas respostas pra memória
	emails = None
	import pickle
	with open('emails.bin', 'rb') as f:
		emails = pickle.load(f)
	
	#gera os vetores para as respostas dos emails
	resposta_vect = numpy.array(vectorizer().fit_transform([x[1] for x in emails]))

	#tokeniza os emails
	dataset = [x[0] for x in emails]
	dataset = [preprocessaTextoEmail(x) for x in dataset]

	#gera o dicionário
	dct = gensim.corpora.Dictionary(dataset)
	
	return dataset, resposta_vect, dct.token2id

def gerarModelo():
	model = Sequential()

	model_word2vec = gensim.models.KeyedVectors.load_word2vec_format('model.bin', binary=True)
	
	#cria camada embedding]
	embedding_matrix = numpy.zeros((len(dicionario), tamanho_vetor_palavra))
	
	for item in dicionario.items():
		if item[0] not in model_word2vec.wv.vocab:
			embedding_matrix[item[1]] = 0
		else:
			embedding_matrix[item[1]] = model_word2vec.wv.word_vec(item[0])

	embedding_layer = Embedding(embedding_matrix.shape[0], 
		                        embedding_matrix.shape[1], 
		                        weights=[embedding_matrix], trainable=True, input_length=tamanho_maximo_texto)
	model.add(embedding_layer)
	model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
	model.add(Attention(tamanho_maximo_texto))
	model.add(Dense(500))
	model.add(Dropout(0.2))
	model.add(Dense(vectorizer.num_topics,activation='sigmoid'))
	return model

def treinarRedeNeural():
	dataset, vetor_resposta, dicionario = getDadosTreino()
	
	#monta os dados de treino
	X = []
	for frase in dataset:
		frase_atual = [0]*tamanho_maximo_texto
		for i, palavra in enumerate(frase):
			if i >= tamanho_maximo_texto:
				break
			if palavra in dicionario:
				frase_atual[i] = dicionario[palavra]
		X.append(frase_atual)
	X = numpy.array(X, dtype=numpy.int32)
	X.reshape((len(dataset), tamanho_maximo_texto, 1))

	Y = vetor_resposta

	#monta a rede neural
	model = gerarModelo()

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.summary()
	

	model.fit(X, Y, epochs=60, batch_size=500, validation_split=0.6, verbose=1)
	
	model.save("meme")
	
def testeRedeNeural():
	dataset, classes, dicionario = getDadosTreino()
	
	model = gerarModelo()
	model.load_weights("meme")
	entrada = input("Texto a ser classificado: ")
	
	print(len(dicionario))
	texto = preprocessaTextoEmail(entrada)
	texto = [ dicionario[x] for x in texto if x in dicionario][0:tamanho_maximo_texto]
	while len(texto) < tamanho_maximo_texto:
		texto.append(0)
	
	X = numpy.zeros((1, tamanho_maximo_texto))
	X[0] = texto
	
	resposta = model.predict(X)
	print(resposta)
	print("#################################")
	print(resposta)
	
	contrario_classes = {classes[chave]: chave for chave in classes.keys()}
	for i in resposta:
		tuplas = []
		print("------------------------------------------------------")
		for numero, probabilidade in enumerate(i):
			tuplas.append((contrario_classes[numero], probabilidade))
		tuplas = sorted(tuplas, key=lambda x: -x[1])
		for row in tuplas:
			print(str(row[0]) + " -> " + str(row[1]))
	
		
def benchmarkRedeNeural():
	dataset, classes, dicionario = getDadosTreino()
	model = gerarModelo()
	model.load_weights("meme")
	
	X = numpy.zeros((len(dataset), tamanho_maximo_texto))
	for i, item in enumerate(dataset):
		X[i] = item[2]
		
	respostas = model.predict(X, verbose=1)
	
	resultados = []
	for i, item in enumerate(respostas):
		resultados.append(dataset[i][0] in numpy.argsort(item)[-1:])
		
	contagem = {}
	for i, item in enumerate(dataset):
		classe_atual = item[0]
		if classe_atual in contagem:
			contagem[classe_atual][1] += 1
			contagem[classe_atual][0] += resultados[i]
		else:
			contagem[classe_atual] = [resultados[i], 1]
	
	print("##############################\nResultado do benchmark:")
	contrario_classes = {classes[chave]: chave for chave in classes.keys()}
	
	soma_acertos = 0
	soma_tudo = 0	
	saida = []
	for item in contagem.keys():
		saida.append((contrario_classes[item], contagem[item][0], contagem[item][1], 100*contagem[item][0]/(contagem[item][1])))			
		soma_acertos += contagem[item][0]
		soma_tudo    += contagem[item][1]
	
	saida = sorted(saida, key=lambda k: -k[3])
	
	for item in saida:
		print("%s: %i/%i (%0.2f%%)" % (item[0], item[1], item[2], item[3]))
	
	print("\nTotal: %i de %i" % (soma_acertos, soma_tudo))
	print("Performance: %0.2f" % (100*soma_acertos/soma_tudo)) 
			
	
		
	
dataset, classes, dicionario = getDadosTreino()
#print("Dicionario: " + str(sorted(classes.items(), key=lambda k: k[0])))

acao = int(input ("1 - treinar;\n2 - testar;\n3 - benchmark\nOp")	)
	
if acao == 1:
	treinarRedeNeural()
elif acao == 2:
	testeRedeNeural()
elif acao == 3:
	benchmarkRedeNeural()
#
#print(model_word2vec.wv.most_similar(positive='explodiu'))
