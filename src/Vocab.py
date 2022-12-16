from collections import defaultdict
import torch

class Vocab:
	def __init__(self, frequencies, max_size=-1, min_freq=0):
		self.itos = [(word, freq) for word, freq in frequencies.items() if freq >= min_freq]	# filter out the words by frequency
		self.itos.sort(key=lambda x: x[1], reverse=True)										# sort by frequency
		self.__freqs__ = [x[1] for x in self.itos]
		self.itos = [x[0] for x in self.itos]													# remove frequency

		for special_symbol in ['<EOS>', '<SOS>', '<UNK>', '<PAD>']:
			self.itos.insert(0, special_symbol)
		for i in range(4):
			self.__freqs__.insert(0, 0)
		
		
		if max_size > -1 and max_size < len(self.itos):
			self.itos = self.itos[0:max_size] 
			
		self.stoi = defaultdict(lambda: 1) 	# value of <UNK>
		for i, word in enumerate(self.itos):
			self.stoi[word] = i
	
	def encode(self, text):
		if isinstance(text, str):
			return self.stoi[text]
		return [self.stoi[word] for word in text]
		
