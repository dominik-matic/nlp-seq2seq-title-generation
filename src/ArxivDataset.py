import torch
import pandas as pd

class ArxivDataset(torch.utils.data.Dataset):
	
	def __init__(self):
		self.instances = []

	def set_vocab(self, vocab):
		self.vocab = vocab
	
	# not used
	def set_abstract_vocab(self, vocab):
		self.abstract_vocab = vocab
	
	# not used
	def set_title_vocab(self, vocab):
		self.title_vocab = vocab
	

	def __getitem__(self, index):
		title, abstract = self.instances[index]
		title_encoded = self.vocab.encode(title)
		title_encoded.append(self.vocab.stoi['<EOS>'])
		return torch.tensor(self.vocab.encode(abstract)), torch.tensor(title_encoded)

	def __len__(self):
		return len(self.instances)

	def from_file(self, filepath):
		data = pd.read_csv(filepath)
		for title, abstract in zip(data['title'], data['abstract']):
			title = title.split(' ')
			abstract = abstract.split(' ')
			self.instances.append((title, abstract))


if __name__ == '__main__':
	ds = ArxivDataset()
	ds.from_file('data_filtered.csv')
	print(ds.instances[0])
