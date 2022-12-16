import random
import torch
from torch import nn

class Encoder(nn.Module):
	def __init__(self,
				input_size,
				hidden_size,
				num_layers=1,
				dropout_p=0.2,
				bidirectional=False):
		super(Encoder, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout = nn.Dropout(dropout_p)
		self.bidirectional = bidirectional
		self.gru = nn.GRU(num_layers=self.num_layers, input_size=self.input_size, hidden_size=hidden_size, bidirectional=bidirectional, batch_first=True)
		
		if bidirectional:
			self.fc = nn.Linear(self.hidden_size * 2 * num_layers, self.hidden_size)


	# x dim: [batch_size, max_sequence_len_in_batch, word_vec_len]
	def forward(self, x, h, lens):
		x = self.dropout(x)
		x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
		y, h = self.gru(x, h)
		y, _ = nn.utils.rnn.pad_packed_sequence(y)

		if self.bidirectional:
			h = self.fc(torch.cat((h[0,:,:], h[1,:,:]), dim=1))
			h = torch.tanh(h)
		else:
			h = h.squeeze(0)

		return y, h
	
	def init_hidden(self, batch_size):
		return torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size)


class Decoder(nn.Module):
	def __init__(self,
				input_size,
				hidden_size,
				output_size,
				bidirectional=True,
				dropout_p = 0.1,
				device= 'cpu'):
		super(Decoder, self).__init__()
		self.device = device

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.dropout = nn.Dropout(dropout_p)
		self.softmax = nn.Softmax(dim=1)		# UNSURE
		self.relu = nn.ReLU()
		self.log_softmax = nn.LogSoftmax(dim=1)
		
		self.attn = nn.Linear(self.hidden_size * (3 if bidirectional else 2), self.hidden_size)
		self.v = nn.Parameter(torch.rand(self.hidden_size))
		self.gru = nn.GRU(self.hidden_size * (3 if bidirectional else 2), self.hidden_size, batch_first=True)
		self.fc = nn.Linear(self.input_size + self.hidden_size * (3 if bidirectional else 2), self.output_size)

	def forward(self, x, h, enc_out, mask):
		"""
		Inputs:
			x: 			input word				[ B H ]
			h: 			encoder hidden state	[ B H ]
			enc_out:	encoder output			[ len B 2H ] or [ len B H ]
		
		Outputs:
			y:
			h:
			a:
		"""
		x = x.unsqueeze(1)								# [ B 1 H ]
		x = self.dropout(x)
		### calculating attention ###
		h = h.unsqueeze(1)								# [ B 1 H ]
		h_r = h.repeat(1, enc_out.shape[0] , 1)			# repeat (1, len, 1) => [ B len H ]
		enc_out = enc_out.permute(1, 0, 2)				# [ B len 2H ] or [ B len H ]
		e = torch.cat((h_r, enc_out), dim=2)			# [B len 3H] or [ B len 2H ]
		e = self.attn(e)								# [ B len H ]
		e = torch.tanh(e)

		e = e.permute(0, 2, 1)							# prepare for BMM => [B H len]
		v = self.v.repeat(x.shape[0], 1)				# repeat (B, 1) => [ B H ]
		v = v.unsqueeze(1)								# [ B 1 H ]

		attention = torch.bmm(v, e)						# [ B 1 H ] x [ B H len] => [B 1 len]
		attention = attention.squeeze(1)				# [ B len ]

		attention = attention.masked_fill(mask == 0, -1e20)

		attention = self.softmax(attention)				# [ B len ]
		### attention calculated ###

		### apply attention ###
		attention = attention.unsqueeze(1)				# [ B 1 len ]
		applied = torch.bmm(attention, enc_out)			# [ B 1 len ] x [ B len 2H/H ] => [B 1 2H/H]
		
		x_with_applied = torch.cat((x, applied), dim=2)	# [ B 1 2H/H ]
		
		h = h.permute(1, 0, 2)							# gru expects hidden in [ 1 B H ]
		y, h = self.gru(x_with_applied, h)				# [ B 1 H ], [ B 1 H ]
		
		y = y.squeeze(1)
		x = x.squeeze(1)
		applied = applied.squeeze(1)
		prediction_input = torch.cat((y, applied, x), dim=1)

		prediction = self.fc(prediction_input)
		return prediction, h.squeeze(0), attention.squeeze(1)


class Seq2Seq(nn.Module):
	def __init__(self,
				input_size,
				hidden_size,
				output_size,
				emb_matrix,
				pad_idx = 0,
				sos_idx = 2,
				eos_idx = 3,
				enc_dropout_p = 0.1,
				dec_dropout_p = 0.1,
				enc_num_layers = 1,
				bidirectional = True,
				out_max_len=100,
				device='cpu'):
		super(Seq2Seq, self).__init__()


		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.enc_dropout_p = enc_dropout_p
		self.dec_dropout_p = dec_dropout_p
		self.emb_matrix = emb_matrix.float()
		self.pad_idx = pad_idx #torch.tensor([pad_idx]).to(device)
		self.sos_idx = sos_idx #torch.tensor([sos_idx]).to(device)
		self.eos_idx = eos_idx #torch.tensor([eos_idx]).to(device)
		self.enc_num_layers = enc_num_layers
		self.out_max_len = out_max_len
		self.device = device

		self.encoder = Encoder(self.input_size, self.hidden_size, num_layers=self.enc_num_layers, dropout_p=self.enc_dropout_p, bidirectional=bidirectional).to(device)
		self.decoder = Decoder(self.input_size, self.hidden_size, self.output_size, dropout_p=self.dec_dropout_p, bidirectional=bidirectional, device=device).to(device)

	def forward(self, x, x_lengths, teacher_forcing=0.3, y_true=None):

		inference = False

		if y_true is not None:
			if teacher_forcing < 0.0:
				raise Exception("Invalid teacher_forcing ratio")
		else:
			inference = True
			teacher_forcing = 0.0
			y_true = torch.zeros((x.shape[0], self.out_max_len)).long().fill_(self.sos_idx).to(self.device)
			

		batch_size = x.shape[0]

		out_max_len = y_true.shape[1]
		outputs = torch.zeros(out_max_len, batch_size, self.output_size)
		atts = torch.zeros(out_max_len, batch_size, x.shape[1])

		mask = (x != self.pad_idx)

		x = self.emb_matrix(x)
		h = self.encoder.init_hidden(batch_size).to(self.device)
		enc_out, h = self.encoder.forward(x, h, x_lengths)

		dec_in = self.emb_matrix(y_true[:, 0])

		for i in range(out_max_len):
			outputs[i], h, atts[i] = self.decoder.forward(dec_in, h, enc_out, mask)
			output = outputs[i].max(1)[1]
			dec_in = (y_true[:, i] if (random.random() < teacher_forcing) else output).to(self.device)
			if inference and dec_in.item() == self.eos_idx:
				break
			dec_in = self.emb_matrix(dec_in)
		
		return outputs


if __name__ == '__main__':
	emb_matrix = nn.Embedding(10000, 300)
	model = Seq2Seq(300, 300, 10000, 100, emb_matrix, bidirectional=True)
	y = model.forward(torch.tensor([[1, 2, 3, 0, 0]]), torch.tensor([5]))
	print(y.shape)