from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
from extras import MatchLSTM,BoundaryDecoder
from masked_cross_entropy import *
from torch.autograd import Variable
from torch import optim


        
class MatchLSTMModel(torch.nn.Module):
    def __init__(self,data_specs,input_p_dim, input_q_dim):
        super(MatchLSTMModel, self).__init__()
        self.data_specs = data_specs
        self.vocab_size = data_specs['Vocab_size']
        self.id2word = data_specs['id2word']
        self.hidden_size=500
        
        self.word_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        
        self.match_lstm=MatchLSTM(input_p_dim, input_q_dim,self.hidden_size,self.hidden_size)                              
        

        self.decoder_init_state_generator = torch.nn.Linear(self.hidden_size[-1], self.hidden_size)
        
        self.boundary_decoder = BoundaryDecoder(input_dim=self.hidden_size[-1], hidden_dim=self.hidden_size)

    def forward(self, passage_batches, passage_lengths,question_batches,question_lengths):
        # word embedding
        passage_embedding= self.word_embedding.forward(passage_batches)  # batch x time x emb
        question_embedding= self.word_embedding.forward(question_batches)  # batch x time x emb
        passage_embedding = torch.nn.utils.rnn.pack_padded_sequence(passage_embedding, passage_lengths)
        question_embedding = torch.nn.utils.rnn.pack_padded_sequence(question_embedding, question_lengths)

        # match lstm
        passage_match_encoding, last_state= self.match_lstm.forward(passage_embedding,question_embedding)
        # generate decoder init state using passage match encoding last state (batch x hid)
        init_state = self.decoder_init_state_generator.forward(last_state)
        init_state = torch.tanh(init_state)  # batch x hid
        # decode
        output = self.boundary_decoder.forward(passage_match_encoding,init_state)  # batch x time x 2

        return output
                
def train(passage_batches, passage_lengths, question_batches, question_lengths,target_batches,target_lengths, encoder,encoder_optimizer,criterion):
    encoder_optimizer.zero_grad()
    loss = 0

    encoder_outputs= encoder(passage_batches, passage_lengths,question_batches, question_lengths)
    
    loss = masked_cross_entropy(
        encoder_outputs.transpose(0, 1).contiguous(),
        target_batches.transpose(0, 1).contiguous(),
        target_lengths)
    loss.backward()
    
    encoder_optimizer.step()
    
    torch.save(encoder.state_dict(), 'encoder_pointer_wob.pt')
    torch.save(decoder.state_dict(), 'decoder_pointer_wob.pt')

    return loss.data[0]

def pad_seq(seq, max_length):
    seq += [0 for i in range(max_length - len(seq))]
    return seq

def indexes_from_sentence(sentence):
    return [Vocab_dict[word] for word in sentence] + [Vocab_dict['<END>']]
    
import random    
def random_batch(batch_size):
    passage_seqs = []
    question_seqs=[]
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        index = random.randint(0,len(pairs)-1)
        pair = pairs[index]

        passage_seqs.append(indexes_from_sentence(pair[0]))
        question_seqs.append(indexes_from_sentence(pair[1]))
        target_seqs.append(indexes_from_sentence(pair[2]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(passage_seqs, question_seqs,target_seqs), key=lambda p: len(p[0]), reverse=True)
    passage_seqs,question_seqs,target_seqs = zip(*seq_pairs)
    
    # For input and target sequences, get array of lengths and pad with 0s to max length
    passage_lengths = [len(s) for s in passage_seqs]
    passage_padded = [pad_seq(s, max(passage_lengths)) for s in passage_seqs]
    question_lengths = [len(s) for s in question_seqs]
    question_padded = [pad_seq(s, max(question_lengths)) for s in question_seqs]

    '''
    Need to create targets! Targets are 1st and last words of the extracted sentences
    
    '''

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    passage_var = Variable(torch.LongTensor(passage_padded)).transpose(0, 1)
    question_var = Variable(torch.LongTensor(question_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_seqs)).transpose(0, 1)

    passage_var = passage_var.cuda()
    question_var = question_var.cuda()
    target_var=target_var.cuda()
        
    return passage_var, passage_lengths, question_var, question_lengths,target_var
    
    
epoch=0
n_epochs=5000
batch_size=50

encoder = MatchLSTMModel(self,data_specs,input_p_dim, input_q_dim)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
encoder.cuda()
print_loss_total=0

print_every=1



while epoch < n_epochs:
    epoch += 1
    
    # Get training data for this cycle
    passage_batches, passage_lengths, question_batches, question_lengths,target_batches = random_batch(batch_size)

    # Run the train function
    loss = train(passage_batches, passage_lengths, question_batches, question_lengths,target_batches,encoder,encoder_optimizer,criterion)

    # Keep track of loss
    print_loss_total += loss

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary =  str(epoch / n_epochs * 100)+'% complete : loss=' + str(print_loss_avg)
        print(print_summary)
        

    