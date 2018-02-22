from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import pickle
import numpy as np
from masked_cross_entropy import *

samples=pickle.load(open('samples_synth.pkl','rb'))
Vocab_dict=pickle.load(open('Vocab_synth.pkl','rb'))
USE_CUDA=True
l_max=100

inv_vocab = {v:k for k, v in Vocab_dict.items()}


embeddings_index = {}
f = open('glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(Vocab_dict.keys()), 300))
for word, i in Vocab_dict.items():
    embedding_vector = embeddings_index.get(str(word))
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        

#print(embedding_matrix)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,n_layers=2, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        #self.embedding.weight.requires_grad = False        

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

        
    def forward(self, input_seqs, input_lengths, hidden=None):
        
        embedded = self.embedding(input_seqs)

        
        #packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(embedded, hidden)

        #outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.w1 = nn.Linear(self.hidden_size, hidden_size)
            self.w2 = nn.Linear(self.hidden_size*2, hidden_size)
            self.w3 = nn.Linear(self.hidden_size*2,self.hidden_size)
            self.v = nn.Linear(self.hidden_size,1)


    def forward(self, hidden, encoder_outputs,question_vector):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S
        
        

        if USE_CUDA:
            attn_energies = attn_energies.cuda()
        

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0),question_vector[b])
                #highest=torch.max(highest,attn_energies[b, i])                
  
            
            attn_energies[b, :] = F.softmax(attn_energies[b, :].clone())

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return attn_energies.unsqueeze(0)
    
    def score(self, hidden, encoder_output,question_vector):

        
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':

            energy=F.tanh(self.w1(hidden)+self.w2(encoder_output)+self.w3(question_vector))
            #energy = F.hardtanh(self.attn(torch.cat((hidden, encoder_output), 1)))
 
            energy1 = self.v(energy)
   
            return energy1

class Attn1(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn1, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.w1 = nn.Linear(self.hidden_size, hidden_size)
            self.w2 = nn.Linear(self.hidden_size*2,self.hidden_size)
            self.v = nn.Linear(self.hidden_size,1)


    def forward(self, hidden, question_vector):
        max_len = question_vector.size(0)
        this_batch_size = question_vector.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S
        
        attn_energies = attn_energies.cuda()
        

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b],question_vector[i,b])
                #highest=torch.max(highest,attn_energies[b, i])                
  
            
            attn_energies[b, :] = F.softmax(attn_energies[b, :].clone())

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return attn_energies.unsqueeze(0)
    
    def score(self, hidden,question_vector):

        
        if self.method == 'dot':
            energy = hidden.dot(question_vector)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(question_vector)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':

            energy=F.tanh(self.w1(hidden)+self.w2(question_vector))

            energy1 = self.v(energy)
   
            return energy1

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn1 = Attn1('concat', hidden_size)
        self.attn2 = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size*3, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size*3, output_size)
        self.wx = nn.Linear(hidden_size,hidden_size)
        self.wh = nn.Linear(hidden_size*2,hidden_size)
        self.ws = nn.Linear(hidden_size,hidden_size)
        self.v = nn.Linear(hidden_size,1)
        
    
    def forward(self, word_input, last_hidden, encoder_outputs,encoder_input,question_embedding,decode=False):
        
        # Note: we run this one step at a time
        # TODO: FIX BATCHING
        
        if(decode):
            bs=1
        else:
            bs=batch_size
        
        # Get the embedding of the current input word (last output word)

        word_embedded = self.embedding(word_input) # S=1 x B x N  
        word_embedded= word_embedded.unsqueeze(0)

        word_embedded = self.dropout(word_embedded)
        word_embedded.permute(1, 0, 2)
        
        quest_attn=self.attn1(last_hidden, question_embedding) 
        question_embedding = quest_attn.transpose(0,1).bmm(question_embedding.transpose(0, 1)) # B x 1 x N



        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn2(last_hidden, encoder_outputs,question_embedding) 


        context = attn_weights.transpose(0,1).bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N

        context = context.transpose(0, 1) # 1 x B x N
        
        # Combine embedded input word and attended context, run through RNN

        rnn_input = torch.cat((word_embedded, context), 2)

        output, hidden = self.gru(rnn_input, last_hidden)

        # Final output layer
        output = output.squeeze(0) # B x N
  
        p_vocab = F.softmax(self.out(torch.cat((hidden.squeeze(0), context.squeeze(0)), 1)))
   
        embedded=word_embedded.squeeze()
        
        p_gen = F.sigmoid(self.v(self.wh(context.squeeze(0)) + self.ws(hidden.squeeze()) + self.wx(embedded)))

        attn = attn_weights.squeeze(0)

		
        in_seq=attn.shape[1]

        
        batch_indices = torch.arange(start=0, end=bs).long()
        
        batch_indices = batch_indices.expand(in_seq, bs).transpose(1, 0).contiguous().view(-1)
      
        idx_repeat = torch.arange(start=0, end=in_seq).repeat(bs).long()
        p_copy = Variable(torch.zeros(bs,30000)).cuda()
        word_indices = encoder_input.data.view(-1)

        
        for i in range(len(batch_indices)):
            p_copy[batch_indices[i],word_indices[i]] = p_copy[batch_indices[i],word_indices[i]].clone()+ attn[batch_indices[i],idx_repeat[i]].clone()
        # Return final output, hidden state, and attention weights (for visualization)
       
        p_out = torch.mul(p_vocab,p_gen) + torch.mul(p_copy,(1-p_gen)) # [b x extended_vocab]
        p_out=torch.log(p_out)  
        
        
        return p_out, hidden, attn_weights
            

def train(passage_input_batches, passage_input_lengths,question_input_batches,question_input_lengths, target_batches,target_lengths, question_encoder,passage_encoder, decoder, question_encoder_optimizer,passage_encoder_optimizer, decoder_optimizer, criterion):
 
        
    question_encoder_optimizer.zero_grad()
    passage_encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    loss = 0
    passage_embedding_batches, _ = passage_encoder(passage_input_batches, passage_input_lengths, None)
    question_embedding, question_hidden = question_encoder(question_input_batches, question_input_lengths, None)

    decoder_input = Variable(torch.LongTensor([29997]*batch_size))# SOS
    decoder_input=decoder_input.cuda()
    decoder_hidden = question_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    max_input_length = max(passage_input_lengths)

    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))        
    coverage_vector = Variable(torch.Tensor(batch_size,max_input_length).zero_())
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
        coverage_vector=coverage_vector.cuda()
    # Run through decoder one time step at a time
    input_words=[]
    output_words=[]    
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, passage_embedding_batches,passage_input_batches,question_embedding,decode=False
        )

        all_decoder_outputs[t] = decoder_output
        
        decoder_input = target_batches[t] # Next input is current target


        topv,topi=decoder_output.data[0,:].topk(1)
        

        ni=topi

    
        input_words.append(inv_vocab[int(target_batches[t,0].data)])
        output_words.append(inv_vocab[int(ni)])
    print(input_words)
    print(output_words)
    
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),
        target_batches.transpose(0, 1).contiguous(),
        target_lengths)

    

    loss.backward()

    question_encoder_optimizer.step()
    passage_encoder_optimizer.step()
    decoder_optimizer.step()
    
    torch.save(question_encoder.state_dict(), 'question_encoder_boundary.pt')
    torch.save(passage_encoder.state_dict(), 'question_encoder_boundary.pt')
    torch.save(decoder.state_dict(), 'decoder_boundary.pt')

    return loss.data[0]
    
# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [0 for i in range(max_length - len(seq))]
    return seq


    
import matplotlib.pyplot as plt

import random    
def random_batch(batch_size):
    passage_seqs = []
    question_seqs=[]
    target_seqs = []
    i=0
    while(i <batch_size):
        index = random.randint(0,len(samples)-1)
        pair = samples[index]

        tar=pair[3]
        if(tar!=[]):
            i=i+1
            passage_seqs.append(pair[0])
            question_seqs.append(pair[1])
            target_seqs.append(pair[2])
            

    # Zip into pairs, sort by length (descending), unzip

    seq_pairs = sorted(zip(passage_seqs, question_seqs,target_seqs), key=lambda p: len(p[0]), reverse=True)
    passage_seqs,question_seqs,target_seqs = zip(*seq_pairs)
    
    # For input and target sequences, get array of lengths and pad with 0s to max length
    passage_lengths = [len(s) for s in passage_seqs]
    passage_padded = [pad_seq(s, max(passage_lengths)) for s in passage_seqs]
    question_lengths = [len(s) for s in question_seqs]
    question_padded = [pad_seq(s, max(question_lengths)) for s in question_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]


    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    passage_var = Variable(torch.LongTensor(passage_padded)).transpose(0, 1)
    question_var = Variable(torch.LongTensor(question_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    passage_var = passage_var.cuda()
    question_var = question_var.cuda()
    target_var=target_var.cuda()
        
    return passage_var, passage_lengths, question_var, question_lengths,target_var,target_lengths



hidden_size = 300
n_layers = 1
dropout = 0.1
batch_size = 10

# Configure training/optimization
clip = 50.0

learning_rate = 1
decoder_learning_ratio = 1
n_epochs = 1000
epoch = 0
plot_every = 1000
print_every = 1
evaluate_every = 10
Vocab_size=len(Vocab_dict.keys())

# Initialize models
question_encoder=EncoderRNN(Vocab_size, hidden_size,n_layers=1, dropout=0.1)
passage_encoder=EncoderRNN(Vocab_size, hidden_size,n_layers=1, dropout=0.1)

decoder = BahdanauAttnDecoderRNN(hidden_size, Vocab_size, dropout_p=0.1)

# Initialize optimizers and criterion


#question_encoder_parameters = filter(lambda p: p.requires_grad, question_encoder.parameters())

#passage_encoder_parameters=parameters = filter(lambda p: p.requires_grad, passage_encoder.parameters())

question_encoder_optimizer = optim.Adadelta(question_encoder.parameters(), lr=learning_rate)
passage_encoder_optimizer = optim.Adadelta(passage_encoder.parameters(), lr=learning_rate)

decoder_optimizer = optim.Adadelta(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()

# Move models to GPU
question_encoder.cuda()
passage_encoder.cuda()
decoder.cuda()


# Keep track of time elapsed and running averages

plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every


ecs = []
dcs = []
eca = 0
dca = 0

while epoch < n_epochs:
    epoch += 1
    
    # Get training data for this cycle
    passage_batches, passage_lengths, question_batches, question_lengths,target_batches,target_lengths = random_batch(batch_size)

    # Run the train function
    loss = train(passage_batches, passage_lengths,question_batches,question_lengths, target_batches,target_lengths, question_encoder,passage_encoder, decoder, question_encoder_optimizer,passage_encoder_optimizer, decoder_optimizer, criterion)
    # Keep track of loss
    print_loss_total += loss

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary =  str(epoch / n_epochs * 100)+'% complete : loss=' + str(print_loss_avg)
        print(print_summary)

    