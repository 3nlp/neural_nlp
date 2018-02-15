import torch
import torch.nn.functional as F
import torch.nn as nn
class BoundaryDecoderAttention(torch.nn.Module):
    '''
        input:  p:          batch x inp_p
                p_mask:     batch
                q:          batch x time x inp_q
                q_mask:     batch x time
                h_tm1:      batch x out
                depth:      int
        output: z:          batch x inp_p+inp_q
    '''

    def __init__(self, input_dim, output_dim):
        super(BoundaryDecoderAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.V = torch.nn.Linear(self.input_dim, self.output_dim)
        self.W_a = torch.nn.Linear(self.output_dim, self.output_dim)
        self.v = torch.nn.Parameter(torch.FloatTensor(self.output_dim))
        self.c = torch.nn.Parameter(torch.FloatTensor(1))
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.V.weight.data, gain=1)
        torch.nn.init.xavier_uniform(self.W_a.weight.data, gain=1)
        self.V.bias.data.fill_(0)
        self.W_a.bias.data.fill_(0)
        torch.nn.init.normal(self.v.data, mean=0, std=0.05)
        self.c.data.fill_(1.0)

    def forward(self, H_r,h_tm1):
        # H_r: batch x time x inp
        # mask_r: batch x time
        # h_tm1: batch x out
        batch_size, time = H_r.size(0), H_r.size(1)
        Fk = self.V.forward(H_r.view(-1, H_r.size(2)))  # batch*time x out
        Fk_prime = self.W_a.forward(h_tm1)  # batch x out
        Fk = Fk.view(batch_size, time, -1)  # batch x time x out
        Fk = torch.tanh(Fk + Fk_prime.unsqueeze(1))  # batch x time x out

        beta = torch.matmul(Fk, self.v)  # batch x time
        beta = beta + self.c.unsqueeze(0)  # batch x time
        beta = F.softmax(beta, axis=-1)  # batch x time
        z = torch.bmm(beta.view(beta.size(0), 1, beta.size(1)), H_r)  # batch x 1 x inp
        z = z.view(z.size(0), -1)  # batch x inp
        return z, beta
        
class BoundaryDecoder(torch.nn.Module):
    '''
    input:  encoded stories:    batch x time x input_dim
            input lengths of the encoded stories      
            init states:        batch x hid
    '''

    def __init__(self, input_dim, hidden_dim):
        super(BoundaryDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers=2
        self.attention_layer = BoundaryDecoderAttention(input_dim=input_dim,
                                                        output_dim=hidden_dim)
        self.rnn = nn.LSTM(self.input_dim,self.hidden_dim , self.n_layers, dropout=self.dropout)
    def forward(self, x,h_0):

        state_stp = [(h_0, h_0)]
        beta_list = []

        for t in range(2):

            previous_h, previous_c = state_stp[t]
            curr_input, beta = self.attention_layer.forward(x,h_tm1=previous_h)
            new_h, new_c = self.rnn(curr_input,previous_h, previous_c)
            state_stp.append((new_h, new_c))
            beta_list.append(beta)

        # beta list: list of batch x time
        res = torch.stack(beta_list, 2)  # batch x time x 2
        #res = res.unsqueeze(2)  # batch x time x 2
        return res
        
class MatchLSTMAttention(torch.nn.Module):
    '''
        input:  p (passage): batch x inp_p
                passage_lengths
                q (question) batch x time x inp_q
                question lengths
                h_tm1:      batch x out  (Output hidden state)
                depth:      int
        output: z:          batch x inp_p+inp_q
    '''

    def __init__(self, input_p_dim, input_q_dim, output_dim):
        super(MatchLSTMAttention, self).__init__()
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.output_dim = output_dim
        self.nlayers = len(self.output_dim)

        W_p = [torch.nn.Linear(self.input_p_dim if i == 0 else self.output_dim[i - 1], self.output_dim[i]) for i in range(self.nlayers)]
        W_q = [torch.nn.Linear(self.input_q_dim, self.output_dim[i]) for i in range(self.nlayers)]
        W_r = [torch.nn.Linear(self.output_dim[i], self.output_dim[i]) for i in range(self.nlayers)]
        w = [torch.nn.Parameter(torch.FloatTensor(self.output_dim[i])) for i in range(self.nlayers)]
        match_b = [torch.nn.Parameter(torch.FloatTensor(1)) for i in range(self.nlayers)]

        self.W_p = torch.nn.ModuleList(W_p)
        self.W_q = torch.nn.ModuleList(W_q)
        self.W_r = torch.nn.ModuleList(W_r)
        self.w = torch.nn.ParameterList(w)
        self.match_b = torch.nn.ParameterList(match_b)
        self.init_weights()

    def init_weights(self):
        for i in range(self.nlayers):
            torch.nn.init.xavier_uniform(self.W_p[i].weight.data, gain=1)
            torch.nn.init.xavier_uniform(self.W_q[i].weight.data, gain=1)
            torch.nn.init.xavier_uniform(self.W_r[i].weight.data, gain=1)
            self.W_p[i].bias.data.fill_(0)
            self.W_q[i].bias.data.fill_(0)
            self.W_r[i].bias.data.fill_(0)
            torch.nn.init.normal(self.w[i].data, mean=0, std=0.05)
            self.match_b[i].data.fill_(1.0)

    def forward(self, input_p, input_q, h_tm1, depth):
        input_p = torch.nn.utils.rnn.pack_padded_sequence(input_p, passage_lengths)
        input_q = torch.nn.utils.rnn.pack_padded_sequence(input_q, question_lengths)
        
        G_p = self.W_p[depth](input_p).unsqueeze(1)  # batch x None x out
        G_q = self.W_q[depth](input_q)  # batch x time x out
        G_r = self.W_r[depth](h_tm1).unsqueeze(1)  # batch x None x out
        G = F.tanh(G_p + G_q + G_r)  # batch x time x out
        alpha = torch.matmul(G, self.w[depth])  # batch x time
        alpha = alpha + self.match_b[depth].unsqueeze(0)  # batch x time
        alpha = F.softmax(alpha,axis=-1)  # batch x time
        alpha = alpha.unsqueeze(1)  # batch x 1 x time
        # batch x time x input_q, batch x 1 x time
        z = torch.bmm(alpha, input_q)  # batch x 1 x input_q
        z = z.squeeze(1)  # batch x input_q
        z = torch.cat([input_p, z], 1)  # batch x input_p+input_q
        return z
        
        
class MatchLSTM(torch.nn.Module):
    '''
    inputs: passages:          batch x time x inp_p
            passage lengths:
            questions:          batch x time x inp_q
            question lengths:     
    outputs:encoding:   batch x time x hid
            last state: batch x hid

    Dropout types:
        dropouth -- dropout on hidden-to-hidden connections
        dropoutw -- hidden-to-hidden weight dropout
    '''

    def __init__(self, input_p_dim, input_q_dim,hidden_dim,output_dim):
        super(MatchLSTM, self).__init__()
        self.hidden = hidden_dim
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.n_layers=2
        self.attention_layer = MatchLSTMAttention(input_p_dim, input_q_dim,
                                                  output_dim=self.hidden)
        self.output=output_dim
                                                  

        self.rnn = nn.GRU(self.hidden, self.output, self.n_layers, dropout=0.1,bidirectional=True)

    def forward(self, input_p,input_q):
        #Initial hidden state should be the question I guess!

        context = self.attn(input_p,input_q, h_tm1, depth)
        states, _ = self.rnn.forward(context)
        last_state = states[:, -1]  # batch x hid
        output = states.unsqueeze(-1)  # batch x time x hid
        return output, last_state
        