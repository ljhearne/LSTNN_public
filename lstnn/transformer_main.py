import torch
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
import lstnn.positionalencoding.sparse_block_attn as sparse_block_attn
import lstnn.positionalencoding.rotary_positionalencoding as rotary_positionalencoding
import lstnn.positionalencoding.learnable_pe as learnable_pe

class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding
    """
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        #position = torch.arange(max_len).unsqueeze(1)
        #div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(max_len*2) / d_model))
        #pe = torch.zeros(1, max_len, d_model)
        #pe[0,:,:] = position / max_len
        ##pe[:, 0, 0::2] = torch.sin(position * div_term)
        ##pe[:, 0, 1::2] = torch.cos(position * div_term)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[0,:x.size(1)]
        return self.dropout(x)

class PositionalEncoding2D(torch.nn.Module):
    """
    2D Positional encoding
    """
    def __init__(self, d_model, height=4, width=4, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))

        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        # flatten
        pe = torch.flatten(pe,start_dim=1,end_dim=2)
        pe = pe.T #seq_length x embed dim
        assert(pe.shape[0]==16)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

# From here:https://jaketae.github.io/study/relative-positional-encoding/
class RelativeGlobalAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.0):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.query = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.Er = torch.nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    
    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )
        
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)
        
        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        
        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)

    def forward_attn(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )
        
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)
        
        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        
        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out), attn
        
    
    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel

class Transformer(torch.nn.Module):
    """
    base transformer model
    Inputs are the tokens of the LST grid
    """
    def __init__(self,
                 input_dim=5, # dimension of input tokens (5, one for each element)
                 output_dim=4,
                 max_tokens=16, # tokens per grid
                 nhead=1,
                 nblocks=1,
                 embedding_dim=160,
                 dropout=0,
                 positional_encoding='absolute',
                 pe_init=1.0
                 ):
        super(Transformer,self).__init__()

        # Define general parameters
        self.input_dim = input_dim
        
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.nhead = nhead
        self.dropout = dropout
        self.positional_encoding = positional_encoding
        
        # a linear embedding
        #self.w_embed = torch.nn.Linear(self.input_dim+1,self.embedding_dim) # additional input dim for EOS token
        #no eos
        self.w_embed = torch.nn.Linear(self.input_dim,self.embedding_dim) 

        #### multimodal transformer block
        #max_tokens = max_tokens + 1 # INCLUDE EOS
        #no eos
        max_tokens = max_tokens 
        if positional_encoding in ['scnope']:
            blocks = []
            for i in range(1,nblocks+1):
                if i%2==0: 
                    blocks.append(TransformerBlock(
                        embedding_dim,
                        max_tokens,
                        nhead=self.nhead,
                        dropout=self.dropout,
                        positional_encoding=positional_encoding,
                        causal='reverse',
                    ))
                else:
                    blocks.append(TransformerBlock(
                        embedding_dim,
                        max_tokens,
                        nhead=self.nhead,
                        dropout=self.dropout,
                        positional_encoding=positional_encoding,
                        causal='forward',
                    ))
            self.blocks = torch.nn.Sequential(*blocks)
        else:
            self.blocks = torch.nn.Sequential(*[TransformerBlock(embedding_dim,
                                                                max_tokens,
                                                                nhead=self.nhead,
                                                                dropout=self.dropout,
                                                                positional_encoding=positional_encoding,
                                                                pe_init=pe_init)
                                                for _ in range(nblocks)])

        self.w_out = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim,self.output_dim),
            torch.nn.LogSoftmax(dim=-1)
        )
        # self.w_out = torch.nn.Sequential(
        #     torch.nn.Linear(self.embedding_dim,self.output_dim)
        # )
        
    def forward(self, task_inputs, noise=False, dropout=False):
        """
        Run a forward pass of a trial by input_elements matrix
        """
        device = task_inputs.device

        # transformer
        # pad inputs for EOS token
        # no eos
        #train_task_pad = torch.nn.functional.pad(task_inputs,(0,1,0,1),'constant')
        #train_task_pad[:,-1,-1] = 1

        train_task_pad = task_inputs
        embedding = self.w_embed(train_task_pad)

        ####
        if self.positional_encoding in ['scnope']:
            l1_reg = 0
            for block in self.blocks:
                embedding, l1_attn = block.forward(embedding)
                l1_reg += l1_attn
            transformer_out = embedding
        else:
            transformer_out = checkpoint_sequential(self.blocks, segments = len(self.blocks), input = embedding)

        outputs = self.w_out(transformer_out[:,-1,:])

        if self.positional_encoding in ['scnope']:
            return outputs, l1_reg/len(self.blocks)
        else:
            return outputs

class TransformerBlock(torch.nn.Module):
    """
    Transformer block
    """
    def __init__(self,
                 embedding_dim,
                 n_tokens,
                 positional_encoding='absolute',
                 nhead=1,
                 dropout=0,
                 learning_rate=0.0001,
                 lossfunc='CrossEntropy',
                 causal='forward',
                 pe_init=1.0):
        super(TransformerBlock,self).__init__()

        # Define general parameters
        self.nhead = nhead
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_embed = torch.nn.Dropout(p=dropout)
        self.positional_encoding = positional_encoding
        
        # positional encoding
        if positional_encoding=='absolute':
            self.pe = PositionalEncoding(self.embedding_dim, max_len=n_tokens)
            self.selfattention = torch.nn.MultiheadAttention(self.embedding_dim,nhead,dropout,batch_first=True)
        elif positional_encoding=='absolute2d':
            self.pe = PositionalEncoding2D(self.embedding_dim, height=4, width=4)
            self.selfattention = torch.nn.MultiheadAttention(self.embedding_dim,nhead,dropout,batch_first=True)
        elif positional_encoding=='relative':
            self.selfattention = RelativeGlobalAttention(self.embedding_dim, nhead, dropout=dropout, max_len=n_tokens)
        elif positional_encoding=='rope':
            self.selfattention = rotary_positionalencoding.RotaryPEMultiHeadAttention(heads=nhead,d_model=self.embedding_dim,
                                                                                      rope_percentage=0.5,dropout_prob=dropout)
        elif positional_encoding=='rope2':
            # different implementation for testing and comparison
            self.selfattention = rotary_positionalencoding.RotaryPE2(self.embedding_dim,nhead,n_tokens,causal=False)
        elif positional_encoding=='nope':
            self.selfattention = torch.nn.MultiheadAttention(self.embedding_dim,nhead,dropout,batch_first=True)
        elif positional_encoding=='cnope':
            self.selfattention = CausalSelfAttention(self.embedding_dim,nhead,positional_encoding='cnope')
        elif positional_encoding=='scnope': #sparse
            #attn_mode = 'local'
            #self.selfattention = sparse_block_attn.SparseAttention(nhead, attn_mode, 8, 8)
            self.selfattention = CausalSelfAttention(self.embedding_dim,nhead,positional_encoding='cnope',causal=causal)
        elif positional_encoding=='rndpe':
            self.pe = RandomPositionalEncoding(n_tokens,self.embedding_dim)
            self.selfattention = torch.nn.MultiheadAttention(self.embedding_dim,nhead,dropout,batch_first=True)
        elif positional_encoding=='rnd2':
            self.pe = learnable_pe.LearnablePositionalEncoding(n_tokens,self.embedding_dim,learnable=False)
            self.selfattention = torch.nn.MultiheadAttention(self.embedding_dim,nhead,dropout,batch_first=True)
        elif positional_encoding=='learn':
            self.pe = learnable_pe.LearnablePositionalEncoding(n_tokens,self.embedding_dim,learnable=True,init=pe_init)
            self.selfattention = torch.nn.MultiheadAttention(self.embedding_dim,nhead,dropout,batch_first=True)
        elif positional_encoding=='clearn':
            self.pe = learnable_pe.LearnablePositionalEncoding(n_tokens,self.embedding_dim,learnable=True,init=1.0)
            self.selfattention = CausalSelfAttention(self.embedding_dim,nhead,positional_encoding='cnope',causal=causal)


        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim,self.embedding_dim*4),
            torch.nn.GELU(),
            torch.nn.Linear(self.embedding_dim*4,self.embedding_dim),
            torch.nn.GELU()
        )

        # layer norm; 1st is after attention (embedding dim); 2nd is after RNN 
        self.layernorm0 = torch.nn.LayerNorm(self.embedding_dim)
        self.layernorm1 = torch.nn.LayerNorm(self.embedding_dim)

    def forward(self, embedding, noise=False, dropout=False):
        """
        Run a forward pass of a trial by input_elements matrix
        For each time window, pass each 
        input (Tensor): batch x seq_length x dim_input x time
        """
        device = embedding.device
        #Add noise to inputs
        if noise:
            embedding = embedding + torch.randn(embedding.shape, device=device, dtype=torch.float)/5

        ####
        # transformer block
        if self.positional_encoding in ['relative']:
            attn_outputs = self.selfattention(embedding)
        elif self.positional_encoding in ['rope']:
            attn_outputs = self.selfattention(query=embedding,key=embedding,value=embedding)
        elif self.positional_encoding in ['nope']:
            attn_outputs, attn_out_weights = self.selfattention(embedding, embedding, embedding, need_weights=False)
        elif self.positional_encoding in ['rope2']:
            attn_outputs = self.selfattention(embedding)
        elif self.positional_encoding in ['cnope']:
            attn_outputs, att = self.selfattention(embedding)
        elif self.positional_encoding in ['scnope']:
            #attn_outputs = self.selfattention(embedding,embedding,embedding)
            attn_outputs, l1_reg = self.selfattention(embedding)
            #l1_reg = torch.mean(torch.abs(att))
        elif self.positional_encoding in ['clearn']:
            embedding = self.pe(embedding) # positional encoding
            embedding = self.dropout_embed(embedding)
            attn_outputs, att = self.selfattention(embedding)
        else:
            embedding = self.pe(embedding) # positional encoding
            embedding = self.dropout_embed(embedding)
            attn_outputs, attn_out_weights = self.selfattention(embedding, embedding, embedding, need_weights=False)
        #attn_outputs = self.layernorm0(attn_outputs)
        attn_outputs = self.layernorm0(attn_outputs+embedding) # w resid connection
        transformer_out = self.mlp(attn_outputs)
        #transformer_out = self.layernorm1(transformer_out)
        transformer_out = self.layernorm1(transformer_out+attn_outputs) # w resid connection

        if self.positional_encoding in ['scnope']:
            return transformer_out, l1_reg
        else:
            return transformer_out


class CausalSelfAttention(torch.nn.Module):
    """
    Causal self attention with no positional encoding    
    """

    def __init__(self, n_embd, n_head, positional_encoding='cnope',dropout=0.0,causal='forward'):
        super().__init__()
        assert n_embd % n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = torch.nn.Linear(n_embd, 3 * n_embd, bias=False)
        # output projection
        self.c_proj = torch.nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

        self.n_head = n_head
        self.n_embd = n_embd
        self.pe = positional_encoding
        self.causal = causal


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch size, sequence length, embedding dimensionality (n_embd)
        (B, T, C) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #temperature=1/100 # sparser probabilities
        temperature = 1/100
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        if self.causal=='forward':
            att = att.masked_fill(torch.ones(T, T, dtype=torch.bool).tril(diagonal=0).logical_not().to(x.device),float("-inf"))
        if self.causal=='reverse':
            att = att.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(diagonal=0).logical_not().to(x.device),float("-inf"))
        att_l1 = torch.mean(torch.abs(att))
        att = F.softmax(att/temperature, dim=-1)
        #l1_reg = torch.mean(torch.abs(att))
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        ## efficient attention using Flash Attention CUDA kernels
        #y = F.scaled_dot_product_attention(
        #    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        #)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y, att_l1

    def forward_attn(self, x: torch.Tensor) -> torch.Tensor:
        # batch size, sequence length, embedding dimensionality (n_embd)
        (B, T, C) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #temperature=1/100 # sparser probabilities
        temperature = 1/100
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        if self.causal=='forward':
            att = att.masked_fill(torch.ones(T, T, dtype=torch.bool).tril(diagonal=0).logical_not().to(x.device),float("-inf"))
        if self.causal=='reverse':
            att = att.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(diagonal=0).logical_not().to(x.device),float("-inf"))
        att_l1 = torch.mean(torch.abs(att))
        att = F.softmax(att/temperature, dim=-1)
        #l1_reg = torch.mean(torch.abs(att))
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        ## efficient attention using Flash Attention CUDA kernels
        #y = F.scaled_dot_product_attention(
        #    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        #)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y, att

class RandomPositionalEncoding(torch.nn.Module):
    def __init__(self, max_seq_len, d_model):
        super(RandomPositionalEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.positional_encoding = self._generate_positional_encoding(max_seq_len, d_model)
        
    def forward(self, x):
        # Add positional encoding to input tensor x
        return x + self.positional_encoding[0, :x.size(1), :].to(x.device)
    
    def _generate_positional_encoding(self, max_seq_len, d_model):
        # Generate random positional encoding matrix
        positional_encoding = torch.randn(1,max_seq_len, d_model)
        return positional_encoding
