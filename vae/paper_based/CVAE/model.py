import os
import inspect
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional



"""
CVAE model for molecule generation.

vocab_size: number of unique characters (atoms) in the dataset
batch_size: number of samples in each training batch
latent_size: dimention of latent vector
lr: learning rate
num_prop: number of properties to condition on (e.g. MW, LogP, TPSA)
    - supports any subset/order based on the property file used for training

NOTE: prior z  
z = N(mean, stddev)
stddev: standard deviation for sampling latent vector (1.0 in the original paper)
mean: mean for sampling latent vector (0.0 in the original paper)

unit_size: number of units in each RNN layer
n_rnn_layer: number of RNN layers in encoder and decoder

device - use GPU if available, otherwise use CPU
embedding - embedding layer to convert input characters to dense vectors




encoder - LSTM encoder that takes embedded input and properties, outputs latent vector
decoder - LSTM decoder that takes latent vector, properties, and previous output, outputs next character

the decoder is applied iteratively to generate a sequence of caracthers.
until it generates a 'E' EOS character or reaches maximum length (120 in the orig paper)

"""

"""
Positional encoding for transformer model.

Has:
d_model: internal Transformer width (unit_size)
dropout: dropout rate for the positional encoding
max_len: maximum length of the input sequence (120 in the original paper)


"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        # pe is calculated using the formula:
        # PE(pos, 2i) = sin(pos / (10000^(2i/d_model)))
        pe = torch.zeros(max_len, d_model) # shape (max_len, d_model)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CVAE(nn.Module):
    def __init__(self, vocab_size: int, args: dict):
        super().__init__()
        self.vocab_size = vocab_size
        self.batch_size = self._get_arg(args, 'batch_size')
        self.latent_size = self._get_arg(args, 'latent_size')
        self.lr = self._get_arg(args, 'lr')
        self.num_prop = self._get_arg(args, 'num_prop')
        self.stddev = self._get_arg(args, 'stddev')
        self.mean = self._get_arg(args, 'mean')
        self.unit_size = self._get_arg(args, 'unit_size')
        self.n_rnn_layer = self._get_arg(args, 'n_rnn_layer')
        self.model_mode = self._get_arg_or_default(args, 'model_mode', 'lstm').lower()
        self.optimizer_name = self._get_arg_or_default(args, 'optimizer', 'adam').lower()
        self.weight_decay = float(self._get_arg_or_default(args, 'weight_decay', 0.0))
        self.use_amp = bool(self._get_arg_or_default(args, 'use_amp', True))
        self.amp_dtype_name = str(self._get_arg_or_default(args, 'amp_dtype', 'float16')).lower()
        self.grad_clip_norm = float(self._get_arg_or_default(args, 'grad_clip_norm', 1.0))
        self.transformer_heads = self._get_arg_or_default(args, 'transformer_heads', 8)
        self.transformer_ff_size = self._get_arg_or_default(args, 'transformer_ff_size', self.unit_size * 4)
        self.transformer_dropout = self._get_arg_or_default(args, 'transformer_dropout', 0.1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.amp_dtype_name not in ('float16', 'bfloat16'):
            raise ValueError("amp_dtype must be either 'float16' or 'bfloat16'.")
        self.amp_dtype = torch.float16 if self.amp_dtype_name == 'float16' else torch.bfloat16
        self.amp_enabled = bool(self.use_amp and self.device.type == 'cuda')
        self.use_grad_scaler = bool(self.amp_enabled and self.amp_dtype == torch.float16)

        # embedding layer to convert input characters to dense vectors
        # embedes using 
        #NOTE: Embedding and encoder and decoder. 

        if self.model_mode == 'lstm':
            self.embedding = nn.Embedding(self.vocab_size, self.latent_size)

            self.encoder = nn.LSTM(
                input_size=self.latent_size + self.num_prop,
                hidden_size=self.unit_size,
                num_layers=self.n_rnn_layer,
                batch_first=True,
            )

            # 2*latent_size because we concatenate mean and stddev to the latent vector for the decoder input
            self.decoder = nn.LSTM(
                input_size=(self.latent_size * 2) + self.num_prop,
                hidden_size=self.unit_size,
                num_layers=self.n_rnn_layer,
                batch_first=True,
            )
        elif self.model_mode == 'transformer':
            # Transformer token embeddings use latent_size.
            # Internal attention/FFN width (d_model) remains unit_size.
            self.embedding = nn.Embedding(self.vocab_size, self.latent_size)
            self.positional_encoding = PositionalEncoding(self.unit_size, dropout=float(self.transformer_dropout))

            self.encoder_input_proj = nn.Linear(self.latent_size + self.num_prop, self.unit_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.unit_size,
                nhead=int(self.transformer_heads),
                dim_feedforward=int(self.transformer_ff_size),
                dropout=float(self.transformer_dropout),
                batch_first=True,
                activation='gelu',
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_rnn_layer)

            self.decoder_input_proj = nn.Linear((self.latent_size * 2) + self.num_prop, self.unit_size)
            self.memory_proj = nn.Linear(self.latent_size + self.num_prop, self.unit_size)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.unit_size,
                nhead=int(self.transformer_heads),
                dim_feedforward=int(self.transformer_ff_size),
                dropout=float(self.transformer_dropout),
                batch_first=True,
                activation='gelu',
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.n_rnn_layer)
        else:
            raise ValueError(f"Unsupported model_mode='{self.model_mode}'. Use 'lstm' or 'transformer'.")

        self.out_mean = nn.Linear(self.unit_size, self.latent_size) # output mean of the latent distribution
        self.out_log_sigma = nn.Linear(self.unit_size, self.latent_size) # output log of variance of the latent distribution (log(sigma^2))
        self.output_layer = nn.Linear(self.unit_size, self.vocab_size) # output layer to predict the next character in the sequence

        if self.optimizer_name == 'adamw':
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer='{self.optimizer_name}'. Use 'adam' or 'adamw'.")

        # Prefer the newer torch.amp API when available (PyTorch 2.x), but keep
        # compatibility with older versions and type stubs.
        amp_mod = getattr(torch, 'amp', None)
        grad_scaler_cls = getattr(amp_mod, 'GradScaler', None) if amp_mod is not None else None
        if grad_scaler_cls is not None:
            self.grad_scaler = grad_scaler_cls('cuda', enabled=self.use_grad_scaler)
        else:
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)

        self.to(self.device)
        amp_status = f"enabled dtype={self.amp_dtype_name}" if self.amp_enabled else "disabled"
        print(f'Network Ready ({self.model_mode}, amp={amp_status})')

    def _autocast_context(self):
        if self.amp_enabled:
            return torch.autocast(device_type='cuda', dtype=self.amp_dtype)
        return contextlib.nullcontext()

    @staticmethod
    def _get_arg(args:dict, key:str):
        if isinstance(args, dict):
            return args[key]
        return getattr(args, key)

    @staticmethod
    def _get_arg_or_default(args:dict, key:str, default):
        if isinstance(args, dict):
            return args.get(key, default)
        return getattr(args, key, default)

    @staticmethod
    def _build_padding_mask(lengths: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
        steps = torch.arange(seq_len, device=device).unsqueeze(0)
        return steps >= lengths.unsqueeze(1)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        # Bool mask keeps mask dtype consistent with key padding masks.
        # True means the position is masked.
        return torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)

    def encode(self, x:torch.Tensor, c:torch.Tensor, l:torch.Tensor) -> tuple:
        if self.model_mode == 'lstm':
            x_emb = self.embedding(x)
            c_seq = c.unsqueeze(1).expand(-1, x_emb.size(1), -1)
            encoder_input = torch.cat([x_emb, c_seq], dim=-1)

            packed = nn.utils.rnn.pack_padded_sequence(
                encoder_input,
                lengths=l.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self.encoder(packed)
            h_last = h_n[-1]
        else:
            x_emb = self.embedding(x)
            c_seq = c.unsqueeze(1).expand(-1, x_emb.size(1), -1)
            encoder_input = torch.cat([x_emb, c_seq], dim=-1)
            encoder_input = self.encoder_input_proj(encoder_input)
            encoder_input = self.positional_encoding(encoder_input)

            src_padding_mask = self._build_padding_mask(l, x_emb.size(1), x.device)
            h_seq = self.encoder(encoder_input, src_key_padding_mask=src_padding_mask)

            last_indices = (l - 1).clamp(min=0)
            h_last = h_seq[torch.arange(h_seq.size(0), device=h_seq.device), last_indices]

        mean = self.out_mean(h_last)
        log_sigma = torch.clamp(self.out_log_sigma(h_last), min=-20.0, max=20.0)
        eps = (torch.randn_like(mean) * self.stddev) + self.mean
        z = mean + torch.exp(log_sigma / 2.0) * eps
        return z, mean, log_sigma

    def decode(self, x:torch.Tensor, z:torch.Tensor, c:torch.Tensor, initial_state=None, lengths:Optional[torch.Tensor]=None) -> tuple:
        if self.model_mode == 'lstm':
            x_emb = self.embedding(x)
            z_seq = z.unsqueeze(1).expand(-1, x_emb.size(1), -1)
            c_seq = c.unsqueeze(1).expand(-1, x_emb.size(1), -1)
            decoder_input = torch.cat([z_seq, x_emb, c_seq], dim=-1)
            y, state = self.decoder(decoder_input, initial_state)
        else:
            x_emb = self.embedding(x)
            z_seq = z.unsqueeze(1).expand(-1, x_emb.size(1), -1)
            c_seq = c.unsqueeze(1).expand(-1, x_emb.size(1), -1)
            decoder_input = torch.cat([x_emb, z_seq, c_seq], dim=-1)
            decoder_input = self.decoder_input_proj(decoder_input)
            decoder_input = self.positional_encoding(decoder_input)

            seq_len = x_emb.size(1)
            tgt_padding_mask = None
            if lengths is not None:
                tgt_padding_mask = self._build_padding_mask(lengths, seq_len, x.device)

            memory = self.memory_proj(torch.cat([z, c], dim=-1)).unsqueeze(1)
            y = self.decoder(
                tgt=decoder_input,
                memory=memory,
                tgt_mask=self._causal_mask(seq_len, x.device),
                tgt_key_padding_mask=tgt_padding_mask,
            )
            state = None

        logits = self.output_layer(y)
        # Softmax is not needed for the CE loss (we use logits). When it is used (argmax/sampling),
        # computing it in fp32 avoids fp16 overflow -> NaNs.
        probs = torch.softmax(logits.float(), dim=-1)
        return probs, logits, state

    def forward(self, x:torch.Tensor, c:torch.Tensor, l:torch.Tensor) -> tuple:
        z, mean, log_sigma = self.encode(x, c, l)
        probs, logits, _ = self.decode(x, z, c, lengths=l)
        return probs, logits, z, mean, log_sigma

    @staticmethod
    def cal_latent_loss(mean:torch.Tensor, log_sigma:torch.Tensor) -> torch.Tensor:
        # calculated KL divergence between the latent distribution and the prior distribution (N(mean, stddev))
        # KL div: 0.5 * sum(1 + log(sigma^2) - mean^2 - sigma^2)
        mean_f = mean.float()
        log_sigma_f = torch.clamp(log_sigma.float(), min=-20.0, max=20.0)
        return torch.mean(-0.5 * (1 + log_sigma_f - torch.square(mean_f) - torch.exp(log_sigma_f)))

    def _clip_gradients(self) -> None:
        if self.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_norm)

    @staticmethod
    def _sequence_loss(logits:torch.Tensor, targets:torch.Tensor, lengths:torch.Tensor) -> torch.Tensor:
        """
        Method to cal reconstruction loss for a batch of sequences with different Lengths.
        loss is calculated using cross entropy Loss for each token 

        """

        batch_size, seq_length, vocab_size = logits.shape
        # Compute CE in fp32 for numerical stability under AMP.
        logits_f = logits.float()
        token_loss = F.cross_entropy(
            logits_f.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction='none',
        ).reshape(batch_size, seq_length)
        steps = torch.arange(seq_length, device=logits.device).unsqueeze(0)
        mask = (steps < lengths.unsqueeze(1)).float()
        return (token_loss * mask).sum() / mask.sum().clamp(min=1.0)

    def _to_tensor_batch(self, x:np.ndarray, y:np.ndarray, l:np.ndarray, c:np.ndarray) -> tuple:
        x_t = torch.as_tensor(x, dtype=torch.long, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        l_t = torch.as_tensor(l, dtype=torch.long, device=self.device)
        c_t = torch.as_tensor(c, dtype=torch.float32, device=self.device)
        return x_t, y_t, l_t, c_t

    def _compute_losses(self, x:torch.Tensor, y:torch.Tensor, l:torch.Tensor, c:torch.Tensor) -> tuple:
        probs, logits, _, mean, log_sigma = self.forward(x, c, l)
        reconstr_loss = self._sequence_loss(logits, y, l)
        latent_loss = self.cal_latent_loss(mean, log_sigma)
        # NOTE: Elbo loss = reconstruction loss + KL divergence loss
        loss = reconstr_loss + latent_loss
        mol_pred = torch.argmax(probs, dim=2)
        return loss, reconstr_loss, latent_loss, mol_pred

    def train_batch(self, x:np.ndarray, y:np.ndarray, l:np.ndarray, c:np.ndarray) -> float:
        self.train(True)
        x_t, y_t, l_t, c_t = self._to_tensor_batch(x, y, l, c)
        self.optimizer.zero_grad()
        with self._autocast_context():
            loss, _, _, _ = self._compute_losses(x_t, y_t, l_t, c_t)
        if self.use_grad_scaler:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self._clip_gradients()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self._clip_gradients()
            self.optimizer.step()
        return float(loss.item())

    def test_batch(self, x:np.ndarray, y:np.ndarray, l:np.ndarray, c:np.ndarray) -> float:
        self.train(False)
        x_t, y_t, l_t, c_t = self._to_tensor_batch(x, y, l, c)
        with torch.no_grad():
            with self._autocast_context():
                loss, _, _, _ = self._compute_losses(x_t, y_t, l_t, c_t)
        return float(loss.item())

    def save(self, ckpt_path:str, global_step:int, model_config:Optional[dict]=None):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        save_path = f'{ckpt_path}-{global_step}.pt'
        torch.save(
            {
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'grad_scaler_state_dict': self.grad_scaler.state_dict(),
                'global_step': global_step,
                'model_config': model_config,
            },
            save_path,
        )

    def restore(self, ckpt_path:str):
        # Newer PyTorch supports weights_only; set it explicitly to avoid warning-prone implicit behavior.
        if 'weights_only' in inspect.signature(torch.load).parameters:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        else:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'grad_scaler_state_dict' in checkpoint and self.use_grad_scaler:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
        else:
            self.load_state_dict(checkpoint)

    def assign_lr(self, learning_rate:float):
        self.lr = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def get_latent_vector(self, x:np.ndarray, c:np.ndarray, l:np.ndarray) -> np.ndarray:
        self.train(False)
        x_t = torch.as_tensor(x, dtype=torch.long, device=self.device)
        c_t = torch.as_tensor(c, dtype=torch.float32, device=self.device)
        l_t = torch.as_tensor(l, dtype=torch.long, device=self.device)
        with torch.no_grad():
            z, _, _ = self.encode(x_t, c_t, l_t)
        return z.detach().cpu().numpy()

    def sample(self, latent_vector:np.ndarray, c:np.ndarray, start_codon:np.ndarray, seq_length:int) -> np.ndarray:
        """
        NOTE: important!!!
        Decoder is applied iterativley to generate seq of carachters until it generates an 'E' char or
        reaches maximum length (120 in the original paper)

        latent_vector: numpy array of shape (batch_size, latent_size) sampled from the prior distribution
        c: numpy array of shape (batch_size, num_prop) properties to condition on
        start_codon: numpy array of shape (batch_size, 1) containing the index of the start token (e.g. 'X') in the vocab
        seq_length: maximum length of the generated sequence (120 in the original paper)

        

        """





        self.train(False)

        # `load_data()` builds the vocabulary by appending 'E' then 'X' at the end.
        # So EOS='E' is always vocab_size-2 and SOS='X' is always vocab_size-1.
        e_index = self.vocab_size - 2

        z = torch.as_tensor(latent_vector, dtype=torch.float32, device=self.device)  # latent vector
        c_t = torch.as_tensor(c, dtype=torch.float32, device=self.device)  # properties
        x = torch.as_tensor(start_codon, dtype=torch.long, device=self.device)  # start token indices

        state = None
        preds = []
        finished = torch.zeros(x.shape[0], dtype=torch.bool, device=self.device)
        e_index_t = torch.tensor(e_index, dtype=torch.long, device=self.device)

        with torch.no_grad():
            # iteratively decode until EOS ('E') is generated or reaches maximum length
            for _ in range(seq_length):
                if self.model_mode == 'lstm':
                    _, logits, state = self.decode(x, z, c_t, initial_state=state)
                    next_x = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                else:
                    _, logits, _ = self.decode(x, z, c_t, lengths=None)
                    next_x = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                # Once a sequence hits EOS, keep it at EOS for the remaining steps.
                next_x = torch.where(
                    finished.unsqueeze(1),
                    e_index_t.view(1, 1).expand_as(next_x),
                    next_x,
                )

                preds.append(next_x)
                finished |= next_x.squeeze(1).eq(e_index_t)
                if self.model_mode == 'lstm':
                    x = next_x
                else:
                    x = torch.cat([x, next_x], dim=1)

                if finished.all():
                    break

        return torch.cat(preds, dim=1).cpu().numpy().astype(int)
