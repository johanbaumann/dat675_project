import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



"""
CVAE model for molecule generation.

vocab_size: number of unique characters (atoms) in the dataset
batch_size: number of samples in each training batch
latent_size: dimention of latent vector
lr: learning rate
num_prop: number of properties to condition on (e.g. MW, LogP, TPSA)
    - will be modified to only use a subset of properties in the future
    - currently using all 3 properties (MW, LogP, TPSA)

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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(self.vocab_size, self.latent_size)
        self.encoder = nn.LSTM(
            input_size=self.latent_size + self.num_prop,
            hidden_size=self.unit_size,
            num_layers=self.n_rnn_layer,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=(self.latent_size * 2) + self.num_prop,
            hidden_size=self.unit_size,
            num_layers=self.n_rnn_layer,
            batch_first=True,
        )
        self.out_mean = nn.Linear(self.unit_size, self.latent_size)
        self.out_log_sigma = nn.Linear(self.unit_size, self.latent_size)
        self.output_layer = nn.Linear(self.unit_size, self.vocab_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.to(self.device)
        print('Network Ready')

    @staticmethod
    def _get_arg(args:dict, key:str):
        if isinstance(args, dict):
            return args[key]
        return getattr(args, key)

    def encode(self, x:torch.Tensor, c:torch.Tensor, l:torch.Tensor) -> tuple:
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

        mean = self.out_mean(h_last)
        log_sigma = self.out_log_sigma(h_last)
        eps = (torch.randn_like(mean) * self.stddev) + self.mean
        z = mean + torch.exp(log_sigma / 2.0) * eps
        return z, mean, log_sigma

    def decode(self, x:torch.Tensor, z:torch.Tensor, c:torch.Tensor, initial_state=None) -> tuple:
        x_emb = self.embedding(x)
        z_seq = z.unsqueeze(1).expand(-1, x_emb.size(1), -1)
        c_seq = c.unsqueeze(1).expand(-1, x_emb.size(1), -1)
        decoder_input = torch.cat([z_seq, x_emb, c_seq], dim=-1)
        y, state = self.decoder(decoder_input, initial_state)
        logits = self.output_layer(y)
        probs = F.softmax(logits, dim=-1)
        return probs, logits, state

    def forward(self, x:torch.Tensor, c:torch.Tensor, l:torch.Tensor) -> tuple:
        z, mean, log_sigma = self.encode(x, c, l)
        probs, logits, _ = self.decode(x, z, c)
        return probs, logits, z, mean, log_sigma

    @staticmethod
    def cal_latent_loss(mean:torch.Tensor, log_sigma:torch.Tensor) -> torch.Tensor:
        return torch.mean(-0.5 * (1 + log_sigma - torch.square(mean) - torch.exp(log_sigma)))

    @staticmethod
    def _sequence_loss(logits:torch.Tensor, targets:torch.Tensor, lengths:torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, vocab_size = logits.shape
        token_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
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
        loss = reconstr_loss + latent_loss
        mol_pred = torch.argmax(probs, dim=2)
        return loss, reconstr_loss, latent_loss, mol_pred

    def train_batch(self, x:np.ndarray, y:np.ndarray, l:np.ndarray, c:np.ndarray) -> float:
        self.train(True)
        x_t, y_t, l_t, c_t = self._to_tensor_batch(x, y, l, c)
        self.optimizer.zero_grad()
        loss, _, _, _ = self._compute_losses(x_t, y_t, l_t, c_t)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def test_batch(self, x:np.ndarray, y:np.ndarray, l:np.ndarray, c:np.ndarray) -> float:
        self.train(False)
        x_t, y_t, l_t, c_t = self._to_tensor_batch(x, y, l, c)
        with torch.no_grad():
            loss, _, _, _ = self._compute_losses(x_t, y_t, l_t, c_t)
        return float(loss.item())

    def save(self, ckpt_path:str, global_step:int):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        save_path = f'{ckpt_path}-{global_step}.pt'
        torch.save(
            {
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'global_step': global_step,
            },
            save_path,
        )

    def restore(self, ckpt_path:str):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.load_state_dict(checkpoint)

    def assign_lr(self, learning_rate:float):
        self.lr = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def get_latent_vector(self, x:np.ndarray, c:np.ndarray, l:np.ndarray):
        self.train(False)
        x_t = torch.as_tensor(x, dtype=torch.long, device=self.device)
        c_t = torch.as_tensor(c, dtype=torch.float32, device=self.device)
        l_t = torch.as_tensor(l, dtype=torch.long, device=self.device)
        with torch.no_grad():
            z, _, _ = self.encode(x_t, c_t, l_t)
        return z.detach().cpu().numpy()

    def sample(self, latent_vector:np.ndarray, c:np.ndarray, start_codon:np.ndarray, seq_length:int) -> np.ndarray:
        self.train(False)
        z = torch.as_tensor(latent_vector, dtype=torch.float32, device=self.device) # convert latent vector to tensor
        c_t = torch.as_tensor(c, dtype=torch.float32, device=self.device) # convert properties to tensor
        x = torch.as_tensor(start_codon, dtype=torch.long, device=self.device) # start codon (e.g. 'X') to tensor

        state = None
        preds: list = []
        with torch.no_grad():
            for _ in range(seq_length):
                _, logits, state = self.decode(x, z, c_t, initial_state=state)
                x = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                preds.append(x)

        return torch.cat(preds, dim=1).cpu().numpy().astype(int)
