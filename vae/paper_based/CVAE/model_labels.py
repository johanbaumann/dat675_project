import os
import inspect
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

"""model_labels.py

CVAE model for molecule generation + optional auxiliary label predictor.

This file is a non-breaking extension of the base CVAE:
- The conditioning vector `c` (properties) is unchanged.
- When disabled (default), the model behaves like the original CVAE.
- When enabled, the model also predicts labels from the latent `z`.

Its a combo of three different papers:
Beta-VAE style KL annealing and optional label prediction loss for multi-task learning.
Its a combo of: "Balancing Exploration and Exploitation:
Disentangled β-CVAE in De Novo Drug Design" by Guang et al https://arxiv.org/pdf/2306.01683

--------------------------------------------------

"Molecular generative model based
on conditional variational autoencoder for de
novo molecular design" by Jaechang et al: https://link.springer.com/article/10.1186/s13321-018-0286-7 
--------------------------

and finaly:
The landmark paper of:
"Automatic Chemical Design Using a Data-Driven
Continuous Representation of Molecules" by Gómez-Bombarelli et al
https://arxiv.org/pdf/1610.02415


CHANGELOG
---------
2026-02-23
- Added optional label prediction head `label_head` (reads latent `z`).
- Added config knobs: `predict_labels`, `label_dim`, `label_loss_weight`.
- Added optional `label_loss` to the total loss and metrics.
- Kept sampling pipeline unchanged; forward signature is backward compatible
    when `predict_labels=False`.
"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
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

        # Optional label predictor head (multi-task CVAE).
        # Defaults are chosen to keep older configs/checkpoints working.
        self.predict_labels = bool(self._get_arg_or_default(args, 'predict_labels', False))
        self.label_dim = int(self._get_arg_or_default(args, 'label_dim', 0))
        self.label_loss_weight = float(self._get_arg_or_default(args, 'label_loss_weight', 1.0))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.amp_dtype_name not in ('float16', 'bfloat16'):
            raise ValueError("amp_dtype must be either 'float16' or 'bfloat16'.")
        self.amp_dtype = torch.float16 if self.amp_dtype_name == 'float16' else torch.bfloat16
        self.amp_enabled = bool(self.use_amp and self.device.type == 'cuda')
        self.use_grad_scaler = bool(self.amp_enabled and self.amp_dtype == torch.float16)

        if self.model_mode == 'lstm':
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
        elif self.model_mode == 'transformer':
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

        self.out_mean = nn.Linear(self.unit_size, self.latent_size)
        self.out_log_sigma = nn.Linear(self.unit_size, self.latent_size)
        self.output_layer = nn.Linear(self.unit_size, self.vocab_size)

        # Label prediction head: y_hat = f_label(z)
        # Only instantiated when enabled to preserve checkpoint compatibility.
        if self.predict_labels:
            if self.label_dim <= 0:
                raise ValueError('label_dim must be a positive integer when predict_labels=True')
            self.label_head = nn.Sequential(
                nn.Linear(self.latent_size, self.unit_size),
                nn.ReLU(),
                nn.Linear(self.unit_size, self.label_dim),
            )

        if self.optimizer_name == 'adamw':
            if self.weight_decay > 0:
                self.optimizer = torch.optim.AdamW(self._build_adamw_param_groups(), lr=self.lr)
            else:
                self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.0)
        elif self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer='{self.optimizer_name}'. Use 'adam' or 'adamw'.")

        amp_mod = getattr(torch, 'amp', None)
        grad_scaler_cls = getattr(amp_mod, 'GradScaler', None) if amp_mod is not None else None
        if grad_scaler_cls is not None:
            self.grad_scaler = grad_scaler_cls('cuda', enabled=self.use_grad_scaler)
        else:
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)

        self.to(self.device)
        amp_status = f"enabled dtype={self.amp_dtype_name}" if self.amp_enabled else "disabled"
        print(
            f'Network Ready ({self.model_mode}, amp={amp_status}, '
            f'predict_labels={self.predict_labels}, label_dim={self.label_dim}, '
            f'label_loss_weight={self.label_loss_weight})'
        )

    def _autocast_context(self):
        if self.amp_enabled:
            return torch.autocast(device_type='cuda', dtype=self.amp_dtype)
        return contextlib.nullcontext()

    def _transformer_fp32_context(self):
        if self.amp_enabled:
            return torch.autocast(device_type='cuda', enabled=False)
        return contextlib.nullcontext()

    @staticmethod
    def _get_arg(args: dict, key: str):
        if isinstance(args, dict):
            return args[key]
        return getattr(args, key)

    @staticmethod
    def _get_arg_or_default(args: dict, key: str, default):
        if isinstance(args, dict):
            return args.get(key, default)
        return getattr(args, key, default)

    @staticmethod
    def _build_padding_mask(lengths: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
        steps = torch.arange(seq_len, device=device).unsqueeze(0)
        return steps >= lengths.unsqueeze(1)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)

    @staticmethod
    def _is_no_decay_param(param_name: str, param: torch.nn.Parameter) -> bool:
        lname = param_name.lower()
        if param.ndim == 1:
            return True
        if lname.endswith('bias'):
            return True
        if 'norm' in lname:
            return True
        if 'embedding' in lname:
            return True
        if lname.startswith('out_mean') or lname.startswith('out_log_sigma'):
            return True
        return False

    def _build_adamw_param_groups(self) -> list:
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_no_decay_param(name, param):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

    def encode(self, x: torch.Tensor, c: torch.Tensor, l: torch.Tensor) -> tuple:
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
            with self._transformer_fp32_context():
                h_seq = self.encoder(
                    encoder_input.float(),
                    src_key_padding_mask=src_padding_mask,
                )

            last_indices = (l - 1).clamp(min=0)
            h_last = h_seq[torch.arange(h_seq.size(0), device=h_seq.device), last_indices]

        mean = self.out_mean(h_last)
        log_sigma = torch.clamp(self.out_log_sigma(h_last), min=-20.0, max=20.0)
        eps = (torch.randn_like(mean) * self.stddev) + self.mean
        z = mean + torch.exp(log_sigma / 2.0) * eps

        return z, mean, log_sigma

    def predict_label(self, z: torch.Tensor) -> torch.Tensor:
        """Predict labels from latent vector.

        initial_state is not used for label prediction since it's only based on 'z', which is derived from the encoder output.
        

        Notes:
          - Only valid if 'self.predict_labels' is True.
          - 'z' should have shape (batch, latent_size).
        """
        if not self.predict_labels:
            raise RuntimeError('predict_label() called but predict_labels=False')
        return self.label_head(z)

    def decode(self, x: torch.Tensor, z: torch.Tensor, c: torch.Tensor, initial_state=None, lengths: Optional[torch.Tensor] = None) -> tuple:
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
            memory = self.memory_proj(torch.cat([z, c], dim=-1)).unsqueeze(1)
            tgt_padding_mask = None
            if lengths is not None:
                tgt_padding_mask = self._build_padding_mask(lengths, seq_len, x.device)
            with self._transformer_fp32_context():
                y = self.decoder(
                    tgt=decoder_input.float(),
                    memory=memory.float(),
                    tgt_mask=self._causal_mask(seq_len, x.device),
                    tgt_key_padding_mask=tgt_padding_mask,
                )
            state = None

        logits = self.output_layer(y)
        probs = torch.softmax(logits.float(), dim=-1)
        return probs, logits, state

    def forward(self, x: torch.Tensor, c: torch.Tensor, l: torch.Tensor) -> tuple:
        """Forward pass.

        Backward compatible return:
          - If predict_labels=False: returns (probs, logits, z, mean, log_sigma)
          - If predict_labels=True:  returns (probs, logits, z, mean, log_sigma, y_hat)
        """
        z, mean, log_sigma = self.encode(x, c, l)
        probs, logits, _ = self.decode(x, z, c, lengths=l)

        if self.predict_labels:
            y_hat = self.predict_label(z)
            return probs, logits, z, mean, log_sigma, y_hat
        return probs, logits, z, mean, log_sigma

    @staticmethod
    def cal_latent_loss(mean: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        mean_f = mean.float()
        log_sigma_f = torch.clamp(log_sigma.float(), min=-20.0, max=20.0)
        return torch.mean(-0.5 * (1 + log_sigma_f - torch.square(mean_f) - torch.exp(log_sigma_f)))

    def _clip_gradients(self) -> float:
        if self.grad_clip_norm > 0:
            total_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_norm)
            return float(total_norm)
        return 0.0

    @staticmethod
    def _sequence_loss(logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, vocab_size = logits.shape
        logits_f = logits.float()
        token_loss = F.cross_entropy(
            logits_f.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction='none',
        ).reshape(batch_size, seq_length)
        steps = torch.arange(seq_length, device=logits.device).unsqueeze(0)
        mask = (steps < lengths.unsqueeze(1)).float()
        return (token_loss * mask).sum() / mask.sum().clamp(min=1.0)

    @staticmethod
    def _label_loss(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Label prediction loss (regression): MSE(y_hat, y_true)."""
        return F.mse_loss(y_hat.float(), y_true.float(), reduction='mean')

    def _to_tensor_batch(self, x: np.ndarray, y: np.ndarray, l: np.ndarray, c: np.ndarray, y_label: Optional[np.ndarray] = None) -> tuple:
        x_t = torch.as_tensor(x, dtype=torch.long, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        l_t = torch.as_tensor(l, dtype=torch.long, device=self.device)
        c_t = torch.as_tensor(c, dtype=torch.float32, device=self.device)
        y_label_t = None
        if y_label is not None:
            y_label_t = torch.as_tensor(y_label, dtype=torch.float32, device=self.device)
        return x_t, y_t, l_t, c_t, y_label_t

    def _compute_losses(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        l: torch.Tensor,
        c: torch.Tensor,
        *,
        y_true: Optional[torch.Tensor] = None,
        beta: float = 1.0,
    ) -> tuple:
        out = self.forward(x, c, l)
        if self.predict_labels:
            probs, logits, _, mean, log_sigma, y_hat = out
        else:
            probs, logits, _, mean, log_sigma = out
            y_hat = None

        reconstr_loss = self._sequence_loss(logits, y, l)
        latent_loss = self.cal_latent_loss(mean, log_sigma)

        loss = reconstr_loss + (float(beta) * latent_loss)

        label_loss = torch.tensor(0.0, device=loss.device)
        # label loss computes what label value the model predicts from the latent vector 'z'
        # and compare with true predictor value (e.g LogP) using MSE. 
        if self.predict_labels and (y_true is not None):
            label_loss = self._label_loss(y_hat, y_true)
            loss = loss + (self.label_loss_weight * label_loss)

        mol_pred = torch.argmax(probs, dim=2)
        stats = {
            'recon_loss': float(reconstr_loss.detach().item()),
            'kl_loss': float(latent_loss.detach().item()),
            'label_loss': float(label_loss.detach().item()),
            'mean_abs': float(mean.detach().abs().mean().item()),
            'log_sigma_mean': float(log_sigma.detach().mean().item()),
            'log_sigma_min': float(log_sigma.detach().min().item()),
            'log_sigma_max': float(log_sigma.detach().max().item()),
            'beta': float(beta),
            'label_loss_weight': float(self.label_loss_weight),
        }
        return loss, reconstr_loss, latent_loss, label_loss, mol_pred, stats

    def train_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        *,
        y_label: Optional[np.ndarray] = None,
        beta: float = 1.0,
        return_metrics: bool = False,
    ):
        self.train(True)
        x_t, y_t, l_t, c_t, y_label_t = self._to_tensor_batch(x, y, l, c, y_label=y_label)
        self.optimizer.zero_grad()
        with self._autocast_context():
            loss, _, _, _, _, stats = self._compute_losses(x_t, y_t, l_t, c_t, y_true=y_label_t, beta=beta)
        grad_norm = 0.0
        if self.use_grad_scaler:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = self._clip_gradients()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            grad_norm = self._clip_gradients()
            self.optimizer.step()
        if return_metrics:
            stats['grad_norm'] = float(grad_norm)
            stats['total_loss'] = float(loss.detach().item())
            return stats
        return float(loss.item())

    def test_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        *,
        y_label: Optional[np.ndarray] = None,
        beta: float = 1.0,
        return_metrics: bool = False,
    ):
        self.train(False)
        x_t, y_t, l_t, c_t, y_label_t = self._to_tensor_batch(x, y, l, c, y_label=y_label)
        with torch.no_grad():
            with self._autocast_context():
                loss, _, _, _, _, stats = self._compute_losses(x_t, y_t, l_t, c_t, y_true=y_label_t, beta=beta)
        if return_metrics:
            stats['total_loss'] = float(loss.detach().item())
            return stats
        return float(loss.item())

    def save(self, ckpt_path: str, global_step: int, model_config: Optional[dict] = None):
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

    def restore(self, ckpt_path: str):
        if 'weights_only' in inspect.signature(torch.load).parameters:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        else:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            # Be tolerant to optional heads (label_head) missing/present across checkpoints.
            try:
                self.load_state_dict(checkpoint['model_state_dict'], strict=True)
            except RuntimeError:
                self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'optimizer_state_dict' in checkpoint:
                # Optimizer state is not required for inference/sampling, and can be incompatible
                # when optional heads are enabled/disabled across checkpoints.
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except (ValueError, RuntimeError):
                    pass
            if 'grad_scaler_state_dict' in checkpoint and self.use_grad_scaler:
                try:
                    self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
                except (ValueError, RuntimeError):
                    pass
        else:
            try:
                self.load_state_dict(checkpoint, strict=True)
            except RuntimeError:
                self.load_state_dict(checkpoint, strict=False)

    def assign_lr(self, learning_rate: float):
        self.lr = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def get_latent_vector(self, x: np.ndarray, c: np.ndarray, l: np.ndarray) -> np.ndarray:
        self.train(False)
        x_t = torch.as_tensor(x, dtype=torch.long, device=self.device)
        c_t = torch.as_tensor(c, dtype=torch.float32, device=self.device)
        l_t = torch.as_tensor(l, dtype=torch.long, device=self.device)
        with torch.no_grad():
            z, _, _ = self.encode(x_t, c_t, l_t)
        return z.detach().cpu().numpy()

    def sample(
        self,
        latent_vector: np.ndarray,
        c: np.ndarray,
        start_codon: np.ndarray,
        seq_length: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> np.ndarray:
        self.train(False)
        e_index = self.vocab_size - 2

        z = torch.as_tensor(latent_vector, dtype=torch.float32, device=self.device)
        c_t = torch.as_tensor(c, dtype=torch.float32, device=self.device)
        x = torch.as_tensor(start_codon, dtype=torch.long, device=self.device)

        state = None
        preds = []
        finished = torch.zeros(x.shape[0], dtype=torch.bool, device=self.device)
        e_index_t = torch.tensor(e_index, dtype=torch.long, device=self.device)

        def _sample_next_from_logits(
            logits_last: torch.Tensor,
            *,
            do_sample: bool,
            temperature: float,
            top_k: Optional[int],
        ) -> torch.Tensor:
            if not do_sample:
                return torch.argmax(logits_last, dim=-1, keepdim=True)

            logits_f = logits_last.float()
            temp = float(temperature)
            if temp <= 0:
                raise ValueError('temperature must be > 0')
            if temp != 1.0:
                logits_f = logits_f / temp

            k = int(top_k) if top_k is not None else 0
            if k > 0 and k < logits_f.size(-1):
                top_vals, _ = torch.topk(logits_f, k, dim=-1)
                kth = top_vals[:, -1].unsqueeze(-1)
                logits_f = torch.where(
                    logits_f < kth,
                    torch.full_like(logits_f, -1e9),
                    logits_f,
                )

            probs = torch.softmax(logits_f, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        with torch.no_grad():
            for _ in range(seq_length):
                if self.model_mode == 'lstm':
                    _, logits, state = self.decode(x, z, c_t, initial_state=state)
                    next_x = _sample_next_from_logits(
                        logits[:, -1, :],
                        do_sample=bool(do_sample),
                        temperature=float(temperature),
                        top_k=top_k,
                    )
                else:
                    _, logits, _ = self.decode(x, z, c_t, lengths=None)
                    next_x = _sample_next_from_logits(
                        logits[:, -1, :],
                        do_sample=bool(do_sample),
                        temperature=float(temperature),
                        top_k=top_k,
                    )

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