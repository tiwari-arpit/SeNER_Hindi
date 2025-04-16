import torch
from torch import nn
from transformers import AutoModel, AutoConfig

class ArrowAttention(nn.Module):
    def __init__(self, hidden_size, window_size, num_heads=8):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.cls_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.local_attention = nn.MultiheadAttention(hidden_size, num_heads)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.size()

        cls_token = hidden_states[:, 0:1, :]
        cls_token = cls_token.transpose(0, 1)
        global_output, _ = self.cls_attention(
            cls_token,
            hidden_states.transpose(0, 1),
            hidden_states.transpose(0, 1)
        )
        global_output = global_output.transpose(0, 1)

        if seq_len > 1:
            local_outputs = []
            for i in range(1, seq_len):
                start = max(1, i - self.window_size)
                end = min(seq_len, i + self.window_size + 1)

                window = hidden_states[:, start:end, :]
                current_token = hidden_states[:, i:i+1, :]

                current_token = current_token.transpose(0, 1)
                window = window.transpose(0, 1)

                local_output, _ = self.local_attention(
                    current_token,
                    window,
                    window
                )
                local_output = local_output.transpose(0, 1)
                local_outputs.append(local_output)

            if local_outputs:
                local_output = torch.cat(local_outputs, dim=1)
                output = torch.cat([global_output, local_output], dim=1)
            else:
                output = global_output
        else:
            output = global_output

        return output

class LogNScaling(nn.Module):
    def __init__(self, hidden_size):
        super(LogNScaling, self).__init__()
        self.scale = nn.Parameter(torch.tensor(1.0 / (hidden_size ** 0.5)))

    def forward(self, attention_scores):
        return attention_scores * self.scale

class BiSPA(nn.Module):
    def __init__(self, hidden_size, window_size, num_heads=8):
        super(BiSPA, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.horizontal_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.vertical_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.size()
        hidden_states_t = hidden_states.transpose(0, 1)

        # Horizontal attention
        horizontal_outputs = []
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            horizontal_input = hidden_states[start:end, :, :]
            query = hidden_states_t[i:i+1, :, :]

            horizontal_output, _ = self.horizontal_attention(
                query,
                horizontal_input,
                horizontal_input
            )
            horizontal_outputs.append(horizontal_output)

        horizontal_output = torch.cat(horizontal_outputs, dim=0)

        # Vertical attention
        vertical_outputs = []
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            vertical_input = hidden_states_t[start:end, :, :]
            query = hidden_states_t[i:i+1, :, :]
            vertical_output, _ = self.vertical_attention(
                query,
                vertical_input,
                vertical_input
            )
            vertical_outputs.append(vertical_output)

        vertical_output = torch.cat(vertical_outputs, dim=0)

        # Combine outputs
        combined_output = torch.cat([horizontal_output, vertical_output], dim=-1)
        output = self.mlp(combined_output)
        output = output.transpose(0, 1)

        return output

class SeNER(nn.Module):
    def __init__(self, model_name, num_labels, window_size=128):
        super(SeNER, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name, config=self.config)
        self.num_labels = num_labels
        self.arrow_attention = ArrowAttention(self.config.hidden_size, window_size)
        self.log_n_scaling = LogNScaling(self.config.hidden_size)
        self.bispa = BiSPA(self.config.hidden_size, window_size)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(self.config.hidden_size_dropout if hasattr(self.config, 'hidden_size_dropout') else 0.1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        attended_output = self.arrow_attention(sequence_output)

        if attended_output.size(1) > 0:
            cls_token = attended_output[:, 0:1, :]
            scaled_cls = self.log_n_scaling(cls_token)

            if attended_output.size(1) > 1:
                attended_output = torch.cat([scaled_cls, attended_output[:, 1:, :]], dim=1)
            else:
                attended_output = scaled_cls

        sequence_output = self.bispa(attended_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}
