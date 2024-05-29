import torch
import torch.nn as nn
from torch.nn import functional as F
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord, stream

block_size = 256
batch_size = 64
max_iter = 1
n_embd = 256
head_size = 16
num_heads = 4
dropout = 0


def create_midi(prediction_output, name):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ("." in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write("midi", fp=f"{name}.mid")


class multiheads(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.drop(x)
        return x


class block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        heads_size = n_embd // num_heads
        self.sa = multiheads(num_heads=num_heads, head_size=heads_size)
        self.net = feedforward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.net(self.ln2(x))
        return x


class head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("trill", torch.tril(torch.ones(block_size, block_size)))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        temp = q @ k.transpose(-2, -1) * C**-0.5
        temp = temp.masked_fill(self.trill[:T, :T] == 0, float("-inf"))
        temp = F.softmax(temp, dim=-1)
        temp = self.drop(temp)
        v = self.value(x)
        out = temp @ v
        return out


class feedforward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class embd(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.position_emb = nn.Embedding(block_size, n_embd)
        self.block = nn.Sequential(
            block(n_embd, num_heads=4),
            block(n_embd, num_heads=4),
            block(n_embd, num_heads=4),
            block(n_embd, num_heads=4),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targate=None):
        B, T = idx.shape
        token_embd = self.token_emb(idx)
        pos_embd = self.position_emb(torch.arange(T))
        x = token_embd + pos_embd
        x = self.block(x)
        x = self.ln1(x)
        logits = self.lm_head(x)
        if targate == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targate = targate.view(B * T)
            loss = F.cross_entropy(logits, targate)
        return logits, loss

    def genrate(self, idx, max_new_token):
        for _ in range(max_new_token):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logit = logits[:, -1, :]
            probs = F.softmax(logit, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=2)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def get_notes():
    with open("data/notes", "rb") as filepath:
        notes = pickle.load(filepath)
    n_vocab = len(set(notes))
    return n_vocab, notes


def prepare_sequences(notes, n_vocab):
    pitchnames = sorted(set(item for item in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    return (note_to_int, int_to_note)


if __name__ == "__main__":
    n_vocab, notes = get_notes()
    noi, ion = prepare_sequences(notes, n_vocab)
    encode = lambda s: [noi[i] for i in s]
    decode = lambda s: [ion[i] for i in s]
    model = embd(n_vocab)
    model.load_state_dict(torch.load("music_weights.pth"))
    op = model.genrate(idx=torch.tensor([[300]]), max_new_token=500)
    op = decode(op[0].tolist())
    create_midi(op, "song7")
