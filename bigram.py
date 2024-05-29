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
dropout = 0.2
torch.manual_seed(257)


def get_notes():
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))

    with open("data/notes", "wb") as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    pitchnames = sorted(set(item for item in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    return (note_to_int, int_to_note)


def create_midi(prediction_output, name):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ("." in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.SnareDrum()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.SnareDrum()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write("midi", fp=f"{name}.mid")


def helper():
    notes = get_notes()

    n_vocab = len(set(notes))
    print(n_vocab)
    return n_vocab, notes


def get_batch(split, data):
    td = int(0.9 * len(data))
    train_data = data[:td]
    val_data = data[td:]
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


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
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def train(m, data):
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for i in range(max_iter):
        xb, yb = get_batch("train", data)
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(loss.item())


if __name__ == "__main__":
    n_vocab, notes = helper()
    noi, ion = prepare_sequences(notes=notes, n_vocab=n_vocab)
    encode = lambda s: [noi[i] for i in s]
    decode = lambda s: [ion[i] for i in s]
    data = torch.tensor(encode(notes), dtype=torch.long)
    model = embd(n_vocab)
    train(model, data)
    torch.save(model.state_dict(), "music_weights.pth")
    op = model.genrate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_token=500)
    op = decode(op[0].tolist())
    create_midi(op, "song5")
