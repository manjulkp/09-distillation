"""Knowledge distillation: train a tiny student to imitate a large teacher.

The teacher's soft probabilities say more than a hard label. "80% positive" tells
the student that a review is mixed, where the label only says "positive". The
teacher can also label extra reviews that have no human labels.
"""

import re
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F

WORD = re.compile(r"[a-z']+")
PAD = 0
UNK = 1


def tokenize(text):
    return WORD.findall(text.lower())


def build_vocab(rows, size=10000):
    counts = Counter(word for row in rows for word in tokenize(row["text"]))
    return {word: i + 2 for i, (word, _) in enumerate(counts.most_common(size - 2))}


def encode(rows, vocab, max_length=128):
    ids = torch.full((len(rows), max_length), PAD, dtype=torch.long)
    for i, row in enumerate(rows):
        tokens = [vocab.get(word, UNK) for word in tokenize(row["text"])][:max_length]
        if tokens:
            ids[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return ids


class Student(nn.Module):
    """Average the word embeddings, then classify. There is no attention at all."""

    def __init__(self, vocab_size, n_embd=32, hidden=32, n_classes=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd, padding_idx=PAD)
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_embd, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, ids):
        mask = (ids != PAD).unsqueeze(-1).float()
        pooled = (self.embedding(ids) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.net(pooled)


def soft_loss(student_logits, teacher_logits, temperature):
    """KL divergence between softened distributions.

    Higher temperature reveals more of the teacher's preferences between classes.
    Softening shrinks gradients by temperature squared, so the loss is multiplied
    by that amount to keep its scale steady.
    """
    student = F.log_softmax(student_logits / temperature, dim=-1)
    teacher = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean") * temperature**2


def distill_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """Mix the soft loss and hard-label loss. `alpha` is the weight of the soft loss.

    Rows labeled -1 have no human label, so they only contribute to the soft loss.
    """
    soft = soft_loss(student_logits, teacher_logits, temperature)
    labeled = labels >= 0
    if alpha >= 1 or not labeled.any():
        return soft if alpha > 0 else torch.zeros(())
    hard = F.cross_entropy(student_logits[labeled], labels[labeled])
    # Unlabeled rows don't contribute to the hard term, so scale it by their share.
    hard = hard * labeled.float().mean()
    return alpha * soft + (1 - alpha) * hard


@torch.no_grad()
def student_accuracy(model, ids, labels, batch_size=512):
    model.eval()
    predictions = torch.cat([model(ids[i : i + batch_size]).argmax(-1) for i in range(0, len(ids), batch_size)])
    return (predictions == labels).float().mean().item()


def train_student(
    model, ids, labels, val_ids, val_labels, teacher_logits=None,
    temperature=2.0, alpha=0.7, epochs=20, batch_size=64, lr=3e-3, seed=0,
):  # fmt: skip
    """Train using hard labels when teacher_logits is None, otherwise distill.

    Returns the best validation accuracy. The model keeps the weights from that epoch.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_val, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(ids), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            logits = model(ids[batch])
            if teacher_logits is None:
                loss = F.cross_entropy(logits, labels[batch])
            else:
                loss = distill_loss(logits, teacher_logits[batch], labels[batch], temperature, alpha)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_acc = student_accuracy(model, val_ids, val_labels)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_val
