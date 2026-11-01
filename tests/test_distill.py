import pytest
import torch
from torch.nn import functional as F

from distill import (
    PAD,
    UNK,
    Student,
    build_vocab,
    distill_loss,
    encode,
    soft_loss,
    student_accuracy,
    train_student,
)


def test_encode_handles_padding_truncation_unknowns_and_empty_text():
    vocab = {"good": 2}
    ids = encode([{"text": "good bad"}, {"text": "123"}, {"text": "good good good"}], vocab, max_length=2)
    assert ids.tolist() == [[2, UNK], [PAD, PAD], [2, 2]]


def test_vocab_keeps_most_common_words():
    vocab = build_vocab([{"text": "a a a b b c"}], size=4)
    assert vocab == {"a": 2, "b": 3}


def test_student_ignores_padding():
    torch.manual_seed(0)
    student = Student(10).eval()
    assert torch.allclose(student(torch.tensor([[2, 3]])), student(torch.tensor([[2, 3, PAD, PAD]])))


def test_student_handles_empty_input():
    assert torch.isfinite(Student(10).eval()(torch.zeros(1, 4, dtype=torch.long))).all()


def test_soft_loss_is_zero_when_student_matches_teacher():
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.2]])
    assert soft_loss(logits, logits, 2.0).item() == pytest.approx(0.0, abs=1e-6)


def test_soft_loss_is_positive_when_they_disagree():
    assert soft_loss(torch.tensor([[2.0, -1.0]]), torch.tensor([[-1.0, 2.0]]), 2.0).item() > 0


def test_temperature_scaling_keeps_gradients_similar_in_size():
    teacher = torch.tensor([[1.0, -1.0]])

    def gradient(temperature):
        student = torch.tensor([[-1.0, 1.0]], requires_grad=True)
        soft_loss(student, teacher, temperature).backward()
        return student.grad.norm().item()

    # Without the temperature-squared factor, this would differ by a factor of 25.
    assert 0.5 < gradient(5.0) / gradient(1.0) < 2.5


def test_alpha_controls_the_mix():
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    teacher = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    labels = torch.tensor([0, 1])
    soft = soft_loss(student, teacher, 2.0)
    hard = F.cross_entropy(student, labels)
    assert distill_loss(student, teacher, labels, 2.0, 1.0).item() == pytest.approx(soft.item())
    assert distill_loss(student, teacher, labels, 2.0, 0.5).item() == pytest.approx(0.5 * soft.item() + 0.5 * hard.item())


def test_unlabeled_rows_only_affect_the_soft_loss():
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    teacher = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    labels = torch.tensor([0, -1])
    soft = soft_loss(student, teacher, 2.0).item()
    hard = F.cross_entropy(student[:1], labels[:1]).item() * 0.5
    assert distill_loss(student, teacher, labels, 2.0, 0.7).item() == pytest.approx(0.7 * soft + 0.3 * hard)


def test_all_unlabeled_batch_uses_soft_loss_without_nan():
    student = torch.tensor([[1.0, 0.0]])
    teacher = torch.tensor([[0.0, 1.0]])
    loss = distill_loss(student, teacher, torch.tensor([-1]), 2.0, 0.7)
    assert loss.item() == pytest.approx(soft_loss(student, teacher, 2.0).item())


def data():
    torch.manual_seed(0)
    # Words 2-5 mean class 0 and words 6-9 mean class 1.
    labels = torch.randint(2, (300,))
    ids = torch.randint(2, 6, (300, 8)) + labels.unsqueeze(1) * 4
    return ids, labels


def test_student_learns_from_hard_labels():
    ids, labels = data()
    model = Student(10)
    train_student(model, ids, labels, ids, labels, epochs=30, seed=0)
    assert student_accuracy(model, ids, labels) > 0.95


def test_student_learns_from_teacher_alone_without_any_labels():
    ids, labels = data()
    teacher = F.one_hot(labels, 2).float() * 8 - 4
    model = Student(10)
    no_labels = torch.full_like(labels, -1)
    train_student(model, ids, no_labels, ids, labels, teacher_logits=teacher, epochs=30, seed=0)
    assert student_accuracy(model, ids, labels) > 0.95
