import unittest
import os
import eval
import torch


class TestEval(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_get_label_log_probs(self):
        b, s, v = 2, 3, 4
        logits = torch.randn(b, s, v)
        labels = torch.tensor(
            [
                [-100, 2, 3],
                [1, -100, 0],
            ]
        )
        expected = torch.zeros(b)
        log_probs = torch.log_softmax(logits, dim=-1)
        for batch_idx in range(b):
            batch_labels = labels[batch_idx]
            batch_log_probs = log_probs[batch_idx]
            mean_log_probs = 0.0
            for seq_idx in range(s - 1):
                label = batch_labels[seq_idx + 1]
                if label == -100:
                    continue
                log_prob = batch_log_probs[seq_idx][label]
                mean_log_probs += log_prob
            expected[batch_idx] = mean_log_probs
        actual = eval.get_label_log_probs(logits, labels)
        if not torch.allclose(expected, actual):
            print(expected)
            print(actual)
        self.assertTrue(torch.allclose(expected, actual))

    def test_are_labels_most_likely(self):
        b, s, v = 2000, 3, 3
        logits = torch.randn(b, s, v)
        labels = []
        for _ in range(b):
            labels += [torch.randint(0, v, (s,))]
        labels = torch.stack(labels)
        labels[:, 0] = -100
        expected = torch.ones(b).bool()
        for batch_idx in range(b):
            batch_labels = labels[batch_idx]
            batch_logits = logits[batch_idx]
            for seq_idx in range(s - 1):
                label = batch_labels[seq_idx + 1]
                if label == -100:
                    continue
                logit = batch_logits[seq_idx][label]
                if not torch.allclose(logit, torch.max(batch_logits[seq_idx])):
                    expected[batch_idx] = False
                    break
        actual = eval.are_labels_most_likely(logits, labels)
        self.assertTrue(torch.all(expected == actual))
