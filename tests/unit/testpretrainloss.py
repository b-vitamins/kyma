import torch
from torch import nn
from torch.nn import functional as F

from kyma.training.pretrain import tokenlossmap


def test_tokenlossmap_matches_transposed_crossentropy_loss() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 11, requires_grad=True)
    tgt = torch.randint(1, 11, (3, 5))
    tgt[:, -1] = 0
    lossfn = nn.CrossEntropyLoss(ignore_index=0, reduction="none")

    expected = lossfn(logits.transpose(1, 2), tgt)
    actual = tokenlossmap(lossfn, logits, tgt)

    torch.testing.assert_close(actual, expected)


def test_tokenlossmap_matches_transposed_crossentropy_gradients() -> None:
    torch.manual_seed(0)
    tgt = torch.randint(1, 11, (3, 5))
    tgt[:, -1] = 0
    mask = (tgt != 0).to(torch.float32)

    logitsa = torch.randn(3, 5, 11, requires_grad=True)
    logitsb = logitsa.detach().clone().requires_grad_(True)
    lossfn = nn.CrossEntropyLoss(ignore_index=0, reduction="none")

    lossa = lossfn(logitsa.transpose(1, 2), tgt)
    lossa = lossa * mask
    lossa = lossa[lossa != 0.0].mean()
    grada = torch.autograd.grad(lossa, logitsa)[0]

    lossb = tokenlossmap(lossfn, logitsb, tgt)
    lossb = lossb * mask
    lossb = lossb[lossb != 0.0].mean()
    gradb = torch.autograd.grad(lossb, logitsb)[0]

    torch.testing.assert_close(lossb, lossa)
    torch.testing.assert_close(gradb, grada)


def test_flat_crossentropy_matches_transposed_contract() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 11)
    tgt = torch.randint(1, 11, (3, 5))
    tgt[:, -1] = 0

    expected = F.cross_entropy(
        logits.transpose(1, 2),
        tgt,
        ignore_index=0,
    )
    actual = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt.reshape(-1),
        ignore_index=0,
    )

    torch.testing.assert_close(actual, expected)
