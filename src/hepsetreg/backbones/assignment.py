"""
Assign measured jets/leptons to truth roles defined by numerical labels.

A config maps target names to entries. `[source, label, ...]` selects the
measured objects of `source` ("jet" or "lepton") carrying those truth
labels; several labels in one entry are exchangeable, e.g. the hadronic W
daughters `[jet, 3, 4]` (or `[jet, 3, 3]` if both carry label 3). A plain
string reuses another target, so

    W_had: [[jet, 3, 4]]
    top_had: [[jet, 1], W_had]
    top_lep: [[jet, 2]]

defines four jet roles, not seven: the same (source, label) is one role
however many targets contain it.
"""

from itertools import permutations, product

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

SOURCES = ("jet", "lepton")


class AssignmentTargets:
    """Roles, exchangeable groups and per-target role indices of a config."""

    def __init__(self, config):
        self.config = config
        self.roles, self.groups, self.targets = [], [], {}
        for name in config:
            roles = self._expand(name, ())
            self.targets[name] = list(dict.fromkeys(roles))

    def _expand(self, name, parents):
        if name in parents or name not in self.config:
            raise ValueError(f"assignment target {name!r} is undefined or refers to itself")
        roles = []
        for entry in self.config[name]:
            if isinstance(entry, str):
                roles += self._expand(entry, parents + (name,))
                continue
            source, *labels = entry
            if source not in SOURCES or not labels or not all(type(l) is int and l > 0 for l in labels):
                raise ValueError(f"{name}: entry {entry} must be [jet|lepton, positive integer label, ...]")
            keys = [(source, l, labels[:i].count(l)) for i, l in enumerate(labels)]
            for key in keys:
                if key not in self.roles:
                    self.roles.append(key)
            group = tuple(sorted(self.roles.index(key) for key in keys))
            if len(group) > 1 and group not in self.groups:
                if any(set(group) & set(other) for other in self.groups):
                    raise ValueError(f"{name}: a role cannot belong to two different exchangeable groups")
                self.groups.append(group)
            roles += [self.roles.index(key) for key in keys]
        return roles

    def permutations(self):
        """Every role order that is physically equivalent, as [P, n_roles]."""
        orders = []
        for choice in product(*(permutations(group) for group in self.groups)):
            order = list(range(len(self.roles)))
            for group, permuted in zip(self.groups, choice):
                for role, other in zip(group, permuted):
                    order[role] = other
            orders.append(order)
        return orders

    def truth(self, labels):
        """Truth object index per role, [B, n_roles] int64.

        `labels[source]` is (values, mask): integer labels and real-object
        mask, both [B, N]. A negative label marks an object whose truth is
        unknown. Each role gets the index of its object within its own
        collection, -1 if no selected object carries its label, or -2 if
        none does but the source has unknown labels.
        """
        n_events = len(next(iter(labels.values()))[0])
        result = np.full((n_events, len(self.roles)), -1, dtype=np.int64)
        for source, label in dict.fromkeys(role[:2] for role in self.roles):
            roles = [i for i, role in enumerate(self.roles) if role[:2] == (source, label)]
            values, mask = labels[source]
            match = (values == label) & mask
            count = match.sum(1)
            if (count > len(roles)).any():
                row = int(np.argmax(count > len(roles)))
                raise ValueError(f"event row {row}: {count[row]} {source}s carry truth label {label}, "
                                 f"but the assignment config has {len(roles)} such role(s)")
            unknown = ((values < 0) & mask).any(1)
            first = np.argsort(~match, axis=1, kind="stable")
            for k, role in enumerate(roles):
                found = first[:, k] if k < first.shape[1] else 0
                result[:, role] = np.where(count > k, found, np.where(unknown, -2, -1))
        return result


class AssignmentHead(nn.Module):
    """Linear role scores per encoder token, S[r, j] = w_r . h_j + b_r.

    Trained with one softmax per role over its compatible, real objects,
    minimised over exchangeable permutations; roles without truth do not
    contribute. Prediction picks distinct objects maximising the summed
    role log-probabilities (Hungarian matching). The returned probabilities
    are these per-role softmax values, not a joint assignment probability.
    """

    def __init__(self, d_model, targets):
        super().__init__()
        self.targets = targets
        self.score = nn.Linear(d_model, len(targets.roles))
        self.register_buffer("lepton_role", torch.tensor([r[0] == "lepton" for r in targets.roles]), persistent=False)
        self.register_buffer("orders", torch.tensor(targets.permutations()), persistent=False)

    def log_probs(self, tokens, batch):
        """[B, n_roles, N_jet + N_lepton] log-probabilities over jet then
        lepton slots (the last tokens of the encoder); -inf if disallowed."""
        jet_mask, lepton_mask = batch["jet_mask"], batch["lepton_mask"]
        mask = torch.cat((jet_mask, lepton_mask), dim=1)
        is_lepton = torch.arange(mask.shape[1], device=mask.device) >= jet_mask.shape[1]
        allowed = (self.lepton_role[:, None] == is_lepton) & mask[:, None]
        logits = self.score(tokens[:, -mask.shape[1]:]).transpose(1, 2)
        return logits.masked_fill(~allowed, -torch.inf).log_softmax(-1).masked_fill(~allowed, -torch.inf)

    def _slots(self, index, batch):
        return torch.where(index >= 0, index + self.lepton_role * batch["jet_mask"].shape[1], -1)

    def loss(self, tokens, batch):
        truth = batch["assignment"]
        logp = self.log_probs(tokens, batch)
        slots = self._slots(truth, batch)[:, self.orders]  # [B, P, R]
        logp = logp.unsqueeze(1).expand(-1, len(self.orders), -1, -1)
        nll = -logp.gather(3, slots.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        nll = nll.masked_fill(slots < 0, 0.).sum(-1).amin(1)
        return nll.sum() / (truth >= 0).sum().clamp(min=1)

    @torch.no_grad()
    def predict(self, tokens, batch):
        """Per-role object index within its collection ([B, n_roles], -1 if
        no compatible object is left) and that choice's role probability."""
        logp = self.log_probs(tokens, batch)
        cost = (-logp).clamp(max=1e9).cpu().numpy()
        choice = np.full(cost.shape[:2], -1)
        for event, matrix in enumerate(cost):
            roles, slots = linear_sum_assignment(matrix)
            keep = matrix[roles, slots] < 1e9
            choice[event, roles[keep]] = slots[keep]
        choice = torch.as_tensor(choice, device=logp.device)
        prob = logp.gather(2, choice.clamp(min=0).unsqueeze(-1)).squeeze(-1).exp().masked_fill(choice < 0, 0.)
        offset = self.lepton_role * batch["jet_mask"].shape[1]
        return torch.where(choice >= 0, choice - offset, -1), prob
