# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set, Union

import torch

from vllm.sequence import ExecuteModelRequest, PromptLogprobs
from vllm.worker.worker_base import WorkerBase


@dataclass
class SpeculativeProposals:
    """Datastructure used to represent proposal tokens from some proposer. It
    also tracks how many speculative tokens each sequence has.
    """

    # Speculative proposal tokens.
    proposal_token_ids: torch.Tensor

    # Probabilities of the proposal tokens according to the proposer.
    # For optimization, we only keep probabilities for the proposed tokens
    # instead of the full vocabulary distribution.
    proposal_probs: torch.Tensor

    # The valid length of each proposal; can be zero.
    proposal_lens: torch.Tensor

    # A flag to mark that there's no available proposals
    no_proposals: bool = False

    def __repr__(self):
        return (f"SpeculativeProposals("
                f"proposal_token_ids={self.proposal_token_ids}, "
                f"proposal_probs={self.proposal_probs.shape}, "
                f"proposal_lens={self.proposal_lens})")
    
    def to(self,device):
        """Move the proposal tensors to the specified device."""
        self.proposal_token_ids = self.proposal_token_ids.to(device)
        self.proposal_probs = self.proposal_probs.to(device)
        self.proposal_lens = self.proposal_lens.to(device)
        return self


@dataclass
class SpeculativeScores:
    """Datastructure used to represent the scores of speculative tokens
    according to the scoring model.
    """

    # Probabilities of the speculative tokens according to the scoring model.
    probs: torch.Tensor

    # Log-probabilities of the speculative tokens according to the scoring
    # model. These values can be used to generate Logprob objects that are
    # returned to the user.
    logprobs: torch.Tensor

    # Token ids sampled from the scoring model. Used for speculative bonus
    # tokens and also non-speculative normal decoding.
    token_ids: torch.Tensor

    # Optional last hidden states from the scoring model.
    hidden_states: Optional[torch.Tensor] = None

    # Scoring model may also return logprobs for prompt tokens
    # for each request, when chunked prefill is enabled.
    prompt_logprobs: Optional[List[PromptLogprobs]] = None

    def __repr__(self):
        return (f"SpeculativeScores("
                f"probs={self.probs.shape}, "
                f"token_ids={self.token_ids.shape})")
    
    def cpu(self):
        """Move the scores tensors to CPU."""
        self.probs = self.probs.cpu()
        self.logprobs = self.logprobs.cpu()
        self.token_ids = self.token_ids.cpu()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.cpu()
        return self
    
    def npu(self):
        """Move the scores tensors to NPU."""
        self.probs = self.probs.npu()
        self.logprobs = self.logprobs.npu()
        self.token_ids = self.token_ids.npu()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.npu()
        return self


class SpeculativeProposer(ABC):

    @abstractmethod
    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        # If set, this contains all sequence IDs that were assigned
        # bonus tokens in their last forward pass.
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        raise NotImplementedError


class SpeculativeScorer(ABC):

    def __init__(self, scorer_worker: WorkerBase,
                 device: Union[torch.device, str], vocab_size: int):
        self._scorer_worker = scorer_worker
        if isinstance(device, torch.device):
            device = device.type
        self._device = device
        self._vocab_size = vocab_size

    @abstractmethod
    def score_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        proposals: SpeculativeProposals,
    ) -> SpeculativeScores:
        raise NotImplementedError
