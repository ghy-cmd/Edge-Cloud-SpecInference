# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import weakref
from typing import Dict, List, Set, Tuple

import torch

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform
from vllm.sequence import (ExecuteModelRequest, HiddenStates, SequenceData,
                           SequenceGroupMetadata)

if current_platform.is_cuda_alike():
    from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner

from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeProposer)
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.worker.worker_base import DelegateWorkerBase

class RemoteDraftWorker(DelegateWorkerBase):
    def __init__(self, *args, **kwargs):
        DelegateWorkerBase.__init__(self, *args, **kwargs)
        self._proposer: SpeculativeProposer