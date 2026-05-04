## @file gcg_attack.py
#  @brief Greedy Coordinate Gradient (GCG) attack implementation (derived from llm-attacks)
#
#  This file contains an implementation of the Greedy Coordinate Gradient (GCG)
#  jailbreak / adversarial-suffix attack. GCG optimizes a discrete "control string"
#  (token sequence) by using gradient information to propose candidate token
#  replacements and selecting the best candidate greedily at each step.
#
#  The implementation includes:
#   - token_gradients(): one-hot embedding trick to compute gradients w.r.t.
#     control-token coordinates
#   - GCGAttackPrompt: AttackPrompt specialization that exposes grad()
#   - GCGPromptManager: gradient-guided sampling of new control candidates
#   - GCGMultiPromptAttack: step() routine aggregating gradients across workers
#     and searching over candidate controls using target/control losses
#
#  IMPORTANT (Attribution):
#   - This file is based on the official llm-attacks implementation of GCG.
#   - Original authorship and the core algorithmic design belong to the
#     llm-attacks contributors and the associated paper authors (see below).
#
#  Local modifications in this version:
#   - Added validity checks for control token IDs (e.g., filtering negative IDs
#     and out-of-vocabulary indices) to prevent scatter_/one-hot failures.
#   - Computes gradients only for valid tokens and stitches them back into a
#     full gradient tensor aligned with the original control slice.
#   - Minor integration changes to use the local attack_manager helpers.
#
#  @author Bc. Petr Kaška (adaptation/integration)
#  @date 1.2.2026
#
#  Source (upstream):
#   - Repository: llm-attacks / llm-attacks
#   - GCG implementation (conceptually corresponding to this module)
#   - https://github.com/llm-attacks/llm-attacks/blob/main/llm_attacks/gcg/gcg_attack.py
#    - Date accessed: 15.1.2024
#
#  Research basis:
#   - GCG / adversarial suffix framework as described in:
#       "Universal and Transferable Adversarial Attacks on Aligned Language Models"
#       arXiv:2307.15043v2
#       Authors: Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr,
#                J. Zico Kolter, Matt Fredrikson
#       https://arxiv.org/abs/2307.15043v2
#

import gc

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from attacks._18_Gcg.helpers.attack_manager import AttackPrompt, MultiPromptAttack, PromptManager
from attacks._18_Gcg.helpers.attack_manager import get_embedding_matrix, get_embeddings


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):
    """
    Computes gradients of the loss with respect to the coordinates.
    """
    embed_weights = get_embedding_matrix(model)
    vocab_size = embed_weights.shape[0]

    # Získání tokenů z úseku určeného pro útok
    attack_token_ids = input_ids[input_slice]
    
    # Vytvoření masky pro platné tokeny (ID musí být >= 0 a < vocab_size)
    valid_mask = (attack_token_ids >= 0) & (attack_token_ids < vocab_size)
    valid_token_ids = attack_token_ids[valid_mask]

    # Vytvoření one-hot tenzoru POUZE pro platné tokeny
    one_hot = torch.zeros(
        valid_token_ids.shape[0],
        vocab_size,
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        valid_token_ids.unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    
    input_embeds_slice = (one_hot @ embed_weights).unsqueeze(0)
    
    # Vytvoření finálního embedding tenzoru
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:, :input_slice.start, :],
            input_embeds_slice,
            embeds[:, input_slice.stop:, :]
        ],
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)
    
    loss.backward()

    # Vytvoření prázdného tenzoru pro gradienty
    full_grads = torch.zeros_like(attack_token_ids, dtype=embed_weights.dtype)
    
    # Přiřazení gradientů na správná místa v tenzoru
    full_grads[valid_mask] = one_hot.grad.clone().detach().squeeze(0)

    return full_grads

class GCGAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def grad(self, model):
        return token_gradients(
            model, 
            self.input_ids.to(model.device), 
            self._control_slice, 
            self._target_slice, 
            self._loss_slice
        )

class GCGPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True):
        
        # 1. Filtrace neplatných ID tokenů z non_ascii_toks, pokud existují
        if not allow_non_ascii and hasattr(self, '_nonascii_toks'):
            # Vytvoření masky pro platné indexy
            vocab_size = grad.shape[1]
            valid_nonascii_toks = self._nonascii_toks[(self._nonascii_toks < vocab_size) & (self._nonascii_toks >= 0)]
            
            # Nastavení gradientů na nekonečno, pouze pro platné indexy
            grad[:, valid_nonascii_toks.to(grad.device)] = np.inf

        # 2. Získání top K indexů s nejvyššími gradienty
        top_indices = (-grad).topk(topk, dim=1).indices
        control_toks = self.control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)

        # 3. Náhodný výběr nového tokenu z top K
        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)

        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1), device=grad.device)
        )
        
        # 4. Nahrazení starých tokenů novými
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks


class GCGMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def step(self, 
             batch_size=1024, 
             topk=256, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1, 
             control_weight=0.1, 
             verbose=False, 
             opt_only=False,
             filter_cand=True):

        
        # GCG currently does not support optimization_only mode, 
        # so opt_only does not change the inner loop.
        opt_only = False

        main_device = self.models[0].device
        control_cands = []

        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "grad", worker.model)

        # Aggregate gradients
        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:
                with torch.no_grad():
                    control_cand = self.prompts[j-1].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
                grad = new_grad
            else:
                grad += new_grad

        with torch.no_grad():
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
        del grad, control_cand ; gc.collect()
        
        # Search
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        with torch.no_grad():
            for j, cand in enumerate(control_cands):
                # Looping through the prompts at this level is less elegant, but
                # we can manage VRAM better this way
                progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
                for i in progress:
                    for k, worker in enumerate(self.workers):
                        worker(self.prompts[k][i], "logits", worker.model, cand, return_ids=True)
                    logits, ids = zip(*[worker.results.get() for worker in self.workers])
                    loss[j*batch_size:(j+1)*batch_size] += sum([
                        target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                        for k, (logit, id) in enumerate(zip(logits, ids))
                    ])
                    if control_weight != 0:
                        loss[j*batch_size:(j+1)*batch_size] += sum([
                            control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                    del logits, ids ; gc.collect()
                    
                    if verbose:
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")

            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_cands[model_idx][batch_idx], loss[min_idx]
        
        del control_cands, loss ; gc.collect()

        print('Current length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print(next_control)

        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)
