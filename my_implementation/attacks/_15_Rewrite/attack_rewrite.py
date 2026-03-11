## @file attack_rewrite.py
#  @brief Simple rewrite-template prompt transformation attack
#
#  This file implements a lightweight rewrite-based attack that transforms an
#  input goal into a rewritten prompt by injecting it into a configurable
#  template. The objective is to preserve the semantic intent of the original
#  request while altering surface form and context, which may help bypass
#  safety filters that are sensitive to specific phrasing patterns.
#
#  Implementation summary:
#   - The attack takes a `goal` string and formats it into `rewrite_template`
#     (expects a "{content}" placeholder).
#   - It returns a minimal chat message list containing a single user message
#     with the rewritten prompt.
#   - A small textual `log` is also returned for experiment traceability.
#
#  @author Bc. Petr Kaška
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This file is an original implementation by Bc. Petr Kaška.
#   - The class design, prompt construction logic, and return format were
#     implemented by the author.
#
#  Research basis:
#   - The idea of rewrite / paraphrase preprocessing as a baseline technique in
#     the context of adversarial/jailbreak prompts is inspired by:
#       "Baseline Defenses for Adversarial Attacks Against Aligned Language Models"
#       arXiv:2309.00614
#       Authors: Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli,
#                John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum,
#                Aniruddha Saha, Jonas Geiping, Tom Goldstein
#       Submitted: 1 Sep 2023; last revised: 4 Sep 2023
#       https://arxiv.org/abs/2309.00614

from typing import List, Dict

class RewriteAttack:

    def __init__(self, rewrite_template: str) -> None:
        self.rewrite_template = rewrite_template

    def generate(self, goal: str) -> tuple[str, List[Dict[str, str]]]:

        attack_prompt = self.rewrite_template.format(content=goal)

        log = f"TASK is '{goal}'"
        prompts = [{"role": "user", "content": attack_prompt}]
        return log, prompts
