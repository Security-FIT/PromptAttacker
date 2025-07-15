"""Defense EA (Evasive Alignment) module

A *toy* defense that can be plugged into the pipeline.  It is implemented as a
class so that you can keep state (e.g. statistics, counters, config) while
processing many prompts.

Current behaviour (placeholder):
  • trims leading/trailing whitespace
  • appends a marker "[DEFENSE_EA]" so you can tell the defense was applied

Example
-------
>>> from defense.defense_EA import DefenseEA
>>> defense = DefenseEA()
>>> defense(" some prompt  ")
'some prompt\n[DEFENSE_EA]'
"""

from __future__ import annotations

class DefenseEA:
    """Very lightweight prompt‑rewriting defense (stateful wrapper)."""

    def __init__(self):
        self.num_processed: int = 0 

    
    def __call__(self, prompt: str) -> str:  
        return self.apply(prompt)

    def apply(self, prompt: str) -> str:
        """Return a *defended* prompt (simple placeholder logic)."""
        self.num_processed += 1
        prompt = prompt.strip()
        prompt += "\n[DEFENSE_EA][DEFENSE_EA][DEFENSE_EA][DEFENSE_EA][DEFENSE_EA][DEFENSE_EA][DEFENSE_EA]"
        return prompt

    def summary(self) -> str:
        return f"DefenseEA processed {self.num_processed} prompts."