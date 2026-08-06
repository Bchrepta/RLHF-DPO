from rlhf_dpo.model.tokenizer import WordTokenizer
from rlhf_dpo.model.transformer import CausalLM, RewardModel

# Back-compat alias used by older imports/tests.
CharTokenizer = WordTokenizer

__all__ = ["WordTokenizer", "CharTokenizer", "CausalLM", "RewardModel"]
