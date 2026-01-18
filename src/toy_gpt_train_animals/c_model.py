"""c_model.py - Simple model module.

Defines a minimal next-token prediction model for a bigram context
(uses two tokens in sequence: previous token, current token).

Responsibilities:
- Represent a simple parameterized model that maps a
  2-tuple of token IDs (previous token, current token)
  to a score for each token in the vocabulary.
- Convert scores into probabilities using softmax.
- Provide a forward pass (no training in this module).

This model is intentionally simple:
- one weight table (conceptually a 3D tensor: prev x curr x next,
  flattened for storage as a 2D matrix)
- one forward computation
- no learning here

Training is handled in a different module.
"""

import logging

from datafun_toolkit.logger import get_logger, log_header
from toy_gpt_train.c_model import SimpleNextTokenModel

__all__ = ["SimpleNextTokenModel"]

LOG: logging.Logger = get_logger("P01", level="INFO")


def main() -> None:
    """Demonstrate a forward pass of the simple model."""
    # Local imports keep modules decoupled.
    from toy_gpt_train_animals.a_tokenizer import DEFAULT_CORPUS_PATH, SimpleTokenizer
    from toy_gpt_train_animals.b_vocab import Vocabulary

    log_header(LOG, "Simple Next-Token Model Demo")

    # Step 1: Tokenize input text.
    tokenizer: SimpleTokenizer = SimpleTokenizer(corpus_path=DEFAULT_CORPUS_PATH)
    tokens: list[str] = tokenizer.get_tokens()

    if not tokens:
        LOG.info("No tokens available for demonstration.")
        return

    # Step 2: Build vocabulary.
    vocab: Vocabulary = Vocabulary(tokens)

    # Step 3: Initialize model.
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=vocab.vocab_size())

    # Step 4: Select previous and current tokens.
    previous_token: str = tokens[0]
    current_token: str = tokens[1]

    previous_id: int | None = vocab.get_token_id(previous_token)
    current_id: int | None = vocab.get_token_id(current_token)

    if previous_id is None or current_id is None:
        LOG.info("One of the sample tokens was not found in vocabulary.")
        return

    # Step 5: Forward pass (bigram context).
    probs: list[float] = model.forward(previous_id, current_id)

    # Step 6: Inspect results.
    LOG.info(
        f"Input tokens: {previous_token!r} (ID {previous_id}), {current_token!r} (ID {current_id})"
    )
    LOG.info("Output probabilities for next token:")
    for idx, prob in enumerate(probs):
        tok: str | None = vocab.get_id_token(idx)
        LOG.info(f"  {tok!r} (ID {idx}) -> {prob:.4f}")


if __name__ == "__main__":
    main()
