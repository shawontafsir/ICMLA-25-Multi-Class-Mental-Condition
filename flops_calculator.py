import torch
from fvcore.nn import FlopCountAnalysis
from transformers import AutoModelForSequenceClassification

from lstm import LSTM  # Your custom LSTM module


def calculate_FLOPs(model, inputs):
    """
    Compute approximate FLOPs, parameters, and peak memory usage.
    """
    model = model.cuda().eval()

    with torch.no_grad():
        flops = FlopCountAnalysis(model, inputs)
        total_flops = flops.total() / (10 ** 9)  # GFLOPs

        unsupported_ops = flops.unsupported_ops()
        if unsupported_ops:
            print("Unsupported ops detected:", unsupported_ops)

        n_params = sum(p.numel() for p in model.parameters())

        print(f"Approx. FLOPs: {total_flops} GFLOPs")
        print(f"Total Parameters: {n_params / (10 ** 6)} Million")

    torch.cuda.reset_peak_memory_stats()
    _ = model(*inputs)
    max_memory_bytes = torch.cuda.max_memory_allocated()
    max_memory_gb = max_memory_bytes / (1024 ** 3)

    print(f"Peak GPU Memory: {max_memory_gb} GB")


if __name__ == "__main__":
    dummy_vocab = {word: idx for idx, word in enumerate(['the', 'cat', 'sat', 'on', 'mat'])}
    batch_size = 16
    seq_len = 128
    inputs = (torch.randint(0, len(dummy_vocab), (batch_size, seq_len)).cuda(),)

    for _embedding, _path in [
        ("bert", None),
        ("glove", "embeddings/glove.6B.300d.txt"),
        ("word2vec", "embeddings/GoogleNews-vectors-negative300.bin")
    ]:
        for _is_bidirectional in [True, False]:
            for _has_attention in [True, False]:
                model = LSTM(
                    hidden_size=128,
                    num_classes=7,
                    is_bidirectional=_is_bidirectional,
                    has_attention=_has_attention,
                    embedding_type=_embedding,
                    vocab=None if _embedding == "bert" else dummy_vocab,
                    pretrained_embedding_path=_path,

                ).cuda().eval()

                print(model.class_name)
                calculate_FLOPs(model, inputs)

    # Simulate real input
    transformer_vocab_size = 30522
    seq_len = 128
    input_ids = torch.randint(0, transformer_vocab_size, (batch_size, seq_len)).cuda()
    albert_input_ids = torch.randint(0, 30000, (batch_size, seq_len)).cuda()
    attention_mask = torch.ones((batch_size, seq_len)).cuda()

    print("Load full RoBERTA")
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2
    ).cuda().eval()

    calculate_FLOPs(model, (input_ids, attention_mask))

    print("Load full BERT")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    ).cuda().eval()

    calculate_FLOPs(model, (input_ids, attention_mask))

    print("Load full DistilBERT")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).cuda().eval()
    calculate_FLOPs(model, (input_ids, attention_mask))

    print("Load full ALBERT")
    model = AutoModelForSequenceClassification.from_pretrained(
        "albert-base-v2", num_labels=2
    ).cuda().eval()

    calculate_FLOPs(model, (albert_input_ids, attention_mask))

    print("Load full Electra")
    model = AutoModelForSequenceClassification.from_pretrained(
        "google/electra-base-discriminator", num_labels=2
    ).cuda().eval()

    calculate_FLOPs(model, (input_ids, attention_mask))
