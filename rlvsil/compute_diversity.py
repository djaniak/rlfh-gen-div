import json
from pprint import pprint

import torch
import typer
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvsil.diversity import DEFAULT_CONFIGS, calculate_diversity_metrics


def generate_samples(
    dataset,
    text_col,
    tokenizer,
    model,
    N,
    temperature,
    summarization_ratio_lenth,
    device,
):
    def build_prompt(summarization_text):
        return f"Twoim zadaniem jest przeczytanie podanego tekstu i napisanie streszczenia w języku polskim. Streszczenie powinno zawierać najważniejsze informacje i wydarzenia opisane w tekście, być zwięzłe i dobrze zorganizowane. Unikaj wprowadzania nowych informacji oraz osobistych opinii.\n\n###\n\nTekst: {summarization_text}\n\nStreszczenie:"

    outputs = []
    for example in tqdm(dataset):
        prompt = build_prompt(example[text_col])
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        max_length = int(len(input_ids[0]) * (1 + summarization_ratio_lenth))
        samples = []
        for _ in tqdm(range(N)):
            output = model.generate(
                input_ids.to(device),
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                max_length=max_length,
                temperature=temperature,
                top_k=0,
                top_p=1,
            )
            generated_text = tokenizer.decode(
                output[0][len(input_ids[0]) :], skip_special_tokens=True
            ).strip()
            samples.append(generated_text)
        outputs.append(samples)
    return outputs


# dataset-id=clarin-knext/summarization-chat-annotated
# model-id=speakleash/Bielik-7B-v0.1


def main(
    model_id: str,
    dataset_id: str,
    dataset_split: str,
    text_col: str,
    output_path: str,
    limit_dataset_samples: int | None = None,
    N: int = 8,
    temperature: float = 1,
    summarization_ratio_lenth: float = 0.1,
    seed: int = 42,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device)

    dataset = load_dataset(dataset_id)[dataset_split]
    if limit_dataset_samples:
        dataset = dataset.select(range(limit_dataset_samples))

    outputs = generate_samples(
        dataset,
        text_col,
        tokenizer,
        model,
        N,
        temperature,
        summarization_ratio_lenth,
        device,
    )
    with open(output_path, "w") as f:
        json.dump({"outputs": outputs}, f)

    metrics = [
        "ead_averaged_distinct_ngrams",
        "nli_sample_from_sim",
        "sent_bert_from_sim",
    ]
    config = DEFAULT_CONFIGS.copy()
    config = {k: v for k, v in config.items() if k in metrics}
    config["sample_overall"] = True
    config["no_overall_input"] = True
    pprint(config)

    results = calculate_diversity_metrics(outputs, metric_configs=config)

    with open(output_path, "w") as f:
        json.dump({"results": results, "outputs": outputs}, f)


if __name__ == "__main__":
    typer.run(main)
