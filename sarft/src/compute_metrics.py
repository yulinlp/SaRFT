import json
import os
import argparse

# logger = logging.getLogger(__name__)
# CURRENT_DIR = os.path.dirname(__file__)
# GPT2TOKENIZER = os.path.join(CURRENT_DIR, "../data/gpt2tokenizer")


# class GPTTokenizer:
#     gpt_tokenizer = AutoTokenizer.from_pretrained(GPT2TOKENIZER, max_length=1e5)

#     def tokenize(self, s):
#         tokens = self.gpt_tokenizer.tokenize(s)
#         # GPT2 uses Byte-level BPE, which will include space as part of the word. 
#         # But for the first word of a sentence, there is no space before it. 
#         # So, we remove all the added spaces ("Ġ"). 
#         tokens = [t.lstrip("Ġ") for t in tokens]
#         return tokens


# xlingual_tokenizer = GPTTokenizer()

# def ngram_of_1D_tensor(X, n):
#         grams = [tuple(X[i:i + n].tolist()) for i in range(X.shape[0] - n + 1)]
#         return grams

# # adapted the flowing from Squad v1.1 evaluation, without removing the articles.
# def normalize_answer(s):
#     """Lower text and remove punctuation, and extra whitespace."""

#     def white_space_fix(text):
#         return ' '.join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_punc(lower(s)))


# def exact_match_score(prediction, ground_truth, xlingual=False):
#     return (normalize_answer((prediction.lower())) == normalize_answer(ground_truth.lower()))


# def rouge1_score(prediction, ground_truth, xlingual=False):
#     if xlingual:
#         scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
#     else:
#         scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
#     scores = scorer.score(prediction=prediction, target=ground_truth)
#     return scores["rouge1"].fmeasure


# def rougeL_score(prediction, ground_truth, xlingual=False):
#     if xlingual:
#         scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
#     else:
#         scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     scores = scorer.score(prediction=prediction, target=ground_truth)
#     return scores["rougeL"].fmeasure


# def EL_score(input_ids, tokenizer, model, N=[10]):
#     batch_size = input_ids.shape[0]
#     max_len = input_ids.shape[1]
#     numerator = {n: [0] * batch_size for n in N}

#     for i in reversed(range(1, max_len)):
#         label = input_ids[..., i:max_len]
#         prompt = input_ids[..., :i]
#         pred = model.generate(input_ids = prompt, max_length=max_len, pad_token_id=tokenizer.eos_token_id)[..., i:]

#         for example_idx in range(batch_size):
#             p, l = pred[example_idx], label[example_idx]
#             # extraction likelihood
#             for n in N:
#                 p_ngram = ngram_of_1D_tensor(p, n)
#                 l_ngram = ngram_of_1D_tensor(l, n)
#                 l_unique = set(l_ngram)
#                 p_tp = [i for i in p_ngram if i in l_unique]
#                 try:
#                     p_acc = len(p_tp) / len(l_ngram)
#                     numerator[n][example_idx] += p_acc
#                 except ZeroDivisionError:  # n-gram isn't defined
#                     pass

#     el_score = {n: [0] * batch_size for n in N}
#     for n in N:
#         for i, _ in enumerate(numerator[n]):
#             el_score[n][i] = numerator[n][i] / \
#                 (max_len - 1 - (n - 1))

#     # return individual example results for logging
#     ret = {}
#     for k in el_score.keys():
#         ret[f'el_{k}-gram'] = torch.Tensor(el_score[k])
    
#     print(el_score)
#     return sum(el_score[N[0]])


# def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
#     scores_for_ground_truths = []
#     for ground_truth in ground_truths:
#         score = metric_fn(prediction, ground_truth, xlingual=xlingual)
#         scores_for_ground_truths.append(score)
#     return max(scores_for_ground_truths)


def compute_metrics(predictions, references, tokenizer, input_ids, model, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    # exact_match, rouge1, rougeL = 0, 0, 0
    # for pred, gold in zip(predictions, references):
    #     gold = [gold]
    #     exact_match += metric_max_over_ground_truths(
    #         exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
    #     )
    #     rouge1 += metric_max_over_ground_truths(
    #         rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
    #     )
    #     rougeL += metric_max_over_ground_truths(
    #         rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
    #     )
    # # EL = EL_score(input_ids=input_ids, tokenizer=tokenizer, model=model)
    # exact_match = 100.0 * exact_match / len(references)
    # rouge1 = 100.0 * rouge1 / len(references)
    # rougeL = 100.0 * rougeL / len(references)
    # metrics = {"exact_match": exact_match, "rouge1": rouge1, "eval_rougeL": rougeL}
    # metrics = {k: round(v, 4) for k, v in metrics.items()}
    metrics = {}
    return metrics


def compute_grouped_metrics(predictions, references, tokenizer, input_ids, model, groups, xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, tokenizer, input_ids, model, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions file.")
    parser.add_argument("--track", choices=["default", "xlingual"], default="default", 
        help="default track or xlingual track. For xlingual, we need to use a different tokenizer."
    )
    parser.add_argument("--compute_per_category_metrics", action="store_true", help="Compute metrics on every evaluation category.")
    parser.add_argument("--compute_per_task_metrics", action="store_true", help="Compute metrics on every evaluation task.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.predictions) as fin:
        examples = [json.loads(l) for l in fin]

    predictions = [e["prediction"] for e in examples]
    references = [e["instance"]["output"] for e in examples]
    tasks = []
    for e in examples:
        if e["task"] == "task121_atomic_question_rewriting":
            e["task"] = "task121_zest_question_rewriting"
        tasks.append(e["Task"])

    results = compute_metrics(predictions, references, xlingual=args.track == "xlingual")
    print("======== Overall Metrics ========")
    print("all_rougeL", results["rougeL"])
    print("all_EM", results["exact_match"])
    print()
    
    category_metrics = [
        ("Textual Entailment", "exact_match"),
        ("Cause Effect Classification", "exact_match"),
        ("Coreference Resolution", "exact_match"),
        ("Dialogue Act Recognition", "exact_match"),
        ("Answerability Classification", "exact_match"),
        ("Word Analogy", "exact_match"),
        ("Overlap Extraction", "rougeL"),
        ("Keyword Tagging", "rougeL"),
        ("Question Rewriting", "rougeL"),
        ("Title Generation", "rougeL"),
        ("Data to Text", "rougeL"),
        ("Grammar Error Correction", "rougeL"),
    ]
    category_metrics = {"_".join(category.lower().split()): metric for category, metric in category_metrics}

    if args.compute_per_category_metrics:
        print("======== Metrics per category ========")
        task_category = {}
        for task in set(tasks):
            with open(os.path.join("./data/tasks/", task+".json")) as fin:
                task_data = json.load(fin)
                task_category[task] = "_".join(task_data["Categories"][0].lower().split())
        categories = [task_category[e["Task"]] for e in examples] 
        results.update(compute_grouped_metrics(predictions, references, categories, xlingual=args.track=="xlingual"))
        
        for category, metric in category_metrics.items():
            # category = "_".join(category.lower().split())
            if f"{metric}_for_{category}" in results:
                print(f"{metric}_for_{category}", results[f"{metric}_for_{category}"])
        print()
            
    if args.compute_per_task_metrics:
        print("======== Metrics per task ========")
        results_by_task = compute_grouped_metrics(predictions, references, tasks, xlingual=args.track=="xlingual")
        for task in sorted(list(set(tasks))):
            category = task_category[task]
            metric = category_metrics[category]
            print(task, results_by_task[f"{metric}_for_{task}"])
        print()