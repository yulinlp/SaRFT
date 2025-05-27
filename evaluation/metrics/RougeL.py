from typing import List
from rouge_score import rouge_scorer

class RougeL:
    def __init__(self, **kwargs) -> None:
        pass

    def score(self, prediction: str, ground_truth: str) -> float:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(prediction, ground_truth)
        return scores["rougeL"].fmeasure

    def metric_max_over_ground_truths(self, prediction: str, ground_truths: List[str]) -> float:
        return max(self.score(prediction, gt) for gt in ground_truths)

    def __call__(self, qa_pairs: List[str]) -> float:
        """
        :param predictions: list of predicted strings
        :param references: list of reference strings
        :return: average rougeL score
        """
        predictions = [qa["output"] for qa in qa_pairs]
        references = [qa["label"] for qa in qa_pairs]
        assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
        
        total_score = sum(self.metric_max_over_ground_truths(pred, [ref]) for pred, ref in zip(predictions, references))
        average_score = 100.0 * total_score / len(references)
        return {"rougeL": average_score}