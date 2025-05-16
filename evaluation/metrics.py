from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu


class EvaluationMetrics:
    @staticmethod
    def vector_similarity(v1, v2):
        """计算向量余弦相似度"""
        return cosine_similarity([v1], [v2])[0][0]

    @staticmethod
    def rouge_score(prediction, reference):
        """计算ROUGE-N分数"""
        if not prediction or not reference:
            return {"rouge-1": {"f": 0}}
        rouge = Rouge()
        return rouge.get_scores(prediction, reference)[0]

    @staticmethod
    def bleu_score(prediction, reference):
        """计算BLEU分数"""
        if not prediction or not reference:
            return 0
        return sentence_bleu([reference.split()], prediction.split())

    @staticmethod
    def hits_at_k(ranks, k=10):
        """计算Hits@K指标"""
        return sum([1 for r in ranks if r <= k]) / len(ranks) if ranks else 0

    @staticmethod
    def mean_reciprocal_rank(ranks):
        """计算平均倒数排名(MRR)"""
        return sum([1 / r for r in ranks if r > 0]) / len(ranks) if ranks else 0