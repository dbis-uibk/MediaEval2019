from dbispipeline.base import Evaluator
from sklearn import metrics


class ClassificationEvaluator(Evaluator):
    """Evaluator that runns basic classification metrics on the pipline."""

    def __init__(self, average='macro'):
        """
        Creates a new instance.

        Args:
            average: see the sklearn doc for e.g. f1_score .
        """
        self.average = average

    def evaluate(self, model, dataloader):
        """
            Evaluates the pipline based on the given dataset.

            Args:
                model: the model given in the pipeline.
                dataloader: the dataloader used to load the dataset.

            Returns: A dict containting f1, accuracy, recall, precision and the
            confusion matrix.
        """
        self._check_loader_methods(dataloader, ['load_train', 'load_test'])

        (xtrain, ytrain) = dataloader.load_train()
        (xtest, ytest) = dataloader.load_test()
        model.fit(xtrain, ytrain)
        pred = model.predict(xtest)
        result = {
            'f1':
                metrics.f1_score(ytest, pred, average=self.average),
            'accuracy':
                metrics.accuracy_score(ytest, pred),
            'recall':
                metrics.recall_score(ytest, pred, average=self.average),
            'precision':
                metrics.precision_score(ytest, pred, average=self.average),
            'confusion_matrix':
                metrics.confusion_matrix(ytest, pred).tolist(),
            'roc_auc_all':
                metrics.roc_auc_score(ytest, pred),
        }

        return result

    @property
    def configuration(self):
        """
        Returns a dict-like representation of the configuration of this loader.
        This is for storing its state in the database.
        """
        return {
            'average': self.average,
        }
