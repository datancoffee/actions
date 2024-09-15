from transformers import pipeline as hf_pipeline
from typing import Any

from core.actions import Action

class InferWithHuggingface(Action):

    def __init__(
            self,
            actionname: str = None,
            *args,
            **kwargs):
        super().__init__(actionname)
        self.hf_pipeline = hf_pipeline(*args, **kwargs)

    def do(self, inputs, *args:Any, **kwargs: Any):
        # pass the baton to __call__ of transformers.Pipeline
        labels = self.hf_pipeline(inputs, *args, truncation=True , **kwargs)
        return labels


class EnrichWithHuggingface(InferWithHuggingface):

    def do(self, inputs:list, field:str, *args:Any, **kwargs: Any):
        """
        Will enrich the input rows with outputs of the model.
        Will preserve all input fields and add the output of the model inference
        :param inputs: Assumed to be a list of dict
        :param field: The field in the dict to run the model on
        :param args:
        :param kwargs:
        :return:
        """

        hf_inputs = [i.get(field,'') for i in inputs]

        # first get the labels and scores
        labels = super().do(hf_inputs, *args, **kwargs)
        # then add the input records, assuming that labels are in the same order
        for l,i in zip(labels,inputs):
            l.update(i)

        return labels