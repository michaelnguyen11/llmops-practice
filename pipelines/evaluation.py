from zenml import pipeline

from steps import evaluation as evaluation_step


@pipeline
def evaluation(is_dummy: bool = False) -> None:
    evaluation_step.evaluate(is_dummy=is_dummy)
