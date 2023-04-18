from luxai_s2 import LuxAI_S2

from model import LuxAIModel


def play(model1: LuxAIModel, model2: LuxAIModel | None):
    if model2 is None:
        model2 = model1

    env = LuxAI_S2()
    env.reset()
