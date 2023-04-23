from luxai_s2 import LuxAI_S2
from model import LinearModel


def play(model1: LinearModel, model2: LinearModel | None):
    if model2 is None:
        model2 = model1

    env = LuxAI_S2()
    env.reset()
