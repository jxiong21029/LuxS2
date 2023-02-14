import torch
from jux.env import JuxEnv
from jux.state import State as JuxState

# try running with XLA_PYTHON_CLIENT_PREALLOCATE=false if there are cudnn-related errors
# and make sure the PATH environment variable makes sense


def state_to_obs(state: JuxState):
    # for now: heuristic behavior for bidding, factory placement, and factory actions
    # only reward function is for supplying factories with water
    # simplified obs reflects all info necessary for this

    state = state.to_torch()
    ret = torch.zeros((7, 48, 48), device=state.env_steps.device)

    # global: [sin t/50, cos t/50, log t, log (1000 - t),]

    # tile: ice, [ore, rubble]
    ret[0] = state.board.ice

    for team in range(2):
        # unit: light, [heavy,] cargo: ice, [ore, water, metal, power]
        # separate obs for ally and enemy
        unit_x = state.units.pos.x[team, : state.n_units[team]].long()
        unit_y = state.units.pos.y[team, : state.n_units[team]].long()
        ret[1 + team, unit_x, unit_y] = 1

        unit_ice = state.units.cargo.ice[team, : state.n_units[team]].float()
        ret[3 + team, unit_x, unit_y] = unit_ice

        # factory: exists, [cargo: ice, ore, water, metal, lichen connected]
        # separate obs for ally and enemy
        factory_x = state.factories.pos.x[team, : state.n_factories[team]].long()
        factory_y = state.factories.pos.y[team, : state.n_factories[team]].long()
        ret[5 + team, factory_x, factory_y] = 1

    return ret


def main():
    pass


if __name__ == "__main__":
    main()
