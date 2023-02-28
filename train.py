import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from jux.actions import bid_action_from_lux, factory_placement_action_from_lux
from jux.env import JuxEnv
from jux.state import State as JuxState
from jux.utils import load_replay

from action_selection import step_best

matplotlib.use("agg")


# TODO: observations should include the units' action queues (in this case, simply the
#   last action taken).
@jax.jit
def state_to_obs(state: JuxState):
    # for now: heuristic behavior for bidding, factory placement, and factory actions
    # only reward function is for supplying factories with water
    # simplified obs reflects all info necessary for this

    ret = jnp.zeros((48, 48, 9))

    # global: [sin t/50, cos t/50, log t, log (1000 - t),]

    # tile: ice, x, y, [ore, rubble]
    ret = ret.at[..., 0].set(jnp.arange(48).reshape(1, 48) / 48)
    ret = ret.at[..., 1].set(jnp.arange(48).reshape(48, 1) / 48)
    ret = ret.at[..., 2].set(state.board.ice)

    for team in range(2):
        # unit: light, [heavy,] cargo: ice, [ore, water, metal]; [power]
        # separate obs for ally and enemy
        unit_x = state.units.pos.x[team]
        unit_y = state.units.pos.y[team]
        cargo_ice = state.units.cargo.ice[team]
        ret = ret.at[unit_x, unit_y, 3 + team].set(1, mode="drop")
        ret = ret.at[unit_x, unit_y, 5 + team].set(cargo_ice / 100, mode="drop")

        # factory: occupancy, [center, cargo: ice, ore, water, metal; lichen connected]
        # separate obs for ally and enemy
        factory_occp_x = state.factories.occupancy.x[team]
        factory_occp_y = state.factories.occupancy.y[team]
        ret = ret.at[factory_occp_x, factory_occp_y, 7 + team].set(1, mode="drop")

    return ret


class QNet(nn.Module):
    # returns predicted Q-values (for each agent)
    # current plan is do things VDN-style and just sum the local Q-values
    @nn.compact
    def __call__(self, obs):
        x = nn.Conv(32, kernel_size=(7, 7))(obs)
        x = nn.relu(x)
        x = nn.Conv(7, kernel_size=(1, 1))(x)
        return x


def main():
    lux_env, lux_actions = load_replay(
        f"https://www.kaggleusercontent.com/episodes/{46215591}.json"
    )
    jux_env, state = JuxEnv.from_lux(lux_env)
    lux_act = next(lux_actions)
    bid, faction = bid_action_from_lux(lux_act)
    state, _ = jux_env.step_bid(state, bid, faction)
    while state.real_env_steps < 0:
        lux_act = next(lux_actions)
        spawn, water, metal = factory_placement_action_from_lux(lux_act)
        state, _ = jux_env.step_factory_placement(state, spawn, water, metal)

    model = QNet()

    key = jax.random.PRNGKey(42)
    params = model.init(key, state_to_obs(state))

    for i in range(1000):
        action_values = model.apply(params, state_to_obs(state))
        state: JuxState = step_best(state, action_values)
        done = (state.n_factories == 0).any()
        if done:
            break
        if i % 20 == 0:
            print(f"Units: {state.n_units.sum()}, Factories: {state.n_factories.sum()}")

        img = jux_env.render(state, mode="rgb_array")
        fig, ax = plt.subplots(constrained_layout=True)
        fig: plt.Figure
        ax: plt.Axes
        ax.axis("off")
        ax.imshow(img)
        fig.savefig(f"images/frame_{i:>03}")
        plt.close(fig)


if __name__ == "__main__":
    main()
