import jax.numpy as jnp
from jax import Array as JaxArray
from jux.state import State as JuxState


from action_masking import get_dig_mask


# non-jitted code
def collision_free_action_argmax(state: JuxState, q_preds: JaxArray):
    for team in range(2):
        # 0: idle
        # 1,2,3,4: N, E, S, W
        # 5: dig
        # 6: dropoff ice

        dig_mask = get_dig_mask(state)

        pickup_mask = jnp.zeros((48, 48))
        pickup_mask.at(state.factories.occupancy.pos.x[team], state.factories.occupancy.pos.y[team]).set(1, mode="drop")

        v = 1 if team == 0 else -1

        center_best = q_preds[..., 0]
        center_best = jnp.where(
            dig_mask & (v * q_preds[..., 5] > v * center_best),
            q_preds[..., 5],
            center_best
        )
        center_best = jnp.where(
            pickup_mask & (v * q_preds[..., 6] > v * center_best),
            q_preds[..., 6],
            center_best,
        )

        agent_pos_q = jnp.zeros(())
