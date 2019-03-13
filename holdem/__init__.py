DEBUG = False
max_limit = 20000

from gym.envs.registration import register
import holdem.env
register(
    id='TexasHoldem-v0',
    entry_point='holdem.env:TexasHoldem',
    kwargs={'n_seats': 6, 'max_limit': max_limit, 'sb': '50/100'},
)
