import gym
import holdem.env
from treys import Evaluator
tenv = gym.make('TexasHoldem-v0')
evaluator = Evaluator()
# print(tenv.action_space.sample())

tenv.reset()
tenv.render()
done = False
while not done:
    # _, _, done, skip_render = tenv.step((1, None))
    observation, reward, done, skip_render = tenv.step((tenv.action_space.sample()))
    if not skip_render:
        tenv.render()
print()
print('---------------------------- Summary ----------------------------')
evaluator.hand_summary(tenv.summary[0], tenv.summary[1])
