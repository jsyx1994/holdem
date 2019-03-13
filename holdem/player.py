# -*- coding: utf-8 -*-
from holdem import max_limit


class Player(object):
    """
    Attributes of players. Players != agents
    """
    def __init__(self, player_id):
        self.id = player_id
        self.stack = max_limit
        self.total_reward = 0   # the accumulative reward of all hands
        # self.total_invest = 0   # the accumulative invest of one episode
        self.pot = 0    # the round pot
        self.is_fold = False
        self.is_all_in = False
        self.last_action = None
        self.is_played = False
        self.prior_player_id = None
        self.hand = []

    def reset(self):
        self.stack = max_limit
        self.pot = 0
        # self.total_invest = 0
        self.is_fold = False
        self.is_all_in = False
        self.last_action = None
        self.is_played = False
        self.prior_player_id = None
        self.hand = []

    def reset_round_info(self):
        self.pot = 0
        self.is_played = False

    def should_pass(self):
        return self.is_fold or self.is_all_in




