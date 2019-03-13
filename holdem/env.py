# -*- coding: utf-8 -*-
from holdem import max_limit, DEBUG
from gym import Env, error, spaces
from treys import Card, Deck, Evaluator
from gym.spaces import prng
from holdem.player import Player


class NormalActions(object):
    """
    the actual actions are 0,1,2. 4 and 5 are used to identify the special case of raise and call
    """
    ACTION_LEVEL = 3
    FOLD = 0
    CALL = 1
    RAISE = 2
    ALL_IN = 3
    CHECK = 4


class CustomActions(object):
    ACTION_LEVEL = 11  # y0-y10(r0-r8,c9,f10)
    ACTION_SPACE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class Round(object):
    ROUND_LEVEL = 4
    PRE_FLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class TexasHoldem(Env):
    # small and big blind
    MAX_ROUND = 4
    SB_BLIND = {
        '10/25': [10, 25],
        '25/50': [25, 50],
        '50/100': [50, 100],
    }

    def __init__(self, n_seats, max_limit, sb='50/100'):
        # constants:
        self.sb = sb  # small/big blind pair
        self.n_seats = n_seats  # the players joined in
        self.max_limit = max_limit  # the max chips of raise
        self._first2play = [2, 0, 0, 0]  # the first player to play in four rounds
        self.small_blind, self.big_blind = TexasHoldem.SB_BLIND[sb][0], TexasHoldem.SB_BLIND[sb][1]

        # global: treys objects
        self._deck = Deck()  # deck manager
        self._evaluator = Evaluator()  # hands evaluator
        self.summary = None  # summary of the whole game of every player
        self.winner = None

        # global: info
        self._round = 0
        self.community_card = []
        self._button = n_seats - 2  # reset inc button(n_seats-1), so small/big blind the 1st/2st
        self.sb_index, self.bb_index, self.utg_index = None, None, None  # index of three special position
        self._player_still_on = n_seats  # number of players on the board
        # self._side_pots = [0] * n_seats   # no side pots in ACPC rule
        self._last_pot = 0
        self._total_pot = 0
        self._last_raise = 0
        self._reward = []
        self._last_raise_times = 0  # preserved
        self._number_of_hands = 0  # the times played

        # about player
        self._seats = [Player(i) for i in range(n_seats)]  # fill with agents
        self._curr_player = None
        self._last_player = None
        self._last_action = None

        self.action_space = spaces.Tuple([
            spaces.Discrete(NormalActions.ACTION_LEVEL),
            spaces.Discrete(max_limit)
        ])

        self.observation_space = spaces.Tuple([
            spaces.Tuple([  # the info containing each player
                             spaces.MultiDiscrete([
                                 NormalActions.ACTION_LEVEL,  # the action of each player
                                 max_limit,  # the stack of every player
                             ])
                         ] * n_seats),
            spaces.Tuple([  # the info of public
                spaces.Discrete(TexasHoldem.MAX_ROUND),  # current round
                spaces.Discrete(max_limit),  # pot amount
                spaces.Discrete(max_limit),  # last raise
                spaces.Discrete(max_limit),  # minimum amount to raise
                spaces.Discrete(max_limit),  # how much needed to call by current player.
                spaces.Discrete(n_seats),  # current player seat location.

            ])
        ]
        )

        self.round_flag = True  # indicate if one round is terminated
        prng.seed()  # this is important for different behaviours to sample actions

    def reset(self):
        # reset deck
        self._deck.shuffle()  # shuffle 52 cards

        # reset global info
        self._round = 0
        self.community_card = []
        # calc locations
        self._button = (self._button + 1) % self.n_seats
        self.sb_index = (self._button + 1) % self.n_seats
        self.bb_index = (self.sb_index + 1) % self.n_seats
        self.utg_index = (self.bb_index + 1) % self.n_seats
        self._player_still_on = self.n_seats
        self._last_pot = self.big_blind
        self._total_pot = self.small_blind + self.big_blind
        self._last_raise = self.big_blind
        self._last_raise_times = 0  # preserved
        self._number_of_hands += 1  # count how many hands played

        # reset player info
        for p in self._seats:  # reset episode information
            p.reset()
        self._curr_player = self._seats[self.utg_index]
        self._last_player = None
        self._last_action = None
        self._reward = [0 for _ in range(self.n_seats)]
        # assign small and big blinds
        self._seats[self.sb_index].pot, self._seats[self.sb_index].stack = self.small_blind, max_limit - self.small_blind
        self._seats[self.bb_index].pot, self._seats[self.bb_index].stack = self.big_blind, max_limit - self.big_blind

        # actions
        self._deal(self.n_seats)  # hands out two private cards for every player

    def step(self, action):
        curr_player = self._curr_player
        skip_render = True
        to_call = self._last_pot - curr_player.pot  # the difference between last players' pot and current's
        if DEBUG:
            print('last_raise:', self._last_raise)
            print('to_call:', to_call)

        # if the player is fold, pass
        if curr_player.is_fold:
            self._curr_player = self._seats[(curr_player.id + 1) % self.n_seats]  # the next player
            return None, None, False, skip_render
        elif curr_player.is_all_in:
            self._curr_player = self._seats[(curr_player.id + 1) % self.n_seats]  # the next player
            return None, None, False, skip_render

        if action[0] == NormalActions.FOLD:
            curr_player.is_fold = True
            curr_player.is_played = True
            self._player_still_on -= 1
        elif action[0] == NormalActions.CALL:
            curr_player.is_played = True
            if self._last_raise == 0:  # special case
                action = (NormalActions.CHECK, None)
            else:   # according to ACPC rules, the player always have enough chips to call
                curr_player.pot += to_call
                curr_player.stack -= to_call
            if curr_player.stack == 0:  # check if player calls an "all in"
                curr_player.is_all_in = True
        elif action[0] == NormalActions.RAISE:
            valid = False
            charge_amount = action[1]
            min2raise = max(self.big_blind, self._last_raise)
            if charge_amount < min2raise:   # special case: fold, the raise amount does not meet the least requirements
                curr_player.is_fold = True  # automatically folds
                curr_player.is_played = True
                action = (NormalActions.FOLD, None)  # work done, change to real action to show in render
                self._player_still_on -= 1
            elif curr_player.stack <= charge_amount:  # special case: all in, the raise amount exceed the player's stack
                valid = True  # convert to all in action
                self._last_raise = curr_player.stack
                curr_player.pot, curr_player.stack = curr_player.stack, 0  # according to ACPC rules
                curr_player.is_all_in = True
                action = (NormalActions.ALL_IN, None)
            else:
                valid = True
                curr_player.pot += charge_amount
                curr_player.stack -= charge_amount
                self._last_raise = charge_amount

            if valid:  # everyone need to consider one more action except the one who raises
                for p in self._seats:
                    p.is_played = False
                curr_player.is_played = True

        curr_player.last_action = action
        self._last_action = action
        self._last_player = self._curr_player.id
        self._last_pot = curr_player.pot if self._last_pot < curr_player.pot else self._last_pot
        self._total_pot = sum([max_limit - p.stack for p in self._seats])

        # the customized state information
        state = {
            'button': self._button,
            'sb': TexasHoldem.SB_BLIND[self.sb],
            'round': self._round,
            'community cards': self.community_card,
            'total pot': self._total_pot,
            'round pot': [p.pot for p in self._seats],
            'player stacks': [p.stack for p in self._seats],
            'last actions': [p.last_action for p in self._seats],
            'next player cards': self._seats[(curr_player.id + 1) % self.n_seats].hand,
            'number of players on': self._player_still_on
        }
        self._reward = [p.stack - max_limit for p in self._seats]  # env reward
        done = False
        skip_render = False

        # check the next valid player and if should go to next round, otherwise next player's turn
        next_pos, quick_step = self.cal_next_player(self._curr_player.id)
        if quick_step:
            for _ in range(self._round, TexasHoldem.MAX_ROUND):  # quickly go through the rounds
                done = self._next_round()
        elif next_pos.is_played:  # the the next player is played before, enter the next round
            done = self._next_round()
        else:
            self._curr_player = next_pos
        done = bool(done)
        return state, self._reward, done, skip_render

    def cal_next_player(self, pos):
        """
        :param pos: the current player's pos
        :return: the next valid player's pos and if can quickly go through the rounds
        """
        quick_step = False
        next_pos = self._seats[(pos + 1) % self.n_seats]
        if self._player_still_on == 1:  # check the easy winner
            return None, True
        while next_pos.is_fold or next_pos.is_all_in:
            if next_pos.id == self._curr_player.id:
                quick_step = True
                # next_pos.is_fold = False  # the current player is the last one, winner cannot fold
                break
            next_pos = self._seats[(next_pos.id + 1) % self.n_seats]
        return next_pos, quick_step

    def _deal(self, n_seats):
        for i in range(n_seats):
            hand = self._deck.draw(2)
            self._seats[i].hand = hand

    def _next_round(self):
        """
        :return: done(TF)
        """
        self.round_flag = True
        self._round += 1
        self._last_pot = 0
        self._last_raise = 0
        for p in self._seats:
            p.reset_round_info()

        if self._round == Round.FLOP:
            self._flop()
        elif self._round == Round.TURN:
            self._turn()
        elif self._round == Round.RIVER:
            self._river()
        else:
            return self._show_down()
        self._curr_player, _ = self.cal_next_player(self._button)
        return False

    def _flop(self):
        self.community_card = self._deck.draw(3)

    def _turn(self):
        self.community_card.append(self._deck.draw(1))

    def _river(self):
        self.community_card.append(self._deck.draw(1))

    def _show_down(self):
        competitors = [p for p in self._seats if not p.is_fold]
        hands = [h.hand for h in competitors]
        hands_id = [(h.hand, h.id) for h in competitors]
        combinations = [(self.community_card, h) for h in hands_id]  # cb[0](cc)  cb[1][0](hands), cb[1][1](id)
        scores_index = [(self._evaluator.evaluate(cb[0], cb[1][0]), cb[1][1]) for index, cb in enumerate(combinations)]
        scores_index = sorted(scores_index, key=lambda x: x[0])
        # calc all the winners and split the pot
        winners_id = [scores_index[0][1]]

        for i in range(len(scores_index) - 1):
            if scores_index[i][0] == scores_index[i + 1][0]:
                winners_id.append(scores_index[i + 1][1])
            else:
                break

        self._reward = [p.stack - max_limit for p in self._seats]
        neg_winner_reward = 0
        for i in range(self.n_seats):
            if i not in winners_id:
                neg_winner_reward += self._reward[i]

        split_reward = -neg_winner_reward / len(winners_id)

        for i in winners_id:
            self._reward[i] = split_reward

        done = True
        self.summary = (self.community_card, hands)
        self.winner = winners_id
        return done

    def render(self, mode='human'):
        from termcolor import cprint, colored
        try:
            text = colored('The last action by player {}:'.format(self._last_player), 'cyan')
            print(text)
            if self._last_action[0] == NormalActions.FOLD:
                cprint('X', 'red')
            elif self._last_action[0] == NormalActions.CALL:
                cprint('-', 'yellow')
            elif self._last_action[0] == NormalActions.RAISE:
                cprint('^ {}'.format(self._last_action[1]), 'magenta')
            elif self._last_action[0] == NormalActions.ALL_IN:
                cprint('A^', 'magenta')
            else:
                print('~')
        except Exception as e:
            pass

        if self.round_flag:
            self.round_flag = False
            if self._round == Round.FLOP:
                print('---------------------------- Flop ----------------------------')
            elif self._round == Round.TURN:
                print('---------------------------- Turn ----------------------------')
            elif self._round == Round.RIVER:
                print('---------------------------- River ----------------------------')
            elif self._round == Round.PRE_FLOP:
                print('---------------------------- Pre flop ----------------------------')
        print()
        print('community cards:')
        if not self.community_card:
            print('-', '[  ],[  ],[  ],[  ],[  ]')
        else:
            print('-' + Card.print_pretty_cards(self.community_card))
        print('Total pot: {}'.format(self._total_pot))

        print('players:')
        for i, p in enumerate(self._seats):
            print('{0}{1}stack: {2} pot:{3} {4} {5}'.format(i,
                                                            Card.print_pretty_cards(p.hand),
                                                            self._seats[i].stack,
                                                            self._seats[i].pot,
                                                            '◉' if i == self._button else '',
                                                            'X' if p.is_fold else ''),
                  )
        if self.winner:
            print("Winner is :")
            for w in self.winner:
                print(w, end=' ')
            print()
            print("Players' reward :")
            print(self._reward)
