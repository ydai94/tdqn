from os.path import basename
from jericho import *
from jericho.template_action_generator import TemplateActionGenerator
from jericho.util import *
from jericho.defines import *
import redis
from textworld.core import EnvInfos

def load_vocab_rev(env):
    vocab = {i+2: str(v) for i, v in enumerate(env.get_dictionary())}
    vocab[0] = ' '
    vocab[1] = '<s>'
    vocab_rev = {v: idx for idx, v in vocab.items()}
    return vocab_rev

class TemplateActionGeneratorJeri:
    '''
    Generates actions using the template-action-space.
    :param rom_bindings: Game-specific bindings from :meth:`jericho.FrotzEnv.bindings`.
    :type rom_bindings: Dictionary
    '''
    def __init__(self, rom_bindings):
        self.rom_bindings = rom_bindings
        grammar = rom_bindings['grammar'].split(';')
        max_word_length = rom_bindings['max_word_length']
        self.templates = self._preprocess_templates(grammar, max_word_length)
        # Enchanter and Spellbreaker only recognize abbreviated directions
        if rom_bindings['name'] in ['enchanter', 'spellbrkr', 'murdac']:
            for act in ['northeast','northwest','southeast','southwest']:
                self.templates.remove(act)
            self.templates.extend(['ne','nw','se','sw'])

    def _preprocess_templates(self, templates, max_word_length):
        '''
        Converts templates with multiple verbs and takes the first verb.
        '''
        out = []
        vb_usage_fn = lambda verb: verb_usage_count(verb, max_word_length)
        p = re.compile(r'\S+(/\S+)+')
        for template in templates:
            if not template:
                continue
            while True:
                match = p.search(template)
                if not match:
                    break
                verb = max(match.group().split('/'), key=vb_usage_fn)
                template = template[:match.start()] + verb + template[match.end():]
            ts = template.split()
            out.append(template)
        return out


    def generate_actions(self, objs):
        '''
        Given a list of objects present at the current location, returns
        a list of possible actions. This list represents all combinations
        of templates filled with the provided objects.
        :param objs: Candidate interactive objects present at the current location.
        :type objs: List of strings
        :returns: List of action-strings.
        :Example:
        >>> import jericho
        >>> env = jericho.FrotzEnv(rom_path)
        >>> interactive_objs = ['phone', 'keys', 'wallet']
        >>> env.act_gen.generate_actions(interactive_objs)
        ['wake', 'wake up', 'wash', ..., 'examine wallet', 'remove phone', 'taste keys']
        '''
        actions = []
        for template in self.templates:
            holes = template.count('OBJ')
            if holes <= 0:
                actions.append(template)
            elif holes == 1:
                actions.extend([template.replace('OBJ', obj) for obj in objs])
            elif holes == 2:
                for o1 in objs:
                    for o2 in objs:
                        if o1 != o2:
                            actions.append(template.replace('OBJ', o1, 1).replace('OBJ', o2, 1))
        return actions


    def generate_template_actions(self, objs, obj_ids):
        '''
        Given a list of objects and their corresponding vocab_ids, returns
        a list of possible TemplateActions. This list represents all combinations
        of templates filled with the provided objects.
        :param objs: Candidate interactive objects present at the current location.
        :type objs: List of strings
        :param obj_ids: List of ids corresponding to the tokens of each object.
        :type obj_ids: List of int
        :returns: List of :class:`jericho.defines.TemplateAction`.
        :Example:
        >>> import jericho
        >>> env = jericho.FrotzEnv(rom_path)
        >>> interactive_objs = ['phone', 'keys', 'wallet']
        >>> interactive_obj_ids = [718, 325, 64]
        >>> env.act_gen.generate_template_actions(interactive_objs, interactive_obj_ids)
        [
          TemplateAction(action='wake', template_id=0, obj_ids=[]),
          TemplateAction(action='wake up', template_id=1, obj_ids=[]),
          ...
          TemplateAction(action='turn phone on', template_id=55, obj_ids=[718]),
          TemplateAction(action='put wallet on keys', template_id=65, obj_ids=[64, 325])
         ]
        '''
        assert len(objs) == len(obj_ids)
        actions = []
        for template_idx, template in enumerate(self.templates):
            holes = template.count('OBJ')
            if holes <= 0:
                actions.append(defines.TemplateAction(template, template_idx, []))
            elif holes == 1:
                for noun, noun_id in zip(objs, obj_ids):
                    actions.append(
                        defines.TemplateAction(template.replace('OBJ', noun),
                                               template_idx, [noun_id]))
            elif holes == 2:
                for o1, o1_id in zip(objs, obj_ids):
                    for o2, o2_id in zip(objs, obj_ids):
                        if o1 != o2:
                            actions.append(
                                defines.TemplateAction(
                                    template.replace('OBJ', o1, 1).replace('OBJ', o2, 1),
                                    template_idx, [o1_id, o2_id]))
        return actions

import jericho
import textworld
import re
from collections import defaultdict

def _load_bindings_from_tw(state, story_file, seed):
    bindings = {}
    g1 = [re.sub('{.*?}', 'OBJ', s) for s in state.command_templates]
    g = list(set([re.sub('go .*', 'go OBJ', s) for s in g1]))
    g.remove('drop OBJ')
    g.remove('examine OBJ')
    g.remove('inventory')
    g.remove('look')
    bindings['grammar'] = ';'.join(g)
    bindings['max_word_length'] = len(max(state.verbs + state.entities, key=len))
    bindings['minimal_actions'] = '/'.join(state['extra.walkthrough'])
    bindings['name'] = state['extra.uuid']
    bindings['rom'] = story_file.split('/')[-1]
    bindings['seed'] = seed
    bindings['walkthrough'] = bindings['minimal_actions']
    return bindings

class JeriWorld:
    def __init__(self, story_file, seed=None, style='jericho', infos = None):
        self.jeri_style = style.lower() == 'jericho'
        if self.jeri_style:
            self._env = textworld.start(story_file, infos=infos)
            state = self._env.reset()
            self.tw_games = True
            self._seed = seed
            self.bindings = None
            if state.command_templates is None:
                self.tw_games = False
                del self._env
                self._env = jericho.FrotzEnv(story_file, seed)
                self.bindings = self._env.bindings
                self._world_changed = self._env._world_changed
                self.act_gen = self._env.act_gen
            else:
                self.bindings = _load_bindings_from_tw(state, story_file, seed)
                self._world_changed = self._env._jericho._world_changed
                self.act_gen = TemplateActionGeneratorJeri(self.bindings)
                self.seed(seed)
        else:
            self._env = textworld.start(story_file, infos=infos)

    def __del__(self):
        del self._env
 
    
    def reset(self):
        if self.jeri_style:
            if self.tw_games:
                state = self._env.reset()
                raw = state['description']
                return raw, {'moves':state.moves, 'score':state.score}
            return self._env.reset()
        else:
            return self._env.reset()
    
    def load(self, story_file, seed=None):
        if self.jeri_style:
            if self.tw_games:
                self._env.load(story_file)
            else:
                self._env.load(story_file, seed)
        else:
            self._env.load(story_file)

    def step(self, action):
        if self.jeri_style:
            if self.tw_games:
                state = self._env.step(action)
                s_action = re.sub(r'\s+', ' ', action.strip())
                # score = self._env.state.score
                # reward = score - old_score
                reward = state[0]['intermediate_reward']
                self._world_changed = self._env._jericho._world_changed
                return state[0].description, reward, state[2],\
                  {'moves':state[0].moves, 'score':state[1]}
            else:
                self._world_changed = self._env._world_changed
            return self._env.step(action)
        else:
            return self._env.step(action)

    def bindings(self):
        if self.jeri_style:
            return self.bindings
        else:
            return None

    def _emulator_halted(self):
        if self.jeri_style:
            if self.tw_games:
                return self._env._env._emulator_halted()
            return self._env._emulator_halted()
        else:
            return None

    def game_over(self):
        if self.jeri_style:
            if self.tw_games:
                self._env.state['lost']
            return self._env.game_over()
        else:
            return None

    def victory(self):
        if self.jeri_style:
            if self.tw_games:
                self._env.state['won']
            return self._env.victory()
        else:
            return None

    def seed(self, seed=None):
        if self.jeri_style:
            self._seed = seed
            return self._env.seed(seed)
        else:
            return None
    
    def close(self):
        if self.jeri_style:
            self._env.close()
        else:
            pass

    def copy(self):
        return self._env.copy()

    def get_walkthrough(self):
        if self.jeri_style:
            if self.tw_games:
                return self._env.state['extra.walkthrough']
            return self._env.get_walkthrough()
        else:
            return None

    def get_score(self):
        if self.jeri_style:
            if self.tw_games:
                return self._env.state['score']
            return self._env.get_score()
        else:
            return None

    def get_dictionary(self):
        if self.jeri_style:
            if self.tw_games:
                state = self._env.state
                return state.entities + state.verbs
            return self._env.get_dictionary()
        else:
            state = self._env.state
            return state.entities + state.verbs

    def get_state(self):
        if self.jeri_style:
            if self.tw_games:
                return self._env._jericho.get_state()
            return self._env.get_state
        else:
            return None
    
    def set_state(self, state):
        if self.jeri_style:
            if self.tw_games:
                self._env._jericho.set_state(state)
            else:
                self._env.get_state
        else:
            pass

    def get_valid_actions(self, use_object_tree=True, use_ctypes=True, use_parallel=True):
        if self.jeri_style:
            if self.tw_games:
                return self._env.state['admissible_commands']
            return self._env.get_valid_actions(use_object_tree, use_ctypes, use_parallel)
        else:
            pass
    
    def _identify_interactive_objects(self, observation='', use_object_tree=False):
        """
        Identifies objects in the current location and inventory that are likely
        to be interactive.
        :param observation: (optional) narrative response to the last action, used to extract candidate objects.
        :type observation: string
        :param use_object_tree: Query the :doc:`object_tree` for names of surrounding objects.
        :type use_object_tree: boolean
        :returns: A list-of-lists containing the name(s) for each interactive object.
        :Example:
        >>> from jericho import *
        >>> env = FrotzEnv('zork1.z5')
        >>> obs, info = env.reset()
        'You are standing in an open field west of a white house with a boarded front door. There is a small mailbox here.'
        >>> env.identify_interactive_objects(obs)
        [['mailbox', 'small'], ['boarded', 'front', 'door'], ['white', 'house']]
        .. note:: Many objects may be referred to in a variety of ways, such as\
        Zork1's brass latern which may be referred to either as *brass* or *lantern*.\
        This method groups all such aliases together into a list for each object.
        """
        if self.jeri_style:
            if self.tw_games:
                objs = set()
                state = self.get_state()

                if observation:
                    # Extract objects from observation
                    obs_objs = extract_objs(observation)
                    obs_objs = [o + ('OBS',) for o in obs_objs]
                    objs = objs.union(obs_objs)

                # Extract objects from location description
                self.set_state(state)
                look = clean(self.step('look')[0])
                look_objs = extract_objs(look)
                look_objs = [o + ('LOC',) for o in look_objs]
                objs = objs.union(look_objs)

                # Extract objects from inventory description
                self.set_state(state)
                inv = clean(self.step('inventory')[0])
                inv_objs = extract_objs(inv)
                inv_objs = [o + ('INV',) for o in inv_objs]
                objs = objs.union(inv_objs)
                self.set_state(state)

                # Filter out the objects that aren't in the dictionary
                dict_words = [w for w in self.get_dictionary()]
                max_word_length = max([len(w) for w in dict_words])
                to_remove = set()
                for obj in objs:
                    if len(obj[0].split()) > 1:
                        continue
                    if obj[0][:max_word_length] not in dict_words:
                        to_remove.add(obj)
                objs.difference_update(to_remove)
                objs_set = set()
                for obj in objs:
                    if obj[0] not in objs_set:
                        objs_set.add(obj[0])
                return objs_set
            return self._env._identify_interactive_objects(observation=observation, use_object_tree=use_object_tree)
        else:
            return None

    def find_valid_actions(self, possible_acts=None):
        if self.jeri_style:
            if self.tw_games:
                diff2acts = {}
                state = self.get_state()
                candidate_actions = self.get_valid_actions()
                for act in candidate_actions:
                    self.set_state(state)
                    self.step(act)
                    diff = self._env._jericho._get_world_diff()
                    if diff in diff2acts:
                        if act not in diff2acts[diff]:
                            diff2acts[diff].append(act)
                    else:
                        diff2acts[diff] = [act]
                self.set_state(state)
                return diff2acts
            else:
                admissible = []
                candidate_acts = self._env._filter_candidate_actions(possible_acts).values()
                true_actions = self._env.get_valid_actions()
                for temp_list in candidate_acts:
                    for template in temp_list:
                        if template.action in true_actions:
                            admissible.append(template)
                return admissible
        else:
            return None


    def _score_object_names(self, interactive_objs):
        """ Attempts to choose a sensible name for an object, typically a noun. """
        if self.jeri_style:
            def score_fn(obj):
                score = -.01 * len(obj[0])
                if obj[1] == 'NOUN':
                    score += 1
                if obj[1] == 'PROPN':
                    score += .5
                if obj[1] == 'ADJ':
                    score += 0
                if obj[2] == 'OBJTREE':
                    score += .1
                return score
            best_names = []
            for desc, objs in interactive_objs.items():
                sorted_objs = sorted(objs, key=score_fn, reverse=True)
                best_names.append(sorted_objs[0][0])
            return best_names
        else:
            return None

    def get_world_state_hash(self):
        if self.jeri_style:
            if self.tw_games:
                return None
            else:
                return self._env.get_world_state_hash()
        else:
            return None

class JerichoEnv:
    ''' Returns valid actions at each step of the game. '''
    def __init__(self, rom_path, seed, step_limit=None):
        self.rom_path = rom_path
        self.seed = seed
        self.steps = 0
        self.step_limit = step_limit
        self.env = None
        self.conn = None
        self.vocab_rev = None

    def create(self):
        info = EnvInfos(objective=True,description=True,inventory=True,feedback=True,intermediate_reward=True,admissible_commands=True)
        self.env = JeriWorld(self.rom_path, seed = self.seed, infos=info)
        self.bindings = self.env.bindings
        self.act_gen = self.env.act_gen
        self.vocab_rev = load_vocab_rev(self.env)
        '''self.conn = redis.Redis(host='localhost', port=6379, db=0)
        self.conn.flushdb()'''

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # Initialize with default values
        info['look'] = 'unknown'
        info['inv'] = 'unknown'
        info['valid'] = ['wait','yes','no']
        if not done:
            try:
                save = self.env.get_state()
                look, _, _, _ = self.env.step('look')
                info['look'] = look
                self.env.set_state(save)
                inv, _, _, _ = self.env.step('inventory')
                info['inv'] = inv
                self.env.set_state(save)
                # Get the valid actions for this state
                '''world_state_hash = self.env.get_world_state_hash()
                valid = self.conn.get(world_state_hash)
                if valid is None:
                    objs = []
                    for inter_objs in self.env._identify_interactive_objects().values():
                        for obj in inter_objs:
                            objs.append(obj[0])
                    obj_ids = [self.vocab_rev[o[:self.bindings['max_word_length']]] for o in objs]
                    acts = self.act_gen.generate_template_actions(objs, obj_ids)
                    valid = self.env.find_valid_actions(acts)
                    redis_valid_value = '/'.join([str(a) for a in valid])
                    self.conn.set(world_state_hash, redis_valid_value)
                    valid = [a.action for a in valid]
                else:
                    valid = valid.decode('cp1252')
                    if valid:
                        valid = [eval(a).action for a in valid.split('/')]
                    else:
                        valid = []'''
                valid = self.env.get_valid_actions()
                if len(valid) == 0:
                    valid = ['wait','yes','no']
                info['valid'] = valid
            except RuntimeError:
                print('RuntimeError: {}, Done: {}, Info: {}'.format(clean(ob), done, info))
        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        return ob, reward, done, info

    def reset(self):
        initial_ob, info = self.env.reset()
        save = self.env.get_state()
        look, _, _, _ = self.env.step('look')
        info['look'] = look
        self.env.set_state(save)
        inv, _, _, _ = self.env.step('inventory')
        info['inv'] = inv
        self.env.set_state(save)
        '''objs = []
        for inter_objs in self.env._identify_interactive_objects().values():
            for obj in inter_objs:
                objs.append(obj[0])
        obj_ids = [self.vocab_rev[o[:self.bindings['max_word_length']]] for o in objs]
        acts = self.act_gen.generate_template_actions(objs, obj_ids)
        valid_temps = self.env.find_valid_actions(acts)
        valid = []
        for tmp in valid_temps:
            valid.append(tmp.action)'''
        valid = self.env.get_valid_actions()
        info['valid'] = valid
        self.steps = 0
        return initial_ob, info

    def get_dictionary(self):
        if not self.env:
            self.create()
        return self.env.get_dictionary()

    def get_action_set(self):
        return None

    def close(self):
        self.env.close()
