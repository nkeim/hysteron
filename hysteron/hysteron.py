"""Event-based simulation of hysterons (hysteretic 2-state subsystems).
"""

# This module requires numba 0.51 or later for optimization. However, by
# commenting out all of the "@numba" lines below, the relevant "import" lines
# above, and the definition of _Recorder_spec, you can eliminate this
# dependency and the code will run, albeit slowly.
#
# This module requires Python 3.5 or later for the "@" operator.
#
# Copyright 2020 Nathan Keim and Joseph Paulsen
#
#   Licensed under the Apache License, Version 2.0 (the "License"); you may not
#   use this file except in compliance with the License. You may obtain a copy
#   of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#   WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#   License for the specific language governing permissions and limitations
#   under the License.

import numba
from numba import float64, optional, boolean
import itertools
import numpy as np


class StabilityError(RuntimeError):
    """Raised if system cannot be made stable."""
    pass


class DegeneracyError(RuntimeError):
    """Raised if multiple hysterons are equally unstable."""
    pass


_Recorder_spec = [
    ('_data', numba.float64[:,:]),
    ('enable_record', numba.boolean),
    ('N', numba.int64),
]

@numba.experimental.jitclass(_Recorder_spec)
class Recorder(object):
    def __init__(self, N=0):
        """Class to hold data captured during a simulation run.
        
        Only records if N > 0 (the number of hysterons).
        """
        if N == 0:
            self.enable_record = False
            self._data = np.array([[np.nan]])
        else:
            self.enable_record = True
            self._data = np.empty((0, N*2 + 1))
        self.N = N
        
    @property
    def data(self):
        return self._data
    
    def record(self, state, H, interactions):
        if not self.enable_record:
            return
        
        r = np.empty((1, self.N * 2 + 1))
        local_field = H + state @ interactions
        r[0, 0] = H
        r[0, 1:(self.N + 1)] = state
        r[0, (self.N + 1):(2*self.N + 1)
                  ] = local_field
        self._data = np.concatenate((self._data, r), axis=0)


@numba.njit
def evolve_event(state, H_target, rising, interactions, Hon, Hoff, 
                 recorder, debug=False, find_next_flip=False):
    """Portion of an event-based simulation that changes field to H_target.
    
    If this results in a hysteron flip, "state" will be modified in-place,
    unless "find_next_flip" is true.
    
    "rising" should be 0 or 1, depending on the direction of field change.
    
    "interactions" is the J_ij matrix, dimensions N x N where N is the number
    of hysterons. The effective local value of the field at each hysteron is
    computed as state * interactions + H, where H is the external field.
    
    "Hon" and "Hoff" are vectors of length N, with Hon >= Hoff, that specify
    the local value of field at which each hysteron changes state.
    
    "recorder" is a Recorder object. Each row in the recorded data relfects 
    the system state and other conditions *before* a spin is flipped.
    
    If "debug", lots of pontentially useful information junk will be spewed 
    to stdout.
    
    If "find_next_flip", return the value of H that will cause the next 
    spin to flip. Use with "H_target" = +/-inf.
    """
    N = len(state)
    # Include enough iterations to flip all the hysterons, plus
    # some stability shenanigans.
    inter_field = state @ interactions
    # Note: if Hon is not greater than Hoff, this can get stuck
    # in an infinite loop where the same spins are flipped on each
    # iteration, and then unflipped by stabilize().
    if debug: print('')
    prev_state = np.empty_like(state)
    while 1:
        prev_state[:] = state
        if debug: print('state', state)
        # "dist" = abs(field required to flip each hysteron)
        if rising:
            dist = Hon - inter_field
            flippable = state < 0
        else:
            dist = inter_field - Hoff
            flippable = state > 0
        if debug: print('dist', dist)
        
        # Find the hysteron that is closest to being flipped
        to_flip = -N  # Impossible
        min_dist = np.inf
        degenerate = False
        for i in range(N):
            if not flippable[i]: continue
            if dist[i] < min_dist:
                min_dist = dist[i]
                to_flip = i
                degenerate = False
            elif dist[i] == min_dist:
                degenerate = True
        if debug: print('flip', to_flip, dist[to_flip], H_target)
        if degenerate:
            raise DegeneracyError('Degeneracy encountered in next unstable hysteron.')
        
        if to_flip < 0:
            # Nobody was flippable. H -> H_target.
            recorder.record(state, H_target, interactions)
            return H_target
        
        # This condition is totally normal. Remember, min_dist is like
        # the field required to flip
        #if min_dist < 0:
        #    raise StabilityError('State not consistent')
                
        if rising:
            # At what value of field will it flip?
            H_flip = dist[to_flip]
            # Check if the target field is even big enough.
            if H_flip > H_target:
                recorder.record(state, H_target, interactions)
                return H_target
            recorder.record(state, H_flip, interactions)
            if find_next_flip:
                return H_flip
            else:
                # Actually change the hysteron
                state[to_flip] = 1
        else:
            H_flip = -dist[to_flip]
            if H_flip < H_target:
                recorder.record(state, H_target, interactions)
                return H_target
            recorder.record(state, H_flip, interactions)
            if find_next_flip:
                return H_flip
            else:
                # Actually change the hysteron
                state[to_flip] = -1
            
        if debug: print('H_flip', H_flip, H_target)
        if debug: print('state_temp', state)
        
        # Recompute interactions, and check for avalanches.
        inter_field = stabilize_event(state, H_flip, 
                                      interactions, Hon, Hoff, recorder)
        
        # Safeguard: If we've reached this point and we're in the same
        # state, we're going to be stuck. 
        # This could probably be avoided by making the present value of H
        # matter more in deciding whether a hysteron is going to get
        # flipped?
        if np.all(state == prev_state):
            raise StabilityError('Simulation got stuck')


@numba.njit
def stabilize_event(state, H, interactions, Hon, Hoff, recorder):
    """Find a stable state, holding magnetic field fixed.
    """
    N = len(state)
    instability = np.empty_like(Hon)
    for it in range(N * 2):
        # Method: Find the most unstable hysteron and flip it
        inter_field = state @ interactions
        local_field = H + inter_field
        
        # Compute instability metric
        for i in range(N):
            if state[i] < 0 and local_field[i] >= Hon[i]:
                instability[i] = local_field[i] - Hon[i]
            elif state[i] > 0 and local_field[i] <= Hoff[i]:
                instability[i] = Hoff[i] - local_field[i]
            else:
                instability[i] = -1.0  # Stable
        
        # Find most unstable hysteron and check for degeneracy
        max_instability = -np.inf
        degenerate = False
        i_max = -N  # Impossible
        for i in range(N):
            if instability[i] > max_instability:
                max_instability = instability[i]
                i_max = i
                degenerate = False
            elif instability[i] == max_instability:
                degenerate = True
        if max_instability > 1 and degenerate:
            raise DegeneracyError('Degeneracy encountered in next unstable hysteron.')

        if instability[i_max] > 0:
            recorder.record(state, H, interactions)
            if state[i_max] > 0:
                state[i_max] = -1
            else:
                state[i_max] = 1
        else:
            return inter_field
    else:
        raise StabilityError("Couldn't find a stable state")
        
        
@numba.njit(locals={'_recorder': numba.typeof(Recorder(0))})
def run_event_extended_fast(state, interactions, Hon, Hoff, amplitude, recorder=None,
                       H_init=-np.inf, debug=False,
                       terminate_on_repeat=True):
    """Stripped-down, numba-only version of run_event_extended().
    
    Call this directly in performance-critical code. For best performance,
    specify all optional arguments explicitly (reason unknown).
    """
    if recorder is None:
        _recorder = Recorder(0)
    else:
        _recorder = recorder
    
    max_cycles = 2**len(Hon) + 1
    history = np.empty((max_cycles + 1, len(state)), dtype=state.dtype)
    periodicity = 0  # Means no periodicity found
    
    # Initialize by bringing field to zero 
    # ("outer hysteresis loop")
    stabilize_event(state, H_init, interactions, Hon, Hoff, _recorder)
    if H_init < 0:
        evolve_event(state, 0, 1, interactions, Hon, Hoff, _recorder, debug=debug, find_next_flip=False)
    else:
        evolve_event(state, 0, 0, interactions, Hon, Hoff, _recorder, debug=debug, find_next_flip=False)
    history[0, :] = state
    
    for i in range(max_cycles):
        if not (i == 0 and H_init > 0):
            evolve_event(state, amplitude, 1, interactions, Hon, Hoff, 
                         _recorder, debug=debug, find_next_flip=False)
        evolve_event(state, -amplitude, 0, interactions, Hon, Hoff, 
                     _recorder, debug=debug, find_next_flip=False)
        evolve_event(state, 0.0, 1, interactions, Hon, Hoff, _recorder,
                     debug=debug, find_next_flip=False)
        history[i + 1, :] = state
        
        # Check for a repeat
        if periodicity == 0:
            for t in range(i + 1):
                if np.all(state == history[t, :]):
                    periodicity = i + 1 - t
                    if terminate_on_repeat:
                        return periodicity
    
    return periodicity


def run_event_extended(state, interactions, Hon, Hoff, amplitude, 
                       recorder=None,
                       H_init=-np.inf, debug=False,
                       terminate_on_repeat=True):
    """Simulate system under oscillatory driving.
    
    "state" is the initial state. "H_init" is the initial field value.
    
    "interactions" is the J_ij matrix, dimensions N x N where N is the number
    of hysterons. The effective local value of the field at each hysteron is
    computed as state * interactions + H, where H is the external field.
    
    "Hon" and "Hoff" are vectors of length N, with Hon >= Hoff.
    
    "amplitude" is the amplitude of cyclic driving in H. The simulation begins
    by ramping the field from "H_init" to +amplitude.
    
    "recorder" is a Recorder object, if recording of data is desired.
    
    "debug" gets evolve_event() to print lots of information.
    
    If "terminate_on_repeat", stop the simulation as soon as a previous state 
    is revisited. Otherwise, run for 2**N + 1 cycles.
    
    Returns the observed period of the limit cycle.
    """
    return run_event_extended_fast(state, interactions, 
                                   Hon, Hoff, amplitude, 
                                   recorder=recorder,
                                   H_init=H_init, debug=debug,
                                   terminate_on_repeat=terminate_on_repeat)


def try_params(interactions, Hon, Hoff, amplitude=1.0, debug=False):
    """Wrapper around run_event_extended() to check for interesting behavior. 
    
    Returns the period of the limit cycle, or -1 if an error was
    encountered.
    """
    N = len(Hon)
    state = np.ones(N) * -1.0
    
    try:
        periodicity = run_event_extended_fast(state, interactions, 
                                              Hon, Hoff, amplitude,
                                              recorder=None, debug=debug,
                                              terminate_on_repeat=True)
    except StabilityError:
        return -1
    
    return periodicity


@numba.njit
def uniform(a, b, N=1):
    """N numbers in a uniform distribution between a and b."""
    return np.random.random(N) * (b - a) + a


@numba.njit
def uniform1(a, b):
    """One number in a uniform distribution between a and b."""
    return np.random.random() * (b - a) + a


def states_from_traces(traces, N, shorten=True):
    """Returns a list of state vectors.
    
    If 'shorten', remove consecutive duplicates."""
    states = [np.array([traces['s%i' % i][t] for i in range(N)]) for 
              t in traces.index]
    if not shorten:
        return states
    else:
        state_trajectory = states[:1]
        for state in states:
            if not np.all(state == state_trajectory[-1]):
                state_trajectory.append(state)
        return state_trajectory
    
    
@numba.njit
def _unique_finite(values):
    """Obtain the set of finite, unique values from "values".
    
    Values must be 1D and sorted.
    
    Returns r, rp, where "r" is the same length as "values", of which
    the first "rp" entries are valid.
    """
    r = np.empty_like(values)
    r[0] = values[0]
    rp = 0  # pointer to the most recent value added
    for i in range(1, len(values)):
        if values[i] != r[rp]:
            if not np.isfinite(values[i]):
                break
            else:
                rp += 1
                r[rp] = values[i]
    return r, rp + 1


def compute_possible_states(N):
    """Return all possible states of N hysterons.
    
    Returns a 2^N x N array.
    """
    return np.array([tup for tup in itertools.product((-1.0, 1.0), 
                                                       repeat=N)])


@numba.njit
def possible_amplitudes(interactions, Hon, Hoff, possible_states):
    """Return the set of distinctive field amplitudes for a system.
    
    These can be used to generate all possible trajectories of the system
    under symmetric driving (excluing the one with no hysteron flips at all).
    
    "possible_states" is an MxN array containing all M possible states of
    N hysterons. It can be found with compute_possible_states()
    
    This finds the midpoints halfway between the absolute values of field 
    at which hysterons can switch.
    """
    H_flips = np.empty((len(possible_states), 2))
    recorder = Recorder(0)
    for i in range(len(possible_states)):
        for H_target, rising in [(np.inf, 1), (-np.inf, 0)]:
            H_flip = evolve_event(possible_states[i], H_target, rising,
                                  interactions, Hon, Hoff, 
                                  recorder=recorder, 
                                  debug=False, find_next_flip=True)
            H_flips[i, rising] = H_flip
    r, rc = _unique_finite(np.sort(np.abs(H_flips.flatten())))
    H_flips = r[:rc]
    return (H_flips[1:] + H_flips[:-1]) / 2


@numba.jit(nopython=True)
def amplitude_sweep(interactions, Hon, Hoff, possible_states):
    """Find the minimum amplitude that causes period > 1.
    
    Returns the observed period and the amplitude.
    
    If the returned period is 1, no period > 1 was found.
    """
    N = len(Hon)
    recorder = Recorder(0)
    pa = possible_amplitudes(interactions, Hon, Hoff, possible_states)
    state = np.empty_like(Hon)
    for H_trial in pa[1:]:
        for i in range(N):
            state[i] = -1.0
        periodicity = run_event_extended_fast(state, interactions, 
                                              Hon, Hoff, H_trial,
                                              recorder=recorder,
                                              H_init=-np.inf, debug=False,
                                              terminate_on_repeat=True)
        if periodicity > 1:
            break
    return periodicity, H_trial


