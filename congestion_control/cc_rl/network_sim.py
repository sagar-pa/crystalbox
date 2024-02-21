# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import List, Tuple, Callable, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import heapq
import json
from cc_rl import sender_obs
from cc_rl.sampler import Sampler
from cc_rl.utils import SharedTestingData, format_float
import csv
import ray

MAX_CWND = 5000
MIN_CWND = 4

MAX_RATE = 1000
MIN_RATE = 0.25

REWARD_SCALE = 1e-3

MAX_STEPS = 100

EVENT_TYPE_SEND = 'S'
EVENT_TYPE_ACK = 'A'

BYTES_PER_PACKET = 1500

LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0
DELTA_SCALE = 1./100 # percent to decimal

USE_LATENCY_NOISE = True
MAX_LATENCY_NOISE = 1.05

USE_CWND = False
STARTING_RATE = 0.25 #unused; slow start ignored, come back to slow start

class Link:

    def __init__(self, bandwidth: float, 
            delay: float, queue_size: int, 
            loss_rate: float, np_random: np.random):
        self.bw = float(bandwidth)
        self.dl = delay
        self.lr = loss_rate
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.max_queue_delay = queue_size / self.bw
        self.np_random = np_random

    def get_cur_queue_delay(self, event_time) -> float:
        return max(0.0, self.queue_delay - (event_time - self.queue_delay_update_time))

    def get_cur_latency(self, event_time):
        return self.dl + self.get_cur_queue_delay(event_time)

    def packet_enters_link(self, event_time) -> bool:
        if self.np_random.random() < self.lr:
            return False
        self.queue_delay = self.get_cur_queue_delay(event_time)
        self.queue_delay_update_time = event_time
        extra_delay = 1.0 / self.bw
        #print("Extra delay: %f, Current delay: %f, Max delay: %f" % (extra_delay, self.queue_delay, self.max_queue_delay))
        if extra_delay + self.queue_delay > self.max_queue_delay:
            #print("\tDrop!")
            return False
        self.queue_delay += extra_delay
        #print("\tNew delay = %f" % self.queue_delay)
        return True

    def print_debug(self) -> None:
        print("Link:")
        print("Bandwidth: %f" % self.bw)
        print("Delay: %f" % self.dl)
        print("Queue Delay: %f" % self.queue_delay)
        print("Max Queue Delay: %f" % self.max_queue_delay)
        print("One Packet Queue Delay: %f" % (1.0 / self.bw))

    def reset(self) -> None:
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0


class Sender:
    
    def __init__(self, rate: float, path: List[Link], dest: int, 
            features: List[str], cwnd: int = 25, history_len: int = 10):
        self.id = Sender._get_next_id()
        self.starting_rate = rate
        self.rate = rate
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = []
        self.sample_time = []
        self.net = None
        self.path = path
        self.dest = dest
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.cwnd = cwnd

    _next_id = 1
    def _get_next_id() -> int:
        result = Sender._next_id
        Sender._next_id += 1
        return result

    def apply_rate_delta(self, delta: float) -> None:
        delta *= DELTA_SCALE
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))

    def apply_cwnd_delta(self, delta: float) -> None:
        delta *= DELTA_SCALE
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_cwnd(self.cwnd * (1.0 + delta))
        else:
            self.set_cwnd(self.cwnd / (1.0 - delta))

    def can_send_packet(self) -> bool:
        if USE_CWND:
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        else:
            return True

    def register_network(self, net) -> None:
        self.net = net

    def on_packet_sent(self) -> None:
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET

    def on_packet_acked(self, rtt: float) -> None:
        self.acked += 1
        self.rtt_samples.append(rtt)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET

    def on_packet_lost(self) -> None:
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET

    def set_rate(self, new_rate: float) -> None:
        self.rate = new_rate
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.rate > MAX_RATE:
            self.rate = MAX_RATE
        if self.rate < MIN_RATE:
            self.rate = MIN_RATE

    def set_cwnd(self, new_cwnd: float) -> None:
        self.cwnd = int(new_cwnd)
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.cwnd > MAX_CWND:
            self.cwnd = MAX_CWND
        if self.cwnd < MIN_CWND:
            self.cwnd = MIN_CWND

    def record_run(self) -> None:
        smi = self.get_run_data()
        self.history.step(smi)

    def get_obs(self) -> np.ndarray:
        return self.history.as_array()

    def get_run_data(self) -> sender_obs.SenderMonitorInterval:
        obs_end_time = self.net.get_cur_time()
        
        #obs_dur = obs_end_time - self.obs_start_time
        #print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        #print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        #print("self.rate = %f" % self.rate)

        return sender_obs.SenderMonitorInterval(
            self.id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            recv_start=self.obs_start_time,
            recv_end=obs_end_time,
            rtt_samples=self.rtt_samples,
            packet_size=BYTES_PER_PACKET
        )

    def reset_obs(self) -> None:
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.rtt_samples = []
        self.obs_start_time = self.net.get_cur_time()

    def print_debug(self) -> None:
        print("Sender:")
        print("Obs: %s" % str(self.get_obs()))
        print("Rate: %f" % self.rate)
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self) -> None:
        #print("Resetting sender!")
        self.rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)

class Network:
    
    def __init__(self, senders: List[Sender], links: List[Link], np_random: np.random):
        self.q = []
        self.cur_time = 0.0
        self.senders = senders
        self.links = links
        self.queue_initial_packets()
        self.np_random = np_random

    def queue_initial_packets(self) -> None:
        for sender in self.senders:
            sender.register_network(self)
            sender.reset_obs()
            heapq.heappush(self.q, (1.0 / sender.rate, sender, EVENT_TYPE_SEND, 0, 0.0, False)) 

    def reset(self) -> None:
        self.cur_time = 0.0
        self.q = []
        [link.reset() for link in self.links]
        [sender.reset() for sender in self.senders]
        self.queue_initial_packets()

    def get_cur_time(self) -> float:
        return self.cur_time

    def run_for_dur(self, dur) -> Tuple[float, float, float, float]:
        end_time = self.cur_time + dur
        for sender in self.senders:
            sender.reset_obs()

        while self.cur_time < end_time:
            event_time, sender, event_type, next_hop, cur_latency, dropped = heapq.heappop(self.q)
            #print("Got event %s, to link %d, latency %f at time %f" % (event_type, next_hop, cur_latency, event_time))
            self.cur_time = event_time
            new_event_time = event_time
            new_event_type = event_type
            new_next_hop = next_hop
            new_latency = cur_latency
            new_dropped = dropped
            push_new_event = False

            if event_type == EVENT_TYPE_ACK:
                if next_hop == len(sender.path):
                    if dropped:
                        sender.on_packet_lost()
                        #print("Packet lost at time %f" % self.cur_time)
                    else:
                        sender.on_packet_acked(cur_latency)
                        #print("Packet acked at time %f" % self.cur_time)
                else:
                    new_next_hop = next_hop + 1
                    link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                    if USE_LATENCY_NOISE:
                        noise = np.clip(1 + self.np_random.normal(0,.05), 0.99, MAX_LATENCY_NOISE)
                        link_latency *= noise
                    new_latency += link_latency
                    new_event_time += link_latency
                    push_new_event = True
            if event_type == EVENT_TYPE_SEND:
                if next_hop == 0:
                    #print("Packet sent at time %f" % self.cur_time)
                    if sender.can_send_packet():
                        sender.on_packet_sent()
                        push_new_event = True
                    heapq.heappush(self.q, (self.cur_time + (1.0 / sender.rate), sender, EVENT_TYPE_SEND, 0, 0.0, False))
                
                else:
                    push_new_event = True

                if next_hop == sender.dest:
                    new_event_type = EVENT_TYPE_ACK
                new_next_hop = next_hop + 1
                
                link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                if USE_LATENCY_NOISE:
                    noise = np.clip(1 + self.np_random.normal(0,.05), 0.99, MAX_LATENCY_NOISE)
                    link_latency *= noise
                new_latency += link_latency
                new_event_time += link_latency
                new_dropped = not sender.path[next_hop].packet_enters_link(self.cur_time)
                   
            if push_new_event:
                heapq.heappush(self.q, (new_event_time, sender, new_event_type, new_next_hop, new_latency, new_dropped))

        sender_mi = self.senders[0].get_run_data()
        throughput = sender_mi.get("recv rate")
        latency = sender_mi.get("avg latency")
        loss = sender_mi.get("loss ratio")
        #bw_cutoff = self.links[0].bw * 0.8
        #lat_cutoff = 2.0 * self.links[0].dl * 1.5
        #loss_cutoff = 2.0 * self.links[0].lr * 1.5
        #print("thpt %f, bw %f" % (throughput, bw_cutoff))
        #reward = 0 if (loss > 0.1 or throughput < bw_cutoff or latency > lat_cutoff or loss > loss_cutoff) else 1 #
        
        # Super high throughput
        #reward = REWARD_SCALE * (20.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        
        # Very high thpt
        reward = 10.0 * throughput / (8 * BYTES_PER_PACKET) - 1e3 * latency - 2e3 * loss
        
        # High thpt
        #reward = REWARD_SCALE * (5.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        
        # Low latency
        #reward = REWARD_SCALE * (2.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        #if reward > 857:
        #print("Reward = %f, thpt = %f, lat = %f, loss = %f" % (reward, throughput, latency, loss))
        
        #reward = (throughput / RATE_OBS_SCALE) * np.exp(-1 * (LATENCY_PENALTY * latency / LAT_OBS_SCALE + LOSS_PENALTY * loss))
        return reward, throughput / 1e6, latency, loss


class SimulatedNetworkEnv(gym.Env):
    
    def __init__(self,
                 history_len: int = 10,
                 features: List[str] = ["send rate",  "recv rate", "avg latency", "loss ratio", 
                    "latency ratio", "send ratio",  
                    "sent latency inflation"],
                    seed = None, sampler_kwargs: dict = None,
                    test: bool = False, is_action_continuous: bool = True, 
                    use_log: bool = False, log_dir: Path = None,
                    shared_testing_data_name: str = None):

        self.sampler = None
        self.links = None
        self.senders = None
        self.net = None # Initialize to none for seeding

        self.use_log = use_log
        self.log_dir = log_dir
        self.log = []
        self.trace_idx = None
        if shared_testing_data_name is None:
            self.shared_testing_data = None
        else:
            self.shared_testing_data = ray.get_actor(shared_testing_data_name, namespace="test")

        super().reset(seed=seed)
        self.set_seed()

        if sampler_kwargs is None:
            sampler_kwargs = {}
        sampler_kwargs["np_random"] = self.np_random
        sampler_kwargs["shared_testing_data_name"] = shared_testing_data_name
        self.test = test
        self.sampler = Sampler(**sampler_kwargs)
        sampling_func = sampler_kwargs.get("sampling_func_cls", None)
        if test or sampling_func == "iterative":
            self.sampling_func = self.sampler.iterative_sample
        else:
            self.sampling_func = self.sampler.sample
        self.history_len = history_len
        self.features = features
        
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links, self.np_random)
        self.run_dur = None
        self.steps_taken = 0
        self.max_steps = MAX_STEPS

        if is_action_continuous:
            self.action_space = spaces.Box(low=-1., high=1.0, shape=(1,), dtype=np.float64)
        else:
            self.action_space = spaces.Discrete(5)
        self.action_lookup = self.create_action_lookup(is_action_continuous)


        self.observation_space = None
        self.obs_min_vec = self.normalize_obs(sender_obs.get_min_obs_vector(self.features), clip=False, scale=False)
        self.obs_max_vec = self.normalize_obs(sender_obs.get_max_obs_vector(self.features), clip=False, scale=False)
        self.observation_space = spaces.Box(np.tile(self.obs_min_vec, (self.history_len, 1)),
                                            np.tile(self.obs_max_vec, (self.history_len, 1)),
                                            dtype=np.float64)

    def change_log_dir(self, log_dir: Union[str, Path], 
            discard_logged_data: bool = True) -> None:
        self.log_dir = log_dir
        if discard_logged_data:
            self.discard_log()

    def discard_log(self) -> None:
        self.log = []

    def setup_log(self, log_file: Union[str, Path], force: bool = False
            ) -> Path:
        log_file = Path(log_file)
        log_file.parent.mkdir(exist_ok=True, parents=True)
        if log_file.exists() and not force:
            raise ValueError("File exists, and force is set to False.")
        return log_file

    def log_episode(self) -> None:
        if len(self.log) > 0:
            if self.shared_testing_data is None:
                progress = 100
            else:
                progress = ray.get(self.shared_testing_data.get_train_progress.remote())
            log_dir = Path(self.log_dir).resolve() / str(progress)
            idx = self.log[-1][0]
            log_file = log_dir / f"test_{idx}.csv"
            columns = ["episode", "step", "prev_sending_rate", 
                "new_sending_rate", "throughput", "latency", "loss", "reward"]
            try:
                log_file = self.setup_log(log_file, force=False)
                with open(log_file, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
                    writer.writerows(self.log)
            except ValueError:
                pass
            self.discard_log()


    def create_action_lookup(self, is_action_continuous: bool) -> Callable:
        if is_action_continuous:
            low, high  = -25, 25
            if USE_CWND:
                def action_lookup(action):
                    # action is in [-1, 1]
                    action = float(action)
                    action = low + (0.5 * (action + 1.0) * (high - low))
                    action = (0, action)
                    return action
            else:
                def action_lookup(action):
                    # action is in [-1, 1]
                    action = float(action)
                    action = low + (0.5 * (action + 1.0) * (high - low))
                    action = (action, 0)
                    return action
        else:
            discrete_lookup = {
                    0: -25,
                    1: -5,
                    2: 0, 
                    3: 5,
                    4: 25
                }
            if USE_CWND:
                def action_lookup(action):
                    action = discrete_lookup[action]
                    action = (0, action)
                    return action
            else:
                def action_lookup(action):
                    action = discrete_lookup[action]
                    action = (action, 0)
                    return action
        return action_lookup

    def set_seed(self) -> None:
        if self.sampler is not None:
            self.sampler.np_random = self.np_random
        if self.links is not None:
            for link in self.links:
                link.np_random = self.np_random
        if self.net is not None:
            self.net.np_random = self.np_random


    def _get_all_sender_obs(self):
        sender_obs = self.senders[0].get_obs()
        sender_obs = np.array(sender_obs)
        #print(sender_obs)
        return sender_obs

    def normalize_obs(self, obs: np.ndarray, 
                            clip: bool = True, scale: bool = True) -> np.ndarray:
        #obs = np.sign(obs) * np.sqrt(np.abs(obs))
        if clip: #Just to make sure
            obs = np.clip(obs, self.obs_min_vec, self.obs_max_vec)
        if scale:
            obs = (obs - self.obs_min_vec) / (self.obs_max_vec - self.obs_min_vec)
        return obs

    def scale_rewards(self, reward: float, reward_scale: float = REWARD_SCALE) -> float:
        reward *= reward_scale
        #reward =  np.sign(reward) * ((np.sqrt(np.abs(reward) + 1) - 1) + 0.001 * reward)
        return reward

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        #print("Actions: %s" % str(actions))
        #print(actions)
        previous_sending_rate = self.senders[0].rate
        for sender in self.senders:
            rate_delta, cwnd = self.action_lookup(action)
            sender.apply_rate_delta(rate_delta)
            if USE_CWND:
                sender.apply_cwnd_delta(cwnd)
        new_sending_rate = self.senders[0].rate
        #print("Running for %fs" % self.run_dur)
        reward, throughput, latency, loss = self.net.run_for_dur(self.run_dur)
        for sender in self.senders:
            sender.record_run()
        self.steps_taken += 1
        sender_obs = self._get_all_sender_obs()
        sender_mi = self.senders[0].get_run_data()
        event = {}
        event["Name"] = "Step"
        event["Time"] = self.steps_taken
        event["Reward"] = reward
        #event["Target Rate"] = sender_mi.target_rate
        event["Send Rate"] = sender_mi.get("send rate")
        event["Throughput"] = sender_mi.get("recv rate")
        event["Latency"] = sender_mi.get("avg latency")
        event["Loss Rate"] = sender_mi.get("loss ratio")
        event["Latency Inflation"] = sender_mi.get("sent latency inflation")
        event["Latency Ratio"] = sender_mi.get("latency ratio")
        event["Send Ratio"] = sender_mi.get("send ratio")
        #event["Cwnd"] = sender_mi.cwnd
        #event["Cwnd Used"] = sender_mi.cwnd_used
        if event["Latency"] > 0.0:
            self.run_dur = 1.0 * sender_mi.get("avg latency")
        #print("Sender obs: %s" % sender_obs)

        sender_obs = self.normalize_obs(sender_obs)
        reward = self.scale_rewards(reward)
        self.sampler.record_action_reward(action = action, reward = reward, 
            throughput = throughput, latency = latency, loss = loss)

        info = {
            "throughput": throughput,
            "latency": latency,
            "loss": loss,
            "trace_idx" : self.trace_idx,
            "previous_sending_rate": previous_sending_rate,
            "sending_rate": new_sending_rate
        }

        done = self.steps_taken >= self.max_steps

        if self.use_log:
            row = [self.trace_idx, 
                self.steps_taken, 
                format_float(previous_sending_rate),
                format_float(new_sending_rate),
                format_float(throughput),
                format_float(latency),
                format_float(loss),
                format_float(reward, 6)]
            if len(self.log) > 1:
                if self.log[-1][0] != row[0]:
                    raise ValueError("Trace Changed while recording logs!")
            self.log.append(row)    
            if done:
                self.log_episode()

        return sender_obs, reward, done, False, info

    def print_debug(self) -> None:
        print("---Link Debug---")
        for link in self.links:
            link.print_debug()
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()

    def set_sending_rate(self, rate: float, should_step: bool = True) -> None:
        """
        Set a new sending rate, for debugging
        Args:
            rate: The rate (in pps) to set the current rate to.
            should_step: Whether or not the simulated network should take a step
                sending at the specified rate
        """
        self.senders[0].set_rate(rate)
        if should_step:
            self.step(action=2)

    def set_sampler_attr(self, attr, value) -> None:
        setattr(self.sampler, attr, value)


    def create_new_links_and_senders(self) -> None:
        (bw, lat, queue, loss), trace_idx = self.sampling_func()
        self.trace_idx = trace_idx
        bw_noise = np.clip(1 + self.np_random.normal(0,.25), 0.3, 1.6)
        starting_rate = bw_noise * bw
        self.links = [Link(bw, lat, queue, loss, self.np_random), Link(bw, lat, queue, loss, self.np_random)]
        #self.senders = [Sender(0.3 * bw, [self.links[0], self.links[1]], 0, self.history_len)]
        #self.senders = [Sender(random.uniform(0.2, 0.7) * bw, [self.links[0], self.links[1]], 0, self.history_len)]
        self.senders = [Sender(starting_rate, [self.links[0], self.links[1]], 0, self.features, history_len=self.history_len)]
        lat_noise = np.clip(1 + self.np_random.normal(0,.025), 1.0, 1.06)
        self.run_dur = lat_noise * lat * 3

    def reset(self, seed: int = None, options = None) -> np.ndarray:
        super().reset(seed=seed)
        self.set_seed()
        if self.trace_idx is not None:
            self.sampler.record_episode(idx = self.trace_idx)
        self.steps_taken = 0
        self.discard_log()
        self.net.reset()
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links, self.np_random)
        self.net.run_for_dur(self.run_dur) # send
        self.net.run_for_dur(self.run_dur) # acks
        return self.normalize_obs(self._get_all_sender_obs()), {
                                                    "trace_idx": self.trace_idx}



