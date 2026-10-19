"""Offline route planning with checked home-return reservations.

An action selects robot, one of its two nearest reachable pending beads, and
direction. A robot's provisional return can be replaced by another bead; its
completed work is immutable. All other robots' reservations are respected.
"""
from __future__ import annotations
import copy
from collections import OrderedDict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from run_pipeline import AssignmentEnv
from waam_validator import validate_trajectory_set, compute_reach_metrics
from waam_validator.collision.simulator import run_collision_analysis

SLOTS = 2
ACTIONS = 3*SLOTS*2
OBS_SIZE = 4+3*5+ACTIONS*13
CONTRACT = dict(id='waam-route-reservations-v2', observation_size=OBS_SIZE,
                action_count=ACTIONS, slots_per_robot=SLOTS,
                actions='(robot*2+nearest_reachable_slot)*2+reverse; offline rescheduling of provisional home return',
                observation='progress/layer/makespan/barrier; robot XYZ/work-finish/return-finish; candidate mask/startXYZ/endXYZ/departure/work-finish/makespan/task-fraction/length/layer',
                normalization='XYZ and length / scenario coord_scale, times / serial worst-case time_scale',
                reward='negative change in makespan / serial worst-case time_scale',
                layer_rule='all reserved returns complete before next layer',
                safety='official trajectory/reach/sampled collision checks of current-layer reservations per candidate; complete home barriers between layers; full validator on export')

class RoutePPOEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, job, tasks, reach_margin_mm=1.0, cache_limit=64):
        super().__init__()
        if not np.isfinite(reach_margin_mm) or reach_margin_mm < 0:
            raise ValueError('Invalid reach margin')
        self.backend = AssignmentEnv(job,tasks)
        self.tasks = self.backend.tasks
        self.margin = float(reach_margin_mm)
        self.starts = np.asarray([t.start for t in tasks],float)
        self.ends = np.asarray([t.end for t in tasks],float)
        self.layers = self.starts[:,2]
        self.eligible = np.zeros((len(tasks),3),bool)
        for r,cfg in enumerate(self.backend.robots):
            radius=cfg.xy_reach_radius_mm-self.margin+1e-7
            self.eligible[:,r] = ((np.linalg.norm(self.starts[:,:2]-self.backend.bases[r,:2],axis=1)<=radius)&
                                  (np.linalg.norm(self.ends[:,:2]-self.backend.bases[r,:2],axis=1)<=radius))
        if not self.eligible.any(axis=1).all():
            raise ValueError('At least one task has no reachable robot')
        self.action_space=spaces.Discrete(ACTIONS)
        self.observation_space=spaces.Box(-np.inf,np.inf,(OBS_SIZE,),dtype=np.float32)
        self.cache_limit=cache_limit
        self.cache=OrderedDict()
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pending=np.ones(len(self.tasks),bool)
        self.rows=[[[0.,*h,'W']] for h in self.backend.homes]
        self.work=copy.deepcopy(self.rows)
        self.completed_rows=copy.deepcopy(self.rows)
        self.z=float(self.layers.min())
        self.barrier=0.
        self.makespan=0.
        self.done=False
        self.trace=[]
        self.history=()
        self.candidates=None
        self._prepare()
        return self._observation(),{}

    def _extend(self, prefix, start, end, departure, robot):
        work=copy.deepcopy(prefix)
        current=np.asarray(work[-1][1:4],float)
        if departure>work[-1][0]+1e-9:
            work.append([departure,*current,'W'])
        process=self.backend.config.process
        for point,mode,speed in ((start,'T',process.travel_speed_mm_s),(end,'D',process.deposition_speed_mm_s)):
            distance=float(np.linalg.norm(np.asarray(point)-np.asarray(work[-1][1:4])))
            if distance>1e-9:
                work[-1][4]=mode
                work.append([work[-1][0]+distance/speed,*point,'W'])
        reserved=copy.deepcopy(work)
        home=self.backend.homes[robot]
        distance=float(np.linalg.norm(np.asarray(reserved[-1][1:4])-home))
        if distance>1e-9:
            reserved[-1][4]='T'
            reserved.append([reserved[-1][0]+distance/process.travel_speed_mm_s,*home,'W'])
        return work,reserved

    def _candidate(self, robot, index, reverse):
        start,end=(self.ends[index],self.starts[index]) if reverse else (self.starts[index],self.ends[index])
        prefix=self.work[robot]
        earliest=prefix[-1][0]
        local_makespan=self.makespan-self.barrier
        # A bounded event search; failure retries after every reserved return.
        events=sorted({earliest,max(earliest,local_makespan),
                       *(s[-1][0] for s in self.rows if s[-1][0]>=earliest)})
        for use_home in (False,True):
            route_prefix=self.rows[robot] if use_home else prefix
            departures=[local_makespan] if use_home else events
            for departure in departures:
                work,reserved=self._extend(route_prefix,start,end,departure,robot)
                rows=list(self.rows)
                rows[robot]=reserved
                trajectory=self.backend._trajectory(rows)
                if (validate_trajectory_set(trajectory,self.backend.config).violations or
                    not compute_reach_metrics(trajectory,self.backend.config).passed or
                    run_collision_analysis(trajectory,self.backend.config).events):
                    continue
                return dict(task=int(index),robot=robot,reverse=bool(reverse),start=start,end=end,
                            departure=departure,work=work,rows=rows,
                            makespan=max(r[-1][0] for r in rows),home_fallback=use_home)
        return None

    def _prepare(self):
        if self.done:
            self.candidates=[None]*ACTIONS
            return
        key=(self.z,self.history)
        if key in self.cache:
            self.candidates=self.cache[key]
            self.cache.move_to_end(key)
            return
        candidates=[None]*ACTIONS
        for robot in range(3):
            indices=np.flatnonzero(self.pending & (self.layers==self.z) & self.eligible[:,robot])
            if not len(indices):
                continue
            current=np.asarray(self.work[robot][-1][1:4],float)
            distances=np.minimum(np.linalg.norm(self.starts[indices]-current,axis=1),
                                 np.linalg.norm(self.ends[indices]-current,axis=1))
            order=indices[np.lexsort((indices,distances))]
            # Fill slots with feasible beads, scanning farther if a near bead is blocked.
            slot=0
            for index in order:
                pair=[self._candidate(robot,int(index),reverse) for reverse in (False,True)]
                if not any(pair):
                    continue
                offset=(robot*SLOTS+slot)*2
                candidates[offset:offset+2]=pair
                slot+=1
                if slot==SLOTS:
                    break
        if not any(candidates):
            raise RuntimeError(f'No safe route at layer {self.z}; remaining={self.pending.sum()}')
        self.candidates=candidates
        if self.cache_limit:
            self.cache[key]=candidates
            while len(self.cache)>self.cache_limit:
                self.cache.popitem(last=False)

    def action_masks(self):
        return np.asarray([c is not None for c in self.candidates],bool)

    def _observation(self):
        if self.done:
            return np.zeros(OBS_SIZE,np.float32)
        cs,ts=self.backend.coord_scale,self.backend.time_scale
        values=[1-self.pending.mean(),self.z/cs,self.makespan/ts,self.barrier/ts]
        for robot in range(3):
            values.extend([*(np.asarray(self.work[robot][-1][1:4])/cs),(self.barrier+self.work[robot][-1][0])/ts,(self.barrier+self.rows[robot][-1][0])/ts])
        for c in self.candidates:
            if c is None:
                values.extend([0.]*13)
            else:
                values.extend([1.,*(c['start']/cs),*(c['end']/cs),(self.barrier+c['departure'])/ts,
                               (self.barrier+c['work'][-1][0])/ts,(self.barrier+c['makespan'])/ts,c['task']/len(self.tasks),
                               np.linalg.norm(c['end']-c['start'])/cs,self.z/cs])
        return np.asarray(values,np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode ended')
        if not self.action_space.contains(action) or not self.action_masks()[int(action)]:
            raise ValueError('Invalid masked route action')
        c=self.candidates[int(action)]
        previous=self.makespan
        self.rows=c['rows']
        self.work=list(self.work)
        self.work[c['robot']]=c['work']
        self.makespan=self.barrier+c['makespan']
        self.pending=self.pending.copy()
        self.pending[c['task']]=False
        self.history=(*self.history,int(action))
        self.trace.append(dict(task=c['task'],robot_id=c['robot']+1,reverse=c['reverse'],
                               departure_s=self.barrier+c['departure'],work_finish_s=self.barrier+c['work'][-1][0],
                               makespan_s=self.makespan,home_fallback=c['home_fallback']))
        self.done=not self.pending.any()
        if not (self.pending & (self.layers==self.z)).any():
            self._finish_layer()
            if not self.done:
                self.z=float(self.layers[self.pending].min())
                self.barrier=self.makespan
                self.rows=[[[0.,*h,'W']] for h in self.backend.homes]
                self.work=copy.deepcopy(self.rows)
                self.history=()
        self._prepare()
        return self._observation(),-(self.makespan-previous)/self.backend.time_scale,self.done,False,{'makespan_s':self.makespan}

    def greedy_action(self):
        return min((c['makespan'],c['work'][-1][0],i) for i,c in enumerate(self.candidates) if c is not None)[2]

    def _finish_layer(self):
        for robot,source in enumerate(self.rows):
            own=self.completed_rows[robot]
            if self.barrier>own[-1][0]:
                own.append([self.barrier,*self.backend.homes[robot],'W'])
            own[-1][4]=source[0][4]
            own.extend([[r[0]+self.barrier,*r[1:]] for r in source[1:]])
            if self.makespan>own[-1][0]:
                own.append([self.makespan,*self.backend.homes[robot],'W'])

    def export(self, out):
        if not self.done:
            raise RuntimeError('Incomplete route cannot be exported')
        self.backend.rows=self.completed_rows
        self.backend.index=len(self.tasks)
        self.backend.makespan=self.makespan
        return self.backend.export(out)

    def close(self):
        self.cache.clear()
        self.backend.close()
