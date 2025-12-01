import numpy as np
import random
import copy
import math
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# LGP System Imports
import dispatching_rules  # auto-register DR
import metaheuristics_impl  # auto-register MH
from lgp_actions import ActionIndividual, ActionLGP, Gene
from typed_action_adapter import run_action_individual

machine_pool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15]

# Danh sách các job ban đầu (jobs_initial)
jobs_initial = {
    1: [
        {'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12}
    ],
    2: [
        {'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12}
    ],
    3: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 1},
        {'op_id': 2, 'candidate_machines': [6], 'processing_time': 8},
        {'op_id': 3, 'candidate_machines': [6], 'processing_time': 8}
    ],
    4: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 7}
    ],
    5: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 1}
    ],
    6: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 1},
        {'op_id': 2, 'candidate_machines': [6], 'processing_time': 8},
        {'op_id': 3, 'candidate_machines': [6], 'processing_time': 8}
    ],
    7: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 7}
    ],
    8: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 1},
        {'op_id': 2, 'candidate_machines': [6], 'processing_time': 8},
        {'op_id': 3, 'candidate_machines': [6], 'processing_time': 8}
    ],
    9: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 7}
    ],
    10: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 7}
    ],
    11: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 7}
    ],
    12: [
        {'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12}
    ],
    13: [
        {'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12}
    ],
    14: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 1},
        {'op_id': 2, 'candidate_machines': [6], 'processing_time': 8},
        {'op_id': 3, 'candidate_machines': [6], 'processing_time': 8}
    ],
    15: [
        {'op_id': 1, 'candidate_machines': [3, 4], 'processing_time': 1},
        {'op_id': 2, 'candidate_machines': [7], 'processing_time': 43},
        {'op_id': 3, 'candidate_machines': [5], 'processing_time': 43}
    ],
    16: [
        {'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12},
        {'op_id': 2, 'candidate_machines': [8], 'processing_time': 8}
    ],
    17: [
        {'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12},
        {'op_id': 2, 'candidate_machines': [8], 'processing_time': 12}
    ],
    18: [
        {'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12},
        {'op_id': 2, 'candidate_machines': [8], 'processing_time': 4}
    ],
    19: [
        {'op_id': 1, 'candidate_machines': [8], 'processing_time': 3}
    ],
    20: [
        {'op_id': 1, 'candidate_machines': [1, 2], 'processing_time': 12},
        {'op_id': 2, 'candidate_machines': [12, 13], 'processing_time': 25}
    ]
}

# due_dates_initial: tất cả các job đều có due_date = 1200, sử dụng dict comprehension
due_dates_initial = {i: 1200 for i in range(1, 51)}

# Dummy simulated_annealing: tạo lịch trình ban đầu dưới dạng dictionary.
def simulated_annealing(jobs, due_dates, lambda_tardiness=1.0, **kwargs):
    # Khởi tạo thời gian sẵn sàng của các máy
    machine_ready = {m: 0 for m in machine_pool}
    schedule = {}
    # Với mỗi job, duyệt các operation theo thứ tự
    for job, ops in jobs.items():
        job_ready = 0  # Thời gian job sẵn sàng cho operation tiếp theo
        for i, op in enumerate(ops):
            best_machine = None
            best_start = None
            best_finish = float('inf')
            # Chọn máy trong candidate có thể bắt đầu sớm nhất
            for m in op['candidate_machines']:
                st = max(job_ready, machine_ready[m])
                ft = st + op['processing_time']
                if ft < best_finish:
                    best_finish = ft
                    best_start = st
                    best_machine = m
            schedule[(job, i)] = (best_start, best_finish, best_machine)
            job_ready = best_finish
            machine_ready[best_machine] = best_finish
    makespan = max(ft for (st, ft, m) in schedule.values())
    total_tardiness = sum(max(0, schedule[(job, i)][1] - due_dates[job])
                          for job in jobs for i in range(len(jobs[job])))
    cost = makespan + lambda_tardiness * total_tardiness
    return None, schedule, makespan, total_tardiness, cost, None

# Chuyển schedule dictionary sang list các event
def schedule_dict_to_list(schedule_dict, jobs_info):
    events = []
    for (job, op_index), (s, f, m) in schedule_dict.items():
        op_info = jobs_info[job]['operations'][op_index]
        event = {
            'job': job,
            'op_index': op_index,
            'start': s,
            'finish': f,
            'machine': m,
            'op_id': op_info['op_id'],
            'candidate_machines': op_info['candidate_machines']
        }
        events.append(event)
    events = sorted(events, key=lambda e: (str(e['job']), e['op_index'], e['start']))
    return events

# Hàm tách schedule (list event) theo current_time
def split_schedule_list(event_list, current_time, jobs_info):
    finished_events = []
    unfinished_jobs = {}
    jobs_events = {}
    for event in event_list:
        jobs_events.setdefault(event['job'], []).append(event)
    for job, events in jobs_events.items():
        events = sorted(events, key=lambda e: (e['op_index'], e['start']))
        ops_list = []
        job_ready = None
        for event in events:
            if event['finish'] <= current_time:
                finished_events.append(event)
                job_ready = event['finish']
            elif event['start'] < current_time < event['finish']:
                finished_part = event.copy()
                finished_part['finish'] = current_time
                finished_events.append(finished_part)
                remaining_time = event['finish'] - current_time
                unfinished_op = {
                    'op_index': event['op_index'],
                    'op_id': event['op_id'],
                    'candidate_machines': event['candidate_machines'],
                    'processing_time': remaining_time
                }
                ops_list.append(unfinished_op)
                job_ready = current_time
                total_ops = len(jobs_info[job]['operations'])
                for op_index in range(event['op_index']+1, total_ops):
                    op = jobs_info[job]['operations'][op_index]
                    new_op = {
                        'op_index': op_index,
                        'op_id': op['op_id'],
                        'candidate_machines': op['candidate_machines'],
                        'processing_time': op['processing_time']
                    }
                    ops_list.append(new_op)
                break
            else:
                unfinished_op = {
                    'op_index': event['op_index'],
                    'op_id': event['op_id'],
                    'candidate_machines': event['candidate_machines'],
                    'processing_time': event['finish'] - event['start']
                }
                ops_list.append(unfinished_op)
                if job_ready is None:
                    job_ready = current_time
        if ops_list:
            unfinished_jobs[job] = {
                'job_ready': job_ready,
                'due_date': jobs_info[job]['due_date'],
                'operations': ops_list
            }
    return finished_events, unfinished_jobs

# Các hàm reschedule heuristic cho phần unfinished:
def reschedule_unfinished_jobs_edd(unfinished_jobs, current_time, finished_events, machine_pool):
    # Áp dụng EDD nhưng có thể kết hợp thêm processing time
    sorted_jobs = sorted(unfinished_jobs.items(), key=lambda x: (x[1]['due_date'], sum(op['processing_time'] for op in x[1]['operations'])))
    new_events = []
    machine_ready = {m: current_time for m in machine_pool}
    for job, info in sorted_jobs:
        job_ready = info['job_ready']
        for op in sorted(info['operations'], key=lambda op: op['op_index']):
            pt = op['processing_time']
            best_start = float('inf')
            best_finish = float('inf')
            best_machine = None
            for m in op['candidate_machines']:
                st = max(job_ready, machine_ready.get(m, current_time))
                ft = st + pt
                if ft < best_finish:
                    best_finish = ft
                    best_start = st
                    best_machine = m
            event = {
                'job': job,
                'op_index': op['op_index'],
                'start': best_start,
                'finish': best_finish,
                'machine': best_machine,
                'op_id': op['op_id'],
                'candidate_machines': op['candidate_machines']
            }
            new_events.append(event)
            job_ready = best_finish
            machine_ready[best_machine] = best_finish
    return new_events

def reschedule_unfinished_jobs_sa(unfinished_jobs, current_time, finished_events, machine_pool, iterations=50):
    # Sử dụng SA với cooling schedule động và số iterations cao hơn.
    current_solution = reschedule_unfinished_jobs_edd(unfinished_jobs, current_time, finished_events, machine_pool)
    current_cost = max(e['finish'] for e in (finished_events + current_solution))
    T = 100  # Nhiệt độ khởi đầu
    cooling_rate = 0.95
    best_solution = current_solution
    best_cost = current_cost
    for i in range(iterations):
        # Tạo neighbor bằng cách thay đổi ngẫu nhiên một vài event trong current_solution
        neighbor = copy.deepcopy(current_solution)
        # Ví dụ: thay đổi finish của một event
        if neighbor:
            idx = random.randint(0, len(neighbor)-1)
            neighbor[idx]['finish'] *= random.uniform(1.0, 1.05)
        merged = finished_events + neighbor
        makespan = max(e['finish'] for e in merged) if merged else 0
        new_cost = makespan  # Giả sử tardiness không thay đổi
        if new_cost < best_cost or random.random() < math.exp(-(new_cost - current_cost)/T):
            current_solution = neighbor
            current_cost = new_cost
            if new_cost < best_cost:
                best_solution = neighbor
                best_cost = new_cost
        T *= cooling_rate
    return best_solution

def reschedule_unfinished_jobs_ga(unfinished_jobs, current_time, finished_events, machine_pool, num_candidates=10, generations=5):
    # Khởi tạo quần thể ban đầu từ hàm EDD
    population = [reschedule_unfinished_jobs_edd(unfinished_jobs, current_time, finished_events, machine_pool) for _ in range(num_candidates)]
    def evaluate(solution):
        merged = finished_events + solution
        return max(e['finish'] for e in merged)  # makespan
    for gen in range(generations):
        # Selection: chọn top 50% cá thể tốt nhất
        population = sorted(population, key=evaluate)[:max(1, num_candidates//2)]
        new_population = []
        # Crossover: tạo ra các cá thể mới từ các cặp
        while len(new_population) < num_candidates:
            parent1, parent2 = random.sample(population, 2)
            child = []
            for e1, e2 in zip(parent1, parent2):
                child.append(e1 if random.random() < 0.5 else e2)
            new_population.append(child)
        # Mutation: thay đổi ngẫu nhiên một vài event trong mỗi cá thể
        for solution in new_population:
            if random.random() < 0.3:
                idx = random.randint(0, len(solution)-1)
                solution[idx]['finish'] *= random.uniform(0.95, 1.05)
        population = new_population
    best_solution = min(population, key=evaluate)
    return best_solution

def reschedule_unfinished_jobs_pso(unfinished_jobs, current_time, finished_events, machine_pool, num_particles=10, iterations=20):
    # Định nghĩa hàm cost: ở đây ta sử dụng makespan của lịch trình (merged)
    def cost_function(candidate):
        merged = finished_events + candidate
        return max(e['finish'] for e in merged) if merged else 0

    # Khởi tạo population (các candidate solution) dựa trên kết quả EDD có nhiễu
    particles = []
    velocities = []
    base_candidate = reschedule_unfinished_jobs_edd(unfinished_jobs, current_time, finished_events, machine_pool)
    for i in range(num_particles):
        candidate = copy.deepcopy(base_candidate)
        # Thêm nhiễu cho mỗi candidate
        for event in candidate:
            event['finish'] *= random.uniform(0.95, 1.05)
        particles.append(candidate)
        velocities.append([0]*len(candidate))

    pbest = copy.deepcopy(particles)
    pbest_costs = [cost_function(p) for p in particles]
    gbest = min(particles, key=cost_function)
    gbest_cost = cost_function(gbest)

    w = 0.5    # inertia weight
    c1 = 1.0   # cognitive coefficient
    c2 = 1.0   # social coefficient

    # PSO loop
    for it in range(iterations):
        for i in range(num_particles):
            for j in range(len(particles[i])):
                current_finish = particles[i][j]['finish']
                pbest_finish = pbest[i][j]['finish']
                gbest_finish = gbest[j]['finish']
                r1 = random.random()
                r2 = random.random()
                new_velocity = w * velocities[i][j] + c1 * r1 * (pbest_finish - current_finish) + c2 * r2 * (gbest_finish - current_finish)
                velocities[i][j] = new_velocity
                particles[i][j]['finish'] = current_finish + new_velocity
            cost_candidate = cost_function(particles[i])
            if cost_candidate < pbest_costs[i]:
                pbest[i] = copy.deepcopy(particles[i])
                pbest_costs[i] = cost_candidate
        candidate_costs = [cost_function(p) for p in particles]
        min_cost = min(candidate_costs)
        if min_cost < gbest_cost:
            gbest = copy.deepcopy(particles[candidate_costs.index(min_cost)])
            gbest_cost = min_cost
    return gbest

# Hàm tạo unified job info cho các job ban đầu
def create_unified_jobs_info(jobs_initial, due_dates_initial):
    info = {}
    for job, ops in jobs_initial.items():
        info[job] = {
            'operations': ops,
            'due_date': due_dates_initial[job]
        }
    return info

class DynamicSchedulingEnv(gym.Env):
    def __init__(self, lambda_tardiness=1.0, action_library=None, action_budget_s=1.0):
        super(DynamicSchedulingEnv, self).__init__()
        self.lambda_tardiness = lambda_tardiness
        self.machine_pool = machine_pool
        self.jobs_initial = jobs_initial
        self.due_dates_initial = due_dates_initial
        self.all_jobs_info = create_unified_jobs_info(self.jobs_initial, self.due_dates_initial)
        
        # LGP Integration: action_library & budget
        self.action_library = action_library if action_library is not None else self._default_action_library()
        self.action_budget_s = action_budget_s
        
        # Tạo initial schedule một lần offline
        _, schedule, _, _, _, _ = simulated_annealing(self.jobs_initial, self.due_dates_initial, lambda_tardiness=self.lambda_tardiness)
        self.initial_schedule_events = schedule_dict_to_list(schedule, self.all_jobs_info)
        # Sử dụng initial schedule cố định cho mỗi episode
        self.current_schedule_events = copy.deepcopy(self.initial_schedule_events)
        self.current_time = 0
        self._generate_dynamic_jobs(num_dynamic=2)
        self.current_dynamic_index = 0
        if random.random() < 0.25:
            job_type = "Urgent"
            etuf = 1.2
        else:
            job_type = "Normal"
            etuf = 1.8
        num_ops = random.randint(min_ops, max_ops)
        operations = []
        total_pt = 0
        for i in range(num_ops):
            candidate_machines = random.sample(self.machine_pool, k=random.randint(1, min(5, len(self.machine_pool))))
            pt = random.randint(min_pt, max_pt)
            total_pt += pt
            op = {
                'op_id': i+1,
                'candidate_machines': candidate_machines,
                'processing_time': pt
            }
            operations.append(op)
        due_date = math.ceil(arrival_time + total_pt * etuf)
        dynamic_job = {
            'job_id': job_id,
            'arrival_time': arrival_time,
            'due_date': due_date,
            'operations': operations,
            'job_type': job_type
        }
        return dynamic_job

    # Cập nhật hàm _generate_dynamic_jobs để đảm bảo các dynamic job không có arrival_time chung
    def _generate_dynamic_jobs(self, num_dynamic=4):
        dynamic_jobs_events = []
        Eave = 20  # Giá trị trung bình của interarrival time; bạn có thể điều chỉnh
        # Tính max_finish dựa trên current_schedule_events
        max_finish = max(e['finish'] for e in self.current_schedule_events)
        margin = 5  # Margin đảm bảo dynamic job xuất hiện trước khi hệ thống kết thúc
        T_max = int(max_finish - margin)
        T_min = self.current_time + 5  # dynamic job sẽ không xuất hiện quá sớm
        if T_min >= T_max:
            T_max = T_min + 10

        # Sinh arrival_time cho dynamic job thứ nhất
        arrival_time = self.current_time + int(np.random.exponential(scale=Eave))
        # Đảm bảo dynamic job thứ nhất có arrival_time nằm trong [T_min, T_max - (num_dynamic-1)]
        arrival_time = max(T_min, min(arrival_time, T_max - (num_dynamic - 1)))
        for i in range(num_dynamic):
            if i > 0:
                arrival_time += int(np.random.exponential(scale=Eave))
            # Đảm bảo rằng dynamic job thứ i có arrival_time <= T_max - (num_dynamic - i - 1)
            max_allowed = T_max - (num_dynamic - i - 1)
            if arrival_time > max_allowed:
                arrival_time = max_allowed
            temp_id = "Temp" + str(i+1)
            dyn_job = self._generate_dynamic_job(temp_id, arrival_time)
            dynamic_jobs_events.append((arrival_time, dyn_job))
        dynamic_jobs_events.sort(key=lambda x: x[0])
        for i, (arrival_time, dj) in enumerate(dynamic_jobs_events):
            dj['job_id'] = "D" + str(i+1)
        self.dynamic_jobs_events = dynamic_jobs_events

    def reset(self):
        self.current_time = 0
        self.all_jobs_info = create_unified_jobs_info(self.jobs_initial, self.due_dates_initial)
        self.current_schedule_events = copy.deepcopy(self.initial_schedule_events)
        self._generate_dynamic_jobs(num_dynamic=2)
        self.current_dynamic_index = 0
        return self._get_state()

    def _get_state(self):
        finished_events, unfinished_jobs = split_schedule_list(self.current_schedule_events, self.current_time, self.all_jobs_info)
        num_unfinished = sum(len(info['operations']) for info in unfinished_jobs.values())
        total_pt = 0
        count = 0
        for info in unfinished_jobs.values():
            for op in info['operations']:
                total_pt += op['processing_time']
                count += 1
        avg_pt = total_pt / count if count > 0 else 0
        return np.array([self.current_time, num_unfinished, avg_pt], dtype=np.float32)

    def get_metrics(self):
            # Tính các chỉ số từ current_schedule_events
            merged = self.current_schedule_events
            makespan = max(e['finish'] for e in merged) if merged else 0
            total_tardiness_normal = 0
            total_tardiness_urgent = 0
            for job, info in self.all_jobs_info.items():
                job_events = [e for e in merged if e['job'] == job]
                if job_events:
                    comp_time = max(e['finish'] for e in job_events)
                    tardiness = max(0, comp_time - info['due_date'])
                    if isinstance(job, int):
                        total_tardiness_normal += tardiness
                    else:
                        if info.get('job_type', 'Normal') == 'Urgent':
                            total_tardiness_urgent += tardiness
                        else:
                            total_tardiness_normal += tardiness
            return {"makespan": makespan, "tardiness_normal": total_tardiness_normal, "tardiness_urgent": total_tardiness_urgent}

    def step(self, action):
        if self.current_dynamic_index >= len(self.dynamic_jobs_events):
            return self._get_state(), 0, True, {}
        arrival_time, dyn_job = self.dynamic_jobs_events[self.current_dynamic_index]
        self.current_time = arrival_time

        finished_events, unfinished_jobs = split_schedule_list(self.current_schedule_events, self.current_time, self.all_jobs_info)

        dyn_info = {
            'job_ready': self.current_time,
            'due_date': dyn_job['due_date'],
            'operations': [{'op_index': i, 'op_id': op['op_id'], 'candidate_machines': op['candidate_machines'], 'processing_time': op['processing_time']}
                           for i, op in enumerate(dyn_job['operations'])]
        }
        self.all_jobs_info[dyn_job['job_id']] = dyn_info
        unfinished_jobs[dyn_job['job_id']] = dyn_info

        # LGP Integration: use run_action_individual instead of hardcoded if-else
        individual = self.action_library[action]
        new_unfinished_events = run_action_individual(
            env=self,
            individual=individual,
            finished_events=finished_events,
            unfinished_jobs=unfinished_jobs,
            total_budget_s=self.action_budget_s
        )

        self.current_schedule_events = finished_events + new_unfinished_events

        merged = self.current_schedule_events
        makespan = max(e['finish'] for e in merged) if merged else 0
        total_tardiness_normal = 0
        total_tardiness_urgent = 0
        for job, info in self.all_jobs_info.items():
            job_events = [e for e in merged if e['job'] == job]
            if job_events:
                comp_time = max(e['finish'] for e in job_events)
                tardiness = max(0, comp_time - info['due_date'])
                if isinstance(job, int):
                    total_tardiness_normal += tardiness
                else:
                    if info.get('job_type', 'Normal') == 'Urgent':
                        total_tardiness_urgent += tardiness
                    else:
                        total_tardiness_normal += tardiness
        alpha = 0.75
        cost = makespan
        reward = -cost

        self.current_dynamic_index += 1
        done = self.current_dynamic_index >= len(self.dynamic_jobs_events)
        next_state = self._get_state()
        return next_state, reward, done, {}

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PPOActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)
    def forward(self, x):
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

def select_action(model, state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    logits, value = model(state_tensor)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action), value

def compute_returns(rewards, masks, gamma=0.9):
    returns = []
    R = 0
    for r, mask in zip(reversed(rewards), reversed(masks)):
        R = r + gamma * R * mask
        returns.insert(0, R)
    return returns

def plot_gantt(schedule_events):
    colors = plt.cm.tab20.colors
    job_colors = {}
    def get_job_color(job):
        job = str(job)
        if job not in job_colors:
            index = len(job_colors) % len(colors)
            job_colors[job] = colors[index]
        return job_colors[job]
    machines = {}
    for event in schedule_events:
        m = event['machine']
        machines.setdefault(m, []).append(event)
    machine_ids = sorted(machines.keys())
    fig, ax = plt.subplots(figsize=(16, 10))
    yticks = []
    ytick_labels = []
    bar_height = 0.8
    for i, m in enumerate(machine_ids):
        yticks.append(i)
        ytick_labels.append(f"Machine {m}")
        events = sorted(machines[m], key=lambda e: e['start'])
        for event in events:
            start = event['start']
            finish = event['finish']
            duration = finish - start
            job = event['job']
            color = get_job_color(job)
            ax.barh(i, duration, left=start, height=bar_height, align='center', color=color, edgecolor='black')
            ax.text(start + duration/2, i, f"{job}", color='black', ha='center', va='center', fontsize=10)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=12)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_title("Gantt Chart for Final Episode Schedule", fontsize=16)
    max_finish = max(e['finish'] for e in schedule_events) if schedule_events else 0
    ax.set_xlim(0, max_finish + 10)
    def sort_key(job_id):
        try:
            return (0, int(job_id))
        except ValueError:
            if job_id.startswith("D"):
                return (1, int(job_id[1:]))
            return (1, job_id)
    sorted_jobs = sorted(job_colors.items(), key=lambda item: sort_key(item[0]))
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=f"Job {job}") for job, color in sorted_jobs]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def initialize_lgp_action_library(pool_size=64, dr_list=None, mh_list=None, seed=0):
    """
    Khởi tạo action library cho PPO sử dụng LGP.
    
    Args:
        pool_size: Số lượng action individuals (mặc định 64)
        dr_list: Danh sách dispatching rules (mặc định ["EDD"])
        mh_list: Danh sách metaheuristics (mặc định ["SA", "GA", "PSO"])
        seed: Random seed
    
    Returns:
        List[ActionIndividual]: Action library
    """
    if dr_list is None:
        dr_list = ["EDD"]
    if mh_list is None:
        mh_list = ["SA", "GA", "PSO"]
    
    lgp = ActionLGP(
        dr_list=dr_list,
        mh_list=mh_list,
        pool_size=pool_size,
        seed=seed
    )
    return lgp.pool


if __name__ == "__main__":
    num_episodes = 1000  # Số lượng episode training
    ppo_epochs = 10
    clip_epsilon = 0.25
    lr = 3e-4
    gamma = 0.9
    entropy_coef = 0.01
    
    # LGP Integration: Initialize action library with 64 portfolio actions
    print("Initializing LGP Action Library...")
    action_library = initialize_lgp_action_library(pool_size=64, seed=42)
    action_budget_s = 1.0  # Time budget for each action execution
    print(f"Action library initialized with {len(action_library)} portfolio actions")

    env = DynamicSchedulingEnv(
        lambda_tardiness=1.0,
        action_library=action_library,
        action_budget_s=action_budget_s
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = PPOActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Helper function to get action name from portfolio
    def get_action_name(action_idx):
        """Trả về mô tả ngắn gọn của action portfolio"""
        individual = action_library[action_idx]
        dr_name = individual.dr_gene.name
        mh_names = [g.name for g in individual.mh_genes]
        return f"{dr_name}+{'+'.join(mh_names)}"

    episode_rewards = []        # Tổng reward của mỗi episode
    policy_loss_history = []      # Trung bình policy loss mỗi episode
    value_loss_history = []       # Trung bình value loss mỗi episode

    # Mở file để ghi kết quả (ghi đè file cũ nếu có)
    with open("training_results_I10_E20.txt", "w") as result_file:
        for episode in range(num_episodes):
            state = env.reset()

            print(f"\n--- Episode {episode+1} ---")
            print("Initial Schedule:")
            for event in env.current_schedule_events:
                print({
                    'job': event['job'],
                    'op_index': event['op_index'],
                    'start': round(event['start'], 2),
                    'finish': round(event['finish'], 2),
                    'machine': event['machine'],
                    'op_id': event['op_id'],
                    'candidate_machines': event['candidate_machines']
                })

            log_probs_list = []
            values_list = []
            rewards = []
            masks = []
            actions_list = []
            states_list = []
            done = False
            step_count = 0
            while not done:
                action, log_prob, value = select_action(model, state)
                print(f"\nStep {step_count+1}: Chosen Action #{action}: {get_action_name(action)}")
                next_state, reward, done, _ = env.step(action)

                dyn_index = env.current_dynamic_index - 1
                if 0 <= dyn_index < len(env.dynamic_jobs_events):
                    arrival_time, dyn_job = env.dynamic_jobs_events[dyn_index]
                    print(f"\nDynamic Job Added (arrival_time: {round(arrival_time,2)}):")
                    # Làm tròn các giá trị thời gian nếu có
                    dj_print = dyn_job.copy()
                    dj_print['arrival_time'] = round(dj_print['arrival_time'], 2)
                    dj_print['due_date'] = round(dj_print['due_date'], 2)
                    print(dj_print)
                print("Final Schedule after reschedule:")
                for event in env.current_schedule_events:
                    print({
                        'job': event['job'],
                        'op_index': event['op_index'],
                        'start': round(event['start'], 2),
                        'finish': round(event['finish'], 2),
                        'machine': event['machine'],
                        'op_id': event['op_id'],
                        'candidate_machines': event['candidate_machines']
                    })

                log_probs_list.append(log_prob)
                values_list.append(value)
                rewards.append(reward)
                masks.append(1 - float(done))
                actions_list.append(action)
                states_list.append(state)
                state = next_state
                step_count += 1

            # Sau khi episode kết thúc, in các metric
            metrics = env.get_metrics()
            print("\nEpisode Metrics:")
            print(f"Makespan: {round(metrics['makespan'], 2)}")
            print(f"Tardiness Normal: {round(metrics['tardiness_normal'], 2)}")
            print(f"Tardiness Urgent: {round(metrics['tardiness_urgent'], 2)}")

            total_reward = sum(rewards)
            print(f"Episode {episode+1}, Total Reward: {round(total_reward, 2)}")

            # Ghi kết quả ra file
            result_file.write(f"Episode {episode+1}:\n")
            result_file.write(f"Total Reward: {round(total_reward, 2)}\n")
            result_file.write(f"Makespan: {round(metrics['makespan'], 2)}\n")
            result_file.write(f"Tardiness Normal: {round(metrics['tardiness_normal'], 2)}\n")
            result_file.write(f"Tardiness Urgent: {round(metrics['tardiness_urgent'], 2)}\n")
            result_file.write("Dynamic Jobs in this episode:\n")
            for arrival_time, dyn_job in env.dynamic_jobs_events:
                result_file.write(f"  Arrival Time: {round(arrival_time,2)} - {dyn_job}\n")
            result_file.write("="*40 + "\n")

            returns = compute_returns(rewards, masks, gamma)
            states_np = np.array(states_list)
            states = torch.FloatTensor(states_np)
            actions = torch.LongTensor(actions_list)
            log_probs = torch.stack(log_probs_list).detach()
            values = torch.stack(values_list).squeeze().detach()
            returns = torch.FloatTensor(returns)

            advantage = returns - values
            
            total_policy_loss = 0.0
            total_value_loss = 0.0
            count_updates = 0

            for _ in range(ppo_epochs):
                logits, value_est = model(states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions)
                ratio = torch.exp(new_log_probs - log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(value_est.squeeze(), returns)
                entropy = dist.entropy().mean()
                loss = policy_loss - entropy_coef * entropy + 0.5 * value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                count_updates += 1
                
            avg_policy_loss = total_policy_loss / count_updates if count_updates > 0 else 0
            avg_value_loss = total_value_loss / count_updates if count_updates > 0 else 0

            policy_loss_history.append(avg_policy_loss)
            value_loss_history.append(avg_value_loss)
            episode_rewards.append(total_reward)

    torch.save(model.state_dict(), "trained_policy_I10_E20_test.pth")
    
    print("\n=== Training Complete ===")
    print(f"Model saved to: trained_policy_I10_E20_test.pth")
    print(f"Results saved to: training_results_I10_E20.txt")
