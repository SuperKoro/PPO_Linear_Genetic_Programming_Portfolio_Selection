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

# LGP imports
import dispatching_rules          # auto-register DR vào DR_REGISTRY
import metaheuristics_impl        # auto-register MH vào MH_REGISTRY

from ga_portfolio import ActionIndividual, ActionLGP, Gene, individual_normalized_weights
from typed_action_adapter import run_action_individual
from coevolution_trainer import CoevolutionConfig, train_with_coevolution
# LGP imports
from lgp_generator import LGPGenerator
from lgp_program import LGPProgram
from lgp_coevolution import train_with_coevolution_lgp


# ==================== Helper for Action Name Display ====================

def format_action_name(individual: ActionIndividual) -> str:
    """
    Format action name cho dễ đọc:
    EDD | SA:0.52, GA:0.23, PSO:0.25
    """
    dr = individual.dr_gene.name
    mh_names = [g.name for g in individual.mh_genes]
    norm_ws = individual_normalized_weights(individual)
    mh_parts = [
        f"{name}:{w:.2f}" for name, w in zip(mh_names, norm_ws)
    ]
    return f"{dr} | " + ", ".join(mh_parts)


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

def reschedule_unfinished_jobs_spt(unfinished_jobs, current_time, finished_events, machine_pool):
    """
    Shortest Processing Time (SPT): Ưu tiên jobs có tổng processing time ngắn nhất.
    """
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: sum(op['processing_time'] for op in x[1]['operations'])
    )
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

def reschedule_unfinished_jobs_lpt(unfinished_jobs, current_time, finished_events, machine_pool):
    """
    Longest Processing Time (LPT): Ưu tiên jobs có tổng processing time dài nhất.
    """
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: sum(op['processing_time'] for op in x[1]['operations']),
        reverse=True  # Descending order
    )
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

def reschedule_unfinished_jobs_fcfs(unfinished_jobs, current_time, finished_events, machine_pool):
    """
    First Come First Served (FCFS/FIFO): Ưu tiên jobs theo thứ tự job_ready time.
    """
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: x[1]['job_ready']
    )
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

def reschedule_unfinished_jobs_cr(unfinished_jobs, current_time, finished_events, machine_pool):
    """
    Critical Ratio (CR): Ưu tiên jobs theo tỷ lệ (due_date - current_time) / remaining_processing_time.
    Tỷ lệ nhỏ = critical hơn = ưu tiên cao hơn.
    """
    def calculate_cr(info):
        remaining_pt = sum(op['processing_time'] for op in info['operations'])
        slack = info['due_date'] - current_time
        if remaining_pt <= 0:
            return float('inf')  # No work left, lowest priority
        return slack / remaining_pt
    
    sorted_jobs = sorted(
        unfinished_jobs.items(),
        key=lambda x: calculate_cr(x[1])
    )
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
        if neighbor and len(neighbor) > 0:  # ← Added len check
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
            if random.random() < 0.3 and len(solution) > 0:  # ← Added len(solution) > 0 check
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
    def __init__(self,
                 lambda_tardiness: float = 1.0,
                 action_library: list = None,
                 action_budget_s: float = 3.0):
        super(DynamicSchedulingEnv, self).__init__()
        self.lambda_tardiness = lambda_tardiness
        self.machine_pool = machine_pool
        self.jobs_initial = jobs_initial
        self.due_dates_initial = due_dates_initial
        self.all_jobs_info = create_unified_jobs_info(self.jobs_initial, self.due_dates_initial)
        
        # Tạo initial schedule một lần offline
        _, schedule, _, _, _, _ = simulated_annealing(
            self.jobs_initial,
            self.due_dates_initial,
            lambda_tardiness=self.lambda_tardiness
        )
        self.initial_schedule_events = schedule_dict_to_list(schedule, self.all_jobs_info)
        self.current_schedule_events = copy.deepcopy(self.initial_schedule_events)
        self.current_time = 0
        self._generate_dynamic_jobs(num_dynamic=4)
        self.current_dynamic_index = 0
        
        # NEW: action_library + budget
        self.action_library = action_library if action_library is not None else self._build_default_action_library()
        self.action_budget_s = float(action_budget_s)
        
        self.observation_space = spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_library))

    def _build_default_action_library(self):
        """
        Tạo 4 action tương đương với code cũ: SA, GA, EDD, PSO
        để backward-compatible khi không dùng LGP.
        """
        def make_one(mh_name: str) -> ActionIndividual:
            genes = [Gene(kind="DR", name="EDD", w_raw=1.0)]
            # 3 MH gene, chỉ gene đầu có weight, 2 gene sau weight=0
            genes.append(Gene(kind="MH", name=mh_name, w_raw=1.0))
            genes.append(Gene(kind="MH", name=mh_name, w_raw=0.0))
            genes.append(Gene(kind="MH", name=mh_name, w_raw=0.0))
            return ActionIndividual(genes=genes)

        actions = [
            make_one("SA"),
            make_one("GA"),
            make_one("EDD"),
            make_one("PSO"),
        ]
        return actions


    def _generate_dynamic_job(self, job_id, arrival_time, min_ops=1, max_ops=5, min_pt=5, max_pt=50):
        # Xác định loại job: 10% chance urgent, 90% normal
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
        """
        action: index trong self.action_library
        """
        # Nếu hết dynamic jobs thì kết thúc
        if self.current_dynamic_index >= len(self.dynamic_jobs_events):
            return self._get_state(), 0.0, True, {}

        # Lấy dynamic job hiện tại và cập nhật thời gian
        arrival_time, dyn_job = self.dynamic_jobs_events[self.current_dynamic_index]
        self.current_time = arrival_time

        finished_events, unfinished_jobs = split_schedule_list(
            self.current_schedule_events,
            self.current_time,
            self.all_jobs_info
        )

        # Thêm dynamic job mới vào unfinished + all_jobs_info
        ops_list = []
        for i, op in enumerate(dyn_job['operations']):
            ops_list.append(
                {
                    'op_index': i,
                    'op_id': op['op_id'],
                    'candidate_machines': op['candidate_machines'],
                    'processing_time': op['processing_time'],
                }
            )
        dyn_info = {
            'job_ready': self.current_time,
            'due_date': dyn_job['due_date'],
            'operations': ops_list,
            'job_type': dyn_job.get('job_type', 'Normal'),
        }
        job_id = dyn_job['job_id']
        self.all_jobs_info[job_id] = dyn_info
        unfinished_jobs[job_id] = dyn_info

        # Gọi action LGP: DR + vector MH
        individual = self.action_library[action]

        new_unfinished_events = run_action_individual(
            env=self,
            individual=individual,
            finished_events=finished_events,
            unfinished_jobs=unfinished_jobs,
            total_budget_s=self.action_budget_s
        )

        self.current_schedule_events = finished_events + new_unfinished_events

        # Tính reward như cũ
        merged = self.current_schedule_events
        makespan = max(e['finish'] for e in merged) if merged else 0

        total_tardiness_normal = 0.0
        total_tardiness_urgent = 0.0
        for job, info in self.all_jobs_info.items():
            job_events = [e for e in merged if e['job'] == job]
            if not job_events:
                continue
            comp_time = max(e['finish'] for e in job_events)
            tardiness = max(0, comp_time - info['due_date'])
            if isinstance(job, int):
                total_tardiness_normal += tardiness
            else:
                if info.get('job_type', 'Normal') == 'Urgent':
                    total_tardiness_urgent += tardiness
                else:
                    total_tardiness_normal += tardiness

        # Tạm thời reward = -makespan (giống code cũ, bạn có thể cộng thêm tardiness)
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
    
    # Save instead of show to avoid blocking
    plt.savefig("results/final/gantt_chart.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Gantt chart saved to: results/final/gantt_chart.png")


def initialize_lgp_action_library(pool_size=64,
                                  dr_list=None,
                                  mh_list=None,
                                  seed=0):
    """
    Tạo pool_size ActionIndividual ngẫu nhiên dùng ActionLGP.
    """
    if dr_list is None:
        dr_list = ["EDD", "SPT", "LPT", "FCFS", "CR"]
    if mh_list is None:
        mh_list = ["SA", "GA", "PSO", "EDD"]

    lgp = ActionLGP(dr_list=dr_list,
                    mh_list=mh_list,
                    pool_size=pool_size,
                    n_mh_genes=3,
                    seed=seed)

    return lgp.pool  # List[ActionIndividual]


if __name__ == "__main__":
    # Import configuration
    from config import (
        PPOConfig, CoevolutionConfig, LGPConfig, EnvironmentConfig,
        RANDOM_SEED, OUTPUT_DIR, MODEL_SAVE_PATH,
        print_config_summary, Presets
    )
    
    # Print configuration summary
    print_config_summary()
    
    # Optional: Use a preset for quick testing
    # Presets.quick_test()  # Uncomment for fast testing
    # Presets.high_quality()  # Uncomment for better results
    
    # PPO hyperparameters from config
    lr = PPOConfig.learning_rate
    gamma = PPOConfig.gamma
    ppo_epochs = PPOConfig.ppo_epochs
    clip_epsilon = PPOConfig.clip_epsilon
    entropy_coef = PPOConfig.entropy_coef

    # 1) Khởi tạo LGP programs ngẫu nhiên
    import random as _random
    lgp_rng = _random.Random(RANDOM_SEED)
    lgp_gen = LGPGenerator(
        max_length=LGPConfig.max_program_length,
        min_length=LGPConfig.min_program_length,
        num_registers=LGPConfig.num_registers,
        rng=lgp_rng,
    )
    lgp_programs = [
        lgp_gen.generate_random_program()
        for _ in range(LGPConfig.pool_size)
    ]
    
    # 2) Khởi tạo action_library ban đầu (placeholder) cho env
    # LGP sẽ overwrite trong training
    action_library = initialize_lgp_action_library(
        pool_size=LGPConfig.pool_size,
        dr_list=LGPConfig.available_dr,
        mh_list=LGPConfig.available_mh,
        seed=RANDOM_SEED
    )

    # 3) Tạo env (cho 1 dataset hiện tại)
    env = DynamicSchedulingEnv(
        lambda_tardiness=EnvironmentConfig.lambda_tardiness,
        action_library=action_library,
        action_budget_s=LGPConfig.action_budget_s
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 4) PPO model + optimizer
    model = PPOActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5) Cấu hình coevolution from config
    from coevolution_trainer import CoevolutionConfig as CoevoCfg
    cfg = CoevoCfg(
        num_generations=CoevolutionConfig.num_generations,
        episodes_per_gen=CoevolutionConfig.episodes_per_gen,
        max_steps_per_episode=CoevolutionConfig.max_steps_per_episode,
        gamma=gamma,
        ppo_epochs=ppo_epochs,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        elite_size=CoevolutionConfig.elite_size,
        n_replace=CoevolutionConfig.n_replace,
        warmup_episodes=CoevolutionConfig.warmup_episodes,
        mutation_sigma=CoevolutionConfig.mutation_sigma,
        dr_mutation_prob=CoevolutionConfig.dr_mutation_prob,
        mh_name_mutation_prob=CoevolutionConfig.mh_name_mutation_prob
    )

    # 6) Train với LGP coevolution (PPO + LGP)
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("🚀 STARTING LGP + PPO COEVOLUTION TRAINING")
    print("="*70)
    
    lgp_programs, final_action_library = train_with_coevolution_lgp(
        env=env,
        lgp_programs=lgp_programs,
        model=model,
        optimizer=optimizer,
        select_action_fn=select_action,
        compute_returns_fn=compute_returns,
        cfg=cfg,
        output_dir=OUTPUT_DIR
    )

    # 7) Lưu model sau khi train
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n=== Training Complete ===")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Results saved to: {OUTPUT_DIR}/")
    
    # Save LGP programs
    programs_data = {
        "num_programs": len(lgp_programs),
        "programs": [prog.to_dict() for prog in lgp_programs]
    }
    lgp_save_path = os.path.join(OUTPUT_DIR, "lgp_programs_final.json")
    import json
    with open(lgp_save_path, 'w') as f:
        json.dump(programs_data, f, indent=2)
    print(f"LGP programs saved to: {lgp_save_path}")





