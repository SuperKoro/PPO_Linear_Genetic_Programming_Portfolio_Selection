import numpy as np
import random
import copy
import math
import gym
from gym import spaces
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Import các hàm và class từ main.py
from main import (
    machine_pool, jobs_initial, due_dates_initial,
    simulated_annealing, schedule_dict_to_list, split_schedule_list,
    reschedule_unfinished_jobs_edd, reschedule_unfinished_jobs_sa,
    reschedule_unfinished_jobs_ga, reschedule_unfinished_jobs_pso,
    create_unified_jobs_info, DynamicSchedulingEnv, PPOActorCritic,
    select_action
)

# Khởi tạo môi trường
env = DynamicSchedulingEnv(lambda_tardiness=1.0)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

# Định nghĩa mapping cho tên action
action_names = {0: "SA", 1: "GA", 2: "EDD", 3: "PSO"}

# Tạo model và load trọng số đã huấn luyện
model = PPOActorCritic(obs_dim, act_dim)
model.load_state_dict(torch.load("trained_policy_I10_E20_test.pth"))
model.eval()

print("=" * 60)
print("INFERENCE - Testing Trained PPO Model")
print("=" * 60)

# Hàm vẽ Gantt chart
def plot_gantt(schedule_events):
    colors = plt.cm.tab20.colors
    job_colors = {}
    
    def get_job_color(job):
        job = str(job)
        if job not in job_colors:
            index = len(job_colors) % len(colors)
            job_colors[job] = colors[index]
        return job_colors[job]
    
    # Nhóm các event theo machine
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
    
    # Tính makespan
    max_finish = max(e['finish'] for e in schedule_events) if schedule_events else 0
    
    # Vẽ đường makespan (đường thẳng đứng màu đỏ)
    ax.axvline(x=max_finish, color='red', linestyle='--', linewidth=2, label=f'Makespan = {max_finish:.2f}')
    
    # Thêm text annotation cho makespan
    ax.text(max_finish, len(machine_ids) - 0.5, f'Makespan\n{max_finish:.2f}', 
            color='red', fontsize=12, fontweight='bold', 
            ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8))
    
    ax.set_title(f"Gantt Chart - Final Schedule (Makespan: {max_finish:.2f})", fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, max_finish + 10)
    
    # Sắp xếp legend theo thứ tự: job số trước, sau đó job dạng "D..."
    def sort_key(job_id):
        try:
            return (0, int(job_id))
        except ValueError:
            if job_id.startswith("D"):
                return (1, int(job_id[1:]))
            return (1, job_id)
    
    sorted_jobs = sorted(job_colors.items(), key=lambda item: sort_key(item[0]))
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=f"Job {job}") for job, color in sorted_jobs]
    
    # Thêm makespan vào legend
    from matplotlib.lines import Line2D
    makespan_line = Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Makespan = {max_finish:.2f}')
    legend_elements.insert(0, makespan_line)
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()

# Inference: sử dụng model đã huấn luyện để chạy một episode thực tế
state = env.reset()
done = False
dynamic_jobs_info = []
step_count = 0

print("\n--- Running Inference Episode ---\n")

while not done:
    action, log_prob, value = select_action(model, state)
    chosen_action = action_names[action]
    
    print(f"Step {step_count + 1}: Chosen Action = {chosen_action}")
    
    next_state, reward, done, _ = env.step(action)
    
    dyn_index = env.current_dynamic_index - 1
    if 0 <= dyn_index < len(env.dynamic_jobs_events):
        arrival_time, dyn_job = env.dynamic_jobs_events[dyn_index]
        dynamic_jobs_info.append({
            "arrival_time": round(arrival_time, 2),
            "job": dyn_job
        })
        print(f"  → Dynamic Job Added: {dyn_job['job_id']} (arrival: {round(arrival_time, 2)}, type: {dyn_job['job_type']})")
    
    state = next_state
    step_count += 1

# Sau khi episode kết thúc, in ra các metric và lịch trình
metrics = env.get_metrics()
print("\n" + "=" * 60)
print("FINAL METRICS")
print("=" * 60)
print(f"Makespan:         {round(metrics['makespan'], 2)}")
print(f"Tardiness Normal: {round(metrics['tardiness_normal'], 2)}")
print(f"Tardiness Urgent: {round(metrics['tardiness_urgent'], 2)}")

print("\n" + "=" * 60)
print("DYNAMIC JOBS IN THIS EPISODE")
print("=" * 60)
for dj in dynamic_jobs_info:
    job_info = dj['job']
    print(f"Job {job_info['job_id']}:")
    print(f"  - Arrival Time: {dj['arrival_time']}")
    print(f"  - Due Date: {job_info['due_date']}")
    print(f"  - Type: {job_info['job_type']}")
    print(f"  - Operations: {len(job_info['operations'])}")

print("\n" + "=" * 60)
print("DISPLAYING GANTT CHART")
print("=" * 60)

# Vẽ Gantt Chart cho lịch trình cuối cùng
plot_gantt(env.current_schedule_events)
