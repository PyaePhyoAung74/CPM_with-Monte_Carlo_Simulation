import tkinter as tk
from tkinter import messagebox, ttk
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import stats
import networkx as nx
import csv
import uuid

root = tk.Tk()
root.title("CPM with Monte Carlo Simulation")

# Global variables
task_entries = []
duration_entries = []
dependencies_entries = []
task_count_entry = tk.Entry(root)
simulations_entry = tk.Entry(root)
seed_entry = tk.Entry(root)  # New entry for random seed

DB_PARAMS = {
    "dbname": "cpm_project",
    "user": "postgres",
    "password": "32926",
    "host": "localhost",
    "port": "5432"
}

def connect_db():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except Exception as e:
        messagebox.showerror("Database Error", f"Failed to connect: {e}")
        return None

def calculate_earliest_start_finish(tasks_data, dependencies_data):
    es = {task_id: 0 for task_id in tasks_data}
    ef = {}
    try:
        G = nx.DiGraph()
        for task_id in tasks_data:
            G.add_node(task_id)
        for task_id, deps in dependencies_data.items():
            for dep in deps:
                if dep not in tasks_data:
                    raise ValueError(f"Dependency {dep} not found for task {task_id}")
                G.add_edge(dep, task_id)
        
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Dependency graph contains cycles")
        
        for task_id in nx.topological_sort(G):
            es[task_id] = max([ef.get(dep, 0) for dep in dependencies_data.get(task_id, [])], default=0)
            ef[task_id] = es[task_id] + tasks_data[task_id]['duration']
    except Exception as e:
        raise ValueError(f"Error in ES/EF calculation: {e}")
    return es, ef

def calculate_latest_start_finish(tasks_data, dependencies_data, ef):
    ls = {}
    lf = {}
    successors = calculate_successors(tasks_data, dependencies_data)
    max_finish = max(ef.values())
    
    for task_id in tasks_data:
        lf[task_id] = max_finish
    
    G = nx.DiGraph()
    for task_id in tasks_data:
        G.add_node(task_id)
    for task_id, deps in dependencies_data.items():
        for dep in deps:
            G.add_edge(dep, task_id)
    
    for task_id in reversed(list(nx.topological_sort(G))):
        for succ_id in successors[task_id]:
            lf[task_id] = min(lf[task_id], ls[succ_id])
        ls[task_id] = lf[task_id] - tasks_data[task_id]['duration']
    
    slack = {task_id: lf[task_id] - ef[task_id] for task_id in tasks_data}
    return ls, lf, slack

def calculate_successors(tasks_data, dependencies_data):
    successors = {task_id: [] for task_id in tasks_data}
    for task_id, deps in dependencies_data.items():
        for dep in deps:
            if dep in successors:
                successors[dep].append(task_id)
    return successors

def find_critical_path(tasks_data, dependencies_data, es, ef, slack):
    G = nx.DiGraph()
    for task_id in tasks_data:
        G.add_node(task_id)
    for task_id, deps in dependencies_data.items():
        for dep in deps:
            G.add_edge(dep, task_id)
    
    critical_tasks = [task_id for task_id, s in slack.items() if abs(s) < 1e-6]
    critical_path = []
    
    if critical_tasks:
        try:
            topo_order = list(nx.topological_sort(G))
            critical_path = [task for task in topo_order if task in critical_tasks]
            # Verify path connectivity
            valid_path = True
            for i in range(len(critical_path) - 1):
                if not G.has_edge(critical_path[i], critical_path[i + 1]) and critical_path[i + 1] not in dependencies_data.get(critical_path[i], []):
                    valid_path = False
                    break
            if not valid_path:
                critical_path = []
        except:
            critical_path = []
    
    if not critical_path:
        # Fallback: longest path based on duration
        try:
            critical_path = nx.dag_longest_path(G, weight=lambda u, v: tasks_data[v]['duration'])
        except:
            critical_path = critical_tasks if critical_tasks else list(tasks_data.keys())
    
    return critical_path

def monte_carlo_cpm(tasks_data, dependencies_data, num_simulations, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    project_durations = np.zeros(num_simulations)
    critical_paths = {}
    task_criticality = {task_id: 0 for task_id in tasks_data}
    avg_durations = {task_id: [] for task_id in tasks_data}
    sample_simulations = []
    
    task_params = {task_id: (info['min'], info['most_likely'], info['max'])
                   for task_id, info in tasks_data.items()}
    
    for i in range(num_simulations):
        sim_tasks_data = {task_id: {'name': info['name'],
                                   'duration': np.random.triangular(*task_params[task_id])}
                         for task_id, info in tasks_data.items()}
        for task_id, dur in sim_tasks_data.items():
            avg_durations[task_id].append(dur['duration'])
        
        try:
            es, ef = calculate_earliest_start_finish(sim_tasks_data, dependencies_data)
            ls, lf, slack = calculate_latest_start_finish(sim_tasks_data, dependencies_data, ef)
            critical_path = find_critical_path(sim_tasks_data, dependencies_data, es, ef, slack)
        except Exception as e:
            messagebox.showerror("Simulation Error", f"Simulation {i+1} failed: {e}")
            continue
        
        project_durations[i] = max(ef.values())
        cp_key = ' -> '.join(critical_path)
        critical_paths[cp_key] = critical_paths.get(cp_key, 0) + 1
        
        for task_id in critical_path:
            task_criticality[task_id] += 1
        
        if i < 5:
            sample_simulations.append({
                'tasks_data': sim_tasks_data,
                'es': es,
                'ef': ef,
                'ls': ls,
                'lf': lf,
                'slack': slack,
                'critical_path': critical_path,
                'project_duration': project_durations[i],
                'dependencies_data': dependencies_data,
                'successors': calculate_successors(sim_tasks_data, dependencies_data)
            })
    
    mean_duration = np.mean(project_durations)
    std_duration = np.std(project_durations)
    percentiles = np.percentile(project_durations, [5, 95])
    median_duration = np.median(project_durations)
    mode_duration = float(stats.mode(project_durations, keepdims=True)[0])
    extra_percentiles = np.percentile(project_durations, [10, 25, 75])
    avg_durations = {task_id: np.mean(durs) for task_id, durs in avg_durations.items()}
    
    # Debugging output for critical path frequencies
    with open("critical_path_log.txt", "w") as f:
        f.write("Critical Path Frequencies:\n")
        for cp, count in sorted(critical_paths.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{cp}: {count} ({count/num_simulations*100:.1f}%)\n")
    
    return (project_durations, critical_paths, sample_simulations,
            task_criticality, mean_duration, std_duration, percentiles,
            avg_durations, median_duration, mode_duration, extra_percentiles)

def on_save_and_calculate():
    try:
        conn = connect_db()
        if not conn:
            return
        cur = conn.cursor()
        cur.execute("DELETE FROM monte_carlo_tasks")
        
        task_count = int(task_count_entry.get() or 0)
        num_simulations = int(simulations_entry.get() or 1000)
        seed = seed_entry.get()
        seed = int(seed) if seed.strip() else None
        
        if task_count <= 0:
            raise ValueError("Number of tasks must be positive.")
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be positive.")
        if num_simulations > 10000:
            raise ValueError("Number of simulations cannot exceed 10,000.")
        
        tasks_data = {}
        dependencies_data = {}
        task_ids = []
        
        for i in range(task_count):
            if i < 26:
                task_id = chr(ord('A') + i)
            else:
                first_char = chr(ord('A') + ((i - 26) // 26))
                second_char = chr(ord('A') + ((i - 26) % 26))
                task_id = first_char + second_char
            task_ids.append(task_id)
        
        for i in range(task_count):
            task_name = task_entries[i].get()
            min_dur = duration_entries[i*3].get()
            most_likely_dur = duration_entries[i*3+1].get()
            max_dur = duration_entries[i*3+2].get()
            task_deps = dependencies_entries[i].get().split(',')
            
            if not task_name or not min_dur or not most_likely_dur or not max_dur:
                raise ValueError(f"Task {i+1} fields cannot be empty.")
            
            min_dur, most_likely_dur, max_dur = int(min_dur), int(most_likely_dur), int(max_dur)
            
            # Validate triangular distribution parameters
            if not (min_dur <= most_likely_dur <= max_dur):
                raise ValueError(f"Task {i+1}: Must have min ({min_dur}) <= most likely ({most_likely_dur}) <= max ({max_dur})")
            
            task_id = task_ids[i]
            tasks_data[task_id] = {
                'name': task_name,
                'min': min_dur,
                'most_likely': most_likely_dur,
                'max': max_dur
            }
            dependencies_data[task_id] = [dep.strip() for dep in task_deps if dep.strip()]
            
            # Validate dependencies
            for dep in dependencies_data[task_id]:
                if dep not in task_ids:
                    raise ValueError(f"Invalid dependency {dep} for task {task_id}")
            
            deps_str = ','.join(dependencies_data[task_id]) if dependencies_data[task_id] else ''
            cur.execute(
                "INSERT INTO monte_carlo_tasks (task_id, name, min_duration, most_likely_duration, max_duration, dependencies) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (task_id, task_name, min_dur, most_likely_dur, max_dur, deps_str)
            )
        
        conn.commit()
        cur.close()
        conn.close()
        
        results = monte_carlo_cpm(tasks_data, dependencies_data, num_simulations, seed)
        display_monte_carlo_results(*results, tasks_data)
    except ValueError as ve:
        messagebox.showerror("Input Error", str(ve))
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def display_monte_carlo_results(project_durations, critical_paths, sample_simulations,
                               task_criticality, mean_duration, std_duration, percentiles,
                               avg_durations, median_duration, mode_duration, extra_percentiles, tasks_data):
    result_window = tk.Toplevel(root)
    result_window.title("Monte Carlo CPM Results")
    result_window.geometry("1200x800")
    
    canvas = tk.Canvas(result_window)
    scrollbar = tk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    result_window.bind_all("<MouseWheel>", _on_mousewheel)
    result_window.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
    result_window.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
    
    tk.Label(scrollable_frame, text=f"Monte Carlo Simulation Results ({len(project_durations)} Simulations)",
             font=("Arial", 12, "bold")).pack(pady=5)
    summary_columns = ("Statistic", "Value")
    summary_tree = ttk.Treeview(scrollable_frame, columns=summary_columns, show="headings", height=7)
    summary_tree.pack(pady=5)
    
    for col in summary_columns:
        summary_tree.heading(col, text=col)
        summary_tree.column(col, width=300, anchor="center")
    
    summary_data = [
        ("Average Project Duration", f"{mean_duration:.2f} days (Std Dev: {std_duration:.2f})"),
        ("Median Project Duration", f"{median_duration:.2f} days"),
        ("Mode Project Duration", f"{mode_duration:.2f} days"),
        ("Min Project Duration", f"{min(project_durations):.2f} days"),
        ("Max Project Duration", f"{max(project_durations):.2f} days"),
        ("90% Confidence Interval", f"[{percentiles[0]:.2f}, {percentiles[1]:.2f}] days"),
        ("10th, 25th, 75th Percentiles", f"[{extra_percentiles[0]:.2f}, {extra_percentiles[1]:.2f}, {extra_percentiles[2]:.2f}] days")
    ]
    
    for stat, value in summary_data:
        summary_tree.insert("", "end", values=(stat, value))
    
    tk.Label(scrollable_frame, text="Critical Path Frequencies:", font=("Arial", 10, "bold")).pack(pady=5)
    cp_columns = ("Critical Path", "Frequency", "Percentage")
    cp_tree = ttk.Treeview(scrollable_frame, columns=cp_columns, show="headings", height=len(critical_paths))
    cp_tree.pack(pady=5)
    
    for col in cp_columns:
        cp_tree.heading(col, text=col)
        cp_tree.column(col, width=600 if col == "Critical Path" else 150, anchor="center")
    
    for cp, count in sorted(critical_paths.items(), key=lambda x: x[1], reverse=True):
        cp_tree.insert("", "end", values=(cp, count, f"{count/len(project_durations)*100:.1f}%"))
    
    tk.Label(scrollable_frame, text="Task Criticality Probabilities:", font=("Arial", 10, "bold")).pack(pady=5)
    criticality_columns = ("Task ID", "Name", "Criticality Count", "Criticality %")
    criticality_tree = ttk.Treeview(scrollable_frame, columns=criticality_columns, show="headings", height=len(tasks_data))
    criticality_tree.pack(pady=5)
    
    for col in criticality_columns:
        criticality_tree.heading(col, text=col)
        criticality_tree.column(col, width=200, anchor="center")
    
    for task_id, count in sorted(task_criticality.items(), key=lambda x: x[1], reverse=True):
        prob = count / len(project_durations) * 100
        criticality_tree.insert("", "end", values=(task_id, tasks_data[task_id]['name'], count, f"{prob:.1f}%"))
    
    tk.Label(scrollable_frame, text="Average Task Durations:", font=("Arial", 10, "bold")).pack(pady=5)
    duration_columns = ("Task ID", "Name", "Average Duration")
    duration_tree = ttk.Treeview(scrollable_frame, columns=duration_columns, show="headings", height=len(tasks_data))
    duration_tree.pack(pady=5)
    
    for col in duration_columns:
        duration_tree.heading(col, text=col)
        duration_tree.column(col, width=200, anchor="center")
    
    for task_id, avg_dur in avg_durations.items():
        duration_tree.insert("", "end", values=(task_id, tasks_data[task_id]['name'], f"{avg_dur:.2f} days"))
    
    tk.Label(scrollable_frame, text="Sample Simulations (First 5):", font=("Arial", 10, "bold")).pack(pady=5)
    for i, sim in enumerate(sample_simulations):
        tk.Label(scrollable_frame, text=f"Simulation {i+1} (Project Duration: {sim['project_duration']:.2f} days)",
                 font=("Arial", 9, "italic")).pack()
        
        columns = ("Task", "Name", "Duration", "ES", "EF", "LS", "LF", "Slack", "Predecessors", "Successors")
        tree = ttk.Treeview(scrollable_frame, columns=columns, show="headings", height=len(tasks_data))
        tree.pack(pady=5)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")
        
        for task_id, task_info in sim['tasks_data'].items():
            preds = ", ".join(sim['dependencies_data'].get(task_id, [])) or "-"
            succs = ", ".join(sim['successors'].get(task_id, [])) or "-"
            tree.insert("", "end", values=(
                task_id, task_info['name'], f"{task_info['duration']:.2f}",
                f"{sim['es'][task_id]:.2f}", f"{sim['ef'][task_id]:.2f}",
                f"{sim['ls'][task_id]:.2f}", f"{sim['lf'][task_id]:.2f}",
                f"{sim['slack'][task_id]:.2f}", preds, succs
            ))
        
        tk.Label(scrollable_frame, text=f"Critical Path: {' -> '.join(sim['critical_path'])}").pack(pady=2)
        tk.Button(scrollable_frame, text="Show Network Diagram",
                  command=lambda s=i: draw_network_diagram(sim['tasks_data'], sim['dependencies_data'], sim['critical_path'],
                                                          sim['es'], sim['ef'], f"Simulation {s+1}")).pack(pady=2)
    
    tk.Label(scrollable_frame, text="Visualizations and Export:", font=("Arial", 10, "bold")).pack(pady=10)
    tk.Button(scrollable_frame, text="Show Duration Histogram with CDF",
              command=lambda: draw_histogram(project_durations)).pack(pady=5)
    tk.Button(scrollable_frame, text="Show Task Criticality Chart",
              command=lambda: draw_criticality_chart(task_criticality, tasks_data, project_durations)).pack(pady=5)
    
    def export_to_csv():
        with open("monte_carlo_results.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Statistic", "Value"])
            writer.writerow(["Mean Duration", f"{mean_duration:.2f}"])
            writer.writerow(["Std Deviation", f"{std_duration:.2f}"])
            writer.writerow(["Median Duration", f"{median_duration:.2f}"])
            writer.writerow(["Mode Duration", f"{mode_duration:.2f}"])
            writer.writerow(["5th Percentile", f"{percentiles[0]:.2f}"])
            writer.writerow(["95th Percentile", f"{percentiles[1]:.2f}"])
            writer.writerow(["10th Percentile", f"{extra_percentiles[0]:.2f}"])
            writer.writerow(["25th Percentile", f"{extra_percentiles[1]:.2f}"])
            writer.writerow(["75th Percentile", f"{extra_percentiles[2]:.2f}"])
            writer.writerow([])
            writer.writerow(["Critical Path", "Frequency", "Percentage"])
            for cp, count in critical_paths.items():
                writer.writerow([cp, count, f"{count/len(project_durations)*100:.1f}%"])
            writer.writerow([])
            writer.writerow(["Task ID", "Name", "Criticality Count", "Criticality %"])
            for task_id, count in task_criticality.items():
                writer.writerow([task_id, tasks_data[task_id]['name'], count, f"{count/len(project_durations)*100:.1f}%"])
        messagebox.showinfo("Export", "Results exported to monte_carlo_results.csv")
    
    tk.Button(scrollable_frame, text="Export Results to CSV", command=export_to_csv).pack(pady=5)

def draw_histogram(project_durations):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.hist(project_durations, bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='Histogram')
    ax1.set_xlabel("Duration (days)")
    ax1.set_ylabel("Frequency", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    percentiles = np.percentile(project_durations, [5, 95])
    ax1.axvline(percentiles[0], color='red', linestyle='--', label='5th Percentile')
    ax1.axvline(percentiles[1], color='green', linestyle='--', label='95th Percentile')
    
    ax2 = ax1.twinx()
    sorted_durations = np.sort(project_durations)
    cdf = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
    ax2.plot(sorted_durations, cdf, color='purple', label='CDF')
    ax2.set_ylabel("Cumulative Probability", color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    
    ax1.set_title(f"Project Duration Distribution (Monte Carlo - {len(project_durations)} Simulations)")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    
    hist_window = tk.Toplevel(root)
    hist_window.title("Duration Histogram with CDF")
    canvas = FigureCanvasTkAgg(fig, master=hist_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

def draw_criticality_chart(task_criticality, tasks_data, project_durations):
    fig, ax = plt.subplots(figsize=(12, 6))
    task_ids = list(task_criticality.keys())
    probs = [task_criticality[task_id] / len(project_durations) * 100 for task_id in task_ids]
    bars = ax.bar(task_ids, probs, color='orange', edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
                ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(range(len(task_ids)))
    ax.set_xticklabels([f"{tid}: {tasks_data[tid]['name']}" for tid in task_ids],
                      rotation=45, ha='right', fontsize=8)
    ax.set_title("Task Criticality Probabilities", fontsize=12)
    ax.set_xlabel("Tasks", fontsize=10)
    ax.set_ylabel("Probability (%)", fontsize=10)
    plt.tight_layout()
    
    chart_window = tk.Toplevel(root)
    chart_window.title("Task Criticality Chart")
    canvas = FigureCanvasTkAgg(fig, master=chart_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

def draw_network_diagram(tasks_data, dependencies_data, critical_path, es, ef, title="Network Diagram"):
    G = nx.DiGraph()
    for task_id in tasks_data:
        G.add_node(task_id)
    for task_id, deps in dependencies_data.items():
        for dep in deps:
            G.add_edge(dep, task_id)
    
    pos = nx.spring_layout(G, k=3.0, iterations=100)
    fig, ax = plt.subplots(figsize=(20, 15))
    node_size = 2000
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_size, ax=ax)
    critical_nodes = set(critical_path)
    nx.draw_networkx_nodes(G, pos, nodelist=critical_nodes, node_color='salmon', node_size=node_size, ax=ax)
    
    edge_colors = ['red' if (u, v) in [(critical_path[i], critical_path[i+1]) for i in range(len(critical_path)-1)] else 'gray' for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, ax=ax)
    
    labels = {task_id: f"{task_id}\n{tasks_data[task_id]['name']}\nES: {es[task_id]:.1f}, EF: {ef[task_id]:.1f}" for task_id in tasks_data}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    plt.title(title, fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    chart_window = tk.Toplevel(root)
    chart_window.title(title)
    canvas = tk.Canvas(chart_window)
    scrollbar_x = tk.Scrollbar(chart_window, orient="horizontal", command=canvas.xview)
    scrollbar_y = tk.Scrollbar(chart_window, orient="vertical", command=canvas.yview)
    canvas.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar_x.pack(side="bottom", fill="x")
    scrollbar_y.pack(side="right", fill="y")
    
    chart_canvas = FigureCanvasTkAgg(fig, master=canvas)
    chart_canvas.draw()
    canvas.create_window((0, 0), window=chart_canvas.get_tk_widget(), anchor="nw")
    
    def configure_canvas(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    chart_canvas.get_tk_widget().bind("<Configure>", configure_canvas)
    
    def _on_mousewheel(event):
        if event.state & 0x1:  # Shift key for horizontal
            canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        else:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    chart_window.bind_all("<MouseWheel>", _on_mousewheel)
    chart_window.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
    chart_window.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
    chart_window.bind_all("<Shift-Button-4>", lambda e: canvas.xview_scroll(-1, "units"))
    chart_window.bind_all("<Shift-Button-5>", lambda e: canvas.xview_scroll(1, "units"))
    
    toolbar_frame = tk.Frame(chart_window)
    toolbar_frame.pack(side="top", fill="x")
    toolbar = NavigationToolbar2Tk(chart_canvas, toolbar_frame)
    toolbar.update()
    
    zoom_frame = tk.Frame(chart_window, bg="white")
    zoom_frame.place(relx=1.0, rely=1.0, anchor="se")
    zoom_in_button = tk.Button(zoom_frame, text="+", font=("Arial", 8, "bold"), width=1, command=toolbar.zoom)
    zoom_in_button.pack(side="left", padx=2, pady=2)
    zoom_out_button = tk.Button(zoom_frame, text="-", font=("Arial", 8, "bold"), width=1, command=lambda: toolbar.zoom(out=True))
    zoom_out_button.pack(side="left", padx=2, pady=2)
    
    ax.autoscale()
    chart_canvas.draw()
    canvas.configure(scrollregion=canvas.bbox("all"))

def create_task_entries():
    try:
        task_count = int(task_count_entry.get())
        if task_count <= 0:
            messagebox.showerror("Error", "Enter a valid number of tasks.")
            return
        
        for widget in root.grid_slaves():
            widget.grid_forget()
        
        task_entries.clear()
        duration_entries.clear()
        dependencies_entries.clear()
        
        canvas = tk.Canvas(root)
        scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=3, column=0, columnspan=10, sticky="nsew")
        scrollbar.grid(row=3, column=10, sticky="ns")
        root.grid_rowconfigure(3, weight=1)
        root.grid_columnconfigure(9, weight=1)
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        root.bind_all("<MouseWheel>", _on_mousewheel)
        root.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        root.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
        
        for i in range(task_count):
            tk.Label(scrollable_frame, text=f"Task {i+1} Name:").grid(row=i, column=0, padx=2, pady=2)
            task_entry = tk.Entry(scrollable_frame)
            task_entry.grid(row=i, column=1, padx=2, pady=2, sticky="ew")
            task_entries.append(task_entry)
            
            tk.Label(scrollable_frame, text="Min Duration:").grid(row=i, column=2, padx=2, pady=2)
            min_entry = tk.Entry(scrollable_frame)
            min_entry.grid(row=i, column=3, padx=2, pady=2, sticky="ew")
            duration_entries.append(min_entry)
            
            tk.Label(scrollable_frame, text="Most Likely:").grid(row=i, column=4, padx=2, pady=2)
            most_entry = tk.Entry(scrollable_frame)
            most_entry.grid(row=i, column=5, padx=2, pady=2, sticky="ew")
            duration_entries.append(most_entry)
            
            tk.Label(scrollable_frame, text="Max Duration:").grid(row=i, column=6, padx=2, pady=2)
            max_entry = tk.Entry(scrollable_frame)
            max_entry.grid(row=i, column=7, padx=2, pady=2, sticky="ew")
            duration_entries.append(max_entry)
            
            tk.Label(scrollable_frame, text="Dependencies:").grid(row=i, column=8, padx=2, pady=2)
            dep_entry = tk.Entry(scrollable_frame)
            dep_entry.grid(row=i, column=9, padx=2, pady=2, sticky="ew")
            dependencies_entries.append(dep_entry)
        
        scrollable_frame.grid_columnconfigure(1, weight=1)
        scrollable_frame.grid_columnconfigure(3, weight=1)
        scrollable_frame.grid_columnconfigure(5, weight=1)
        scrollable_frame.grid_columnconfigure(7, weight=1)
        scrollable_frame.grid_columnconfigure(9, weight=1)
        
        tk.Button(root, text="Save and Run Monte Carlo", command=on_save_and_calculate).grid(row=4, column=0, columnspan=10, pady=10)
        tk.Button(root, text="Reset", command=reset_input_fields).grid(row=5, column=0, columnspan=10, pady=10)
        
        tk.Label(root, text="Enter the number of tasks:").grid(row=0, column=0, padx=5, pady=5)
        task_count_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(root, text="Number of simulations:").grid(row=1, column=0, padx=5, pady=5)
        simulations_entry.grid(row=1, column=1, padx=5, pady=5)
        tk.Label(root, text="Random seed (optional):").grid(row=2, column=0, padx=5, pady=5)
        seed_entry.grid(row=2, column=1, padx=5, pady=5)
        tk.Button(root, text="Create Task Entries", command=create_task_entries).grid(row=0, column=2, columnspan=2, pady=10)
        tk.Button(root, text="Load Large Example", command=load_large_example_project).grid(row=1, column=2, columnspan=2, pady=10)
    except ValueError:
        messagebox.showerror("Error", "Enter a valid number of tasks.")

def reset_input_fields():
    for entry in task_entries + duration_entries + dependencies_entries:
        entry.delete(0, tk.END)
    task_count_entry.delete(0, tk.END)
    simulations_entry.delete(0, tk.END)
    seed_entry.delete(0, tk.END)
    for widget in root.grid_slaves(row=3):
        widget.grid_forget()

def load_large_example_project():
    tasks = {}
    task_definitions = [
        ('A', 'Project Initiation', 1, 2, 3, ''),
        ('B', 'Requirements Gathering', 3, 5, 7, 'A'),
        ('C', 'Stakeholder Interviews', 2, 4, 6, 'B'),
        ('D', 'Requirements Documentation', 4, 6, 8, 'C'),
        ('E', 'Architecture Design', 5, 7, 10, 'D'),
        ('F', 'Database Schema Design', 4, 6, 8, 'E'),
        ('G', 'API Specification', 3, 5, 7, 'E'),
        ('H', 'UI/UX Wireframing', 3, 4, 6, 'D'),
        ('I', 'Frontend Development Setup', 2, 3, 5, 'H'),
        ('J', 'Backend Development Setup', 2, 3, 5, 'G'),
        ('K', 'Database Implementation', 5, 7, 9, 'F'),
        ('L', 'API Development', 6, 8, 11, 'G,J'),
        ('M', 'Frontend Feature 1', 4, 6, 8, 'I'),
        ('N', 'Frontend Feature 2', 5, 7, 10, 'I'),
        ('O', 'Backend Feature 1', 6, 8, 12, 'L,K'),
        ('P', 'Backend Feature 2', 5, 7, 10, 'L,K'),
        ('Q', 'Integration Testing Preparation', 3, 4, 6, 'O,P'),
        ('R', 'Frontend Unit Testing', 4, 5, 7, 'M,N'),
        ('S', 'Backend Unit Testing', 4, 5, 7, 'O,P'),
        ('T', 'Integration Testing', 6, 8, 10, 'Q,R,S'),
        ('U', 'Bug Fixing Phase 1', 3, 5, 7, 'T'),
        ('V', 'Performance Optimization', 4, 6, 8, 'U'),
        ('W', 'Security Audit', 3, 5, 7, 'U'),
        ('X', 'User Acceptance Testing', 5, 7, 9, 'V,W'),
        ('Y', 'Documentation Writing', 4, 6, 8, 'X'),
        ('Z', 'Training Material Preparation', 3, 4, 6, 'X'),
        ('AA', 'Deployment Planning', 2, 3, 5, 'X'),
        ('AB', 'Initial Deployment', 3, 4, 6, 'AA'),
        ('AC', 'Load Testing', 4, 5, 7, 'AB'),
        ('AD', 'Final Bug Fixing', 3, 5, 7, 'AC'),
        ('AE', 'Production Deployment', 2, 3, 4, 'AD'),
        ('AF', 'Post-Deployment Monitoring', 2, 3, 5, 'AE'),
        ('AG', 'Customer Training', 3, 4, 6, 'Z,AE'),
        ('AH', 'Feedback Collection', 2, 3, 4, 'AF'),
        ('AI', 'Final Adjustments', 3, 4, 6, 'AH'),
        ('AJ', 'Project Closure', 1, 2, 3, 'AI'),
        ('AK', 'Marketing Preparation', 4, 6, 8, 'X'),
        ('AL', 'Launch Event Planning', 3, 5, 7, 'AK'),
        ('AM', 'Launch Execution', 2, 3, 5, 'AL,AE'),
        ('AN', 'Post-Launch Review', 2, 3, 4, 'AM'),
    ]
    
    for i, (task_id, name, min_dur, most_likely, max_dur, deps) in enumerate(task_definitions):
        tasks[task_id] = {
            'name': name,
            'min': min_dur,
            'most_likely': most_likely,
            'max': max_dur,
            'deps': deps
        }
    
    task_count_entry.delete(0, tk.END)
    task_count_entry.insert(0, str(len(tasks)))
    simulations_entry.delete(0, tk.END)
    simulations_entry.insert(0, "1000")
    seed_entry.delete(0, tk.END)
    seed_entry.insert(0, "42")  # Default seed for reproducibility
    create_task_entries()
    
    for i, (task_id, info) in enumerate(tasks.items()):
        task_entries[i].insert(0, info['name'])
        duration_entries[i*3].insert(0, str(info['min']))
        duration_entries[i*3+1].insert(0, str(info['most_likely']))
        duration_entries[i*3+2].insert(0, str(info['max']))
        dependencies_entries[i].insert(0, info['deps'])

# Initial UI setup
tk.Label(root, text="Enter the number of tasks:").grid(row=0, column=0, padx=5, pady=5)
task_count_entry.grid(row=0, column=1, padx=5, pady=5)
tk.Label(root, text="Number of simulations:").grid(row=1, column=0, padx=5, pady=5)
simulations_entry.grid(row=1, column=1, padx=5, pady=5)
tk.Label(root, text="Random seed (optional):").grid(row=2, column=0, padx=5, pady=5)
seed_entry.grid(row=2, column=1, padx=5, pady=5)
tk.Button(root, text="Create Task Entries", command=create_task_entries).grid(row=0, column=2, columnspan=2, pady=10)
tk.Button(root, text="Load Large Example", command=load_large_example_project).grid(row=1, column=2, columnspan=2, pady=10)

root.mainloop()