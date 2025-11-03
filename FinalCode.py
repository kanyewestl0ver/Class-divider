# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 01:37:14 2025

@author: braed
"""

#!/usr/bin/env python3
# Final script: Ensures all students are assigned to teams,
# includes strict initial rules, subsequent optimization, and results visualization.
# Version: Iteration count is dynamically and precisely calculated based on the problem's combinatorial complexity.

# --- Imports ---
# csv: For reading/writing CSV files (student records).
# math: For math.sqrt(), used in standard deviation calculation.
# random: For random.shuffle() (fairness) and random.sample() (optimization).
# collections.Counter: Used in the summary to count gender combinations.
import csv
import math
import random
from collections import defaultdict, Counter
# matplotlib.pyplot: Used to create and save plot visualizations.
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration Section ---
# This defines the "batch size" for the algorithm. The script will process
# students in chunks of this size, running the entire algorithm on each chunk.
TUTORIAL_GROUP_SIZE = 50 # Number of students per tutorial group.

# This is a tuning knob for the optimizer.
# It multiplies the calculated "base complexity" to get the final iteration count.
# Alpha < 1.0: Faster, lower quality results.
# Alpha = 1.0: A standard run.
# Alpha > 1.0: Slower, but allows the optimizer more time to find better swaps.
EXPLORATION_ALPHA = 1.5 

# --- Core Logic ---

# <--- Major Change: Rewrote this function with more robust, mathematical logic ---
def build_templates(num_teams, minority_count):
    """
    Creates a "template" or plan for distributing minority gender members.
    New logic: Distributes minority_count as evenly as possible across num_teams.
    
    Example: 23 minority students, 10 teams.
    - base_num = 23 // 10 = 2 (Every team gets *at least* 2)
    - num_teams_with_flex = 23 % 10 = 3 (3 teams must take one *extra* member)
    - num_teams_with_base = 10 - 3 = 7 (7 teams take only the base amount)
    - Result: [3, 3, 3, 2, 2, 2, 2, 2, 2, 2]
    Note: This function will be used in the function gender_balanced_teams()
    """
    
    # Ensure there are teams to assign to, avoiding division by zero
    if num_teams == 0:
        return []

    # Base number of minority members per team (e.g., 23 // 10 = 2)
    base_num = minority_count // num_teams
    
    # Number of teams that need one extra member (e.g., 23 % 10 = 3)
    num_teams_with_flex = minority_count % num_teams
    
    # Number of remaining teams that only get the base number (e.g., 10 - 3 = 7)
    num_teams_with_base = num_teams - num_teams_with_flex
    
    # Create the template list
    templates = []
    # (e.g., 3 teams get 2+1=3 members)
    templates.extend([base_num + 1] * num_teams_with_flex)
    # (e.g., 7 teams get 2 members)
    templates.extend([base_num] * num_teams_with_base)
    
    # Shuffle the template to ensure fairness. This prevents
    # teams 0, 1, 2 from *always* being the ones to get the extra member.
    random.shuffle(templates)
    return templates

def find_valid_student_and_remove(student_pool, team_schools):
    """
    Finds the best student from a given pool who satisfies the school constraints.
    - "Constraint" = Their school is not already in the team.

    This function has a critical side effect: it .pop()s the student
    from the pool, removing them from future consideration.
    Note: This function will be used in the function gender_balanced_teams()
    """
    
    # The student_pool is assumed to be pre-sorted by CGPA, descending.
    for i, student in enumerate(student_pool):
        # This is the school diversity constraint check.
        if student['School'] not in team_schools:
            # Found a valid student. Remove them from the pool and return them.
            return student_pool.pop(i)
    # No student in the pool satisfies the diversity constraint.
    return None

def get_cgpa(student):
    """A simple helper function to get the CGPA value from a student dictionary. Used as a key for sorting."""
    return student["CGPA"]

# <--- Change: Removed default value for team_size=5 ---
def gender_balanced_teams(students, team_size):
    """
    Phase 1: Form "ideal" teams.
    This phase greedily assigns students based on three rules, in order:
    1. Gender Balance (following the 'templates' plan).
    2. School Diversity (using `find_valid_student_and_remove`).
    3. CGPA (by picking the highest-CGPA student who fits 1 & 2).
    """
    # Separate students into gender pools.
    males = [s for s in students if s["Gender"] == "Male"]
    females = [s for s in students if s["Gender"] == "Female"]
    
    # Sort both pools by CGPA (highest first). This is crucial
    # for `find_valid_student_and_remove` to work greedily.
    males.sort(key=get_cgpa, reverse=True)
    females.sort(key=get_cgpa, reverse=True)
    
    # Calculate the number of teams to create within this tutorial group.
    num_teams = len(students) // team_size
    if num_teams == 0:
        # Not enough students to make a single team.
        return [], students
    
    teams = [[] for i in range(num_teams)] 
    
    # Identify which gender is the minority.
    if len(males) <= len(females):
        minority_pool, majority_pool = males, females
        minority_gender = "Male"
    else:
        minority_pool, majority_pool = females, males
        minority_gender = "Female"
    
    # <--- Change: build_templates function now uses new logic ---
    # Get the "plan" for distributing the minority gender.
    templates = build_templates(num_teams, len(minority_pool))
    
    # Loop to fill every slot (e.g., 10 teams * 5 members/team = 50 slots).
    for i in range(num_teams * team_size):
        # `i % num_teams` implements a "round-robin" assignment.
        # (Team 0, Team 1, ... Team 9, Team 0, Team 1, ...)
        team_idx = i % num_teams
        current_team = teams[team_idx]
        # Note that the teams variable contains the [] for all teams
        
        # If this team is already full, skip it.
        if len(current_team) >= team_size:
            continue
            
        # Check how many minority members this team *already* has.
        minority_in_team_count = sum(1 for p in current_team if p['Gender'] == minority_gender)
        
        # Check if templates list is empty (edge case). #Error Debugging
        if team_idx >= len(templates):
            continue
            
        # Compare the team's state to the plan.
        # `needs_minority` is True if the team *must* take a minority member. 
        needs_minority = minority_in_team_count < templates[team_idx]
        
        # Get a set of schools already in this team.
        team_schools = {p['School'] for p in current_team}
        student_to_add = None
        
        # --- Core Selection Logic ---
        # Priority 1: Try to add a student of the *required* gender.
        if needs_minority and minority_pool:
            student_to_add = find_valid_student_and_remove(minority_pool, team_schools)
        # If minority criteria fulfilled, fill in the majority_pool
        elif not needs_minority and majority_pool:
            student_to_add = find_valid_student_and_remove(majority_pool, team_schools)
            
        # Priority 2 (Fallback): If no student from the *required* pool was valid
        # (e.g., all remaining minority students had school conflicts),
        # try to fill the slot with a student from the *other* pool.
        # Note that we prioritise school distribution over gender distribution
        if not student_to_add:
            if majority_pool:
                student_to_add = find_valid_student_and_remove(majority_pool, team_schools)
            if not student_to_add and minority_pool:
                # Last resort: try the minority pool again (if the first attempt was on the majority pool).
                student_to_add = find_valid_student_and_remove(minority_pool, team_schools)
                
        # If a student was successfully found, add them to the team.
        if student_to_add:
            current_team.append(student_to_add)
            
    # After the loop, any students left in the pools are unassigned.
    unassigned_students = minority_pool + majority_pool
    return teams, unassigned_students

def calculate_std_dev(teams):
    """
    Calculates the standard deviation of the "average CGPAs" for a list of teams.
    This is the **cost function** for the optimizer. A lower value is better,
    as it means the average CGPAs of the teams are closer to each other (i.e., more balanced).
    """
    team_avg_cgpas = []
    for team in teams:
        if team: # Avoid empty teams
            avg = sum(member['CGPA'] for member in team) / len(team)
            team_avg_cgpas.append(avg)
            
    # Need at least 2 data points to calculate std dev.
    if len(team_avg_cgpas) < 2:
        return 0.0
        
    # Standard deviation calculation:
    # 1. Calculate the mean of the team averages.
    overall_avg = sum(team_avg_cgpas) / len(team_avg_cgpas)
    # 2. Calculate the variance (average of squared differences from the mean).
    variance = sum((avg - overall_avg) ** 2 for avg in team_avg_cgpas) / len(team_avg_cgpas)
    # 3. Standard deviation is the square root of variance.
    return math.sqrt(variance)

def optimize_teams_by_cgpa(teams, max_iterations):
    """
    Phase 2: Optimize the formed teams by swapping members to balance CGPA.
    This is an implementation of a **Stochastic Hill Climbing** algorithm.
    It randomly picks two teams and tries to find the *best possible swap*
    between them that reduces the overall CGPA standard deviation.
    """
    if len(teams) < 2:
        # Cannot optimize if there's only 0 or 1 team.
        return teams
        
    # Get the initial "score" (cost) of the teams from Phase 1.
    current_std_dev = calculate_std_dev(teams)
    
    # Run the optimization loop.
    for _ in range(max_iterations):
        if current_std_dev == 0:
            # Teams are perfectly balanced. No need to continue.
            break
            
        # 1. SELECTION: Pick two different teams at random.
        # teams = [(..),(..),(..),(..),(..),(..)]
        team1_idx, team2_idx = random.sample(range(len(teams)), 2)
        
        team1, team2 = teams[team1_idx], teams[team2_idx]
        
        if not team1 or not team2: #ensure both teams are not empty 
            continue
            
        best_swap_indices = None
        best_new_std_dev = current_std_dev
        
        # 2. EXPLORATION: Check *every possible 1-for-1 swap* between team1 and team2.
        for p1_idx, p1 in enumerate(team1):
            for p2_idx, p2 in enumerate(team2):
            
                # --- CONSTRAINT CHECKING ---
                
                # CRITICAL CONSTRAINT 1: Only swap students of the SAME GENDER.
                # This ensures that the hard-won gender balance from Phase 1 is
                # *preserved* during this CGPA-balancing phase.
                if p1['Gender'] != p2['Gender']:
                 continue
                    
                # CRITICAL CONSTRAINT 2: Only swap if it does NOT create a school conflict.
                team1_schools_without_p1 = {p['School'] for p in team1 if p != p1}
                team2_schools_without_p2 = {p['School'] for p in team2 if p != p2}
                
                # Check if p2's school is already in team1, or p1's school is in team2.
                if p2['School'] in team1_schools_without_p1 or p1['School'] in team2_schools_without_p2:
                    continue
                    
                # --- SIMULATION & EVALUATION ---
                # If both constraints are met, *simulate* the swap.
                team1[p1_idx], team2[p2_idx] = team2[p2_idx], team1[p1_idx]
                
                # Evaluate the cost (std dev) of this new configuration.
                new_std_dev = calculate_std_dev(teams)
                
                # If this swap is *better* (lower std dev) than the best one
                # we've found *so far*, record it.
                if new_std_dev < best_new_std_dev:
                    best_new_std_dev = new_std_dev
                    best_swap_indices = (p1_idx, p2_idx)
                
                # <--- Fix: Correctly undo the simulated swap ---
                # This is essential. We are only *testing* the swap. We undo it
                # immediately so the next iteration of the inner loop tests
                # from the original state.
                team1[p1_idx], team2[p2_idx] = p1, p2
                
        # 3. COMMIT: After checking all $N \times M$ possible swaps, we find the best swap that can help improve the std the most!
        if best_swap_indices:
            p1_idx, p2_idx = best_swap_indices
            team1[p1_idx], team2[p2_idx] = team2[p2_idx], team1[p1_idx]
            # Update the score to beat for the next iteration.
            current_std_dev = best_new_std_dev
            
    return teams

# <--- Change: Removed default values for team_size and max_iterations ---
def form_teams(input_csv, output_csv, team_size, max_iterations):
    """
    Main controller function, responsible for managing the entire process
    of reading, processing, and writing data.
    """
    try:
        # Read all student records from the CSV into a list of dictionaries.
        with open(input_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            records = [row for row in reader]
            # Ensure CGPA is a float for mathematical operations.
            for row in records:
                row["CGPA"] = float(row["CGPA"])
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error reading input file: {e}")
        return
        
    all_assigned_students = []
    num_students = len(records)
    # Calculate how many batches (TGs) to process.
    num_tutorial_groups = num_students // TUTORIAL_GROUP_SIZE
    print(f"Found {num_students} students in total, divided into {num_tutorial_groups} tutorial groups.")
    
    # Process each Tutorial Group (batch) independently.
    for i in range(num_tutorial_groups):
        start_index = i * TUTORIAL_GROUP_SIZE
        end_index = start_index + TUTORIAL_GROUP_SIZE
        students_in_tg = records[start_index:end_index]
        
        if not students_in_tg:
            continue # Should not happen if logic is correct, but safe to have.
            
        tg_name = students_in_tg[0]["Tutorial Group"]
        print(f"\n--- Processing Tutorial Group: {tg_name} ---")
        
        # --- Run Phase 1 ---
        # The team_size passed in here comes from the user input in main.
        initial_teams, unassigned_phase1 = gender_balanced_teams(students_in_tg, team_size)
        
        # --- Filter Phase 1 Results ---
        complete_teams = []
        all_leftover_students = list(unassigned_phase1)
        # Any teams from Phase 1 that are *not* full are dissolved.
        # Their members are added to the leftover pool.
        for team in initial_teams:
            if len(team) == team_size:
                complete_teams.append(team)
            else:
                all_leftover_students.extend(team)
        print(f"Initial grouping complete: {len(complete_teams)} full teams, {len(all_leftover_students)} leftover students.")
        
        # --- Run Phase 2 ---
        # The max_iterations passed in here comes from the dynamic calculation in main.
        # Only the "complete" teams are optimized.
        optimized_teams = optimize_teams_by_cgpa(complete_teams, max_iterations)
        
        # --- Assign Team Names & Handle Leftovers ---
        team_number = 1
        # Assign names to the optimized teams.
        for team in optimized_teams:
            team_name = f"{tg_name}-T{team_number:02d}"
            for member in team:
                member["Team Assigned"] = team_name
            all_assigned_students.extend(team)
            team_number += 1
            
        # Handle all leftover students (from unassigned_phase1 + dissolved teams).
       ## if all_leftover_students:
            # A simple heuristic for leftovers: sort by CGPA
            # and chunk them into teams.
            all_leftover_students.sort(key=get_cgpa, reverse=True)
            student_idx = 0
            while student_idx < len(all_leftover_students):
                # Create new teams from slices of the leftover list.
                new_team = all_leftover_students[student_idx : student_idx + team_size]
                team_name = f"{tg_name}-T{team_number:02d}"
                for member in new_team:
                    member["Team Assigned"] = team_name
                all_assigned_students.extend(new_team)
                team_number += 1
                student_idx += team_size ## Modify 
                ##
    # --- Write Final Output ---
    if all_assigned_students:
        # Get all column headers from the first student, including the new one.
        fieldnames = list(all_assigned_students[0].keys())
        if "Team Assigned" not in fieldnames:
             fieldnames.append("Team Assigned")
             
        # Write all processed students to the final output file.
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_assigned_students)
        print(f"\nSuccessfully wrote assignment results for {len(all_assigned_students)} students to {output_csv}")
    else:
        print("No students were assigned. Output file is empty.")

# <--- Fix: Fixed the logic for extracting the group name ---
def get_tutorial_group_from_team_name(team_name):
    """Helper function to extract the tutorial group name from the full team name."""
    # Example: 'TG-01-T01' split by '-T' gives ['TG-01', '01'].
    # We take the first element [0], which is 'TG-01'.
    return team_name.split('-T')[0]

def summarise(output_csv):
    """Reads the output CSV file and prints a summary report."""
    try:
        # Read the CSV file using standard csv module
        with open(output_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            records = [row for row in reader]
            # Convert CGPA to float
            for row in records:
                row['CGPA'] = float(row['CGPA'])
    except FileNotFoundError:
        print(f"Error: '{output_csv}' file not found. Cannot generate summary.")
        return
    
    if not records:
        print("Summary: No teams were created.")
        return
    
    # Group records by Tutorial Group
    teams_by_tg = defaultdict(list)
    for record in records:
        tg_name = get_tutorial_group_from_team_name(record['Team Assigned'])
        record['Tutorial Group'] = tg_name
        teams_by_tg[tg_name].append(record)
    
    print("\n--- Team Formation Summary ---")
    print(f"Total students assigned: {len(records)}")
    
    # Count unique teams
    unique_teams = set(record['Team Assigned'] for record in records)
    print(f"Total teams created: {len(unique_teams)}")
    
    tg_std_devs = []
    # Analyze and print stats for each Tutorial Group
    for tg_name in sorted(teams_by_tg.keys()):
        group_records = teams_by_tg[tg_name]
        print(f"\n--- Tutorial Group: {tg_name} ---")
        
        # Group by team within this TG
        teams_dict = defaultdict(list)
        for record in group_records:
            teams_dict[record['Team Assigned']].append(record)
        
        # Convert to list of teams
        teams_in_group = list(teams_dict.values())
        
        # --- Report Metrics ---
        # 1. Academic Balance:
        std_dev = calculate_std_dev(teams_in_group)
        tg_std_devs.append(std_dev)
        print(f"  Std. Dev. of team average CGPAs within group: {std_dev:.6f} (Lower is more balanced)")
        
        # 2. School Diversity:
        diverse_teams_count = 0
        for team in teams_in_group:
            # A team is "fully diverse" if the number of unique schools
            # equals the number of members.
            if len(set(member['School'] for member in team)) == len(team):
                diverse_teams_count += 1
        print(f"  Teams with full school diversity: {diverse_teams_count} / {len(teams_in_group)}")
        
        # 3. Gender Balance:
        # Use Counter to get a frequency count of gender patterns.
        gender_patterns = Counter(
            tuple(sorted(member['Gender'] for member in team))
            for team in teams_in_group
        )
        print("  Gender Combination Distribution:")
        for pattern, count in sorted(gender_patterns.items()):
            f_count = pattern.count('Female')
            m_count = pattern.count('Male')
            print(f"    ({f_count} Female, {m_count} Male): {count} teams")
            
    # Print the overall average (mean of means) for std. dev.
    if tg_std_devs:
        avg_std_dev = sum(tg_std_devs) / len(tg_std_devs)
        print(f"\n--- Overall Performance ---")
        print(f"Average of internal standard deviations across all tutorial groups: {avg_std_dev:.6f}")

def create_visualizations(output_csv):
    """Generates and saves visualizations summarizing the results from the output file."""
    try:
        # Read the CSV file using standard csv module
        with open(output_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            records = [row for row in reader]
            # Convert CGPA to float
            for row in records:
                row['CGPA'] = float(row['CGPA'])
        
        if not records:
            print("\nCannot create visualizations because the output file is empty.")
            return
    except FileNotFoundError:
        print(f"\nCannot create visualizations because '{output_csv}' file not found.")
        return
    
    # Use a clean plotting style.
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Add Tutorial Group column
    for record in records:
        record['Tutorial Group'] = get_tutorial_group_from_team_name(record['Team Assigned'])
    
    # --- Plot 1: Overall CGPA and School Distribution (Histogram) ---
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # --- Top plot: CGPA Distribution ---
    cgpas = [record['CGPA'] for record in records]
    axs[0].hist(cgpas, bins=20, color='skyblue', edgecolor='black')
    axs[0].set_title('CGPA Distribution of All Students')
    axs[0].set_xlabel('CGPA')
    axs[0].set_ylabel('Number of Students')
    
    # --- Bottom plot: School Representation ---
    school_counts = Counter(record['School'] for record in records)
    schools = list(school_counts.keys())
    counts = list(school_counts.values())
    
    # Sort by count for better visualization
    sorted_pairs = sorted(zip(schools, counts), key=lambda x: x[1])
    schools_sorted = [pair[0] for pair in sorted_pairs]
    counts_sorted = [pair[1] for pair in sorted_pairs]
    
    axs[1].barh(schools_sorted, counts_sorted, color='salmon')
    axs[1].set_title('Number of Students per School')
    axs[1].set_xlabel('Number of Students')
    axs[1].set_ylabel('School')
    
    # --- Finalize and Save ---
    plt.tight_layout()
    plt.savefig('combined_vertical_plots.png')
    plt.close(fig)
    
    print("Saved combined vertical chart 'combined_vertical_plots.png'")

    
    # --- Plot 2: Team Balance (Box Plot) ---
    if records:
        # Calculate average CGPA for each team
        team_cgpas = defaultdict(list)
        team_tgs = {}
        for record in records:
            team_name = record['Team Assigned']
            team_cgpas[team_name].append(record['CGPA'])
            team_tgs[team_name] = record['Tutorial Group']
        
        # Create list of team average CGPAs with their TG
        team_avg_data = []
        for team_name, cgpas in team_cgpas.items():
            avg_cgpa = sum(cgpas) / len(cgpas)
            team_avg_data.append({
                'Team Assigned': team_name,
                'CGPA': avg_cgpa,
                'Tutorial Group': team_tgs[team_name]
            })
        
        n_teams = len(team_avg_data)
        sample_size = min(20, n_teams)
        
        if sample_size > 0:
            # Random sample
            sampled_data = random.sample(team_avg_data, sample_size)
            print(f"Total teams found: {n_teams}. Randomly sampled {sample_size} teams for the plot.")
            
            # Group by Tutorial Group for plotting
            tg_cgpas = defaultdict(list)
            for item in sampled_data:
                tg_cgpas[item['Tutorial Group']].append(item['CGPA'])
            
            plt.figure(figsize=(15, 8))
            
            # Prepare data for seaborn boxplot
            plot_data = []
            for tg, cgpas in tg_cgpas.items():
                for cgpa in cgpas:
                    plot_data.append({'Tutorial Group': tg, 'CGPA': cgpa})
            
            # Extract x and y values
            tgs = [item['Tutorial Group'] for item in plot_data]
            cgpas_plot = [item['CGPA'] for item in plot_data]
            
            # Create boxplot
            tg_unique = sorted(set(tgs))
            tg_data = {tg: [] for tg in tg_unique}
            for tg, cgpa in zip(tgs, cgpas_plot):
                tg_data[tg].append(cgpa)
            
            positions = list(range(len(tg_unique)))
            box_data = [tg_data[tg] for tg in tg_unique]
            
            plt.boxplot(box_data, positions=positions, vert=False, labels=tg_unique)
            
            # Add strip plot (individual points)
            for i, tg in enumerate(tg_unique):
                y_vals = [i + 1] * len(tg_data[tg])
                # Add jitter
                y_vals_jitter = [y + random.uniform(-0.1, 0.1) for y in y_vals]
                plt.scatter(tg_data[tg], y_vals_jitter, color='black', alpha=0.7, s=30)
                
            plt.title(f'Distribution of Team Average CGPAs (Random Sample of {sample_size} Teams)')
            plt.xlabel('Team Average CGPA')
            plt.ylabel('Tutorial Group')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.savefig('team_cgpa_balance_by_tg_SAMPLE.png')
            plt.close()
            print("Saved chart 'team_cgpa_balance_by_tg_SAMPLE.png'")
        else:
            print("Skipping plot: No teams to sample.")
    else:
        print("Skipping plot: No data found.")

    
    # --- Plot 3: Team Composition by Tutorial Group (Faculty-Gender 100% Stacked Bar) ---
        # File name is known
    file_name = "teams_assigned_final.csv"
    
    # --- Data Aggregation and Filtering ---
    
    # 1. Read all data and identify unique tutorial groups
    all_data = []
    all_tutorial_groups = set()
    try:
        with open(file_name, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                all_data.append(row)
                all_tutorial_groups.add(row['Tutorial Group'])
    except Exception as e:
        print(f"Error reading file: {e}")
        
    # 2. Select 5 random tutorial groups
    num_groups_to_select = 5
    # Note: sorted() is used to ensure random.sample takes a sequence
    selected_groups = random.sample(sorted(list(all_tutorial_groups)), min(num_groups_to_select, len(all_tutorial_groups)))
    
    # 3. Filter the data and aggregate counts for the selected groups
    filtered_data = [row for row in all_data if row['Tutorial Group'] in selected_groups]
    
    gender_counts = {}
    school_counts = {}
    
    for row in filtered_data:
        team = row['Team Assigned']
        gender = row['Gender']
        school = row['School']
        
        # Aggregate Gender Counts
        if team not in gender_counts:
            # Initialize counts for the team
            gender_counts[team] = {'Male': 0, 'Female': 0, 'Other': 0}
        if gender in gender_counts[team]:
            gender_counts[team][gender] += 1
        else:
            gender_counts[team]['Other'] += 1
    
        # Aggregate School Counts
        if team not in school_counts:
            school_counts[team] = {}
        school_counts[team][school] = school_counts[team].get(school, 0) + 1
    
    # --- Visualization 1: Gender Distribution ---
    
    gender_plot_file = 'random_5_groups_gender_distribution.png'
    if gender_counts:
        teams = list(gender_counts.keys())
        teams.sort()
        
        males = [gender_counts[t].get('Male', 0) for t in teams]
        females = [gender_counts[t].get('Female', 0) for t in teams]
        
        plt.figure(figsize=(12, 6))
        x = range(len(teams))
        
        plt.bar(x, males, label='Male', color='#377eb8')
        plt.bar(x, females, bottom=males, label='Female', color='#e41a1c')
        
        plt.ylabel('Number of Students')
        plt.xlabel('Assigned Team (Within Randomly Selected Groups)')
        plt.title(f'Gender Distribution in Randomly Selected Tutorial Groups: {", ".join(selected_groups)}')
        plt.xticks(x, teams, rotation=90)
        plt.legend(title='Gender')
        plt.tight_layout()
        plt.show() # Note: In the actual script, this is replaced by savefig
        plt.close()
    
    # --- Visualization 2: School/Faculty Distribution (Top 10) ---
    
    school_plot_file = 'random_5_groups_school_distribution.png'
    if school_counts:
        all_schools = set()
        for counts in school_counts.values():
            all_schools.update(counts.keys())
        
        teams = list(school_counts.keys())
        teams.sort()
        
        school_data = {school: [] for school in all_schools}
        for team in teams:
            for school in all_schools:
                school_data[school].append(school_counts[team].get(school, 0))
    
        # Identify top schools among the selected groups for readability
        school_totals = {school: sum(school_data[school]) for school in all_schools}
        top_schools = sorted(school_totals, key=school_totals.get, reverse=True)[:10]
        plot_school_data = {school: school_data[school] for school in top_schools}
    
        plt.figure(figsize=(15, 8))
        bottoms = [0] * len(teams)
        colors = plt.cm.get_cmap('tab20', len(top_schools))
    
        x = range(len(teams))
        
        for i, school in enumerate(top_schools):
            counts = plot_school_data[school]
            plt.bar(x, counts, bottom=bottoms, label=school, color=colors(i))
            bottoms = [bottoms[j] + counts[j] for j in range(len(teams))]
        plt.ylabel('Number of Students')
        plt.xlabel('Assigned Team (Within Randomly Selected Groups)')
        plt.title(f'School/Faculty Distribution in Randomly Selected Tutorial Groups: {", ".join(selected_groups)}')
        plt.xticks(x, teams, rotation=90)
        plt.legend(title='School/Faculty', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show() # Note: In the actual script, this is replaced by savefig
        plt.close()
            
    # --- End of Plot 3 ---


# --- Main Program Entry Point ---
if __name__ == "__main__":
    # Define input and output files.
    INPUT_FILE = "records.csv"
    OUTPUT_FILE = "teams_assigned_final.csv"

    # --- Get User Input ---
    user_team_size = 0
    while True:
        try:
            raw_input = input("Please enter the number of members per team (Recommended 4-10, press Enter for default 5): ")
            if not raw_input:
                user_team_size = 5
                print("No input detected, using default value: 5 members/team.")
                break
            user_team_size = int(raw_input)
            # Add validation checks.
            if user_team_size < 2 or user_team_size > TUTORIAL_GROUP_SIZE:
                print(f"Error: Invalid input. Team size must be between 2 and {TUTORIAL_GROUP_SIZE}. Please try again.")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid integer. Please try again.")

    # ==================== Dynamic Iteration Calculation ====================
    # This is the "smart" part of the script. It calculates the number of
    # optimization iterations based on the *combinatorial complexity* of
    # the problem, which is defined by the user's chosen team size.
    
    # 1. How many teams will be in each batch?
    num_teams_per_group = TUTORIAL_GROUP_SIZE // user_team_size
    
    if num_teams_per_group < 2:
        # Not enough teams to optimize (can't swap).
        dynamic_iterations = 0  
    else: 
       # 2. How many unique *pairs* of teams are there? 
       # This is the "N choose 2" formula: (N * (N-1)) / 2 
       num_team_pairs = (num_teams_per_group * (num_teams_per_group - 1)) // 2 
        
       # 3. How many possible 1-for-1 swaps are there *between* any two teams? 
       # (team_size * team_size) 
       swaps_per_pair = user_team_size * user_team_size 
        
       # 4. What is the *total* number of possible swaps in the search space? 
       total_complexity = num_team_pairs * swaps_per_pair 
        
       # 5. Scale this by our ALPHA to get the final iteration count. 
       # This auto-tunes the optimizer: 
       # - Small team size (e.g., 4) -> 12 teams/group -> fewer iterations needed. 
       # - Large team size (e.g., 10) -> 5 teams/group -> more iterations needed. 
       dynamic_iterations = int(total_complexity * EXPLORATION_ALPHA) 
   # ======================================================================= 

print(f"\nSetup successful! Grouping will proceed with {user_team_size} members/team.") 
print(f"Based on the problem complexity ({num_teams_per_group} teams/group) and exploration factor (Alpha={EXPLORATION_ALPHA}),") 
print(f"the optimization process will run for {dynamic_iterations} iterations.") 

   # --- Execute all steps in order --- 
    
   # Pass the user-inputted user_team_size and calculated dynamic_iterations 
   # to the main controller function. 
form_teams(INPUT_FILE, OUTPUT_FILE, team_size=user_team_size, max_iterations=dynamic_iterations) 
    
   # Run the reporting functions on the final output file. 
summarise(OUTPUT_FILE) 
create_visualizations(OUTPUT_FILE)