import re
from transformers import pipeline
from datetime import datetime, timedelta

# Initialize the lightweight DistilGPT-2 model for text generation
generator = pipeline("text-generation", model="distilgpt2", device=0)  # Use CUDA if available

# Function to parse tasks and durations
def parse_tasks(tasks_input):
    tasks = []
    for task in tasks_input.split(","):
        task = task.strip()
        if ":" in task:
            # Handle explicit duration (e.g., "gym: 1h" or "write report: 90m")
            task_name, duration = [x.strip() for x in task.split(":", 1)]
            if duration.endswith("h"):
                hours = float(duration[:-1])
            elif duration.endswith("m"):
                hours = float(duration[:-1]) / 60
            else:
                hours = infer_duration(task_name)
            tasks.append({"name": task_name, "duration": hours})
        else:
            # Infer duration for tasks without explicit duration
            tasks.append({"name": task, "duration": infer_duration(task)})
    return tasks

# Function to infer duration based on task type
def infer_duration(task):
    task = task.lower()
    # Short tasks (~30-60 minutes)
    if any(keyword in task for keyword in ["call", "email", "shop", "walk", "lunch"]):
        return 0.5  # 30 minutes
    # Medium tasks (~1-2 hours)
    elif any(keyword in task for keyword in ["gym", "read", "meet", "plan", "study"]):
        return 1.0  # 1 hour
    # Long tasks (~2-3 hours)
    elif any(keyword in task for keyword in ["write", "code", "design", "prepare", "project"]):
        return 2.0  # 2 hours
    return 1.0  # Default: 1 hour

# Function to generate a daily schedule
def generate_schedule(tasks_input, time_range):
    # Parse tasks and durations
    tasks = parse_tasks(tasks_input)
    # Create a prompt for the model
    prompt = f"""
    Create a daily schedule for the following tasks: {', '.join(f'{t["name"]} ({t["duration"]}h)' for t in tasks)}.
    Available time: {time_range}.
    Format the schedule as a list with time slots and tasks, e.g., '8:00 AM - 9:00 AM: Task'.
    Respect the task durations and spread tasks within the time range.
    """
    
    # Generate text using DistilGPT-2
    response = generator(
        prompt,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        truncation=True
    )[0]["generated_text"]
    
    # Extract and clean the schedule from the response
    schedule = clean_schedule(response, tasks, time_range)
    return schedule

# Function to clean and format the generated schedule
def clean_schedule(response, tasks, time_range):
    # Default to a dynamic schedule if the model output is poor
    default_schedule = generate_default_schedule(tasks, time_range)
    
    # Try to extract time slots and tasks from the response
    lines = response.split("\n")
    schedule_lines = []
    for line in lines:
        # Look for lines resembling a schedule (e.g., "8:00 AM - 9:00 AM: Task")
        if re.match(r"\d{1,2}:\d{2}\s[AP]M\s*-\s*\d{1,2}:\d{2}\s[AP]M:.*", line):
            schedule_lines.append(line.strip())
    
    # If no valid schedule lines are found, return the default schedule
    if not schedule_lines:
        return default_schedule
    
    # Ensure all tasks are included
    included_tasks = [line.split(": ")[-1].lower() for line in schedule_lines]
    for task in tasks:
        if not any(task["name"].lower() in included for included in included_tasks):
            schedule_lines.append(f"Extra slot: {task['name']}")
    
    return "\n".join(schedule_lines) if schedule_lines else default_schedule

# Function to generate a fallback dynamic schedule
def generate_default_schedule(tasks, time_range):
    # Parse the time range (e.g., "8 AM–6 PM")
    try:
        start_time, end_time = re.findall(r"\d{1,2}\s*[AP]M", time_range)
        start_hour = int(re.findall(r"\d{1,2}", start_time)[0])
        if "PM" in start_time and start_hour != 12:
            start_hour += 12
        end_hour = int(re.findall(r"\d{1,2}", end_time)[0])
        if "PM" in end_time and end_hour != 12:
            end_hour += 12
        if "AM" in end_time and end_hour == 12:
            end_hour = 0
    except:
        start_hour, end_hour = 8, 18  # Default to 8 AM - 6 PM
    
    total_hours = end_hour - start_hour
    if total_hours <= 0:
        total_hours = 10  # Default to 10 hours if invalid
    
    # Check if total task durations fit within the time range
    total_task_hours = sum(task["duration"] for task in tasks)
    if total_task_hours > total_hours:
        return f"Error: Total task duration ({total_task_hours}h) exceeds available time ({total_hours}h)."
    
    # Generate schedule respecting task durations
    schedule = []
    current_time = datetime.strptime(f"{start_hour}:00", "%H:%M")
    
    for task in tasks:
        duration_hours = task["duration"]
        start_time = current_time
        end_time = start_time + timedelta(hours=duration_hours)
        
        # Format times
        start_str = start_time.strftime("%I:%M %p").lstrip("0")
        end_str = end_time.strftime("%I:%M %p").lstrip("0")
        schedule.append(f"{start_str} - {end_str}: {task['name']}")
        
        current_time = end_time
    
    # Check if schedule exceeds time range
    last_end_hour = current_time.hour + current_time.minute / 60
    if last_end_hour > end_hour:
        return f"Error: Schedule exceeds available time range ({time_range})."
    
    return "\n".join(schedule)

# Console-based main function
def main():
    print("AI Daily Planner Generator")
    print("Enter your tasks (optionally with durations, e.g., 'gym: 1h') and available time.")
    print("Example tasks: write report: 2h, gym: 1h, call friend")
    print("Example time range: 8 AM–6 PM")
    
    # Get user input
    tasks_input = input("Tasks (comma-separated): ").strip()
    time_range = input("Available time: ").strip()
    
    # Validate input
    if not tasks_input or not time_range:
        print("Error: Please provide both tasks and a time range.")
        return
    
    # Generate and display schedule
    print("\nGenerating your daily schedule...")
    schedule = generate_schedule(tasks_input, time_range)
    print("\nYour Daily Schedule:")
    print(schedule)

# Run the script
if __name__ == "__main__":
    main()