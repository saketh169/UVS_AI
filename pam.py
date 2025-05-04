import logging
from data import Task, Vehicle

logging.basicConfig(level=logging.DEBUG)

def pam_algorithm(tasks: list[Task], vehicles: list[Vehicle], station: str, current_time: float) -> list[str]:
    assignments = []
    
    # Filter tasks and vehicles at the given station
    available_tasks = [
        task for task in tasks
        if task.origin == station and not task.assigned and task.deadline >= current_time
    ]
    available_vehicles = [v for v in vehicles if v.station == station]
    
    # Sort tasks by fee (descending)
    available_tasks.sort(key=lambda x: x.fee, reverse=True)
    
    logging.debug(f"Available tasks at {station}: {[f'{t.origin}->{t.dest}: ${t.fee}' for t in available_tasks]}")
    logging.debug(f"Available vehicles at {station}: {[f'{v.id}: {v.electricity}' for v in available_vehicles]}")
    
    # Assign tasks to vehicles
    for task in available_tasks:
        # Sort vehicles by electricity (descending) before each assignment
        available_vehicles.sort(key=lambda x: x.electricity, reverse=True)
        assigned = False
        for vehicle in available_vehicles:
            if vehicle.capacity >= 1 and vehicle.electricity >= task.service_time * 2:
                assignments.append(
                    f"{vehicle.id} -> 1 tasks to {task.dest} (Fees: {task.fee})"
                )
                vehicle.station = task.dest
                vehicle.capacity -= 1
                vehicle.electricity = max(0, vehicle.electricity - task.service_time * 2)
                task.assigned = True
                assigned = True
                logging.debug(f"Assigned {vehicle.id} to task {task.origin}->{task.dest}, Fee: {task.fee}, Battery: {vehicle.electricity}")
                break
        if not assigned:
            logging.debug(f"No vehicle assigned to task {task.origin}->{task.dest}")
    
    return assignments