import logging
import random
from data import Station, Task, Vehicle

logging.basicConfig(level=logging.DEBUG)

def reposition_vehicles(stations: list[Station], vehicles: list[Vehicle], tasks: list[Task], assignments: list[str], algorithm: str) -> list[str]:
    repositioning = []
    
    # Calculate demand (unassigned tasks' fees) per station
    demand = {station.id: 0.0 for station in stations}
    for task in tasks:
        if not task.assigned:
            demand[task.origin] += float(task.fee)
    
    # Calculate supply (vehicles) per station
    supply = {station.id: 0 for station in stations}
    for vehicle in vehicles:
        supply[vehicle.station] += 1
    
    logging.debug(f"Demand: {demand}")
    logging.debug(f"Supply: {supply}")

    # vehical capacity at each station 
    vehicle_capacity = {station.id: 0 for station in stations}
    for vehicle in vehicles:
        vehicle_capacity[vehicle.station] += vehicle.capacity
    logging.debug(f"Vehicle capacity: {vehicle_capacity}")

    
    if algorithm == "rar":  # Random Assignment Reposition
        # Identify stations with any demand
        needs_vehicles = [(s.id, demand[s.id]) for s in stations if demand[s.id] > 0]
        if not needs_vehicles:
            logging.debug("No stations need vehicles for rar")
            return repositioning
        
        # Total demand for probability weights
        total_demand = sum(d for _, d in needs_vehicles)
        if total_demand == 0:
            logging.debug("Total demand is zero for rar")
            return repositioning
        
        # Normalize weights
        weights = [d / total_demand for _, d in needs_vehicles]
        logging.debug(f"rar weights: {dict(zip([s_id for s_id, _ in needs_vehicles], weights))}")
        
        # Reposition idle vehicles
        for vehicle in vehicles:
            assigned = any(vehicle.id in a.split()[0] for a in assignments)
            if not assigned and vehicle.electricity >= 10:
                # Randomly select target station based on demand weights
                target_station_id = random.choices(
                    [s_id for s_id, _ in needs_vehicles],
                    weights=weights,
                    k=1
                )[0]
                # needs_vehicles.sort(key=lambda x: x[1], reverse=True)
                # target_station_id = needs_vehicles[0][0]
                if vehicle.station != target_station_id:
                    repositioning.append(f"{vehicle.id} moves from {vehicle.station} to {target_station_id}")
                    vehicle.electricity = max(0, vehicle.electricity - 10)
                    vehicle.station = target_station_id
                    logging.debug(f"rar: Repositioned {vehicle.id} from {vehicle.station} to {target_station_id}, Battery: {vehicle.electricity}")
    
    elif algorithm == "rdr":  # Random Demand Reposition
        # Calculate demand-to-supply deficit
        deficits = {s.id: max(0, demand[s.id] - supply[s.id] * vehicle_capacity[s.id]) for s in stations}
        total_deficit = sum(deficits.values())
        if total_deficit == 0:
            logging.debug("No deficit for rdr")
            return repositioning
        
        # Normalize weights
        weights = [deficits[s.id] / total_deficit for s in stations]
        logging.debug(f"rdr weights: {dict(zip([s.id for s in stations], weights))}")
        
        # Reposition idle vehicles
        for vehicle in vehicles:
            assigned = any(vehicle.id in a.split()[0] for a in assignments)
            if not assigned and vehicle.electricity >= 10:
                # Randomly select target station based on deficit weights
                target_station_id = random.choices(
                    [s.id for s in stations],
                    weights=weights,
                    k=1
                )[0]
                if vehicle.station != target_station_id:
                    repositioning.append(f"{vehicle.id} moves from {vehicle.station} to {target_station_id}")
                    vehicle.electricity = max(0, vehicle.electricity - 10)
                    vehicle.station = target_station_id
                    logging.debug(f"rdr: Repositioned {vehicle.id} from {vehicle.station} to {target_station_id}, Battery: {vehicle.electricity}")
    
    return repositioning