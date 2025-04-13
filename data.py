class Station:
    def __init__(self, id, location, charging_points):
        self.id = id
        self.location = location  # e.g., "0-0" -> (0,0)
        self.charging_points = int(charging_points)

class Task:
    def __init__(self, origin, dest, fee, deadline, service_time):
        self.origin = origin
        self.dest = dest
        self.fee = float(fee)
        self.deadline = float(deadline)
        self.service_time = float(service_time)
        self.assigned = False

class Vehicle:
    def __init__(self, id, station, electricity, capacity):
        self.id = id
        self.station = station
        self.electricity = float(electricity)
        self.capacity = int(capacity)