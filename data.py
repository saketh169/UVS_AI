class Station:
    def __init__(self, id, location, charging_points):
        self.id = id
        self.location = location  # (x, y) tuple
        self.charging_points = int(charging_points)

class Task:
    def __init__(self, origin, dest, fee, deadline, service_time, stations=None):
        self.origin = origin
        self.dest = dest
        self.fee = float(fee)
        self.deadline = float(deadline)
        self.service_time = float(service_time)
        # Set location based on destination station, with fallback
        self.location = next((s.location for s in stations or [] if s.id == dest), (0, 0))

class Vehicle:
    def __init__(self, id, station, electricity, capacity, stations=None):
        self.id = id
        self.station = station
        self.electricity = float(electricity)
        self.capacity = int(capacity)
        # Set location based on station, with fallback
        self.location = next((s.location for s in stations or [] if s.id == station), (0, 0))