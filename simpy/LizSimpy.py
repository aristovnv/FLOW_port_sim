import simpy
import random

class Port:
    def __init__(self, env, terminals, shared_yard_capacity):
        self.env = env
        
        # Terminals
        self.terminals = terminals
        
        # Shared regular yard for all terminals
        self.shared_yard = simpy.FilterStore(env, capacity=shared_yard_capacity)
    
    def process_ship(self, ship, terminal):
        yield self.env.process(terminal.unload_ship(ship))
        yield self.env.process(terminal.load_ship(ship))

    def process_train(self, train, terminal):
        yield self.env.process(terminal.unload_train(train))
        yield self.env.process(terminal.load_train(train))

    def route_container_to_yard(self, container, terminal):
        # Try to place in terminal-specific regular yards
        for yard in terminal.regular_yards:
            if len(yard.items) < yard.capacity:
                yield yard.put(container)
                print(f"Container placed in {terminal.name}'s regular yard at {self.env.now}")
                return
        
        # If terminal-specific yards are full, place in shared yard
        if len(self.shared_yard.items) < self.shared_yard.capacity:
            yield self.shared_yard.put(container)
            print(f"Container placed in shared regular yard at {self.env.now}")
        else:
            print(f"No space in {terminal.name}'s yards or shared yard at {self.env.now}")


class Terminal:
    def __init__(self, env, name, num_cranes_ship, num_cranes_yard, num_chassis, num_train_cranes, yard_capacity):
        self.env = env
        self.name = name
        
        # Resources
        self.ship_cranes = simpy.Resource(env, num_cranes_ship)
        self.yard_cranes = simpy.Resource(env, num_cranes_yard)
        self.train_cranes = simpy.Resource(env, num_train_cranes)
        self.chassis = simpy.Resource(env, num_chassis)
        
        # Queues
        self.ship_queue = simpy.Store(env)
        self.train_queue = simpy.Store(env)
        self.truck_queue = simpy.Store(env)
        
        # Yard sections
        self.yards = {
            'reefer': simpy.FilterStore(env, capacity=yard_capacity['reefer']),
            'hazardous': simpy.FilterStore(env, capacity=yard_capacity['hazardous']),
        }
        self.regular_yards = [simpy.FilterStore(env, capacity=yard_capacity['regular']) for _ in range(2)]  # 2 regular yards per terminal
    
    def unload_ship(self, ship):
        with self.ship_cranes.request() as req:
            yield req
            containers_to_unload = min(ship.num_containers, len(ship.containers))
            print(f"Unloading {containers_to_unload} containers from ship {ship.name} at {self.env.now} in {self.name}")
            for _ in range(containers_to_unload):
                container = ship.containers.pop(0)
                yield self.env.process(self.handle_container(container))
                yield self.env.timeout(1)

    def load_ship(self, ship):
        with self.ship_cranes.request() as req:
            yield req
            containers_to_load = random.randint(50, 150)  # Load some random containers back to ship
            print(f"Loading {containers_to_load} containers onto ship {ship.name} at {self.env.now} in {self.name}")
            for _ in range(containers_to_load):
                # Get container from terminal's regular yards
                for yard in self.regular_yards:
                    if len(yard.items) > 0:
                        container = yield yard.get()
                        ship.containers.append(container)
                        yield self.env.timeout(1)
                        break

    def unload_train(self, train):
        with self.train_cranes.request() as req:
            yield req
            containers_to_unload = min(train.num_containers, len(train.containers))
            print(f"Unloading {containers_to_unload} containers from train {train.name} at {self.env.now} in {self.name}")
            for _ in range(containers_to_unload):
                container = train.containers.pop(0)
                yield self.env.process(self.handle_container(container))
                yield self.env.timeout(1)

    def load_train(self, train):
        with self.train_cranes.request() as req:
            yield req
            containers_to_load = random.randint(20, 50)  # Load some random containers back to train
            print(f"Loading {containers_to_load} containers onto train {train.name} at {self.env.now} in {self.name}")
            for _ in range(containers_to_load):
                # Get container from terminal's regular yards
                for yard in self.regular_yards:
                    if len(yard.items) > 0:
                        container = yield yard.get()
                        train.containers.append(container)
                        yield self.env.timeout(1)
                        break

    def handle_container(self, container):
        # Use chassis to move containers
        with self.chassis.request() as req:
            yield req
            print(f"Container moved by chassis at {self.env.now} in {self.name}")
            if container.is_reefer:
                yield self.yards['reefer'].put(container)
                print(f"Container placed in reefer yard at {self.env.now} in {self.name}")
            elif container.is_hazard:
                yield self.yards['hazardous'].put(container)
                print(f"Container placed in hazardous yard at {self.env.now} in {self.name}")
            else:
                # For regular containers, they need to be routed either to terminal-specific or shared yard
                yield self.env.process(port.route_container_to_yard(container, self))


class Container:
    def __init__(self, is_reefer=False, is_hazard=False, is_empty=False):
        self.is_reefer = is_reefer
        self.is_hazard = is_hazard
        self.is_empty = is_empty

class Ship:
    def __init__(self, name, num_containers):
        self.name = name
        self.num_containers = num_containers
        self.containers = [Container(random.choice([True, False]), random.choice([True, False])) for _ in range(num_containers)]

class Train_simple:
    def __init__(self, name, num_containers):
        self.name = name
        self.num_containers = num_containers
        self.containers = [Container(random.choice([True, False]), random.choice([True, False])) for _ in range(num_containers)]

class Train:
    def __init__(self, name, num_containers, departure_time):
        self.name = name
        self.num_containers = num_containers  # The target number of containers to load
        self.containers_loaded = 0  # Start with zero containers
        self.departure_time = departure_time  # The time the train will leave
        self.containers = []  # Containers will be loaded during the stay

    def load_container(self, container):
        self.containers.append(container)
        self.containers_loaded += 1

def ship_arrival(env, port):
    while True:
        terminal = random.choice(port.terminals)
        ship = Ship(f"Ship_{random.randint(1,100)}", random.randint(100, 500))
        print(f"Ship {ship.name} arrives at {env.now} with {ship.num_containers} containers to {terminal.name}")
        env.process(port.process_ship(ship, terminal))
        yield env.timeout(random.randint(30, 50))

def train_arrival_simple(env, port):
    while True:
        terminal = random.choice(port.terminals)
        train = Train(f"Train_{random.randint(1,100)}", random.randint(50, 200))
        print(f"Train {train.name} arrives at {env.now} with {train.num_containers} containers to {terminal.name}")
        env.process(port.process_train(train, terminal))
        yield env.timeout(random.randint(60, 100))

def train_arrival(env, port, train_schedule):
    for schedule in train_schedule:
        # Wait until the scheduled time for the train to arrive
        yield env.timeout(schedule['arrival_time'] - env.now)
        
        # Find the appropriate terminal based on the schedule
        terminal = next(t for t in port.terminals if t.name == schedule['terminal'])
        
        # Create the train and process its arrival
        train = Train(schedule['train_name'], schedule['num_containers'], schedule['departure_time'])
        print(f"Train {train.name} arrives at {env.now} with {train.num_containers} containers to {terminal.name}. Departure at {train.departure_time}.")
        env.process(port.process_train(train, terminal))


def train_loading(env, train, terminal):
    while env.now < train.departure_time:
        # Check if the train has reached its full capacity
        if train.containers_loaded >= train.num_containers:
            print(f"Train {train.name} at {terminal.name} is fully loaded at {env.now}. Ready to depart.")
            break
        
        # Load containers onto the train
        if len(terminal.yard['regular']) > 0:  # Assume regular containers are loaded onto trains
            container = terminal.yard['regular'].pop(0)
            train.load_container(container)
            print(f"Loading container onto {train.name} at {terminal.name}. Containers loaded: {train.containers_loaded}/{train.num_containers}.")
            yield env.timeout(5)  # Assume it takes 5 units of time to load a container
        else:
            # If no containers are available, wait for more to arrive
            yield env.timeout(1)

    # Train departs, whether fully loaded or not
    print(f"Train {train.name} departs from {terminal.name} at {env.now}. Loaded {train.containers_loaded}/{train.num_containers} containers.")

def process_train(env, port, train, terminal):
    # Start loading process for the train
    loading_process = env.process(train_loading(env, train, terminal))
    
    # Wait until the train is scheduled to depart
    yield env.timeout(train.departure_time - env.now)
    
    # Ensure the train departs on time
    if not loading_process.triggered:
        loading_process.interrupt()
    
    print(f"Train {train.name} leaves {terminal.name} at {env.now} with {train.containers_loaded}/{train.num_containers} containers loaded.")

# Environment setup
env = simpy.Environment()

# Terminal yard capacities
yard_capacity = {'reefer': 100, 'hazardous': 50, 'regular': 300}

# Initialize terminals
terminals = [
    Terminal(env, "Maher", num_cranes_ship=2, num_cranes_yard=2, num_chassis=3, num_train_cranes=1, yard_capacity=yard_capacity),
    Terminal(env, "APM", num_cranes_ship=2, num_cranes_yard=2, num_chassis=3, num_train_cranes=1, yard_capacity=yard_capacity),
    Terminal(env, "Newark", num_cranes_ship=2, num_cranes_yard=2, num_chassis=3, num_train_cranes=1, yard_capacity=yard_capacity)
]

# Shared yard capacity
shared_yard_capacity = 500

# Initialize port
port = Port(env, terminals, shared_yard_capacity)

train_schedule = [
    {'arrival_time': 50, 'terminal': 'Maher', 'train_name': 'Train_1', 'num_containers': 120, 'departure_time': 200},
    {'arrival_time': 150, 'terminal': 'APM', 'train_name': 'Train_2', 'num_containers': 80, 'departure_time': 300},
    {'arrival_time': 300, 'terminal': 'Newark', 'train_name': 'Train_3', 'num_containers': 100, 'departure_time': 450},
    {'arrival_time': 450, 'terminal': 'Maher', 'train_name': 'Train_4', 'num_containers': 90, 'departure_time': 600},
    # Add more scheduled trains as needed
]

# Process for ships and trains
env.process(ship_arrival(env, port))
env.process(train_arrival(env, port, train_schedule))

# Run simulation
env.run(until=500)
