"""
Port Simulation using SimPy
"""
import simpy
import random
import numpy as np
from datetime import datetime, timedelta
from config import *
import matplotlib.pyplot as plt
from collections import defaultdict

class Container:
    def __init__(self, id, arrival_time):
        self.id = id
        
        # Use normal distribution for reefer percentage (mean=5%, std=0.5%)
        reefer_prob = max(0, min(1, random.normalvariate(0.05, 0.005)))
        self.type = 'reefer' if random.random() < reefer_prob else 'dry'
        
        # Use normal distribution for train percentage (mean=15%, std=1%)
        train_prob = max(0, min(1, random.normalvariate(0.15, 0.01)))
        self.modal = 'train' if random.random() < train_prob else 'truck'
        
        self.arrival_time = arrival_time
        self.yard_entry_time = None
        self.ready_time = None
        self.departure_time = None
        
        # Set planned dwell time based on type
        if self.type == 'reefer':
            # 2.1 ± 0.8 days
            self.dwell_time = random.normalvariate(2.1, 0.8) * 24 * 60
        else:
            # 2.47 ± 1.5 days
            self.dwell_time = random.normalvariate(2.47, 1.5) * 24 * 60
        self.dwell_time = max(60, self.dwell_time)  # Minimum 1 hour dwell time

class Ship:
    def __init__(self, env, id, container_count):
        self.env = env
        self.id = id
        self.container_count = container_count
        self.containers = self._generate_containers()

    def _generate_containers(self):
        containers = []
        for i in range(self.container_count):
            containers.append(Container(
                f"{self.id}_container_{i}",
                self.env.now
            ))
        return containers

class Statistics:
    def __init__(self, env):
        self.env = env
        self.containers = {
            'total': 0,
            'dry': {'truck': 0, 'train': 0},
            'reefer': {'truck': 0, 'train': 0}
        }
        self.arrivals = {
            'ships': 0,
            'containers': {
                'total': 0,
                'dry': 0,
                'reefer': 0
            }
        }
        self.departures = {
            'truck': {
                'total': 0,
                'by_hour': [0] * 24,  # Track departures by hour of day
                'dry': 0,
                'reefer': 0
            },
            'train': {
                'total': 0,
                'by_hour': [0] * 24,
                'dry': 0,
                'reefer': 0
            }
        }
        self.yard_state = {
            'current': {'dry': 0, 'reefer': 0},
            'max': {'dry': 0, 'reefer': 0}
        }
        self.dwell_times = {
            'dry': [],
            'reefer': []
        }
        self.wait_times = {
            'ship': [],
            'gate': [],
            'train': []
        }
        self.yard_full_events = 0
        self.missed_train_connections = 0
        
        # Add time series tracking
        self.hourly_stats = {
            'arrivals': defaultdict(lambda: defaultdict(int)),  # {hour: {'dry': count, 'reefer': count}}
            'departures': {
                'truck': defaultdict(lambda: defaultdict(int)),  # {hour: {'dry': count, 'reefer': count}}
                'train': defaultdict(lambda: defaultdict(int))
            }
        }
        
        # Add yard utilization tracking
        self.yard_utilization = {
            'dry': {'truck': defaultdict(int), 'train': defaultdict(int)},
            'reefer': {'truck': defaultdict(int), 'train': defaultdict(int)}
        }

    def log_ship_arrival(self):
        self.arrivals['ships'] += 1
        
    def log_container_arrival(self, container_type):
        self.arrivals['containers']['total'] += 1
        self.arrivals['containers'][container_type] += 1
        
        # Add hourly tracking
        current_hour = int(self.env.now / 60)  # Convert minutes to hours
        self.hourly_stats['arrivals'][current_hour][container_type] += 1

    def log_container_departure(self, container_type, modal, hour):
        self.departures[modal]['total'] += 1
        self.departures[modal]['by_hour'][hour] += 1
        self.departures[modal][container_type] += 1
        
        # Add hourly tracking
        current_hour = int(self.env.now / 60)  # Convert minutes to hours
        self.hourly_stats['departures'][modal][current_hour][container_type] += 1

    def update_yard_state(self, dry_count, reefer_count):
        """Update yard state with current levels"""
        self.yard_state['current']['dry'] = dry_count
        self.yard_state['current']['reefer'] = reefer_count
        self.yard_state['max']['dry'] = max(self.yard_state['max']['dry'], dry_count)
        self.yard_state['max']['reefer'] = max(self.yard_state['max']['reefer'], reefer_count)
        
        # Track yard utilization hourly
        current_hour = int(self.env.now / 60)
        for container_type in ['dry', 'reefer']:
            for modal in ['truck', 'train']:
                self.yard_utilization[container_type][modal][current_hour] = \
                    self.yard_state['current'][container_type]

    def log_container(self, container_type, modal):
        self.containers['total'] += 1
        self.containers[container_type][modal] += 1
        
    def log_dwell_time(self, container_type, dwell_time):
        self.dwell_times[container_type].append(dwell_time)
        
    def log_wait_time(self, wait_type, wait_time):
        self.wait_times[wait_type].append(wait_time)
        
    def log_yard_full(self):
        self.yard_full_events += 1
        
    def log_missed_train(self):
        self.missed_train_connections += 1
        
    def get_summary(self):
        total = self.containers['total']
        dry_total = self.containers['dry']['truck'] + self.containers['dry']['train']
        reefer_total = self.containers['reefer']['truck'] + self.containers['reefer']['train']
        
        return {
            'Arrivals': {
                'Ships': self.arrivals['ships'],
                'Containers': {
                    'Total': self.arrivals['containers']['total'],
                    'Dry': self.arrivals['containers']['dry'],
                    'Reefer': self.arrivals['containers']['reefer']
                }
            },
            'Departures': {
                'Truck': {
                    'Total': self.departures['truck']['total'],
                    'Dry': self.departures['truck']['dry'],
                    'Reefer': self.departures['truck']['reefer'],
                    'Peak Hour': max(enumerate(self.departures['truck']['by_hour']), 
                                   key=lambda x: x[1])[0]
                },
                'Train': {
                    'Total': self.departures['train']['total'],
                    'Dry': self.departures['train']['dry'],
                    'Reefer': self.departures['train']['reefer'],
                    'Peak Hour': max(enumerate(self.departures['train']['by_hour']), 
                                   key=lambda x: x[1])[0]
                }
            },
            'Yard State': {
                'Current': self.yard_state['current'],
                'Maximum': self.yard_state['max']
            },
            'Container Counts': {
                'Total Processed': total,
                'By Type': {
                    'Dry': {
                        'Total': dry_total,
                        'Truck': self.containers['dry']['truck'],
                        'Train': self.containers['dry']['train']
                    },
                    'Reefer': {
                        'Total': reefer_total,
                        'Truck': self.containers['reefer']['truck'],
                        'Train': self.containers['reefer']['train']
                    }
                },
                'Modal Split': {
                    'Truck': (self.containers['dry']['truck'] + self.containers['reefer']['truck']) / total if total > 0 else 0,
                    'Train': (self.containers['dry']['train'] + self.containers['reefer']['train']) / total if total > 0 else 0
                }
            },
            'Dwell Times (days)': {
                'Dry': {
                    'Mean': np.mean(self.dwell_times['dry']) / (24 * 60) if self.dwell_times['dry'] else 0,
                    'Median': np.median(self.dwell_times['dry']) / (24 * 60) if self.dwell_times['dry'] else 0,
                    'Std Dev': np.std(self.dwell_times['dry']) / (24 * 60) if self.dwell_times['dry'] else 0,
                    '95th Percentile': np.percentile(self.dwell_times['dry'], 95) / (24 * 60) if self.dwell_times['dry'] else 0
                },
                'Reefer': {
                    'Mean': np.mean(self.dwell_times['reefer']) / (24 * 60) if self.dwell_times['reefer'] else 0,
                    'Median': np.median(self.dwell_times['reefer']) / (24 * 60) if self.dwell_times['reefer'] else 0,
                    'Std Dev': np.std(self.dwell_times['reefer']) / (24 * 60) if self.dwell_times['reefer'] else 0,
                    '95th Percentile': np.percentile(self.dwell_times['reefer'], 95) / (24 * 60) if self.dwell_times['reefer'] else 0
                }
            },
            'Wait Times (hours)': {
                'Ship': {
                    'Mean': np.mean(self.wait_times['ship']) / 60 if self.wait_times['ship'] else 0,
                    '95th Percentile': np.percentile(self.wait_times['ship'], 95) / 60 if self.wait_times['ship'] else 0
                },
                'Gate': {
                    'Mean': np.mean(self.wait_times['gate']) / 60 if self.wait_times['gate'] else 0,
                    '95th Percentile': np.percentile(self.wait_times['gate'], 95) / 60 if self.wait_times['gate'] else 0
                },
                'Train': {
                    'Mean': np.mean(self.wait_times['train']) / 60 if self.wait_times['train'] else 0,
                    '95th Percentile': np.percentile(self.wait_times['train'], 95) / 60 if self.wait_times['train'] else 0
                }
            },
            'Operational Issues': {
                'Yard Full Events': self.yard_full_events,
                'Missed Train Connections': self.missed_train_connections
            }
        }

    def plot_statistics(self):
        """Generate visualizations of key statistics"""
        #plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 15))  # Made figure taller for 6 subplots
        
        # 1. Container Arrivals Time Series
        ax1 = plt.subplot(3, 2, 1)
        hours = sorted(self.hourly_stats['arrivals'].keys())
        dry_arrivals = [self.hourly_stats['arrivals'][h]['dry'] for h in hours]
        reefer_arrivals = [self.hourly_stats['arrivals'][h]['reefer'] for h in hours]
        
        ax1.plot(hours, dry_arrivals, label='Dry', color='blue')
        ax1.plot(hours, reefer_arrivals, label='Reefer', color='red')
        ax1.set_title('Container Arrivals Over Time')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Number of Containers')
        ax1.legend()
        
        # 2. Container Departures Time Series
        ax2 = plt.subplot(3, 2, 2)
        truck_dry = [self.hourly_stats['departures']['truck'][h]['dry'] for h in hours]
        truck_reefer = [self.hourly_stats['departures']['truck'][h]['reefer'] for h in hours]
        train_dry = [self.hourly_stats['departures']['train'][h]['dry'] for h in hours]
        train_reefer = [self.hourly_stats['departures']['train'][h]['reefer'] for h in hours]
        
        ax2.plot(hours, truck_dry, label='Truck-Dry', color='blue')
        ax2.plot(hours, truck_reefer, label='Truck-Reefer', color='red')
        ax2.plot(hours, train_dry, label='Train-Dry', color='green')
        ax2.plot(hours, train_reefer, label='Train-Reefer', color='orange')
        ax2.set_title('Container Departures Over Time')
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Number of Containers')
        ax2.legend()
        
        # 3. Modal Split Pie Chart
        ax3 = plt.subplot(3, 2, 3)
        modal_data = [
            self.departures['truck']['total'],
            self.departures['train']['total']
        ]
        ax3.pie(modal_data, labels=['Truck', 'Train'], autopct='%1.1f%%')
        ax3.set_title('Modal Split of Departures')
        
        # 4. Container Type Split Pie Chart
        ax4 = plt.subplot(3, 2, 4)
        type_data = [
            self.arrivals['containers']['dry'],
            self.arrivals['containers']['reefer']
        ]
        ax4.pie(type_data, labels=['Dry', 'Reefer'], autopct='%1.1f%%')
        ax4.set_title('Container Type Split')
        
        # 5. Dry Container Yard Utilization
        ax5 = plt.subplot(3, 2, 5)
        hours = sorted(self.yard_utilization['dry']['truck'].keys())
        
        dry_truck = [self.yard_utilization['dry']['truck'][h] for h in hours]
        dry_train = [self.yard_utilization['dry']['train'][h] for h in hours]
        
        ax5.plot(hours, dry_truck, label='Truck', color='blue')
        ax5.plot(hours, dry_train, label='Train', color='green')
        ax5.set_title('Dry Container Yard Utilization')
        ax5.set_xlabel('Hour')
        ax5.set_ylabel('Container Count')
        ax5.legend()
        
        # 6. Reefer Container Yard Utilization
        ax6 = plt.subplot(3, 2, 6)
        reefer_truck = [self.yard_utilization['reefer']['truck'][h] for h in hours]
        reefer_train = [self.yard_utilization['reefer']['train'][h] for h in hours]
        
        ax6.plot(hours, reefer_truck, label='Truck', color='red')
        ax6.plot(hours, reefer_train, label='Train', color='orange')
        ax6.set_title('Reefer Container Yard Utilization')
        ax6.set_xlabel('Hour')
        ax6.set_ylabel('Container Count')
        ax6.legend()
        
        plt.tight_layout()
        plt.show()

class PortSimulation:
    def __init__(self):
        self.env = simpy.Environment()
        self.berths = simpy.Resource(self.env, capacity=MAX_BERTHS)
        self.cranes = simpy.Resource(self.env, capacity=MAX_BERTHS * CRANES_PER_BERTH)
        self.gate = simpy.Resource(self.env, capacity=50)
        
        # Initialize yard containers tracking
        self.yard_containers = {
            'dry': {
                'truck': [],
                'train': []
            },
            'reefer': {
                'truck': [],
                'train': []
            }
        }
        
        # Create yards with modal split
        self.regular_yard = {
            'truck': simpy.Container(self.env, capacity=int(25000 * 0.85)),  # 21,250
            'train': simpy.Container(self.env, capacity=int(25000 * 0.15))   # 3,750
        }
        self.reefer_yard = {
            'truck': simpy.Container(self.env, capacity=int(2000 * 0.85)),  # 1,700
            'train': simpy.Container(self.env, capacity=int(2000 * 0.15))   # 300
        }
        
        # Train schedule (4 trains per day)
        self.train_times = [6, 12, 18, 24]  # 6am, 12pm, 6pm, 12am
        
        self.stats = Statistics(self.env)
        self.env.process(self.schedule_trains())

    def schedule_trains(self):
        """Schedule 4 trains per day at fixed times"""
        while True:
            current_hour = (self.env.now / 60) % 24  # Current hour of day
            
            # Find next train time
            next_train = next((t for t in self.train_times if t > current_hour), self.train_times[0])
            
            # Calculate wait time until next train
            wait_time = (next_train - current_hour) * 60 if next_train > current_hour else (24 - current_hour + next_train) * 60
            
            yield self.env.timeout(wait_time)
            # Process train arrival
            self.env.process(self.handle_train_departure())

    def generate_ship_arrival(self):
        """Generate ships based on historical data distribution"""
        ship_id = 0
        while True:
            # Generate container count based on historical data
            container_count = np.random.choice([ship["containers"] for ship in SHIPS_DATA])
            ship = Ship(self.env, f"ship_{ship_id}", container_count)
            self.env.process(self.handle_ship(ship))
            
            # Random time between ship arrivals (4-12 hours)
            yield self.env.timeout(random.uniform(4 * 60, 12 * 60))
            ship_id += 1

    def handle_ship(self, ship):
        """Process a ship's arrival and unloading"""
        arrival_time = self.env.now
        self.stats.log_ship_arrival()
        
        # Request a berth
        with self.berths.request() as berth_req:
            yield berth_req
            
            # Start unloading containers
            for container in ship.containers:
                self.stats.log_container_arrival(container.type)
                yield self.env.process(self.unload_container(container))
            
            # Wait for berth transition
            yield self.env.timeout(BERTH_TRANSITION_TIME)
            
        # Record statistics
        self.stats.log_wait_time('ship', self.env.now - arrival_time)

    def unload_container(self, container):
        """Process container unloading from ship"""
        with self.cranes.request() as crane_req:
            yield crane_req
            
            # Unloading time
            unload_time = random.normalvariate(
                CONTAINER_UNLOAD_TIME["mean"],
                CONTAINER_UNLOAD_TIME["std"]
            )
            yield self.env.timeout(max(1, unload_time))
            
            # Process container after unloading
            self.env.process(self.process_container(container))

    def process_container(self, container):
        """Process container after unloading"""
        yard = self.reefer_yard if container.type == 'reefer' else self.regular_yard
        container_list = self.yard_containers[container.type][container.modal]
        
        if yard[container.modal].level < yard[container.modal].capacity:
            container.yard_entry_time = self.env.now
            
            # First add to tracking list
            if container not in container_list:
                container_list.append(container)
            
            # Then add to physical yard
            yield yard[container.modal].put(1)
            
            # Update yard statistics
            self.stats.update_yard_state(
                self.regular_yard['truck'].level + self.regular_yard['train'].level,
                self.reefer_yard['truck'].level + self.reefer_yard['train'].level
            )
            
            # Set container ready time
            container.ready_time = self.env.now + container.dwell_time
            
            # Process departure after dwell time
            yield self.env.timeout(container.dwell_time)
            
            if container.modal == "truck":
                if container in container_list:
                    try:
                        # Use a flag to track if departure was successful
                        container.departure_started = True
                        yield self.env.process(self.handle_truck_departure(container))
                    except Exception as e:
                        print(f"Error during truck departure for container {container.id}: {e}")
                # Remove warning as it's not needed - container might have already departed
            
            # Record container processed
            self.stats.log_container(container.type, container.modal)
        else:
            self.stats.log_yard_full()
            return

    def handle_truck_departure(self, container):
        """Process truck departures during gate hours"""
        while True:
            current_hour = int((self.env.now / 60) % 24)
            
            if current_hour < GATE_HOURS["open"] or current_hour >= GATE_HOURS["close"]:
                if current_hour >= GATE_HOURS["close"]:
                    next_opening = ((24 - current_hour) + GATE_HOURS["open"]) * 60
                else:
                    next_opening = (GATE_HOURS["open"] - current_hour) * 60
                yield self.env.timeout(next_opening)
                continue
            
            with self.gate.request() as gate_req:
                start_wait = self.env.now
                yield gate_req
                
                current_hour = int((self.env.now / 60) % 24)
                if current_hour < GATE_HOURS["open"] or current_hour >= GATE_HOURS["close"]:
                    continue
                
                container_list = self.yard_containers[container.type]['truck']
                
                # Only process if container hasn't already departed
                if container in container_list and not hasattr(container, 'departure_processed'):
                    # Process time at gate
                    yield self.env.timeout(10)
                    container.departure_time = self.env.now
                    
                    # Mark container as processed to prevent duplicate processing
                    container.departure_processed = True
                    
                    # Remove from tracking list
                    container_list.remove(container)
                    
                    # Remove from physical yard
                    if container.type == 'reefer':
                        yield self.reefer_yard['truck'].get(1)
                    else:
                        yield self.regular_yard['truck'].get(1)
                    
                    # Calculate actual dwell time
                    actual_dwell_time = container.departure_time - container.yard_entry_time
                    
                    departure_hour = int((self.env.now / 60) % 24)
                    if GATE_HOURS["open"] <= departure_hour < GATE_HOURS["close"]:
                        self.stats.log_wait_time('gate', self.env.now - start_wait)
                        self.stats.log_container_departure(container.type, 'truck', departure_hour)
                        self.stats.log_dwell_time(container.type, actual_dwell_time)
                        break
                else:
                    # Container has already been processed or removed, exit quietly
                    break

    def handle_train_departure(self):
        """Process train departure with up to 250 containers"""
        current_hour = int((self.env.now / 60) % 24)
        
        # Get current train container counts in yards
        regular_count = self.regular_yard['train'].level
        reefer_count = self.reefer_yard['train'].level
        
        # Target 150 containers per train, max 250
        target_containers = min(150, regular_count + reefer_count)
        max_containers = min(250, regular_count + reefer_count)
        
        # If we have less than 100 containers, skip this train
        if target_containers < 100:
            self.stats.log_missed_train()
            return
            
        # Calculate proportional split between dry and reefer based on yard levels
        if regular_count + reefer_count > 0:
            regular_ratio = regular_count / (regular_count + reefer_count)
            regular_to_take = min(regular_count, int(target_containers * regular_ratio))
            reefer_to_take = min(reefer_count, target_containers - regular_to_take)
            
            # Remove containers from train yards only
            if regular_to_take > 0:
                yield self.regular_yard['train'].get(regular_to_take)
                for _ in range(regular_to_take):
                    self.stats.log_container_departure('dry', 'train', current_hour)
                
            if reefer_to_take > 0:
                yield self.reefer_yard['train'].get(reefer_to_take)
                for _ in range(reefer_to_take):
                    self.stats.log_container_departure('reefer', 'train', current_hour)
            
            total_loaded = regular_to_take + reefer_to_take
            self.stats.log_wait_time('train', 30)  # Average wait time for loading
        
            # Process time for train departure (30 minutes)
            yield self.env.timeout(30)
            
            # Update yard state with current levels
            self.stats.update_yard_state(
                self.regular_yard['train'].level,
                self.reefer_yard['train'].level
            )
        else:
            self.stats.log_missed_train()

    def run(self, duration):
        """Run the simulation for specified duration (in minutes)"""
        self.env.process(self.generate_ship_arrival())
        self.env.run(until=duration)

if __name__ == "__main__":
    # Run simulation for 30 days
    sim = PortSimulation()
    sim.run(60 * 24 * 60)  # 30 days in minutes
    
    # Print hourly truck departure distribution
    print("\nHourly Truck Departure Distribution:")
    print("Hour | Count")
    print("-" * 20)
    for hour, count in enumerate(sim.stats.departures['truck']['by_hour']):
        if count > 0:  # Only show hours with departures
            print(f"{hour:02d}:00 | {count}")
    
    # Get and print detailed statistics
    stats = sim.stats.get_summary()
    
    print("\n=== Port Simulation Statistics (30 Days) ===\n")
    
    print("Arrivals:")
    print(f"Ships: {stats['Arrivals']['Ships']}")
    print("\nContainers:")
    print(f"Total: {stats['Arrivals']['Containers']['Total']}")
    print(f"Dry: {stats['Arrivals']['Containers']['Dry']}")
    print(f"Reefer: {stats['Arrivals']['Containers']['Reefer']}")
    
    print("\nDepartures:")
    print("\nTruck:")
    print(f"Total: {stats['Departures']['Truck']['Total']}")
    print(f"Dry: {stats['Departures']['Truck']['Dry']}")
    print(f"Reefer: {stats['Departures']['Truck']['Reefer']}")
    print(f"Peak Hour: {stats['Departures']['Truck']['Peak Hour']}")
    print("\nTrain:")
    print(f"Total: {stats['Departures']['Train']['Total']}")
    print(f"Dry: {stats['Departures']['Train']['Dry']}")
    print(f"Reefer: {stats['Departures']['Train']['Reefer']}")
    print(f"Peak Hour: {stats['Departures']['Train']['Peak Hour']}")
    
    print("\nYard State:")
    print(f"Current: Dry - {stats['Yard State']['Current']['dry']}, Reefer - {stats['Yard State']['Current']['reefer']}")
    print(f"Maximum: Dry - {stats['Yard State']['Maximum']['dry']}, Reefer - {stats['Yard State']['Maximum']['reefer']}")
    
    print("\nContainer Counts:")
    print(f"Total Processed: {stats['Container Counts']['Total Processed']}")
    print("\nBy Type:")
    for container_type, data in stats['Container Counts']['By Type'].items():
        print(f"\n{container_type}:")
        print(f"  Total: {data['Total']}")
        print(f"  Truck: {data['Truck']}")
        print(f"  Train: {data['Train']}")
    
    print("\nModal Split:")
    print(f"Truck: {stats['Container Counts']['Modal Split']['Truck']:.2%}")
    print(f"Train: {stats['Container Counts']['Modal Split']['Train']:.2%}")
    
    print("\nDwell Times (days):")
    for container_type, metrics in stats['Dwell Times (days)'].items():
        print(f"\n{container_type}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")
    
    print("\nWait Times (hours):")
    for wait_type, metrics in stats['Wait Times (hours)'].items():
        print(f"\n{wait_type}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")
    
    print("\nOperational Issues:")
    print(f"Yard Full Events: {stats['Operational Issues']['Yard Full Events']}")
    print(f"Missed Train Connections: {stats['Operational Issues']['Missed Train Connections']}")
    
    # Generate visualizations
    sim.stats.plot_statistics()
