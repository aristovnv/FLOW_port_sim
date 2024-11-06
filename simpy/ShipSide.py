import random
import datetime 
from collections import defaultdict
from constants import TERMINAL_BERTH_CONFIG
from datetime import time, timedelta

def simulate_teu_feu_split(total_containers, teu_feu_ratio):
    # Calculate the total number of containers with a bit of randomness
    if teu_feu_ratio > 0:
        teu_proportion = 1 / (1 + teu_feu_ratio)  # For example, 0.5 ratio makes TEU = 2/3 of total
        feu_proportion = teu_feu_ratio / (1 + teu_feu_ratio)  # FEU = 1/3 of total in 2:1 ratio
    else:
        raise ValueError("teu_feu_ratio must be positive.")
    
    # Generate counts with a bit of variability
    teu_count = max(0, int(total_containers * teu_proportion + random.uniform(-2, 2)))
    feu_count = max(0, int(total_containers * feu_proportion + random.uniform(-1, 1)))
    
    # Ensure the counts don't exceed total containers
    if teu_count + feu_count > total_containers:
        # Adjust in case of overshooting due to randomness
        teu_count = int(total_containers * teu_proportion)
        feu_count = total_containers - teu_count

    return teu_count, feu_count

# if we need to change truck schedule it is in this class (including hazard)
class TruckSchedule:
    def __init__(self):
        self.weekday_start = time(7, 0)
        self.weekend_start = time(8, 0)
        self.weekday_end = time(20, 0)
        self.weekend_end = time(18, 0)
        self.hazard_days = [0, 1, 2, 3, 4]  # Monday to Friday
        self.reduced_hours_days = [5]  # Saturday
        self.closed_days = [6]  # Sunday

    def is_open(self, current_time):
        day_of_week = current_time.weekday()
        if day_of_week in self.closed_days:
            return False
        if day_of_week in self.reduced_hours_days:
            return self.weekend_start <= current_time.time() < self.weekend_end
        return self.weekday_start <= current_time.time() < self.weekday_end

    def can_handle_hazard(self, current_time):
        return current_time.weekday() in self.hazard_days


class TruckOperations:
    def __init__(self, terminal):
        self.terminal = terminal
        self.schedule = TruckSchedule()

    def NOprocess_trucks_old(self, current_time, load_mean, load_std, unload_mean, unload_std):
        if not self.schedule.is_open(current_time):
            return 0, 0

        # Generate number of containers to load and unload
        containers_to_load = max(0, int(random.normalvariate(load_mean, load_std)))  # Example values
        containers_to_unload = max(0, int(random.normalvariate(unload_mean, unload_std)))  # Example values

        # Process loading (container types logic!!!!)
        actual_loaded = 0
        for _ in range(containers_to_load):
            container_types = list(self.terminal.yard.keys())
            random.shuffle(container_types)
            for container_type in container_types:
                if self.terminal.yard[container_type]['import'] > 0:
                    self.terminal.yard[container_type]['import'] -= 1
                    actual_loaded += 1
                    break

        # Process unloading
        actual_unloaded = 0
        for _ in range(containers_to_unload):
            container_type = random.choice(['reefer', 'regular', 'hazard'])
            if container_type == 'hazard' and not self.schedule.can_handle_hazard(current_time):
                container_type = random.choice(['reefer', 'regular'])
            if self.terminal.yard[container_type]['export'] + self.terminal.yard[container_type]['import'] < self.terminal.yard_capacity[container_type]:
                self.terminal.yard[container_type]['export'] += 1
                actual_unloaded += 1

        return actual_loaded, actual_unloaded    
    
    def process_trucks(self, current_time, load_mean, load_std, unload_mean, unload_std, is_feu):
        if not self.schedule.is_open(current_time):
            return 0, 0, 0 ,0

        # Generate number of containers to load and unload
        container_multiplier = 2 if is_feu else 1  # FEU is equivalent to 2 TEUs
        containers_to_load = max(0, int(random.normalvariate(load_mean, load_std))) * container_multiplier
        containers_to_unload = max(0, int(random.normalvariate(unload_mean, unload_std))) * container_multiplier

        # Process loading
        actual_loaded = 0
        actual_loaded_cnt = 0
        for _ in range(containers_to_load):
            container_types = list(self.terminal.yard.keys())
            random.shuffle(container_types)
            for container_type in container_types:
                actual_qty = terminal.remove_container('import', container_type, is_feu=is_feu, count=1, current_date=current_time)
                if is_feu:
                    if self.terminal.yard[container_type]['import_FEU'] > 0:
                        self.terminal.yard[container_type]['import_FEU'] -= actual_qty
                        self.terminal.yard[container_type]['import'] -= 2 * actual_qty  # Update total import/export count for FEU
                        actual_loaded += 2 * actual_qty
                        actual_loaded_cnt += actual_qty
                        break
                else:
                    if self.terminal.yard[container_type]['import_TEU'] > 0:
                        self.terminal.yard[container_type]['import_TEU'] -= actual_qty
                        self.terminal.yard[container_type]['import'] -= actual_qty  # Update total import/export count for TEU
                        actual_loaded += actual_qty
                        actual_loaded_cnt += actual_qty
                        break

        # Process unloading
        actual_unloaded = 0
        actual_unloaded_cnt = 0
        for _ in range(containers_to_unload):
            container_type = random.choice(['reefer', 'regular', 'hazard'])
            if container_type == 'hazard' and not self.schedule.can_handle_hazard(current_time):
                container_type = random.choice(['reefer', 'regular'])
            
            total_capacity = 2 * (self.terminal.yard[container_type]['export_FEU'] + self.terminal.yard[container_type]['import_FEU'])+ \
            self.terminal.yard[container_type]['export_TEU'] + self.terminal.yard[container_type]['import_TEU']
            if is_feu:
                if (total_capacity + 1 < self.terminal.yard_capacity[container_type]):
                    self.terminal.yard[container_type]['import_FEU'] += 1
                    self.terminal.yard[container_type]['import'] += 2
                    terminal.add_container('export', container_type, is_feu=is_feu, count=1, current_date=current_time)
                    actual_unloaded += 2
                    actual_unloaded_cnt += 1
                    
            else:
                if (total_capacity < self.terminal.yard_capacity[container_type]):
                    self.terminal.yard[container_type]['import_TEU'] += 1
                    self.terminal.yard[container_type]['import'] += 1
                    terminal.add_container('import', container_type, is_feu=is_feu, count=1, current_date=current_time)
                    actual_unloaded += 1
                    actual_unloaded_cnt += 1
                    

        return actual_loaded, actual_unloaded, actual_loaded_cnt, actual_unloaded_cnt

# Define Berth class
class Berth:
    load_speed_mean = 0
    unload_speed_mean = 0
    load_speed_dev = 0
    unload_speed_dev = 0
    def __init__(self, berth_id, categories, load_speed_mean, load_speed_dev, unload_speed_mean, unload_speed_dev):
        self.berth_id = berth_id
        self.categories = categories
        self.load_speed_mean = load_speed_mean
        self.load_speed_dev = load_speed_dev
        self.unload_speed_mean = unload_speed_mean
        self.unload_speed_dev = unload_speed_dev
        self.current_ship = None
    
    def generate_dynamic_speed(self, ship, time_factors, is_unload):
        """
        Generate dynamic load/unload speeds based on ship and time factors.
        :param ship: A dictionary containing ship characteristics like size, operator, origin.
        :param time_factors: A dictionary containing time characteristics like day, month, season, etc.
        :return: Tuple of dynamic load and unload speeds.
        """
        #ship['size_factor'] = 1
        #time_factors['time_multiplier'] = 1
        #ship['operator_factor'] = 1
        # Example: Adjust mean and deviation based on ship and time factors (customizable)
        if is_unload:
            adjusted_speed_mean = self.unload_speed_mean #* ship['size_factor'] * time_factors['time_multiplier']
            adjusted_speed_dev = self.unload_speed_dev #* ship['operator_factor']
        else: 
            adjusted_speed_mean = self.load_speed_mean #* ship['size_factor'] * time_factors['time_multiplier']
            adjusted_speed_dev = self.load_speed_dev #* ship['operator_factor']
        
        # Generate dynamic load/unload speeds based on the adjusted mean and deviation
        return abs(round(random.gauss(adjusted_speed_mean, adjusted_speed_dev)))

    def get_unload_speed (self, ship, time):
        return self.generate_dynamic_speed(ship, time, True)
    
    def get_load_speed (self, ship, time):
        return self.generate_dynamic_speed(ship, time, False)


# Define Ship class
class Ship:
    def __init__(self, operator, size, origin, scheduled_arrival):
        self.operator = operator
        self.size = size
        self.origin = origin
        self.scheduled_arrival = scheduled_arrival
        self.actual_arrival = None
        self.containers_to_unload = 0
        self.containers_to_load = 0

class Train:
    def __init__(self, train_id, max_capacity, unload_mean, unload_std, load_mean, load_std, destination, scheduled_arrival, scheduled_departure, \
                 TEU_FEU_ratio):
        self.train_id = train_id
        self.destination = destination
        self.max_capacity = max_capacity
        self.origin = destination
        self.scheduled_arrival = scheduled_arrival
        self.scheduled_departure = scheduled_departure
        self.containers_to_unload = 0
        self.containers_to_load = 0
        self.loading_speed = 50
        self.unloading_speed = 50
        self.unload_mean = unload_mean
        self.unload_std = unload_std
        self.load_mean = load_mean
        self.load_std = load_std

        self.containers_to_unload = self.generate_containers(self.unload_mean, self.unload_std)
        self.containers_to_load = self.generate_containers(self.load_mean, self.load_std)
        self.TEU_FEU_ratio = TEU_FEU_ratio
    def generate_containers(self, mean, std):
        return max(0, min(int(random.normalvariate(mean, std)), self.max_capacity))      
      
    def reset_containers(self):
        self.containers_to_unload = self.generate_containers(self.unload_mean, self.unload_std)
        self.containers_to_load = self.generate_containers(self.load_mean, self.load_std)    

# Define Terminal class
class Terminal:
    def __init__(self, terminal_name):
        self.container_schedule = {'import': {}, 'export': {}}
        self.terminal_name = terminal_name
        
        # Load terminal-specific configurations
        terminal_config = TERMINAL_BERTH_CONFIG.get(terminal_name)
        if not terminal_config:
            raise ValueError(f"Terminal {terminal_name} is not configured.")
        self.min_days_on_custom = terminal_config['min_days_on_custom']
        self.custom_mean_days = terminal_config['custom_mean_days']
        self.custom_std_days = terminal_config['custom_std_days']
        self.yard_capacity = terminal_config['yard_capacity']
        self.berths = []
        self.trains = []

        # Create berths dynamically based on terminal configuration
        for berth_id, berth_info in terminal_config['berths'].items():
            load_speed = berth_info['load_speed']['mean']
            load_speed_dev = berth_info['unload_speed']['deviation']
            unload_speed = berth_info['load_speed']['mean']
            unload_speed_dev = berth_info['unload_speed']['deviation']
            berth_sizes = terminal_config['berth_categories'][berth_id]  # Example berth limits, can be fetched dynamically too
            
            berth = Berth(berth_id, berth_sizes, load_speed, load_speed_dev, unload_speed, unload_speed_dev)
            self.berths.append(berth)
        for train_id, train_info in terminal_config['trains'].items():
            self.trains.append(Train(train_id, train_info['capacity'], train_info['unload_mean'], train_info['unload_std'], train_info['load_mean'], train_info['load_std'],train_info['destination'], \
                                     time(train_info['scheduled_arrival'], 0), time(train_info['scheduled_departure'], 0 ), train_info['TEU_to_FEU'])) 
        self.yard = terminal_config['initial_yard_stocks']
        self.hourly_stats = []
        self.daily_stats = []
        self.trucks_schedule_TEU = terminal_config['trucks_TEU']
        self.trucks_schedule_FEU = terminal_config['trucks_FEU']
    def generate_normal_value(self, mean, deviation):
        # Generate a value from a normal distribution and round it to the nearest integer
        return round(random.gauss(mean, deviation))

    def get_available_berth(self, ship_size):
        for berth in self.berths:
            if berth.current_ship is None and ship_size in berth.categories:
                return berth
        return None

    def add_customs_delay(self):
        # Calculate customs delay ensuring it's at least MIN_DAYS_ON_CUSTOM
        days = max(self.min_days_on_custom, int(random.normalvariate(self.custom_mean_days, self.custom_std_days)))
        return timedelta(days=days)

    def add_container(self, operation, container_type, is_feu, count, current_date):
        # Calculate the customs hold date
         #{'import': {'date': {'reefer': {'TEU': count, 'FEU': count}, ... }}, 'export': {...}}
        customs_release_date = current_date + self.add_customs_delay()
        date_str = customs_release_date.strftime("%Y-%m-%d")
        
        # Initialize date and container type entry if not present
        if date_str not in self.container_schedule[operation]:
            self.container_schedule[operation][date_str] = {}
        if container_type not in self.container_schedule[operation][date_str]:
            self.container_schedule[operation][date_str][container_type] = {'TEU': 0, 'FEU': 0}
        
        # Update the count based on container type (TEU or FEU)
        container_key = 'FEU' if is_feu else 'TEU'
        self.container_schedule[operation][date_str][container_type][container_key] += count

    def remove_container(self, operation, container_type, is_feu, count, current_date):

        date_str = current_date.strftime("%Y-%m-%d")
                
        # Update the count based on container type (TEU or FEU)
        container_key = 'FEU' if is_feu else 'TEU'
        remove_cnt = min(self.container_schedule.get(operation, {}).get(date_str, {}).get(container_type, {}).get(container_key, 0), count)
        self.container_schedule[operation][date_str][container_type][container_key] -= remove_cnt
        return remove_cnt

    def init_stocks(self, current_date):
        date_str = current_date.strftime("%Y-%m-%d")
        # Initialize date and container type entry if not present
        for container_type in ['reefer','regular', 'hazard']:
            self.container_schedule.setdefault('import', {}).setdefault(date_str, {}).setdefault(container_type, {})
            self.container_schedule.setdefault('export', {}).setdefault(date_str, {}).setdefault(container_type, {})
            self.container_schedule['import'][date_str][container_type] = {'TEU': self.yard[container_type][f"import_TEU"], 'FEU': self.yard[container_type][f"import_FEU"]}
            self.container_schedule['export'][date_str][container_type] = {'TEU': self.yard[container_type][f"import_TEU"], 'FEU': self.yard[container_type][f"import_FEU"]}

    def end_of_day_transfer(self, current_date):
        next_date = current_date + timedelta(days=1)  # Move to the next day

        # Iterate over each operation type (import/export)
        for operation, dates in self.container_schedule.items():
            # Check if there are containers scheduled for the current date
            if current_date in dates:
                for container_type, sizes in dates[current_date].items():
                    for size, count in sizes.items():
                        if count > 0:  # Only transfer if there are containers left to move
                            is_feu = size == 'FEU'
                            
                            # Add containers to the next day's schedule
                            self.add_container(operation, container_type, is_feu, count, next_date)

                            # Remove containers from the current day's schedule
                            self.remove_container(operation, container_type, is_feu, count, current_date)

def generate_ship_arrival(scheduled_arrival, ship, deviation_hours=48):
    deviation = random.randint(-deviation_hours, deviation_hours)
    return scheduled_arrival + datetime.timedelta(hours=deviation)

def get_container_numbers(mean, std_dev):
    return max(0, int(random.normalvariate(mean, std_dev)))

def get_container_distribution(ship, time, total_containers, is_unloading):
    if is_unloading:
        base_distribution = [3, 6, 1]
    else:
        base_distribution = [3, 5, 2]
    
    # Apply some randomness to the distribution
    #distribution = [max(0, x + random.randint(-1, 1)) for x in base_distribution]
    total = sum(base_distribution)
    
    return {
        'reefer': int(total_containers * base_distribution[0] / total),
        'regular': int(total_containers * base_distribution[1] / total),
        'hazard': int(total_containers * base_distribution[2] / total)
    }


def process_trains(terminal, current_time, trains):
    train_unloaded = 0
    train_loaded = 0
    train_unloaded_cnt = 0
    train_loaded_cnt = 0
    #terminal.add_container('import', 'container_type', is_feu=is_feu, count=1, current_date=current_time)
    for train in trains:
        if (current_time.time() >= train.scheduled_arrival and current_time.time() < (time(23, 59) if train.scheduled_departure < train.scheduled_arrival else train.scheduled_departure)) \
            or (current_time.time() < train.scheduled_departure and current_time.time() >= (time(0, 0) if train.scheduled_departure < train.scheduled_arrival else train.scheduled_arrival)):
            for container_type, amount in get_container_distribution(train, current_time, min(train.containers_to_unload, train.loading_speed), True).items():
                use_ct = container_type
                if container_type == 'empty':
                    use_ct = 'regular'
                teu, feu = simulate_teu_feu_split(amount, train.TEU_FEU_ratio)

                unloaded_teu = min(teu, terminal.yard_capacity[use_ct] - (terminal.yard[use_ct]['import']+terminal.yard[use_ct]['export'])) 
                terminal.add_container('export', use_ct, is_feu=False, count=unloaded_teu, current_date=current_time)
                terminal.yard[use_ct]['export'] += unloaded_teu
                train.containers_to_unload -= unloaded_teu
                train_unloaded += unloaded_teu
                train_unloaded_cnt += unloaded_teu

                unloaded_feu = min(teu, terminal.yard_capacity[use_ct] - (terminal.yard[use_ct]['import']+terminal.yard[use_ct]['export'])) 
                terminal.add_container('export', use_ct, is_feu=True, count=unloaded_feu, current_date=current_time)
                terminal.yard[use_ct]['export'] += unloaded_feu * 2
                train.containers_to_unload -= unloaded_feu * 2
                train_unloaded += unloaded_feu * 2


            for container_type, amount in get_container_distribution(train, current_time, min(train.containers_to_load, train.unloading_speed), False).items():
                use_ct = container_type
                if container_type == 'empty':
                    use_ct = 'regular'
                teu, feu = simulate_teu_feu_split(amount, train.TEU_FEU_ratio)
                loaded_teu = min(teu, terminal.yard[use_ct]['import'])
                actual = terminal.remove_container('import', use_ct, is_feu=False, count=loaded_teu, current_date=current_time)
                terminal.yard[use_ct]['import'] -= actual 
                train.containers_to_load -= actual 
                train_loaded += actual
                loaded_feu = min(feu, terminal.yard[use_ct]['import'])
                actual = terminal.remove_container('import', use_ct, is_feu=True, count=loaded_feu, current_date=current_time)
                terminal.yard[use_ct]['import'] -= actual * 2
                train.containers_to_load -= actual * 2
                train_loaded += actual * 2
                train_loaded_cnt += actual

        elif current_time.time() == train.scheduled_departure:
            train.reset_containers()
    return train_loaded, train_unloaded, train_loaded_cnt, train_unloaded_cnt

def get_container_distribution(train, time, total_containers, is_unloading):
    if is_unloading:
        base_distribution = [2, 6, 1, 1]
    else:
        base_distribution = [3, 5, 1, 2]
    
    # Apply some randomness to the distribution
    #distribution = [max(0, x + random.randint(-1, 1)) for x in base_distribution]
    total = sum(base_distribution)
    
    return {
        'reefer': int(total_containers * base_distribution[0] / total),
        'regular': int(total_containers * base_distribution[1] / total),
        'hazard': int(total_containers * base_distribution[2] / total)
    }


def simulate(terminal, ships, start_date, end_date):
    current_time = start_date
    terminal.init_stocks(current_time)
    ship_queue = []
    truck_ops = TruckOperations(terminal)
    while current_time <= end_date:
        # Check for new ship arrivals
        hourly_loaded = {container_type: 0 for container_type in terminal.yard.keys()}
        hourly_unloaded = {container_type: 0 for container_type in terminal.yard.keys()}        
        hourly_loaded_cnt = {container_type: 0 for container_type in terminal.yard.keys()}
        hourly_unloaded_cnt = {container_type: 0 for container_type in terminal.yard.keys()}        

        for ship in ships:
            if ship.scheduled_arrival.date() == current_time.date():
                ship.actual_arrival = generate_ship_arrival(ship.scheduled_arrival, ship)
                ship_queue.append(ship)
        #should be hourly
        trucks_config_TEU = terminal.trucks_schedule_TEU[7]
        trucks_config_FEU = terminal.trucks_schedule_FEU[7]
        loaded_TEU, unloaded_TEU, loaded_cnt_TEU, unloaded_cnt_TEU = truck_ops.process_trucks(current_time, trucks_config_TEU['load_mean'],trucks_config_TEU['load_std'],trucks_config_TEU['unload_mean'],trucks_config_TEU['unload_std'], False)
        loaded_FEU, unloaded_FEU, loaded_cnt_FEU, unloaded_cnt_FEU = truck_ops.process_trucks(current_time, trucks_config_FEU['load_mean'],trucks_config_FEU['load_std'],trucks_config_FEU['unload_mean'],trucks_config_FEU['unload_std'], True)
        loaded = loaded_TEU + loaded_FEU 
        unloaded = unloaded_TEU + unloaded_FEU 
        loaded_cnt = loaded_cnt_TEU + loaded_cnt_FEU 
        unloaded_cnt = unloaded_cnt_TEU + unloaded_cnt_FEU
        train_loaded, train_unloaded, train_loaded_cnt, train_unloaded_cnt = process_trains(terminal, current_time, terminal.trains)


        # Process ships in queue
        for ship in ship_queue:
            if ship.actual_arrival <= current_time:
                berth = terminal.get_available_berth(ship.size)
                if berth:
                    berth.current_ship = ship
                    ship.containers_to_unload = abs(get_container_numbers(500, 20))  # Example values                    
                    ship.containers_to_load = abs(get_container_numbers(490, 30))  # Example values
                    ship_queue.remove(ship)

        # Process loading/unloading for each berth
        for berth in terminal.berths:
            if berth.current_ship:
                ship = berth.current_ship
                unload_distribution = get_container_distribution(ship, current_time, ship.containers_to_unload, True)
                load_distribution = get_container_distribution(ship, current_time, ship.containers_to_load, False)
                if ship.containers_to_unload > 0:
                    # Unloading                    
                    for container_type, amount in unload_distribution.items():
                        
                        unload_speed = berth.get_unload_speed(ship, current_time)
                        unloaded = min(amount, unload_speed)
                        teu, feu = simulate_teu_feu_split(unloaded, 0.5)
                        ship.containers_to_unload -= unloaded
                        terminal.add_container('import', container_type, is_feu=False, count=teu, current_date=current_time)
                        terminal.add_container('import', container_type, is_feu=True, count=feu, current_date=current_time)
                        terminal.yard[container_type]['import'] = min(terminal.yard[container_type]['import'] + terminal.yard[container_type]['export'] + teu + feu * 2, terminal.yard_capacity[container_type])
                        hourly_unloaded_cnt[container_type] += teu + feu
                        hourly_unloaded[container_type] += teu + feu * 2 
                if ship.containers_to_load > 0:
                    # Loading
                    for container_type, amount in load_distribution.items():
                        load_speed = berth.get_load_speed(ship, current_time)
                        loaded = min(amount, load_speed, terminal.yard[container_type]['export'])
                        teu, feu = simulate_teu_feu_split(loaded, 0.5)
                        actual = terminal.remove_container('export', container_type, is_feu=False, count=teu, current_date=current_time)
                        ship.containers_to_load -= actual
                        hourly_loaded[container_type] += actual
                        hourly_loaded_cnt[container_type] += actual
                        terminal.yard[container_type]['export'] -= actual
                        actual = terminal.remove_container('export', container_type, is_feu=True, count=feu, current_date=current_time)
                        ship.containers_to_load -= actual
                        hourly_loaded[container_type] += 2 * actual
                        hourly_loaded_cnt[container_type] += actual
                        terminal.yard[container_type]['export'] -= 2 * actual
                        
                        if loaded < 0:
                            print(f"load_speed {load_speed} and mean is {berth.load_speed_mean}  and {berth.load_speed_dev}")

                        

                # Check if ship is finished
                if ship.containers_to_unload <= 0 and ship.containers_to_load <= 0:
                    berth.current_ship = None

        # Record hourly stats
        terminal.hourly_stats.append({
            'time': current_time,
            'yard_occupation': sum(sum(inner_dict.values()) for inner_dict in terminal.yard.values()),# sum(x for a, x in terminal.yard.items()[1].items()),
            'containers_loaded': sum(x for a, x in hourly_loaded.items()),
            'containers_unloaded': sum(x for a, x in hourly_unloaded.items()),
            'containers_cnt_loaded': sum(x for a, x in hourly_loaded_cnt.items()),
            'containers_cnt_unloaded': sum(x for a, x in hourly_unloaded_cnt.items()),
            'truck_containers_loaded': loaded,
            'truck_containers_unloaded': unloaded,
            'truck_containers_cnt_loaded': loaded_cnt,
            'truck_containers_cnt_unloaded': unloaded_cnt,
            'train_loaded': train_loaded,
            'train_unloaded': train_unloaded,
            'train_cnt_loaded': train_loaded_cnt,
            'train_cnt_unloaded': train_unloaded_cnt
        })

        # Record daily stats at the end of each day
        if current_time.hour == 23:
            daily_stats = {
                'date': current_time.date(),
                'total_loaded': sum(stat['containers_loaded'] for stat in terminal.hourly_stats[-24:]),
                'total_unloaded': sum(stat['containers_unloaded'] for stat in terminal.hourly_stats[-24:]),
                'total_cnt_loaded': sum(stat['containers_cnt_loaded'] for stat in terminal.hourly_stats[-24:]),
                'total_cnt_unloaded': sum(stat['containers_cnt_unloaded'] for stat in terminal.hourly_stats[-24:]),
                'avg_yard_occupation': sum(stat['yard_occupation'] for stat in terminal.hourly_stats[-24:]) / 24,
                'total_truck_loaded': sum(stat['truck_containers_loaded'] for stat in terminal.hourly_stats[-24:]),
                'total_truck_unloaded': sum(stat['truck_containers_unloaded'] for stat in terminal.hourly_stats[-24:]),
                'total_truck_cnt_loaded': sum(stat['truck_containers_cnt_loaded'] for stat in terminal.hourly_stats[-24:]),
                'total_truck_cnt_unloaded': sum(stat['truck_containers_cnt_unloaded'] for stat in terminal.hourly_stats[-24:]),
                'train_loaded': sum(stat['train_loaded'] for stat in terminal.hourly_stats[-24:]),
                'train_unloaded': sum(stat['train_unloaded'] for stat in terminal.hourly_stats[-24:]),
                'train_cnt_loaded': sum(stat['train_cnt_loaded'] for stat in terminal.hourly_stats[-24:]),
                'train_cnt_unloaded': sum(stat['train_cnt_unloaded'] for stat in terminal.hourly_stats[-24:])
            }
            terminal.daily_stats.append(daily_stats)
            terminal.end_of_day_transfer(current_time)
        current_time += datetime.timedelta(hours=1)

# Example usage
start_date = datetime.datetime(2024, 11, 6)
end_date = datetime.datetime(2024, 11, 30)
terminal = Terminal('Maher')

# Generate some example ships
ships = [
    Ship('Operator1', 4, 'Origin1', start_date + datetime.timedelta(days=random.randint(0, 30))),
    Ship('Operator2', 5, 'Origin2', start_date + datetime.timedelta(days=random.randint(0, 30))),
    Ship('Operator3', 6, 'Origin3', start_date + datetime.timedelta(days=random.randint(0, 30))),
    # Add more ships as needed
]

trains = [

]

simulate(terminal, ships, start_date, end_date)
'''
print("Hourly Stats:")
for stat in terminal.hourly_stats:  # Print first 5 days
    print(f"Time: {stat['time']}, Loaded: {stat['containers_loaded']}, Unloaded: {stat['containers_unloaded']}, Avg Yard Occupation: {stat['yard_occupation']:.2f}")
'''

# Print some results
print("Daily Stats:")
for stat in terminal.daily_stats:  # Print first 5 days
    print(f"Date: {stat['date']}, Loaded (Ship): {stat['total_loaded']}, Unloaded (Ship): {stat['total_unloaded']}, \
          Loaded (Ship, cnt): {stat['total_cnt_loaded']}, Unloaded (Ship, cnt): {stat['total_cnt_unloaded']},  Avg Yard Occupation: \
          {stat['avg_yard_occupation']:.2f}, total_truck_loaded: {stat['total_truck_loaded']}, total_truck_unloaded: \
          {stat['total_truck_unloaded']}, total_truck_loaded_cnt: {stat['total_truck_cnt_loaded']}, total_truck_unloaded_cnt: \
          {stat['total_truck_cnt_unloaded']}, train loaded: {stat['train_loaded']}, train unloaded: {stat['train_unloaded']} \
            , train loaded cnt: {stat['train_cnt_loaded']}, train unloaded cnt: {stat['train_cnt_unloaded']}")
