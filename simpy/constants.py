TERMINAL_BERTH_CONFIG = {
    'Maher': {
        'min_days_on_custom' : 1,
        'custom_mean_days': 3,
        'custom_std_days': 1,
        'berths': {
            1: {'load_speed': {'mean': 50, 'deviation': 2}, 'unload_speed': {'mean': 50, 'deviation': 3}},
            2: {'load_speed': {'mean': 40, 'deviation': 2}, 'unload_speed': {'mean': 40, 'deviation': 2}},
            3: {'load_speed': {'mean': 50, 'deviation': 2}, 'unload_speed': {'mean': 50, 'deviation': 2}},           # Add more berths dynamically as needed
            4: {'load_speed': {'mean': 53, 'deviation': 2}, 'unload_speed': {'mean': 53, 'deviation': 2}},
            5: {'load_speed': {'mean': 40, 'deviation': 2}, 'unload_speed': {'mean': 40, 'deviation': 2}},
            6: {'load_speed': {'mean': 51, 'deviation': 2}, 'unload_speed': {'mean': 49, 'deviation': 2}},           # Add more berths dynamically as needed

        },
        'trains': {
            1: {'capacity': 600, 'unload_mean': 500, 'unload_std': 50, 'load_mean': 500, 'load_std': 50, 'destination': 'LA', 'scheduled_arrival': 1, 'scheduled_departure': 13, 'TEU_to_FEU':1},
            2: {'capacity': 600, 'unload_mean': 550, 'unload_std': 50, 'load_mean': 500, 'load_std': 50, 'destination': 'LA', 'scheduled_arrival': 15, 'scheduled_departure': 3, 'TEU_to_FEU':2},
            3: {'capacity': 600, 'unload_mean': 540, 'unload_std': 50, 'load_mean': 500, 'load_std': 50, 'destination': 'LA', 'scheduled_arrival': 3, 'scheduled_departure': 15, 'TEU_to_FEU':0.5},
            4: {'capacity': 600, 'unload_mean': 520, 'unload_std': 50, 'load_mean': 500, 'load_std': 50, 'destination': 'LA', 'scheduled_arrival': 8, 'scheduled_departure': 20, 'TEU_to_FEU':1},
        },

        'yard_capacity':{
            'reefer': 5000,
            'regular': 50000,
            'hazard': 100
        },  # Example yard capacity for 'Maher'
        'berth_categories': {
            1: [0, 1, 2, 3, 4],
            2: [0, 2, 4],
            3: [0, 1, 2, 4,5],
            4: [0, 1, 2, 3, 4, 5],
            5: [0, 1, 2, 3, 4, 5 ,6],
            6: [0, 1, 2, 3, 4, 5 ,6],
        },
        'initial_yard_stocks':{
            'reefer': {'import_TEU': 1000, 'import_FEU': 1000, 'export_TEU': 1000, 'export_FEU': 1000 , 'import': 3000, 'export':3000 },
            'regular': {'import_TEU': 500, 'import_FEU': 500, 'export_TEU': 500, 'export_FEU': 500 , 'import': 1500, 'export':1500 },
            'hazard': {'import_TEU': 30, 'import_FEU': 10, 'export_TEU': 20, 'export_FEU': 10 , 'import': 50, 'export':40 }
        },
        'trucks_TEU':{ 7 : {'load_mean':30, 'load_std': 2, 'unload_mean': 30, 'unload_std':3,},},
        'trucks_FEU':{ 7 : {'load_mean':15, 'load_std': 3, 'unload_mean': 15, 'unload_std':4,},}
    },
    'PortX': {
        'berths': {
            1: {'load_speed': (5, 8), 'unload_speed': (5, 8)},
        },
        'yard_capacity': 1200
    },
}
