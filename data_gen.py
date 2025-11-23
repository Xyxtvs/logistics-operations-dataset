"""
Logistics Operations Data Generator
Generates realistic transportation/supply chain data for SQL analytics practice
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 12, 31)
NUM_DRIVERS = 150
NUM_TRUCKS = 120
NUM_TRAILERS = 180
NUM_CUSTOMERS = 200
NUM_FACILITIES = 50
OUTPUT_DIR = Path("logistics_data")

print("Generating logistics database...")
print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
print(f"Fleet size: {NUM_TRUCKS} trucks, {NUM_TRAILERS} trailers, {NUM_DRIVERS} drivers\n")

# Core reference data
CITIES = [
    ('Atlanta', 'GA', 33.7490, -84.3880),
    ('Chicago', 'IL', 41.8781, -87.6298),
    ('Dallas', 'TX', 32.7767, -96.7970),
    ('Los Angeles', 'CA', 34.0522, -118.2437),
    ('New York', 'NY', 40.7128, -74.0060),
    ('Phoenix', 'AZ', 33.4484, -112.0740),
    ('Philadelphia', 'PA', 39.9526, -75.1652),
    ('Houston', 'TX', 29.7604, -95.3698),
    ('Miami', 'FL', 25.7617, -80.1918),
    ('Detroit', 'MI', 42.3314, -83.0458),
    ('Seattle', 'WA', 47.6062, -122.3321),
    ('Denver', 'CO', 39.7392, -104.9903),
    ('Portland', 'OR', 45.5152, -122.6784),
    ('Las Vegas', 'NV', 36.1699, -115.1398),
    ('Minneapolis', 'MN', 44.9778, -93.2650),
    ('Charlotte', 'NC', 35.2271, -80.8431),
    ('Indianapolis', 'IN', 39.7684, -86.1581),
    ('Columbus', 'OH', 39.9612, -82.9988),
    ('Memphis', 'TN', 35.1495, -90.0490),
    ('Kansas City', 'MO', 39.0997, -94.5786),
    ('Salt Lake City', 'UT', 40.7608, -111.8910),
    ('Nashville', 'TN', 36.1627, -86.7816),
    ('Milwaukee', 'WI', 43.0389, -87.9065),
    ('Oklahoma City', 'OK', 35.4676, -97.5164),
    ('Omaha', 'NE', 41.2565, -95.9345)
]

TRUCK_MAKES = ['Freightliner', 'Kenworth', 'Peterbilt', 'Volvo', 'International', 'Mack']
TRAILER_TYPES = ['Dry Van', 'Refrigerated']
CUSTOMER_TYPES = ['Dedicated', 'Spot', 'Contract']
FREIGHT_TYPES = ['General', 'Food/Beverage', 'Automotive', 'Retail', 'Electronics', 'Consumer Goods']
INCIDENT_TYPES = ['Accident', 'Moving Violation', 'DOT Violation', 'Customer Complaint', 'Equipment Damage']
MAINTENANCE_TYPES = ['Preventive', 'Repair', 'Inspection', 'Tire', 'Brake', 'Engine', 'Transmission']

# Helper functions
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in miles"""
    R = 3959  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def generate_date_range(start, end, n):
    """Generate n random dates between start and end"""
    delta = (end - start).total_seconds()
    return [start + timedelta(seconds=random.random() * delta) for _ in range(n)]

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. DRIVERS TABLE
print("Generating drivers table...")
driver_ids = [f"DRV{str(i+1).zfill(5)}" for i in range(NUM_DRIVERS)]
hire_dates = [START_DATE - timedelta(days=random.randint(0, 3650)) for _ in range(NUM_DRIVERS)]

# Pre-calculate termination dates
termination_dates = []
for i in range(NUM_DRIVERS):
    if random.random() < 0.15 and hire_dates[i] < END_DATE - timedelta(days=180):
        termination_dates.append(hire_dates[i] + timedelta(days=random.randint(180, 900)))
    else:
        termination_dates.append(None)

drivers = pd.DataFrame({
    'driver_id': driver_ids,
    'first_name': [random.choice(['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard',
                                   'Joseph', 'Thomas', 'Charles', 'Mary', 'Patricia', 'Jennifer', 'Linda',
                                   'Barbara', 'Elizabeth', 'Susan', 'Jessica', 'Sarah', 'Karen'])
                   for _ in range(NUM_DRIVERS)],
    'last_name': [random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
                                 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
                                 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin'])
                  for _ in range(NUM_DRIVERS)],
    'hire_date': hire_dates,
    'termination_date': termination_dates,
    'license_number': [f"DL{random.randint(100000000, 999999999)}" for _ in range(NUM_DRIVERS)],
    'license_state': [random.choice([city[1] for city in CITIES]) for _ in range(NUM_DRIVERS)],
    'date_of_birth': [datetime(random.randint(1960, 1995), random.randint(1, 12), random.randint(1, 28))
                      for _ in range(NUM_DRIVERS)],
    'home_terminal': [random.choice([city[0] for city in CITIES]) for _ in range(NUM_DRIVERS)],
    'employment_status': ['Active' if term is None else 'Terminated' for term in termination_dates],
    'cdl_class': ['A'] * NUM_DRIVERS,
    'years_experience': [random.randint(2, 25) for _ in range(NUM_DRIVERS)]
})

drivers.to_csv(OUTPUT_DIR / 'drivers.csv', index=False)
print(f"  ✓ Generated {len(drivers)} driver records")

# 2. TRUCKS TABLE
print("Generating trucks table...")
truck_ids = [f"TRK{str(i+1).zfill(5)}" for i in range(NUM_TRUCKS)]
truck_acquisition_dates = [START_DATE - timedelta(days=random.randint(0, 2555)) for _ in range(NUM_TRUCKS)]

trucks = pd.DataFrame({
    'truck_id': truck_ids,
    'unit_number': [f"{random.randint(1000, 9999)}" for _ in range(NUM_TRUCKS)],
    'make': [random.choice(TRUCK_MAKES) for _ in range(NUM_TRUCKS)],
    'model_year': [max(2015, date.year - random.randint(0, 7)) for date in truck_acquisition_dates],
    'vin': [f"1{random.choice(['F', 'N', 'V', 'M'])}{random.choice(['T', 'U', 'V'])}"
            f"{random.randint(100000000000000, 999999999999999)}" for _ in range(NUM_TRUCKS)],
    'acquisition_date': truck_acquisition_dates,
    'acquisition_mileage': [random.randint(0, 50000) for _ in range(NUM_TRUCKS)],
    'fuel_type': ['Diesel'] * NUM_TRUCKS,
    'tank_capacity_gallons': [random.choice([150, 200, 250]) for _ in range(NUM_TRUCKS)],
    'status': [random.choice(['Active', 'Active', 'Active', 'Active', 'Active',
                              'Active', 'Active', 'Active', 'Maintenance', 'Inactive'])
               for _ in range(NUM_TRUCKS)],
    'home_terminal': [random.choice([city[0] for city in CITIES]) for _ in range(NUM_TRUCKS)]
})

trucks.to_csv(OUTPUT_DIR / 'trucks.csv', index=False)
print(f"  ✓ Generated {len(trucks)} truck records")

# 3. TRAILERS TABLE
print("Generating trailers table...")
trailer_ids = [f"TRL{str(i+1).zfill(5)}" for i in range(NUM_TRAILERS)]

trailers = pd.DataFrame({
    'trailer_id': trailer_ids,
    'trailer_number': [f"{random.randint(1000, 9999)}" for _ in range(NUM_TRAILERS)],
    'trailer_type': [random.choice(TRAILER_TYPES) for _ in range(NUM_TRAILERS)],
    'length_feet': [53] * NUM_TRAILERS,
    'model_year': [random.randint(2015, 2024) for _ in range(NUM_TRAILERS)],
    'vin': [f"1{random.choice(['A', 'B', 'C'])}{random.choice(['T', 'U', 'V'])}"
            f"{random.randint(100000000000000, 999999999999999)}" for _ in range(NUM_TRAILERS)],
    'acquisition_date': [START_DATE - timedelta(days=random.randint(0, 2555)) for _ in range(NUM_TRAILERS)],
    'status': ['Active'] * NUM_TRAILERS,
    'current_location': [random.choice([city[0] for city in CITIES]) for _ in range(NUM_TRAILERS)]
})

trailers.to_csv(OUTPUT_DIR / 'trailers.csv', index=False)
print(f"  ✓ Generated {len(trailers)} trailer records")

# 4. CUSTOMERS TABLE
print("Generating customers table...")
customer_ids = [f"CUST{str(i+1).zfill(5)}" for i in range(NUM_CUSTOMERS)]

company_prefixes = ['ABC', 'XYZ', 'Global', 'United', 'Premier', 'Superior', 'National',
                    'American', 'First', 'Elite', 'Metro', 'Continental', 'Pacific']
company_types = ['Logistics', 'Distribution', 'Manufacturing', 'Retail', 'Foods',
                 'Supply Chain', 'Wholesale', 'Industries', 'Group', 'Corp']

customers = pd.DataFrame({
    'customer_id': customer_ids,
    'customer_name': [f"{random.choice(company_prefixes)} {random.choice(company_types)}"
                      for _ in range(NUM_CUSTOMERS)],
    'customer_type': [random.choice(CUSTOMER_TYPES) for _ in range(NUM_CUSTOMERS)],
    'credit_terms_days': [random.choice([15, 30, 45, 60]) for _ in range(NUM_CUSTOMERS)],
    'primary_freight_type': [random.choice(FREIGHT_TYPES) for _ in range(NUM_CUSTOMERS)],
    'account_status': [random.choice(['Active', 'Active', 'Active', 'Active', 'Inactive'])
                       for _ in range(NUM_CUSTOMERS)],
    'contract_start_date': [START_DATE - timedelta(days=random.randint(0, 730))
                            for _ in range(NUM_CUSTOMERS)],
    'annual_revenue_potential': [random.randint(100000, 5000000) for _ in range(NUM_CUSTOMERS)]
})

customers.to_csv(OUTPUT_DIR / 'customers.csv', index=False)
print(f"  ✓ Generated {len(customers)} customer records")

# 5. FACILITIES TABLE
print("Generating facilities table...")
facility_ids = [f"FAC{str(i+1).zfill(5)}" for i in range(NUM_FACILITIES)]
facility_cities = random.choices(CITIES, k=NUM_FACILITIES)

facilities = pd.DataFrame({
    'facility_id': facility_ids,
    'facility_name': [f"{city[0]} {random.choice(['Terminal', 'Distribution Center', 'Warehouse', 'Hub'])}"
                      for city in facility_cities],
    'facility_type': [random.choice(['Terminal', 'Distribution Center', 'Warehouse', 'Cross-Dock'])
                      for _ in range(NUM_FACILITIES)],
    'city': [city[0] for city in facility_cities],
    'state': [city[1] for city in facility_cities],
    'latitude': [city[2] for city in facility_cities],
    'longitude': [city[3] for city in facility_cities],
    'dock_doors': [random.randint(10, 150) for _ in range(NUM_FACILITIES)],
    'operating_hours': [random.choice(['24/7', '6AM-10PM', '7AM-7PM', '8AM-5PM'])
                        for _ in range(NUM_FACILITIES)]
})

facilities.to_csv(OUTPUT_DIR / 'facilities.csv', index=False)
print(f"  ✓ Generated {len(facilities)} facility records")

# 6. ROUTES TABLE
print("Generating routes table...")
route_pairs = []
route_ids = []
route_counter = 1

for i, origin in enumerate(CITIES[:20]):
    for dest in CITIES[:20]:
        if origin != dest and random.random() < 0.15:
            route_pairs.append((origin, dest))
            route_ids.append(f"RTE{str(route_counter).zfill(5)}")
            route_counter += 1

NUM_ROUTES = len(route_pairs)

routes = pd.DataFrame({
    'route_id': route_ids,
    'origin_city': [pair[0][0] for pair in route_pairs],
    'origin_state': [pair[0][1] for pair in route_pairs],
    'destination_city': [pair[1][0] for pair in route_pairs],
    'destination_state': [pair[1][1] for pair in route_pairs],
    'typical_distance_miles': [
        int(haversine_distance(pair[0][2], pair[0][3], pair[1][2], pair[1][3]) * 1.15)
        for pair in route_pairs
    ],
    'base_rate_per_mile': [round(random.uniform(1.50, 2.80), 2) for _ in range(NUM_ROUTES)],
    'fuel_surcharge_rate': [round(random.uniform(0.15, 0.35), 2) for _ in range(NUM_ROUTES)],
    'typical_transit_days': [
        max(1, int(haversine_distance(pair[0][2], pair[0][3], pair[1][2], pair[1][3]) / 500))
        for pair in route_pairs
    ]
})

routes.to_csv(OUTPUT_DIR / 'routes.csv', index=False)
print(f"  ✓ Generated {len(routes)} route records")

# 7. LOADS TABLE
print("Generating loads table (this will take a moment)...")

total_days = (END_DATE - START_DATE).days
NUM_LOADS = int(NUM_TRUCKS * total_days * 0.65)

load_ids = [f"LOAD{str(i+1).zfill(8)}" for i in range(NUM_LOADS)]
load_dates = sorted([START_DATE + timedelta(days=random.randint(0, total_days)) for _ in range(NUM_LOADS)])

load_routes = random.choices(routes['route_id'].tolist(), k=NUM_LOADS)
load_customers = random.choices(customers['customer_id'].tolist(), k=NUM_LOADS)

# Pre-calculate revenue data
route_lookup = routes.set_index('route_id')[['typical_distance_miles', 'base_rate_per_mile', 'fuel_surcharge_rate']].to_dict('index')

revenues = []
fuel_surcharges = []

for route_id in load_routes:
    route_info = route_lookup[route_id]
    distance = route_info['typical_distance_miles']
    base_rate = route_info['base_rate_per_mile']
    fsc_rate = route_info['fuel_surcharge_rate']

    actual_rate = base_rate * random.uniform(0.85, 1.15)
    revenues.append(round(distance * actual_rate, 2))
    fuel_surcharges.append(round(distance * fsc_rate, 2))

loads = pd.DataFrame({
    'load_id': load_ids,
    'customer_id': load_customers,
    'route_id': load_routes,
    'load_date': load_dates,
    'load_type': [random.choice(['Dry Van', 'Refrigerated']) for _ in range(NUM_LOADS)],
    'weight_lbs': [random.randint(10000, 45000) for _ in range(NUM_LOADS)],
    'pieces': [random.randint(1, 28) for _ in range(NUM_LOADS)],
    'revenue': revenues,
    'fuel_surcharge': fuel_surcharges,
    'accessorial_charges': [random.choice([0, 0, 0, 50, 75, 100, 150, 200]) for _ in range(NUM_LOADS)],
    'load_status': ['Completed'] * NUM_LOADS,
    'booking_type': [random.choice(['Dedicated', 'Dedicated', 'Contract', 'Spot']) for _ in range(NUM_LOADS)]
})

loads.to_csv(OUTPUT_DIR / 'loads.csv', index=False)
print(f"  ✓ Generated {len(loads)} load records")

# 8. TRIPS TABLE
print("Generating trips table...")

trip_ids = [f"TRIP{str(i+1).zfill(8)}" for i in range(NUM_LOADS)]

active_drivers = drivers[drivers['employment_status'] == 'Active']['driver_id'].tolist()
active_trucks = trucks[trucks['status'] == 'Active']['truck_id'].tolist()
active_trailers = trailers[trailers['status'] == 'Active']['trailer_id'].tolist()

# Pre-calculate trip metrics
actual_distances = []
actual_durations = []
fuel_gallons = []
avg_mpgs = []
idle_times = []

for route_id in load_routes:
    route_info = route_lookup[route_id]
    planned_distance = route_info['typical_distance_miles']

    actual_distance = int(planned_distance * random.uniform(0.98, 1.08))
    actual_distances.append(actual_distance)

    actual_durations.append(round(actual_distance / random.uniform(50, 65), 1))

    mpg = round(random.uniform(5.5, 7.5), 2)
    avg_mpgs.append(mpg)
    fuel_gallons.append(round(actual_distance / mpg, 1))

    idle_times.append(round(random.uniform(2, 12), 1))

trips = pd.DataFrame({
    'trip_id': trip_ids,
    'load_id': load_ids,
    'driver_id': [random.choice(active_drivers) if random.random() > 0.02 else None
                  for _ in range(NUM_LOADS)],
    'truck_id': [random.choice(active_trucks) if random.random() > 0.02 else None
                 for _ in range(NUM_LOADS)],
    'trailer_id': [random.choice(active_trailers) if random.random() > 0.02 else None
                   for _ in range(NUM_LOADS)],
    'dispatch_date': load_dates,
    'actual_distance_miles': actual_distances,
    'actual_duration_hours': actual_durations,
    'fuel_gallons_used': fuel_gallons,
    'average_mpg': avg_mpgs,
    'idle_time_hours': idle_times,
    'trip_status': ['Completed'] * NUM_LOADS
})

trips.to_csv(OUTPUT_DIR / 'trips.csv', index=False)
print(f"  ✓ Generated {len(trips)} trip records")

# 9. FUEL PURCHASES TABLE
print("Generating fuel purchases table...")

NUM_FUEL_PURCHASES = int(NUM_LOADS * 2.3)
fuel_purchase_ids = [f"FUEL{str(i+1).zfill(8)}" for i in range(NUM_FUEL_PURCHASES)]

fuel_trips = random.choices(trip_ids, k=NUM_FUEL_PURCHASES)

# Pre-fetch trip data for fuel purchases
trip_lookup = trips.set_index('trip_id')[['truck_id', 'driver_id', 'dispatch_date']].to_dict('index')

fuel_trucks = []
fuel_drivers = []
fuel_dates = []

for trip_id in fuel_trips:
    trip_info = trip_lookup[trip_id]
    fuel_trucks.append(trip_info['truck_id'])
    fuel_drivers.append(trip_info['driver_id'])
    fuel_dates.append(trip_info['dispatch_date'] + timedelta(hours=random.randint(0, 72)))

def get_fuel_price(date):
    base_price = 4.20
    if date.year == 2022:
        return base_price + random.uniform(-0.80, 0.80)
    elif date.year == 2023:
        return 3.85 + random.uniform(-0.60, 0.60)
    else:
        return 3.65 + random.uniform(-0.50, 0.50)

gallons_list = [round(random.uniform(50, 200), 1) for _ in range(NUM_FUEL_PURCHASES)]
prices_list = [round(get_fuel_price(date), 3) for date in fuel_dates]

fuel_purchases = pd.DataFrame({
    'fuel_purchase_id': fuel_purchase_ids,
    'trip_id': fuel_trips,
    'truck_id': fuel_trucks,
    'driver_id': fuel_drivers,
    'purchase_date': fuel_dates,
    'location_city': [random.choice([city[0] for city in CITIES]) for _ in range(NUM_FUEL_PURCHASES)],
    'location_state': [random.choice([city[1] for city in CITIES]) for _ in range(NUM_FUEL_PURCHASES)],
    'gallons': gallons_list,
    'price_per_gallon': prices_list,
    'total_cost': [round(g * p, 2) for g, p in zip(gallons_list, prices_list)],
    'fuel_card_number': [f"FC{random.randint(100000, 999999)}" for _ in range(NUM_FUEL_PURCHASES)]
})

fuel_purchases.to_csv(OUTPUT_DIR / 'fuel_purchases.csv', index=False)
print(f"  ✓ Generated {len(fuel_purchases)} fuel purchase records")

# 10. MAINTENANCE RECORDS TABLE
print("Generating maintenance records table...")

NUM_MAINTENANCE = int(NUM_TRUCKS * total_days / 45)
maintenance_ids = [f"MAINT{str(i+1).zfill(8)}" for i in range(NUM_MAINTENANCE)]

maintenance_trucks = random.choices(truck_ids, k=NUM_MAINTENANCE)
maintenance_dates = sorted([START_DATE + timedelta(days=random.randint(0, total_days))
                            for _ in range(NUM_MAINTENANCE)])

labor_hours_list = [round(random.uniform(0.5, 8.0), 1) for _ in range(NUM_MAINTENANCE)]
maintenance_types_list = [random.choice(MAINTENANCE_TYPES) for _ in range(NUM_MAINTENANCE)]

labor_costs = [round(hours * random.uniform(85, 125), 2) for hours in labor_hours_list]
parts_costs = [round(random.uniform(50, 3500), 2) if mtype != 'Inspection'
               else round(random.uniform(0, 150), 2)
               for mtype in maintenance_types_list]

maintenance = pd.DataFrame({
    'maintenance_id': maintenance_ids,
    'truck_id': maintenance_trucks,
    'maintenance_date': maintenance_dates,
    'maintenance_type': maintenance_types_list,
    'odometer_reading': [random.randint(50000, 750000) for _ in range(NUM_MAINTENANCE)],
    'labor_hours': labor_hours_list,
    'labor_cost': labor_costs,
    'parts_cost': parts_costs,
    'total_cost': [round(lc + pc, 2) for lc, pc in zip(labor_costs, parts_costs)],
    'facility_location': [random.choice([city[0] for city in CITIES]) for _ in range(NUM_MAINTENANCE)],
    'downtime_hours': [round(random.uniform(2, 48), 1) for _ in range(NUM_MAINTENANCE)],
    'service_description': [f"{random.choice(['Routine', 'Emergency', 'Scheduled'])} {mtype}"
                           for mtype in maintenance_types_list]
})

maintenance.to_csv(OUTPUT_DIR / 'maintenance_records.csv', index=False)
print(f"  ✓ Generated {len(maintenance)} maintenance records")

# 11. DELIVERY EVENTS TABLE
print("Generating delivery events table...")

NUM_EVENTS = NUM_LOADS * 2
event_ids = [f"EVT{str(i+1).zfill(8)}" for i in range(NUM_EVENTS)]

delivery_events_data = []

# Create lookup for efficient access
load_lookup = loads.set_index('load_id')['route_id'].to_dict()
route_df = routes.set_index('route_id')
trip_df = trips.set_index('load_id')

for i, load_id in enumerate(load_ids):
    trip = trip_df.loc[load_id]
    route_id = load_lookup[load_id]
    route_info = route_df.loc[route_id]

    dispatch_date = trip['dispatch_date']
    transit_hours = trip['actual_duration_hours']

    pickup_scheduled = dispatch_date + timedelta(hours=random.randint(6, 18))
    pickup_actual = pickup_scheduled + timedelta(hours=random.uniform(-2, 4))

    delivery_scheduled = pickup_actual + timedelta(hours=float(transit_hours))
    delivery_actual = delivery_scheduled + timedelta(hours=random.uniform(-3, 6))

    pickup_detention = max(0, random.uniform(-30, 180))
    delivery_detention = max(0, random.uniform(-30, 240))

    pickup_ontime = abs((pickup_actual - pickup_scheduled).total_seconds() / 3600) <= 2
    delivery_ontime = abs((delivery_actual - delivery_scheduled).total_seconds() / 3600) <= 2

    delivery_events_data.extend([
        {
            'event_id': event_ids[i*2],
            'load_id': load_id,
            'trip_id': trip['trip_id'],
            'event_type': 'Pickup',
            'facility_id': random.choice(facility_ids),
            'scheduled_datetime': pickup_scheduled,
            'actual_datetime': pickup_actual,
            'detention_minutes': int(pickup_detention),
            'on_time_flag': pickup_ontime,
            'location_city': route_info['origin_city'],
            'location_state': route_info['origin_state']
        },
        {
            'event_id': event_ids[i*2 + 1],
            'load_id': load_id,
            'trip_id': trip['trip_id'],
            'event_type': 'Delivery',
            'facility_id': random.choice(facility_ids),
            'scheduled_datetime': delivery_scheduled,
            'actual_datetime': delivery_actual,
            'detention_minutes': int(delivery_detention),
            'on_time_flag': delivery_ontime,
            'location_city': route_info['destination_city'],
            'location_state': route_info['destination_state']
        }
    ])

delivery_events = pd.DataFrame(delivery_events_data)
delivery_events.to_csv(OUTPUT_DIR / 'delivery_events.csv', index=False)
print(f"  ✓ Generated {len(delivery_events)} delivery event records")

# 12. SAFETY INCIDENTS TABLE
print("Generating safety incidents table...")

NUM_INCIDENTS = max(int(NUM_LOADS * 0.002), 50)
incident_ids = [f"INC{str(i+1).zfill(8)}" for i in range(NUM_INCIDENTS)]

incident_trips = random.choices(trip_ids, k=NUM_INCIDENTS)

incident_trucks = []
incident_drivers = []
incident_dates = []

for trip_id in incident_trips:
    trip_info = trip_lookup[trip_id]
    incident_trucks.append(trip_info['truck_id'])
    incident_drivers.append(trip_info['driver_id'])
    incident_dates.append(trip_info['dispatch_date'] + timedelta(hours=random.randint(0, 72)))

vehicle_damages = [round(random.uniform(0, 25000), 2) if random.random() > 0.3 else 0.0
                   for _ in range(NUM_INCIDENTS)]
cargo_damages = [round(random.uniform(0, 50000), 2) if random.random() > 0.7 else 0.0
                 for _ in range(NUM_INCIDENTS)]

safety_incidents = pd.DataFrame({
    'incident_id': incident_ids,
    'trip_id': incident_trips,
    'truck_id': incident_trucks,
    'driver_id': incident_drivers,
    'incident_date': incident_dates,
    'incident_type': [random.choice(INCIDENT_TYPES) for _ in range(NUM_INCIDENTS)],
    'location_city': [random.choice([city[0] for city in CITIES]) for _ in range(NUM_INCIDENTS)],
    'location_state': [random.choice([city[1] for city in CITIES]) for _ in range(NUM_INCIDENTS)],
    'at_fault_flag': [random.choice([True, False, False]) for _ in range(NUM_INCIDENTS)],
    'injury_flag': [random.choice([False, False, False, False, True]) for _ in range(NUM_INCIDENTS)],
    'vehicle_damage_cost': vehicle_damages,
    'cargo_damage_cost': cargo_damages,
    'claim_amount': [round(v + c, 2) for v, c in zip(vehicle_damages, cargo_damages)],
    'preventable_flag': [random.choice([True, False, False]) for _ in range(NUM_INCIDENTS)],
    'description': [f"{random.choice(['Minor', 'Moderate', 'Severe'])} incident involving {random.choice(['traffic', 'weather', 'equipment', 'other driver'])}"
                    for _ in range(NUM_INCIDENTS)]
})

safety_incidents.to_csv(OUTPUT_DIR / 'safety_incidents.csv', index=False)
print(f"  ✓ Generated {len(safety_incidents)} safety incident records")

# 13. DRIVER MONTHLY METRICS TABLE
print("Generating driver monthly metrics table...")

active_driver_ids = drivers[drivers['employment_status'] == 'Active']['driver_id'].tolist()
months = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')

driver_metrics_data = []

trip_df_full = trips.set_index('driver_id')
load_df = loads.set_index('load_id')
delivery_df = delivery_events[delivery_events['event_type'] == 'Delivery'].set_index('trip_id')

for driver_id in active_driver_ids:
    if driver_id not in trip_df_full.index:
        continue

    driver_trips = trip_df_full.loc[[driver_id]]

    for month in months:
        month_end = month + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        month_trips = driver_trips[
            (driver_trips['dispatch_date'] >= month) &
            (driver_trips['dispatch_date'] <= month_end)
        ]

        if len(month_trips) > 0:
            trip_load_ids = month_trips['load_id'].tolist()
            month_loads = load_df.loc[load_df.index.isin(trip_load_ids)]

            trip_ids_list = month_trips['trip_id'].tolist()
            delivery_evts = delivery_df.loc[delivery_df.index.isin(trip_ids_list)]

            driver_metrics_data.append({
                'driver_id': driver_id,
                'month': month.date(),
                'trips_completed': len(month_trips),
                'total_miles': int(month_trips['actual_distance_miles'].sum()),
                'total_revenue': round(month_loads['revenue'].sum(), 2),
                'average_mpg': round(month_trips['average_mpg'].mean(), 2),
                'total_fuel_gallons': round(month_trips['fuel_gallons_used'].sum(), 1),
                'on_time_delivery_rate': round(delivery_evts['on_time_flag'].mean(), 3) if len(delivery_evts) > 0 else None,
                'average_idle_hours': round(month_trips['idle_time_hours'].mean(), 1)
            })

driver_monthly_metrics = pd.DataFrame(driver_metrics_data)
driver_monthly_metrics.to_csv(OUTPUT_DIR / 'driver_monthly_metrics.csv', index=False)
print(f"  ✓ Generated {len(driver_monthly_metrics)} driver monthly metric records")

# 14. TRUCK UTILIZATION METRICS TABLE
print("Generating truck utilization metrics table...")

truck_metrics_data = []

trip_df_trucks = trips.set_index('truck_id')
maintenance_df = maintenance.set_index('truck_id')

for truck_id in truck_ids:
    if truck_id not in trip_df_trucks.index:
        continue

    truck_trips = trip_df_trucks.loc[[truck_id]]
    truck_maint = maintenance_df.loc[[truck_id]] if truck_id in maintenance_df.index else pd.DataFrame()

    for month in months:
        month_end = month + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        month_trips = truck_trips[
            (truck_trips['dispatch_date'] >= month) &
            (truck_trips['dispatch_date'] <= month_end)
        ]

        if len(month_trips) > 0:
            month_maint = truck_maint[
                (truck_maint['maintenance_date'] >= month) &
                (truck_maint['maintenance_date'] <= month_end)
            ] if len(truck_maint) > 0 else pd.DataFrame()

            days_in_month = (month_end - month).days + 1

            trip_load_ids = month_trips['load_id'].tolist()
            month_loads = load_df.loc[load_df.index.isin(trip_load_ids)]

            truck_metrics_data.append({
                'truck_id': truck_id,
                'month': month.date(),
                'trips_completed': len(month_trips),
                'total_miles': int(month_trips['actual_distance_miles'].sum()),
                'total_revenue': round(month_loads['revenue'].sum(), 2),
                'average_mpg': round(month_trips['average_mpg'].mean(), 2),
                'maintenance_events': len(month_maint),
                'maintenance_cost': round(month_maint['total_cost'].sum(), 2) if len(month_maint) > 0 else 0.0,
                'downtime_hours': round(month_maint['downtime_hours'].sum(), 1) if len(month_maint) > 0 else 0.0,
                'utilization_rate': round(len(month_trips) / days_in_month, 3)
            })

truck_utilization_metrics = pd.DataFrame(truck_metrics_data)
truck_utilization_metrics.to_csv(OUTPUT_DIR / 'truck_utilization_metrics.csv', index=False)
print(f"  ✓ Generated {len(truck_utilization_metrics)} truck utilization metric records")

# Generate summary statistics
print("\n" + "="*60)
print("DATABASE GENERATION COMPLETE")
print("="*60)

summary = {
    'Table': [],
    'Records': [],
    'Date Range': [],
    'File Size (KB)': []
}

for csv_file in sorted(OUTPUT_DIR.glob('*.csv')):
    df = pd.read_csv(csv_file)
    summary['Table'].append(csv_file.stem)
    summary['Records'].append(len(df))

    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        try:
            dates = pd.to_datetime(df[date_cols[0]])
            summary['Date Range'].append(f"{dates.min().date()} to {dates.max().date()}")
        except:
            summary['Date Range'].append('N/A')
    else:
        summary['Date Range'].append('N/A')

    summary['File Size (KB)'].append(round(csv_file.stat().st_size / 1024, 1))

summary_df = pd.DataFrame(summary)
print("\n" + summary_df.to_string(index=False))

# Create schema documentation
schema_doc = """
LOGISTICS DATABASE SCHEMA
=========================

1. DRIVERS
   - Primary Key: driver_id
   - Contains: Driver demographics, employment history, license info
   
2. TRUCKS
   - Primary Key: truck_id
   - Contains: Fleet equipment details, acquisition info, status
   
3. TRAILERS
   - Primary Key: trailer_id
   - Contains: Trailer inventory, types, status
   
4. CUSTOMERS
   - Primary Key: customer_id
   - Contains: Customer accounts, contract types, revenue potential
   
5. FACILITIES
   - Primary Key: facility_id
   - Contains: Terminal and warehouse locations, capacity
   
6. ROUTES
   - Primary Key: route_id
   - Contains: Origin-destination pairs, distances, rate structures
   
7. LOADS
   - Primary Key: load_id
   - Foreign Keys: customer_id, route_id
   - Contains: Shipment details, revenue, booking type
   
8. TRIPS
   - Primary Key: trip_id
   - Foreign Keys: load_id, driver_id, truck_id, trailer_id
   - Contains: Actual trip performance, fuel consumption, duration
   
9. FUEL_PURCHASES
   - Primary Key: fuel_purchase_id
   - Foreign Keys: trip_id, truck_id, driver_id
   - Contains: Fuel transactions, prices, locations
   
10. MAINTENANCE_RECORDS
    - Primary Key: maintenance_id
    - Foreign Keys: truck_id
    - Contains: Service history, costs, downtime
    
11. DELIVERY_EVENTS
    - Primary Key: event_id
    - Foreign Keys: load_id, trip_id, facility_id
    - Contains: Pickup/delivery timestamps, detention, on-time status
    
12. SAFETY_INCIDENTS
    - Primary Key: incident_id
    - Foreign Keys: trip_id, truck_id, driver_id
    - Contains: Accidents, violations, damage costs
    
13. DRIVER_MONTHLY_METRICS (Aggregated)
    - Composite Key: driver_id, month
    - Contains: Monthly performance summaries per driver
    
14. TRUCK_UTILIZATION_METRICS (Aggregated)
    - Composite Key: truck_id, month
    - Contains: Monthly equipment utilization summaries

KEY RELATIONSHIPS:
- loads -> customers (many-to-one)
- loads -> routes (many-to-one)
- trips -> loads (one-to-one)
- trips -> drivers (many-to-one)
- trips -> trucks (many-to-one)
- trips -> trailers (many-to-one)
- fuel_purchases -> trips (many-to-one)
- maintenance_records -> trucks (many-to-one)
- delivery_events -> trips (many-to-one)
- safety_incidents -> trips (many-to-one)

ANALYTICAL USE CASES:
1. Driver performance: On-time rates, MPG, revenue per mile
2. Route profitability: Revenue vs costs by lane
3. Fleet utilization: Miles per truck, revenue per asset
4. Maintenance analysis: Cost per mile, downtime impact
5. Fuel efficiency: MPG trends, fuel cost by route
6. Customer analysis: Revenue by customer, service levels
7. Safety metrics: Incident rates, preventable accidents
8. Seasonal patterns: Load volume, rate fluctuations
"""

with open(OUTPUT_DIR / 'DATABASE_SCHEMA.txt', 'w') as f:
    f.write(schema_doc)

print(f"\nAll files saved to: {OUTPUT_DIR.absolute()}")
print(f"Schema documentation: {(OUTPUT_DIR / 'DATABASE_SCHEMA.txt').absolute()}")
print("\nReady for PostgreSQL import and SQL practice!")