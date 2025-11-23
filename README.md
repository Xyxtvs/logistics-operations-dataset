# Logistics Operations Database Generator

A synthetic data generator that creates realistic, highly normalized transportation and supply chain datasets for SQL analytics practice, data science education, and operations research.

## Overview

This tool generates a complete logistics operations database spanning 3 years (2022-2024) with 85,000+ records across 14 normalized tables. The data models real-world transportation operations including fleet management, route optimization, driver performance, maintenance scheduling, and customer relationships.

Unlike generic datasets, this generator produces domain-specific operational data with authentic patterns found in Class 8 trucking operations: seasonal load variations, realistic fuel price trends, proper maintenance intervals, driver turnover patterns, and equipment utilization rates matching industry standards.

## Dataset Characteristics

**Scale:**
- 85,000+ operational records
- 14 interconnected tables
- 3-year temporal coverage (2022-2024)
- National geographic scope (25+ cities)

**Realism Features:**
- Seasonal freight volume fluctuations
- Historical fuel price patterns (2022 diesel price spike reflected)
- Equipment depreciation and maintenance cycles
- Driver attrition rates (~15% annually)
- On-time delivery performance (85-95% range)
- Load-to-truck ratios reflecting actual market conditions

**Domain Coverage:**
- Fleet operations (120 trucks, 180 trailers)
- Driver management (150 drivers with turnover)
- Customer accounts (200 shippers across contract types)
- Route networks (60+ city pairs)
- Financial transactions (revenue, fuel costs, maintenance)
- Safety and compliance (incident tracking)

## Use Cases

**SQL Practice:**
- Window functions (LAG, LEAD, RANK, NTILE)
- Complex JOINs (6+ table queries)
- CTEs and subqueries
- Time-series analysis
- Aggregation and grouping patterns

**Analytics Training:**
- Driver performance scorecards
- Route profitability analysis
- Fleet utilization optimization
- Predictive maintenance modeling
- Customer segmentation
- Cost-per-mile calculations

**Operations Research:**
- Route optimization algorithms
- Resource allocation modeling
- Scheduling constraint problems
- Capacity planning scenarios

**Data Engineering:**
- ETL pipeline development
- Data warehouse design
- Star schema implementation
- Fact/dimension modeling

## Generated Tables

### Core Entities
- `drivers` - Personnel records, licenses, employment history
- `trucks` - Fleet inventory, specifications, status
- `trailers` - Equipment types, assignments
- `customers` - Shipper accounts, contract terms
- `facilities` - Terminals, warehouses, dock capacity
- `routes` - Origin-destination pairs, rate structures

### Transactional Data
- `loads` - Shipment details, revenue, weight
- `trips` - Driver-truck assignments, actual performance
- `fuel_purchases` - Transaction-level fuel data with historical pricing
- `maintenance_records` - Service history, costs, downtime
- `delivery_events` - Pickup/delivery timestamps, detention
- `safety_incidents` - Accidents, violations, claims

### Aggregated Metrics
- `driver_monthly_metrics` - Performance summaries
- `truck_utilization_metrics` - Equipment efficiency

## Installation

### Requirements

```
pandas>=2.0.0
numpy>=1.24.0
```

### Setup

```bash
# Clone or download the script
git clone https://github.com/Xyxtvs/logistics-operations-dataset.git
cd logistics-data-generator

# Install dependencies
pip install pandas numpy

# Run the generator
python data_gen.py
```

## Usage

### Basic Generation

```python
python data_gen.py
```

Output: 14 CSV files in `./logistics_data/` directory

### Customization

Edit configuration variables at the top of the script:

```python
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 12, 31)
NUM_DRIVERS = 150
NUM_TRUCKS = 120
NUM_TRAILERS = 180
NUM_CUSTOMERS = 200
NUM_FACILITIES = 50
```

### Database Import

**PostgreSQL:**

```sql
-- Create schema
CREATE SCHEMA logistics;

-- Import example (drivers table)
COPY logistics.drivers
FROM '/path/to/logistics_data/drivers.csv'
DELIMITER ',' CSV HEADER;

-- Repeat for all 14 tables
```

**Python (pandas):**

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:pass@localhost/db')

for csv_file in Path('logistics_data').glob('*.csv'):
    df = pd.read_csv(csv_file)
    df.to_sql(csv_file.stem, engine, if_exists='replace', index=False)
```

## Sample Queries

### Driver Performance Ranking

```sql
WITH monthly_performance AS (
  SELECT 
    driver_id,
    DATE_TRUNC('month', dispatch_date) AS month,
    COUNT(*) AS trips,
    AVG(average_mpg) AS avg_mpg,
    SUM(actual_distance_miles) AS total_miles
  FROM trips
  WHERE driver_id IS NOT NULL
  GROUP BY driver_id, DATE_TRUNC('month', dispatch_date)
)
SELECT 
  driver_id,
  month,
  DENSE_RANK() OVER (PARTITION BY month ORDER BY avg_mpg DESC) AS mpg_rank,
  avg_mpg,
  total_miles
FROM monthly_performance
WHERE trips >= 10
ORDER BY month DESC, mpg_rank;
```

### Route Profitability Analysis

```sql
SELECT 
  r.route_id,
  r.origin_city,
  r.destination_city,
  COUNT(l.load_id) AS load_count,
  AVG(l.revenue) AS avg_revenue,
  AVG(t.fuel_gallons_used * 3.85) AS avg_fuel_cost,
  AVG(l.revenue - (t.fuel_gallons_used * 3.85)) AS avg_net_margin
FROM routes r
JOIN loads l ON r.route_id = l.route_id
JOIN trips t ON l.load_id = t.load_id
GROUP BY r.route_id, r.origin_city, r.destination_city
HAVING COUNT(l.load_id) >= 20
ORDER BY avg_net_margin DESC;
```

### On-Time Delivery Trends

```sql
SELECT 
  DATE_TRUNC('month', scheduled_datetime) AS month,
  event_type,
  COUNT(*) AS total_events,
  SUM(CASE WHEN on_time_flag THEN 1 ELSE 0 END) AS on_time_count,
  ROUND(AVG(CASE WHEN on_time_flag THEN 1.0 ELSE 0.0 END) * 100, 2) AS on_time_pct
FROM delivery_events
GROUP BY DATE_TRUNC('month', scheduled_datetime), event_type
ORDER BY month, event_type;
```

## Data Quality Notes

**Intentional Variations:**
- 2% of trips have unassigned drivers/trucks (operational reality)
- 15% driver turnover reflects industry attrition
- Fuel prices mirror actual 2022-2024 market trends
- Detention time and delivery windows include realistic variability

**Synthetic Limitations:**
- Weather patterns not modeled
- Hours of service (HOS) regulations simplified
- Customer credit/payment data excluded
- Cargo claims simplified to basic damage costs

**Foreign Key Integrity:**
- All relationships properly maintained
- No orphaned records
- Referential integrity enforced through generation logic

## Performance Considerations

**Generation Time:**
- ~60-90 seconds on modern hardware
- Memory usage: ~500MB peak
- Output size: ~45MB total (all CSVs)

**Database Import:**
- Recommend creating indexes on foreign keys
- Consider partitioning trips/loads by date
- Aggregated tables speed up dashboard queries

## Advanced Analytics Examples

The dataset supports complex analytical scenarios:

- **Predictive maintenance:** Correlate maintenance_records with trip performance to predict failures
- **Driver retention modeling:** Analyze termination_date patterns against performance metrics
- **Dynamic pricing:** Model fuel_surcharge adjustments based on historical fuel_purchases data
- **Customer lifetime value:** Calculate revenue concentration and contract renewals
- **Safety scoring:** Build risk profiles from safety_incidents linked to driver/equipment

## License

MIT License - Free for commercial and educational use.

## Attribution

If using this dataset in academic work or publications, please cite:

```
Logistics Operations Database Generator (2025)
Synthetic transportation data for analytics education
https://github.com/Xyxtvs/logistics-operations-dataset/
```

## Contributing

Contributions welcome:
- Additional tables (dispatch logs, payroll, insurance)
- Enhanced realism (HOS compliance, weather delays)
- Industry-specific variants (LTL, intermodal, last-mile)
- Validation scripts for data quality checks

## Support

For questions or issues:
- Open a GitHub issue
- Review the DATABASE_SCHEMA.txt file included in output
- Check that all foreign key relationships resolve correctly

## Version History

**v1.0.0** (2025-11-23)
- Initial release
- 14 normalized tables
- 3-year dataset (2022-2024)
- 85,000+ records

---

**Maintained by:**  [Yogape Rodriguez](www.linkedin.com/in/yogape)  
**Domain Expertise:** 12+ years in Class 8 trucking operations.     
**Purpose:** Transition tool for logistics professionals entering data analytics.
