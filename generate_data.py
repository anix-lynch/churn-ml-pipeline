#!/usr/bin/env python3
"""
Generate synthetic data for churn ML pipeline
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# Constants
N_CUSTOMERS = 10000
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2024, 12, 31)

def generate_customers():
    """Generate customers.csv: customer_id, signup_date, country, segment, age, gender"""
    customer_ids = [f"CUST_{i:05d}" for i in range(1, N_CUSTOMERS + 1)]

    # Signup dates - biased towards more recent
    signup_dates = []
    for _ in range(N_CUSTOMERS):
        days_range = (END_DATE - START_DATE).days
        random_days = int(np.random.exponential(scale=days_range/3))
        signup_date = START_DATE + timedelta(days=min(random_days, days_range))
        signup_dates.append(signup_date.strftime('%Y-%m-%d'))

    countries = ['US', 'UK', 'CA', 'DE', 'FR', 'AU', 'JP', 'IN', 'BR', 'MX']
    segments = ['premium', 'standard', 'basic', 'enterprise']
    genders = ['M', 'F', 'Other']

    data = {
        'customer_id': customer_ids,
        'signup_date': signup_dates,
        'country': np.random.choice(countries, N_CUSTOMERS, p=[0.4, 0.15, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]),
        'segment': np.random.choice(segments, N_CUSTOMERS, p=[0.2, 0.4, 0.3, 0.1]),
        'age': np.random.normal(35, 12, N_CUSTOMERS).astype(int).clip(18, 80),
        'gender': np.random.choice(genders, N_CUSTOMERS, p=[0.48, 0.48, 0.04])
    }

    df = pd.DataFrame(data)
    df.to_csv('/Users/anixlynch/dev/Takehome1_endtoend/data/raw/customers.csv', index=False)
    print(f"Generated {len(df)} customers")

def generate_transactions():
    """Generate transactions.csv: transaction_id, customer_id, amount, timestamp, channel"""
    # Each customer has 0-50 transactions
    transactions = []

    customer_ids = pd.read_csv('/Users/anixlynch/dev/Takehome1_endtoend/data/raw/customers.csv')['customer_id'].tolist()

    transaction_id = 1
    channels = ['web', 'mobile', 'api', 'in_app']

    for cust_id in customer_ids:
        n_transactions = np.random.poisson(5) + 1  # At least 1 transaction

        for _ in range(n_transactions):
            # Transaction date after signup
            signup_date = pd.to_datetime(pd.read_csv('/Users/anixlynch/dev/Takehome1_endtoend/data/raw/customers.csv',
                                                   usecols=['customer_id', 'signup_date']).set_index('customer_id').loc[cust_id, 'signup_date'])

            days_since_signup = (END_DATE - signup_date).days
            if days_since_signup > 0:
                transaction_days = np.random.randint(0, days_since_signup)
                transaction_date = signup_date + timedelta(days=transaction_days)
            else:
                transaction_date = signup_date

            transactions.append({
                'transaction_id': f"TXN_{transaction_id:07d}",
                'customer_id': cust_id,
                'amount': round(np.random.exponential(50) + 5, 2),  # Most transactions small
                'timestamp': transaction_date.strftime('%Y-%m-%d %H:%M:%S'),
                'channel': np.random.choice(channels, p=[0.4, 0.3, 0.2, 0.1])
            })
            transaction_id += 1

    df = pd.DataFrame(transactions)
    df.to_csv('/Users/anixlynch/dev/Takehome1_endtoend/data/raw/transactions.csv', index=False)
    print(f"Generated {len(df)} transactions")

def generate_sessions():
    """Generate sessions.csv: session_id, customer_id, session_start, session_end, pages_viewed, device"""
    sessions = []

    customer_ids = pd.read_csv('/Users/anixlynch/dev/Takehome1_endtoend/data/raw/customers.csv')['customer_id'].tolist()

    session_id = 1
    devices = ['desktop', 'mobile', 'tablet']

    for cust_id in customer_ids:
        # Each customer has 0-100 sessions
        n_sessions = np.random.poisson(15)

        signup_date = pd.to_datetime(pd.read_csv('/Users/anixlynch/dev/Takehome1_endtoend/data/raw/customers.csv',
                                               usecols=['customer_id', 'signup_date']).set_index('customer_id').loc[cust_id, 'signup_date'])

        for _ in range(n_sessions):
            days_since_signup = (END_DATE - signup_date).days
            if days_since_signup > 0:
                session_days = np.random.randint(0, days_since_signup)
                session_start = signup_date + timedelta(days=session_days)
            else:
                session_start = signup_date

            # Session duration 1-120 minutes
            duration_minutes = np.random.exponential(20) + 1
            session_end = session_start + timedelta(minutes=int(duration_minutes))

            sessions.append({
                'session_id': f"SESS_{session_id:07d}",
                'customer_id': cust_id,
                'session_start': session_start.strftime('%Y-%m-%d %H:%M:%S'),
                'session_end': session_end.strftime('%Y-%m-%d %H:%M:%S'),
                'pages_viewed': np.random.poisson(5) + 1,
                'device': np.random.choice(devices, p=[0.5, 0.4, 0.1])
            })
            session_id += 1

    df = pd.DataFrame(sessions)
    df.to_csv('/Users/anixlynch/dev/Takehome1_endtoend/data/raw/sessions.csv', index=False)
    print(f"Generated {len(df)} sessions")

def generate_support_tickets():
    """Generate support_tickets.csv: ticket_id, customer_id, created_at, resolved_at, category, satisfaction_score"""
    tickets = []

    customer_ids = pd.read_csv('/Users/anixlynch/dev/Takehome1_endtoend/data/raw/customers.csv')['customer_id'].tolist()

    ticket_id = 1
    categories = ['billing', 'technical', 'account', 'feature_request', 'general']
    satisfaction_scores = [1, 2, 3, 4, 5, None]  # Some unresolved

    for cust_id in customer_ids:
        # Each customer has 0-10 tickets
        n_tickets = np.random.poisson(2)

        signup_date = pd.to_datetime(pd.read_csv('/Users/anixlynch/dev/Takehome1_endtoend/data/raw/customers.csv',
                                               usecols=['customer_id', 'signup_date']).set_index('customer_id').loc[cust_id, 'signup_date'])

        for _ in range(n_tickets):
            days_since_signup = (END_DATE - signup_date).days
            if days_since_signup > 0:
                ticket_days = np.random.randint(0, days_since_signup)
                created_at = signup_date + timedelta(days=ticket_days)
            else:
                created_at = signup_date

            # Resolution time 1-30 days, or None (unresolved)
            is_resolved = np.random.choice([True, False], p=[0.8, 0.2])
            if is_resolved:
                resolution_days = np.random.exponential(7) + 1
                resolved_at = created_at + timedelta(days=int(resolution_days))
                satisfaction = np.random.choice(satisfaction_scores[:-1], p=[0.1, 0.1, 0.2, 0.3, 0.3])
            else:
                resolved_at = None
                satisfaction = None

            tickets.append({
                'ticket_id': f"TICK_{ticket_id:06d}",
                'customer_id': cust_id,
                'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'resolved_at': resolved_at.strftime('%Y-%m-%d %H:%M:%S') if resolved_at else None,
                'category': np.random.choice(categories, p=[0.25, 0.3, 0.2, 0.15, 0.1]),
                'satisfaction_score': satisfaction
            })
            ticket_id += 1

    df = pd.DataFrame(tickets)
    df.to_csv('/Users/anixlynch/dev/Takehome1_endtoend/data/raw/support_tickets.csv', index=False)
    print(f"Generated {len(df)} support tickets")

if __name__ == "__main__":
    print("Generating synthetic data...")
    generate_customers()
    generate_transactions()
    generate_sessions()
    generate_support_tickets()
    print("Data generation complete!")
