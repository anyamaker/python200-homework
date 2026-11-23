# Week 8 Project — Supabase Setup and Cloud Cost Analysis

## Supabase Setup

I created a Supabase project called `python200` in the West US (Oregon) region. I created
two tables, `weather_raw` and `weather_enriched`, with the exact SQL from the
assignment, and disabled Row Level Security on both as instructed. I confirmed
in the Table Editor that both tables exist with the expected columns.

## Cloud Cost Analysis

All estimates use the **US East (N. Virginia)** region and **On-Demand** pricing.

### Scenario A — Lightweight compute

- **EC2 t3.micro** (2 vCPU, 1 GB RAM), Linux, 160 hours/month: **$1.66/month**

### Scenario B — Heavy analytics workload

- **EC2 p3.2xlarge** (8 vCPU, 1 GPU V100), Linux, 730 hours/month (24/7): **$2,233.80/month**
- **RDS for PostgreSQL db.m5.large** (2 vCPU, 8 GB RAM), 20 GB gp2 storage, Single-AZ: **$132.24/month**
- **S3 Standard**, 1024 GB (1 TB): **$23.55/month**
- **Total Scenario B: $2,389.59/month**

## Comparison

The GPU instance alone in Scenario B costs about **1,350 times more** than the
whole of Scenario A. The total difference is about **1,439 times**. This tells
me two things:

1. **A GPU instance is worth it only for workloads that actually need it**
   (model training, large-scale inference) and only for the hours you use it.
   For a web API or a light script, a t3.micro is more than enough and a GPU
   would be wasted money.

2. **Cloud costs grow with every checkbox and every hour.** Turning on Multi-AZ,
   RDS Proxy, Database Insights, and Extended Support would have added several
   more dollars per month to the RDS instance. Leaving a GPU instance running
   for a full month (730 hours) is what makes Scenario B expensive — if I only
   needed it for 4 hours of training, the cost would drop from ~$2,234 to ~$12.

## What surprised me

The GPU instance dominates the entire bill. RDS and S3 together are less than
7% of Scenario B. This matches what the lesson said about cost alerts and
cleanup: one forgotten GPU instance left running overnight can cost more than
the entire rest of the project combined.

## Video Walkthrough

[Week 8 walkthrough — Supabase and AWS Cost Analysis](https://www.youtube.com/watch?v=w-ja3FseLeE)