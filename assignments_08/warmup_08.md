# Week 8 Warmup — Cloud Concepts

## Cloud Concepts Question 1

The core economic model of cloud computing is pay-as-you-go: you rent computing resources (compute, storage, network) from a provider and pay only for what you actually use, measured in seconds, hours, or gigabytes. When you own your own servers, you pay a large fixed cost up front for hardware, plus ongoing costs for power, cooling, space, and staff to maintain it — whether the server is busy or idle. With the cloud, that fixed cost becomes a variable cost, and you can scale it up or down to match demand. The tradeoff is that you no longer own the hardware, so you depend on the provider's pricing and availability.

## Cloud Concepts Question 2

Vertical scaling means making a single machine bigger — more CPU, more RAM, a faster GPU. You hit a ceiling at the largest machine the provider offers, and the upgrade often requires downtime. Horizontal scaling means adding more machines and splitting the work across them. There is no hard ceiling (you just add more instances), and it scales well for unpredictable traffic, but the software has to be designed to split work across machines.

Scenario 1 — web app spike from 1,000 to 100,000 users per day: **horizontal scaling**. You add more instances behind a load balancer. A single bigger machine would hit a limit quickly, and you can wind the extra instances back down after the spike.

Scenario 2 — model training too slow, wants faster GPU and more RAM: **vertical scaling**. Training a single model is usually one big job that does not split cleanly across many machines. You just need one bigger machine with a better GPU.

Scenario 3 — data pipeline from 10 to 10,000 files per run, work can be split: **horizontal scaling**. Each file can be processed independently, so you add more workers and distribute the files across them.

## Cloud Concepts Question 3

Classification:
- Gmail → **SaaS**. It is a finished application you log into and use. You manage nothing on the server side.
- Azure Virtual Machines → **IaaS**. You get a raw VM and manage the OS, software, and updates yourself.
- AWS S3 → **IaaS** (or more precisely a managed storage service, but from the three-tier model it fits closest to IaaS — you get raw storage you control). You manage what goes in, who can access it, and the lifecycle rules.
- GitHub Codespaces → **PaaS**. You get a ready-made development environment with code editor and runtime; you manage your code, not the machine.
- Snowflake → **PaaS** (managed data warehouse). The provider runs the infrastructure; you manage your data, schemas, and queries.
- Supabase → **BaaS** (backend-as-a-service). It gives you a database, auth, and file storage behind an API. You manage your schema and data, not the servers.

Definitions:
- **IaaS (Infrastructure as a Service)**: you rent raw computing resources — a virtual machine, storage, network — and you set up everything from the operating system up. Example: Azure Virtual Machines. You manage the OS, installed software, patches, security, and the application itself.
- **PaaS (Platform as a Service)**: the provider manages the infrastructure, and you bring your code. Example: GitHub Codespaces. You manage your code and configuration; the provider handles the machine, OS, and runtime.
- **SaaS (Software as a Service)**: the provider builds, runs, and maintains the whole application. Example: Gmail. You just log in and use it; you manage nothing on the infrastructure side.

## Cloud Concepts Question 4

A managed data platform like Databricks or Snowflake is a layer on top of a cloud provider. It is not a separate cloud — Databricks runs on AWS, GCP, or Azure, and Snowflake runs on top of AWS, Azure, or GCP. The platform pre-wires the data and analytics pieces for you: storage, compute, SQL engine, notebooks, and orchestration, tuned for data workloads. What you gain: much faster start with data pipelines and ML, less infrastructure work, and a familiar interface for analysts. What you give up: flexibility (you cannot use any cloud service, only what the platform exposes), some control over cost and configuration, and potentially higher price than building the same stack yourself on the raw cloud provider.

## Cloud Concepts Question 5

The lesson describes the situation where the data fits comfortably on a single machine and there are no massive compute demands. In that case, local processing is often faster and cheaper than setting up cloud infrastructure. This is a good fit for early prototyping and small projects. The cloud becomes the right choice when you outgrow a single machine, need scale, or need availability that a laptop cannot provide.

# Part 2: Warmup — Cloud Landscape

## Cloud Landscape Question 1

- **Amazon Web Services (AWS)** — the oldest and largest provider, with the broadest service catalog. Primary strength: breadth and ecosystem maturity. Most likely used by large enterprises, startups with engineering teams, and any organization that has been on the cloud for a long time.
- **Google Cloud Platform (GCP)** — strongest in data and machine learning. Primary strength: analytics and ML infrastructure (BigQuery, Vertex AI). Most likely used by data-heavy teams, ML organizations, and companies already using Google's ecosystem.
- **Microsoft Azure** — dominant in enterprise and government because of its integration with Windows, Active Directory, and Microsoft 365. Primary strength: enterprise integration. Most likely used by large nonprofits, public-sector organizations, and companies already tied to Microsoft agreements.

## Cloud Landscape Question 2

Three reasons the course switched from Azure to Supabase:
- **Access**: Azure requires organizational provisioning (tenant, invitation, auth setup), which can block students for days. Supabase accounts are self-provisioned in under two minutes and the free tier covers the course.
- **Pedagogical fit**: Azure Blob Storage stores data as opaque files. Supabase stores rows and columns in a relational database, which is a more transferable skill and maps directly to how the pipeline in weeks 9-11 needs to work.
- **Pipeline coherence**: The ETL pipeline has a raw zone and an enriched zone, and in Supabase these become two tables with a clear foreign-key relationship. That reinforces data model concepts and makes each stage easy to inspect.

Reflection: When evaluating a cloud tool for a new project, ask what the tool actually teaches you or lets you do, not just what has the biggest brand. A tool that is easy to start with and uses primitives you can carry to other providers is often a better choice than a heavier tool that matches what a big enterprise uses — even if the enterprise tool is the more "professional" one long term.

## Cloud Landscape Question 3

1. Store 10 TB of image files and retrieve by filename from any machine → **object storage**. Provider: AWS S3, GCP Cloud Storage, or Cloudflare R2.
2. Run an ML training job on a GPU for four hours, then shut it down → **compute** (GPU instance) or **ML platform**. Provider: AWS EC2 with GPU instances (like p3.2xlarge), GCP Vertex AI, or Azure Machine Learning.
3. Host a web API that auto-scales up and down → **serverless compute** or **container compute**. Provider: AWS Lambda, GCP Cloud Run, or Azure Functions.
4. Send structured data to an LLM and get text back → **LLM API**. Provider: OpenAI API directly, Azure OpenAI Service, AWS Bedrock, or GCP Vertex AI.

## Cloud Landscape Question 4

Simple project: a hobby dashboard that tracks a few stocks, stores historical prices, and asks an LLM to summarize weekly price movements.

Plausible stack using two providers:
- **Data storage**: Supabase (managed Postgres) for price history and user notes — cheap, simple, relational, easy to query.
- **LLM summary**: OpenAI API directly for the weekly text summary.
- **Optional edge deployment**: Cloudflare Workers to serve the dashboard cheaply at the edge.

Consolidating to one provider would reduce the number of accounts, credentials, and bills to manage, and would give you one support relationship. But you would give up the best tool for each job — Supabase's simple relational DB is better than managing a raw AWS RDS for a small hobby project, and Cloudflare's edge compute is cheaper than an equivalent on AWS. For a small project, mixing is fine. For a larger organization, the consolidation question becomes a real cost and risk tradeoff.