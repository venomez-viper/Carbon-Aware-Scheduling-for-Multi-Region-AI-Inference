# Carbon-Aware Scheduling for Multi-Region AI Inference Workloads: Quantifying Latency and Sustainability Trade-Offs in Cloud Systems

**Authors:** Akash Anipakalu Giridhar, Brandon Youngkrantz, Alexandre Corret, Yogith Ramanan

**Date:** February 2026

***

## Abstract

As AI inference workloads continue to scale across globally distributed cloud infrastructure, the environmental cost of computation has become a pressing concern. Data centers are projected to consume over 500 TWh of energy by 2025, emitting roughly 225 megatons of CO₂. This paper investigates carbon-aware scheduling strategies for routing AI inference requests across multiple cloud regions while balancing latency constraints and sustainability goals. Three scheduling policies are evaluated through simulation — Latency-First (baseline), Carbon-First, and a Hybrid approach — using real-world carbon intensity data from Electricity Maps and inter-region latency measurements. Results from the simulation are evaluated against established benchmarks from CASPER and Microsoft's carbon-aware computing research, providing practical guidance for enterprises seeking to incorporate sustainability into their cloud architecture without violating Service Level Objectives (SLOs).[^1][^2][^3]

***

## 1. Introduction

The rapid growth of artificial intelligence has driven unprecedented demand for cloud computing resources. AI inference — the process of using trained models to generate predictions for live requests — now accounts for a significant and growing share of data center workloads. Unlike training, which is a one-time cost, inference runs continuously and can accumulate substantial energy consumption over a model's lifetime. A single ChatGPT query using GPT-4o consumes approximately 0.3 watt-hours of energy, and when scaled to billions of daily requests, inference workloads represent a significant environmental footprint.[^4][^5]

Different cloud regions have fundamentally different carbon profiles. The carbon intensity of electricity — measured in grams of CO₂ equivalent per kilowatt-hour (gCO₂eq/kWh) — varies dramatically based on the local energy grid's generation mix. For example, Google Cloud's Stockholm region operates at just 3 gCO₂eq/kWh due to hydro and wind power, while the Mumbai region reaches 679 gCO₂eq/kWh because of coal-heavy generation. This means the same computation can produce vastly different emissions depending on where it executes — a variation of more than 200×.[^6]

Simultaneously, cloud regions exhibit different network latency characteristics. Inter-region latencies typically range from 10 to 100 milliseconds, and many AI inference applications operate under strict SLOs that define maximum acceptable response times (e.g., 95% of requests must complete within 100–200 ms). The tension between routing requests to the "greenest" region versus the "fastest" region creates the central trade-off this paper investigates.[^7]

This research addresses three core questions:

- **RQ1:** How do network latency and carbon intensity vary across selected cloud regions relevant for AI inference workloads?
- **RQ2:** What are the trade-offs between latency, SLO violations, and carbon emissions when applying different scheduling strategies?
- **RQ3:** Under what conditions is carbon-aware scheduling practically viable for enterprises?

***

## 2. Background and Related Work

### 2.1 Carbon-Aware Computing

Carbon-aware computing is the practice of aligning workload timing and placement with real-time carbon signals to minimize greenhouse gas emissions. This approach exploits both **spatial variability** (different regions have different carbon intensities at any given time) and **temporal variability** (a single region's carbon intensity fluctuates throughout the day based on renewable energy availability).[^8][^9]

Google pioneered large-scale carbon-aware computing with its Carbon-Intelligent Computing System (CICS), which delays temporally flexible workloads to align with periods of low-carbon energy availability. The system uses day-ahead carbon intensity forecasts from Electricity Maps combined with internal power demand predictions to generate Virtual Capacity Curves (VCCs) that throttle resource availability during high-carbon periods. Google Cloud now uses Electricity Maps data to provide hourly emissions reporting and region-level carbon-free energy (CFE%) metrics to customers.[^9][^10][^11]

Microsoft has also advanced carbon-aware practices. A joint study between Microsoft and UBS demonstrated that time-shifting workloads based on carbon intensity forecasts could reduce Software Carbon Intensity (SCI) by approximately 15%, while location-shifting to appropriate regions could achieve reductions of nearly 75%. The Green Software Foundation's carbon-aware SDK, co-developed with Microsoft, provides practical tools for implementing these strategies.[^3][^12]

### 2.2 Scheduling Frameworks

Several research frameworks have formalized the carbon-latency trade-off in scheduling. **CASPER** (Carbon-Aware Scheduling and Provisioning for Distributed Web Services) formulates the problem as a multi-objective optimization, simultaneously minimizing carbon footprint and server provisioning costs while respecting latency constraints. Evaluated across six AWS regions using real Wikimedia workload traces and Electricity Maps carbon data, CASPER demonstrated up to 70% carbon emission reductions compared to a latency-first baseline, with controllable and often negligible performance degradation.[^2]

**CASA** (Carbon- and SLO-Aware Autoscaling and Scheduling) addresses serverless cloud platforms, reducing the operational carbon footprint of Function-as-a-Service clusters by up to 2.6× while simultaneously reducing SLO violation rates by up to 1.4× compared to state-of-the-art approaches. **LinTS** proposes a Linear Programming-based approach for carbon-optimal temporal scheduling of inter-datacenter data transfers.[^13][^1]

A hybrid carbon- and energy-aware scheduling approach for green cloud computing has demonstrated that integrating real-time carbon intensity data with SLO-aware control mechanisms can achieve a 35% reduction in carbon footprint with only a 5% increase in average job latency.[^14]

### 2.3 AI Inference Characteristics

AI inference workloads have distinct characteristics that influence scheduling decisions. BERT-base model inference on GPU instances (e.g., NVIDIA T4) achieves 2.6–5× lower latency compared to CPU instances, with optimized runtimes offering approximately 1.45× better throughput and 1.13× better latency. Inference latency varies by model complexity, input size, batch size, and hardware configuration.[^15][^16]

The energy consumption per inference request varies widely. Estimates for ChatGPT-like systems range from approximately 0.3 Wh per query for GPT-4o to 0.00289 kWh per request for earlier estimates. For simpler models like BERT, energy consumption per request is significantly lower, making the carbon cost more dependent on grid carbon intensity than absolute energy draw.[^5][^4]

### 2.4 Market Context

The global carbon-aware workload scheduling market reached USD 1.42 billion in 2024 and is projected to grow to USD 8.92 billion by 2033, representing a compound annual growth rate of 21.3%. This growth is driven by increasing regulatory pressure, enterprise sustainability commitments (ESG reporting), and growing recognition that carbon-aware computing can be implemented without significant performance penalties.[^17]

***

## 3. Methodology

### 3.1 Workloads and SLO Definitions

The simulation focuses on AI inference workloads representative of common enterprise deployments:

| Workload | Model Type | Typical Latency Target | SLO Definition |
|----------|-----------|----------------------|----------------|
| Text Classification | BERT-base | 80–120 ms | 95% of requests < 100 ms |
| Question Answering | BERT-large | 100–200 ms | 95% of requests < 150 ms |
| Image Embedding | ResNet-50 (optional) | 50–100 ms | 95% of requests < 80 ms |

These workloads are selected because they represent small-to-medium models commonly deployed for inference, with well-understood latency profiles and meaningful SLO requirements.

### 3.2 Cloud Regions

Five cloud regions are selected to represent diverse latency and carbon intensity profiles:

| Region | Location | Approx. Carbon Intensity (gCO₂eq/kWh) | Carbon Profile |
|--------|----------|----------------------------------------|----------------|
| US-East | N. Virginia | 323 | Mixed grid (gas, nuclear, growing renewables)[^6] |
| US-West | Oregon | 79 | Hydro-dominant, low carbon[^6][^18] |
| EU-West | Frankfurt | 276 | Mixed with significant renewables[^6] |
| EU-North | Finland/Stockholm | 3–39 | Hydro/wind-dominant, ultra-low carbon[^6] |
| Asia | Singapore | 367 | Fossil-fuel heavy, higher carbon[^6] |

These regions capture the key dynamics: US-West and EU-North serve as "green" regions with low carbon intensity; US-East and EU-West are moderate; Singapore represents a high-carbon option with strategic geographic positioning for Asian users.[^18]

### 3.3 Latency Data

Inter-region network latency is modeled using a matrix derived from publicly available measurements (CloudPing and AWS Infrastructure Performance data). Latency values represent round-trip times and include both network propagation delay and processing overhead.[^19][^20]

| User Location → Region | US-East | US-West | EU-West | EU-North | Singapore |
|------------------------|---------|---------|---------|----------|-----------|
| US-East | 5 ms | 65 ms | 85 ms | 95 ms | 230 ms |
| US-West | 65 ms | 5 ms | 145 ms | 155 ms | 170 ms |
| EU | 85 ms | 145 ms | 5 ms | 25 ms | 165 ms |
| Asia | 230 ms | 170 ms | 165 ms | 175 ms | 5 ms |

Inter-region latencies generally fall in the 10–100 ms range for nearby regions, extending to 150–230 ms for cross-continental routes.[^7]

### 3.4 Carbon Intensity Data

Carbon intensity data is sourced from Electricity Maps, which provides real-time and historical hourly carbon intensity measurements for electricity grids worldwide. For the simulation, hourly carbon intensity traces for each selected region are used, capturing both the average levels and temporal variability driven by renewable energy intermittency.[^10][^21][^22]

Key observations from the data:

- Carbon intensity exhibits up to 6× spatial variation between regions (e.g., France/Nordics at ~40 gCO₂eq/kWh vs. Singapore at ~370 gCO₂eq/kWh)[^2]
- Temporal variability within a single region can cause carbon intensity to fluctuate by up to 40% over 24 hours, driven by diurnal renewable generation patterns[^2]
- Regions with high renewable penetration (e.g., California, Germany) show greater temporal variability than fossil-fuel-dominated or hydro-stable regions[^2]

### 3.5 Scheduling Policies

Three scheduling policies are implemented and evaluated:

**Policy 1: Latency-First (Baseline)**
Each request is routed to the region with the lowest network latency from the user's location. This policy represents the default behavior of most cloud load balancers and ignores carbon intensity entirely.

**Policy 2: Carbon-First**
Each request is routed to the region with the lowest current carbon intensity, regardless of latency. This policy maximizes carbon savings but may cause significant SLO violations when the greenest region is geographically distant from the user.

**Policy 3: Hybrid**
This policy combines latency and carbon intensity using a weighted scoring function:

score = α × normalized\_latency + (1 − α) × normalized\_carbon

where α ∈  controls the trade-off between latency optimization (α = 1, equivalent to Latency-First) and carbon optimization (α = 0, equivalent to Carbon-First). Additionally, a constrained variant is tested: among regions satisfying the latency SLO threshold, select the one with the lowest carbon intensity. The parameter α is varied across experiments (0.2, 0.3, 0.5, 0.7) to generate trade-off curves.[^23][^2]

### 3.6 Evaluation Metrics

Each policy is evaluated using the following metrics:

- **Average Latency (ms):** Mean response time across all requests
- **P95 Latency (ms):** 95th percentile latency, capturing tail behavior
- **SLO Violation Rate (%):** Percentage of requests exceeding the defined latency threshold
- **Average Carbon Intensity per Request (gCO₂eq/kWh):** Weighted average carbon intensity of the regions where requests are served
- **Relative Carbon Reduction vs. Baseline (%):** Emission reduction compared to the Latency-First policy

### 3.7 Simulation Design

The simulation generates synthetic request traffic across user locations (US, EU, Asia) following realistic diurnal patterns. For each time step (hourly granularity over a 7-day simulation window), the simulator:

1. Generates incoming requests proportional to regional demand patterns
2. Applies the scheduling policy to determine the target region for each request
3. Computes end-to-end latency (network latency + simulated model inference time)
4. Records the carbon cost of each request using the target region's current carbon intensity
5. Aggregates metrics across the simulation period

This approach follows established simulation methodologies used in carbon-aware scheduling research.[^24][^14][^2]

***

## 4. Experiments and Results

### 4.1 RQ1: Regional Variation in Latency and Carbon Intensity

<!-- ⚠️ SWAP ZONE: Update with actual values from your carbon intensity dataset and latency measurements -->

The selected regions exhibit dramatic variation in both dimensions. EU-North (Finland/Stockholm) operates at 25 gCO₂eq/kWh on average during the simulation period, making it significantly cleaner than Singapore (367 gCO₂eq/kWh) and US-East (323 gCO₂eq/kWh). US-West (Oregon) also stands out as a green option at 79 gCO₂eq/kWh, benefiting from the Pacific Northwest's substantial hydroelectric capacity.[^6][^18]

However, network latency does not favor the same regions. For US-based users, US-East to US-East represents typical intra-area connection delays averaging 5 ms local latency but has moderate-to-high carbon intensity. Routing those requests to EU-North would add approximately 90 ms of latency penalty—a potentially acceptable trade-off for non-real-time workloads but problematic for latency-sensitive applications.

The carbon intensity data confirms a finding from prior literature: executing the same job in different cloud regions can lead to nearly 15× variation in operational carbon emissions across our selected extremes (Singapore vs EU-North). The US average grid carbon intensity stands at approximately 384 gCO₂eq/kWh, while the EU averages 213 gCO₂eq/kWh, underscoring the geographic dimension of the problem.[^25][^2]

<!-- END SWAP ZONE -->

### 4.2 RQ2: Trade-Offs Under Different Scheduling Policies

The simulation results reveal a clear spectrum of trade-offs across the three policies:

<!-- Updated from Simulation Results -->
<!-- Updated from Simulation Results -->
<!-- Updated from Simulation Results -->
| Policy | Avg. Latency (ms) | P95 Latency (ms) | SLO Violation Rate (%) | Avg. Carbon Intensity (gCO₂eq/kWh) | Carbon Reduction (%) |
|--------|-------------------|-------------------|------------------------|------------------------------------|----------------------|
| Latency-First | 37.50 | 79.50 | 0.00% | 268.70 | 0.00% |
| Carbon-First | 132.70 | 225.90 | 71.11% | 24.90 | 90.70% |
| Hybrid (α=0.7) | 62.80 | 127.50 | 1.33% | 117.70 | 56.20% |
| Hybrid (α=0.5) | 101.70 | 201.10 | 48.41% | 37.10 | 86.20% |
| Hybrid (α=0.3) | 102.60 | 201.20 | 51.02% | 35.70 | 86.70% |
| Hybrid (α=0.2) | 102.60 | 201.10 | 51.17% | 35.70 | 86.70% |
| Constrained-Hybrid | 65.90 | 137.60 | 1.01% | 108.60 | 59.60% |

### Visualizing the Trade-offs

*(Below are automatically generated graphical analyses of the simulation results, illustrating the data documented above).*

**Figure 1: Policy Trade-off Analysis**
![Trade-off Curve](Carbon-Aware-Scheduling-for-Multi-Region-AI-Inference/outputs/graphs/tradeoff_curve.png)
*This scatter plot visualizes the core tension: optimizing for carbon reduction correlates directly with increased Service Level Objective (SLO) violations unless constrained architectures are applied.*

**Figure 2: Per-Workload Penalty**
![Workload SLO Violations](Carbon-Aware-Scheduling-for-Multi-Region-AI-Inference/outputs/graphs/workload_slo_violations.png)
*This grouped chart highlights how indiscriminate carbon-first routing punishes strict SLO AI endpoints (like ResNet-50 vs. BERT-large).*

**Figure 3: Simulated Regional Carbon Trace Fluctuations**
![Carbon Traces](Carbon-Aware-Scheduling-for-Multi-Region-AI-Inference/outputs/graphs/carbon_traces.png)
*The temporal and geographical variance of carbon intensity across the grid nodes evaluated in the simulation window.*

<!-- ⚠️ SWAP ZONE: Update these observations with your actual simulation findings -->
<!-- Compare your numbers to CASPER benchmarks below and note similarities/differences -->

These results can be compared with findings from CASPER, which demonstrated that even small latency relaxations (e.g., 20 ms) achieve approximately 25% carbon reduction, while larger relaxations (400–500 ms) reach up to 70% reduction. The diminishing returns beyond α = 0.3 mirror CASPER's finding that Carbon-400 and Carbon-500 policies converge in effectiveness.[^2]

**Key observations:**

<!-- Update the specific numbers below with your simulation output -->

- **The Latency-First policy** delivers the best performance (37.50 ms average, 0.00% SLO violations) but at the highest carbon cost (268.70 gCO₂eq/kWh). This is expected because requests are served locally in high-carbon regions like US-East and Singapore.
- **The Carbon-First policy** achieves the greatest carbon savings (90.70% reduction) but at a drastically high SLO violation rate (71.11%). This policy routes nearly all traffic to EU-North and US-West regardless of user location, creating cross-continental latency penalties that impact model delivery.
- **Hybrid policies** offer the most practical trade-offs. At α = 0.7, carbon emissions drop by an impressive 56.20% while keeping collective SLO violations at merely 1.33%. Our simulation results show that a Constrained-Hybrid policy achieves a 59.60% carbon reduction while maintaining SLO violations below 1.01%. This aligns with CASPER's findings that small latency relaxations (20 ms) achieve approximately 25% carbon reduction, and both studies confirm that the carbon-latency trade-off is manageable in practice.

<!-- END SWAP ZONE -->

### 4.3 RQ3: Viability Conditions for Carbon-Aware Scheduling

<!-- ⚠️ SWAP ZONE: Update findings below with interpretations from your actual simulation results -->

The simulation reveals that the viability of carbon-aware scheduling depends on three key factors:

**1. SLO Strictness**
The trade-off between carbon reduction and SLO compliance varies significantly by workload type. BERT-base requests (SLO: 100ms) achieved 86.20% carbon reduction under Hybrid α=0.5 with 51.70% SLO violations, while BERT-large requests (SLO: 150ms) achieved the same 86.20% network-wide carbon reduction profile with only 40.23% violations. This difference in carbon savings with equivalent violation rates demonstrates that workload-specific SLO tuning is essential for practical carbon-aware scheduling. Workloads with tight SLOs (< 80 ms) benefit primarily from conservative hybrid policies (α ≥ 0.7), yielding 56.2% to 59.6% carbon reductions seamlessly across our test bed. Workloads with relaxed SLOs (> 150 ms) can use more aggressive tuning limits.

**2. User Geographic Distribution**
The benefit of carbon-aware scheduling is greatest when users are geographically distributed and when low-carbon regions exist near major user populations.

**3. Carbon Intensity Contrast Between Regions**
Carbon-aware scheduling is most effective when there is substantial variation in carbon intensity across available regions. The 15× variation observed across the selected regions creates significant optimization opportunity. If all regions had similar carbon intensities, spatial shifting would provide minimal benefit, and temporal shifting would become the primary strategy.[^26][^2]

<!-- END SWAP ZONE -->

### 4.4 Constrained Hybrid Policy

<!-- ⚠️ SWAP ZONE: Update with your constrained hybrid results -->

The constrained variant — selecting the lowest-carbon region among those meeting the SLO threshold — proves highly effective in the simulation. By pre-filtering regions that satisfy latency requirements, this policy bounds the absolute negative outcome of spatial shifting, producing zero major deviations over time to keep its aggregate SLO violation rate tightly bound at 1.01% while still capturing sweeping 59.60% carbon reductions across the userbase. This approach mirrors CASPER's finding that combining hard latency constraints with carbon optimization delivers the best practical outcomes.[^2]

<!-- END SWAP ZONE -->

***

## 5. Discussion

### 5.1 Enterprise Implications

The results carry several practical implications for cloud architects and enterprises:

**Carbon-aware routing is low-hanging fruit.** Conservative hybrid policies (α = 0.5–0.7) deliver meaningful carbon reductions (20–46%) with minimal performance impact. Organizations can implement these policies in existing load balancers with relatively simple modifications — no new infrastructure is required.[^8]

**Location selection matters more than timing.** Microsoft's research found that choosing an appropriate cloud region can achieve up to 75% SCI reduction, far exceeding the ~15% from time-shifting alone. This finding is corroborated by the simulation results: spatial shifting to green regions dominates the carbon savings.[^3]

**The market is ready.** Cloud providers are increasingly surfacing carbon data to customers. Google Cloud publishes CFE% and grid carbon intensity for all regions, ElectricityMaps provides real-time API access to carbon data, and AWS publishes sustainability resources to help customers make informed decisions. The tools and data needed for carbon-aware scheduling are available today.[^27][^22][^6]

### 5.2 Limitations

This study has several limitations:

- **Simulation vs. Production:** The evaluation uses simulated request routing rather than live production traffic. Real deployments face additional complexities including data sovereignty requirements (GDPR, HIPAA), cold-start latencies, and variable server load.[^2]
- **Scope 2 only:** The analysis focuses on operational carbon emissions (Scope 2) and does not account for embodied emissions from server manufacturing (Scope 3), which can represent a significant portion of total lifecycle emissions.[^28]
- **Simplified inference model:** The simulation assumes uniform inference times across regions. In practice, inference latency depends on hardware heterogeneity, GPU availability, and batch scheduling.
- **Limited regions:** Only five regions are evaluated. Production deployments across 20+ regions would create more optimization opportunities but also more complexity.

### 5.3 Future Work

Several extensions could strengthen and generalize these findings:

- **Temporal + spatial co-optimization:** Combining location-shifting with time-shifting strategies could yield additional carbon savings, particularly for batch or deferrable inference workloads.[^26][^3]
- **Dynamic alpha adjustment:** Adapting the hybrid policy's α parameter in real-time based on current carbon intensity conditions, load patterns, and SLO headroom.
- **Multi-cloud strategies:** Extending the simulation across providers (AWS, Azure, GCP) to exploit cross-provider carbon intensity differences.
- **Reinforcement learning schedulers:** Training adaptive policies that learn optimal routing strategies from historical carbon and latency data.
- **Embodied carbon integration:** Incorporating Scope 3 emissions into the scheduling objective, following frameworks like Microsoft's GreenSKU approach which reduces carbon per core by 15–28%.[^29]

***

## 6. Conclusion

<!-- ⚠️ SWAP ZONE: Update conclusion numbers with your final simulation results -->

This research demonstrates that carbon-aware scheduling for multi-region AI inference workloads is both technically feasible and practically valuable. The three-policy evaluation framework — Latency-First, Carbon-First, and Hybrid — reveals a clear trade-off spectrum, with hybrid approaches offering the most actionable path for enterprises. A constrained-hybrid policy formulation achieves 59.60% carbon reduction with only 1.01% SLO violations, representing a compelling sustainability gain at practically unnoticeable performance cost.

The geographic variation in carbon intensity across cloud regions — ranging from 25 gCO₂eq/kWh in Nordic regions to 367 gCO₂eq/kWh in high-carbon grids like Singapore — provides substantial optimization potential. Even conservative policies that preserve strict SLO compliance can reduce carbon emissions by over 50%, making carbon-aware scheduling an accessible sustainability tool for any organization operating multi-region cloud infrastructure.[^6]

The findings align with and extend prior work from Google, Microsoft, and academic frameworks like CASPER and CASA, confirming that the carbon-latency trade-off is manageable and that meaningful emission reductions are achievable without sacrificing user experience. As the carbon-aware workload scheduling market grows toward USD 8.92 billion by 2033, adopting these strategies represents both an environmental responsibility and an emerging competitive advantage.[^17][^9][^13][^3][^2]

<!-- END SWAP ZONE -->

***

## References and Data Sources

**Carbon Intensity Data:** Electricity Maps (electricitymaps.com), Google Cloud Carbon Data[^22][^6]

**Latency Data:** CloudPing (cloudping.co), AWS Infrastructure Performance[^20][^19]

**Key Research:**
- CASPER: Souza et al., "Carbon-Aware Scheduling and Provisioning for Distributed Web Services," IGSC 2023[^2]
- CASA: Qi et al., "A Framework for SLO and Carbon-Aware Autoscaling and Scheduling," 2024[^13]
- Google CICS: Radovanovic et al., "Carbon-Aware Computing for Datacenters," 2021[^9]
- Microsoft Carbon-Aware Whitepaper, 2023[^3]
- Wiesner et al., "How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud," 2021[^26]
- Lacoste et al., "Measuring the Carbon Intensity of AI in Cloud Instances," 2022[^30]

**Tools:** Python, NumPy, Pandas, Matplotlib, Seaborn, Jupyter Notebooks

---

## References

1. [[PDF] Carbon-Aware Temporal Data Transfer Scheduling Across Cloud ...](https://arxiv.org/pdf/2506.04117.pdf) - The datacenters are projected to consume over 500 TWh of energy in 2025, emitting roughly 225 metric...

2. [[PDF] CASPER: Carbon-Aware Scheduling and Provisioning for ...](http://www.ecs.umass.edu/irwin/casper.pdf) - This section provides background information on the grid carbon intensity, cloud model, carbon-aware...

3. [[PDF] Carbon-aware computing - the white paper](https://msftstories.thesourcemediaassets.com/sites/418/2023/01/carbon_aware_computing_whitepaper.pdf) - As the study showed, the highest carbon-aware reductions are to be gained from time- shifting comput...

4. [How much energy does ChatGPT use? - Epoch AI](https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use) - We find that typical ChatGPT queries using GPT-4o likely consume roughly 0.3 watt-hours, which is te...

5. [How Much Energy Will It Take To Power AI? - Contrary Research](https://research.contrary.com/foundations-and-frontiers/ai-inference) - ... energy consumption of 0.00289 kilowatt-hours per request? That would require 26,010,000 kilowatt...

6. [Carbon free energy for Google Cloud regions](https://cloud.google.com/sustainability/region-carbon) - Grid carbon intensity (gCO2eq/kWh): This metric indicates the average operational gross emissions pe...

7. [Multi-AZ vs. Multi-Region in the Cloud - FlashGrid.io](https://www.flashgrid.io/news/multi-az-vs-multi-region-in-the-cloud/) - Network latency between availability zones usually varies between 0.3 and 2 ms (milliseconds) depend...

8. [Carbon-Aware Scheduling - Emergent Mind](https://www.emergentmind.com/topics/carbon-aware-scheduling) - Carbon-aware scheduling is the practice of aligning workload timing and placement with real-time car...

9. [Carbon-Aware Computing for Datacenters | Green AI](https://luiscruz.github.io/green-ai/publications/2021-07-radovanovic-carbon.html) - This paper presents Google's Carbon-Intelligent Computing System (CICS), that uses carbon-intensity ...

10. [Google Cloud reports its carbon footprint to its customers](https://www.electricitymaps.com/technology/google-cloud) - Google Cloud uses Electricity Maps to provide hourly emissions data for its Carbon Footprint Report....

11. [Our data centers now work harder when the sun shines and wind ...](https://blog.google/innovation-and-ai/infrastructure-and-cloud/global-network/data-centers-work-harder-sun-shines-wind-blows/) - The first version of this carbon-intelligent computing platform focuses on shifting tasks to differe...

12. [Carbon-aware computing: Measuring and reducing the ...](https://news.microsoft.com/de-ch/2023/01/10/carbon-aware-computing-whitepaper/) - A foundational principle of Green Software is known as carbon-aware computing, which involves shifti...

13. [[2409.00550] CASA: A Framework for SLO and Carbon-Aware ...](https://arxiv.org/abs/2409.00550) - Experimental results indicate that CASA reduces the operational carbon footprint of a FaaS cluster b...

14. [[PDF] Energy-aware workload scheduling in snowflake for sustainable big ...](https://wjaets.com/sites/default/files/fulltext_pdf/WJAETS-2025-0717.pdf) - By aligning workload scheduling with grid carbon intensity and energy-efficiency considerations, thi...

15. [Speed up model inference with Vertex AI Predictions' optimized ...](https://cloud.google.com/blog/topics/developers-practitioners/speed-model-inference-vertex-ai-predictions-optimized-tensorflow-runtime) - For the BERT base model the optimized TensorFlow runtime offers approximately 1.45 times better thro...

16. [BERT inference on G4 instances using Apache MXNet and GluonNLP](https://aws.amazon.com/blogs/machine-learning/bert-inference-on-g4-instances-using-apache-mxnet-and-gluonnlp-1-million-requests-for-20-cents/) - When running inference on the BERT-base model, the g4dn.xlarge GPU instance achieves between 2.6–5 t...

17. [Carbon‑Aware Workload Scheduling Market Research Report 2033](https://dataintelo.com/report/carbonaware-workload-scheduling-market) - According to our latest research, the global Carbon‑Aware Workload Scheduling market size has reache...

18. [How to Choose AWS Regions for Lower Carbon Footprint](https://oneuptime.com/blog/post/2026-02-12-choose-aws-regions-for-lower-carbon-footprint/view) - A data-driven guide to selecting AWS regions based on carbon intensity to minimize the environmental...

19. [How Infrastructure Performance for AWS Network Manager works](https://docs.aws.amazon.com/network-manager/latest/infrastructure-performance/how-nmip-works.html) - Inter-Region latency metrics, generated by aggregating latency measurements from probes located acro...

20. [AWS Region Latency Matrix](https://www.cloudping.co) - AWS Region Latency Matrix ; af-south-1. 1.29ms. 301.72ms ; ap-east-1. 302.82ms. 1.49ms ; ap-east-2. ...

21. [Google data centers shift their computations to cleaner times and ...](https://www.electricitymaps.com/technology/google-data-centers) - Google uses Electricity Maps data to calculate the carbon intensity of the electricity powering Goog...

22. [Electricity Maps | The world's most comprehensive electricity data ...](https://www.electricitymaps.com) - Electricity Maps provides global access to electricity mix, prices, load, and carbon intensity. Avai...

23. [Research-Topic.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/127698555/c8770ae9-763d-4ce1-8095-618c7ea2a624/Research-Topic.pdf?AWSAccessKeyId=ASIA2F3EMEYESYP7A7NT&Signature=HThF6mh0BweEDAwKD02MfkFuT50%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEO3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQC0YQvT6Y9DRcQvcDZYyQ6PH7j63P3UXE%2Fbch2KdvqTFQIgLlSSp8KXjaBovTH%2Fs%2BGkgqkn9dJjXQjRQLtP2x56phUq%2FAQItf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDI5wg4wit8iBFOUlaSrQBJ4S%2F4S6inch%2Byw6wmjQXDKuGjPpynIf8i1cE%2B0OwhnaCaq7yPQw07rQErqvnn5yLnspvcEFNEtKoecoqv%2FmYw8BH0XF4s%2FsCnf%2BvKHziHT%2BtpFZT1gLWscqhfrLlIadXRtgVp6kzbd8JmtIbFfQSoELivFHpI7mQTuA97S0bct4Jtp3MYXVMaqhVf6kFhBrszxyAysA%2BvsdJDN6e5%2FgJWtbWZTrF%2B24a8yCAlbKEgF9xmz1MlHH2hh2AZCTntYi7Wo%2BNF8tSK1oj5YDZI0N2W16wCTNQbwS6eL7XF8b7IKPHTXlLtfihaQdbI61K0rbJIoGOMl6X%2FxUGSQV3Qj7wC7jdqk5lmv0U4VTD8SARBZ1Cm1STJarBs0T6koegbNcYzClJ0Pea1APS0rxeulVyGCi9ifYQjSbW4pT2LaTT3e2zJ0Wnn6VBN0wy5MMPhyIy1I8IeeVLArf6sPu0vSpmxEzyt%2BgQA6YeObKy8lyyVoj%2BTu7AabCiL2DKEut89q9TVtyTH9pvw2cHYPoq17TEJw3jAYWST4M331dCBRTesLp0R8hEtHcBM1e142JZnhU%2F7nVBCR%2B4fSBJx5r1PKrvMYHK%2FTBl4xGS4lkQFbijWqfBl4Hka2pTLRwpgTR2IUnZwAtk%2B75Mg577bhv2PHiaevg2o37zMgJ2iKoEGdNaNDy4tLI%2FC0kzQegQoiolcSYqVrY%2B86jntfpp%2Bbq2TWY9beR%2B7s0z7IscVEw8LU3aRXCi1YtlivH79GZxNjAw5TUp%2F4sYVC%2BasbDbJfex8fFBKkw1afozAY6mAFbXOaO73COsIGMA8j7cqfmG2HYeh0s26SynLsxyEHdp82yhcdnKPR31vqCKaoXgMpnkqTZfEKs8nELQ%2F3WpYlm4iw3IdVx4ZSUkrAyOwdOXkczmAkTsUOUWalnh%2BxgCakoXLBTKmPPD%2BLolps%2BdyNKpG0%2BoZhB2M0LN43ikoWPDnV5R8U82yVsiy4nUW%2B8JDpzWGkg0Yc5hQ%3D%3D&Expires=1771710626) - Research Project Scope Document                                                                     ...

24. [[PDF] Simulation framework for carbon-aware workload shifting in the cloud](http://www.diva-portal.org/smash/get/diva2:1942206/FULLTEXT02.pdf) - Carbon-aware workload shifting algorithms are a method to utilize the global scale and flexibility o...

25. [Major Countries and Regions - Global Electricity Review 2025 | Ember](https://ember-energy.org/latest-insights/global-electricity-review-2025/major-countries-and-regions/) - The carbon intensity of US electricity generation was 384 gCO2/kWh ... The carbon intensity of elect...

26. [[PDF] How Temporal Workload Shifting Can Reduce Carbon Emissions in ...](https://arxiv.org/pdf/2110.13234.pdf) - ABSTRACT. Depending on energy sources and demand, the carbon intensity of the public power grid fluc...

27. [[PDF] How moving onto the AWS cloud reduces carbon emissions](https://sustainability.aboutamazon.com/carbon-reduction-aws.pdf) - The Software Carbon Intensity (SCI) specification from the Green Software Foundation (GSF) aims to a...

28. [Microsoft quantifies environmental impacts of datacenter ...](https://news.microsoft.com/source/features/sustainability/microsoft-quantifies-environmental-impacts-of-datacenter-cooling-from-cradle-to-grave-in-new-nature-study/) - Microsoft developed a new approach to assess the carbon, water and energy impacts of different cooli...

29. [[PDF] Designing Cloud Servers for Lower Carbon](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/03/2024-GreenSKU-ISCA2024.pdf) - They are: (1) a target data center workload, represented as a record of VM deployments over a time p...

30. [[PDF] Measuring the Carbon Intensity of AI in Cloud Instances - arXiv](https://arxiv.org/pdf/2206.05229.pdf) - We provide measurements of operational software carbon intensity for a set of modern models covering...

